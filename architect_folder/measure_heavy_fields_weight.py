"""
A1, риск раздела 11: "Тяжёлые поля весят больше ожидаемого — замер на
десяти вопросах перед A4." Генерирует ~10 примерных RAG-промптов через
подтверждённый генератор (verify_generator_backend.py дал PASS), извлекает
ровно те поля, что заморожены в dump_schema_v1.md (`hidden_states_last`,
`attention_by_group`, `attention_by_passage`), пишет JSONL и меряет
реальный байт-размер — не оценку "на глаз".

Промпты — не из настоящего датасета (он ещё не выбран, зависимость B4),
а представительные заглушки: важен объём на токен и на вопрос, а не
содержание. Экстраполирует на размер пилота (300 dev, ещё ~300 test).

Запуск (Kaggle/HPC, нужен GPU):
    python measure_heavy_fields_weight.py --model meta-llama/Llama-3.1-8B-Instruct
    python measure_heavy_fields_weight.py --model hf-internal-testing/tiny-random-LlamaForCausalLM --dtype float32
        (второй вариант — быстрая проверка логики на CPU, без реальной модели)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_PASSAGES = 3
MAX_NEW_TOKENS = 50

EXAMPLE_QUESTIONS = [
    ("What is the capital of France?",
     ["Paris is the capital and most populous city of France.",
      "France is a country in Western Europe with several overseas territories.",
      "The Eiffel Tower is located in Paris, completed in 1889."]),
    ("How does photosynthesis work?",
     ["Photosynthesis is the process by which green plants convert light energy into chemical energy.",
      "Chlorophyll absorbs sunlight, primarily in the blue and red wavelengths.",
      "The process produces oxygen as a byproduct and consumes carbon dioxide."]),
    ("Who wrote the novel Moby Dick?",
     ["Moby-Dick is a novel by American writer Herman Melville, published in 1851.",
      "The novel is considered a leading work of American Romanticism.",
      "It tells the story of Captain Ahab's obsessive quest for revenge on a white whale."]),
    ("What causes earthquakes?",
     ["Earthquakes are caused by the sudden release of energy in the Earth's crust.",
      "Most earthquakes occur along tectonic plate boundaries due to friction and stress buildup.",
      "The point where an earthquake originates underground is called the hypocenter."]),
    ("What is the boiling point of water at sea level?",
     ["Water boils at 100 degrees Celsius (212 Fahrenheit) at standard atmospheric pressure.",
      "Boiling point decreases with altitude due to lower atmospheric pressure.",
      "At the summit of Mount Everest, water boils at around 71 degrees Celsius."]),
    ("Explain how vaccines work.",
     ["Vaccines train the immune system to recognize and fight specific pathogens.",
      "They typically contain a weakened or inactive part of a pathogen, or genetic instructions for it.",
      "After vaccination, the immune system can respond faster if exposed to the real pathogen."]),
    ("What is the largest planet in the solar system?",
     ["Jupiter is the largest planet in the Solar System by both mass and volume.",
      "It is a gas giant composed primarily of hydrogen and helium.",
      "Jupiter has at least 95 known moons, including the four large Galilean moons."]),
    ("What is machine learning?",
     ["Machine learning is a field of AI focused on algorithms that learn patterns from data.",
      "It includes supervised, unsupervised, and reinforcement learning paradigms.",
      "Applications include image recognition, natural language processing, and recommendation systems."]),
    ("Why is the sky blue?",
     ["The sky appears blue due to Rayleigh scattering of sunlight by the atmosphere.",
      "Shorter (blue) wavelengths of light are scattered more than longer (red) wavelengths.",
      "At sunset, light travels through more atmosphere, scattering away blue and leaving red/orange."]),
    ("What is the function of the human liver?",
     ["The liver filters blood, metabolizes nutrients, and detoxifies harmful substances.",
      "It produces bile, which aids in digestion of fats.",
      "The liver also stores glycogen and plays a role in blood clotting."]),
]

INSTRUCTION = "Using the context passages provided below, answer the question concisely."


def build_prompt_ids(tokenizer, question: str, passages: list[str]):
    """Собирает промпт вручную по кускам, чтобы точно знать token-спаны
    (как prompt_spans в dump_schema_v1.md), а не восстанавливать их
    регулярками постфактум."""
    def toks(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    ids = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)

    instr_ids = toks(f"Instruction: {INSTRUCTION}\n\n")
    a = len(ids)
    ids += instr_ids
    b = len(ids)

    passage_spans = []
    for i, p in enumerate(passages):
        p_ids = toks(f"Passage {i + 1}: {p}\n")
        c = len(ids)
        ids += p_ids
        d = len(ids)
        passage_spans.append((c, d))

    q_ids = toks(f"\nQuestion: {question}\nAnswer:")
    e = len(ids)
    ids += q_ids
    f = len(ids)

    spans = {"instruction": (a, b), "passages": passage_spans, "question": (e, f)}
    return ids, spans


def group_for_position(pos: int, prompt_len: int, spans: dict) -> int:
    """0=instruction, 1=context, 2=question, 3=generated."""
    if pos >= prompt_len:
        return 3
    a, b = spans["instruction"]
    if a <= pos < b:
        return 0
    e, f = spans["question"]
    if e <= pos < f:
        return 2
    return 1  # пассажи и разделители между ними


def passage_index_for_position(pos: int, spans: dict):
    for i, (c, d) in enumerate(spans["passages"]):
        if c <= pos < d:
            return i
    return None


@torch.no_grad()
def run_one(model, tokenizer, question: str, passages: list[str], device: str) -> dict:
    prompt_ids, spans = build_prompt_ids(tokenizer, question, passages)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        output_attentions=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )

    n_generated = out.sequences.shape[1] - prompt_len
    n_passages = len(spans["passages"])

    hidden_states_last = []
    attention_by_group = []
    attention_by_passage = []

    for t in range(n_generated):
        h_last_layer = out.hidden_states[t][-1]  # (batch, seq_at_step, hidden)
        hidden_states_last.append(h_last_layer[0, -1, :].float().cpu().numpy().tolist())

        a_last_layer = out.attentions[t][-1]  # (batch, heads, query_len, key_len)
        a_avg_heads = a_last_layer[0].mean(dim=0)  # (query_len, key_len)
        a_row = a_avg_heads[-1, :].float().cpu().numpy()  # (key_len,)
        key_len = a_row.shape[0]

        group_sums = [0.0, 0.0, 0.0, 0.0]
        passage_sums = [0.0] * n_passages
        for pos in range(key_len):
            g = group_for_position(pos, prompt_len, spans)
            group_sums[g] += float(a_row[pos])
            if g == 1:
                pidx = passage_index_for_position(pos, spans)
                if pidx is not None:
                    passage_sums[pidx] += float(a_row[pos])

        attention_by_group.append(group_sums)
        attention_by_passage.append(passage_sums)

    return {
        "question": question,
        "n_prompt_tokens": prompt_len,
        "n_generated_tokens": n_generated,
        "n_passages": n_passages,
        "hidden_states_last": hidden_states_last,
        "attention_by_group": attention_by_group,
        "attention_by_passage": attention_by_passage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--n-questions", type=int, default=len(EXAMPLE_QUESTIONS))
    parser.add_argument("--out", type=Path, default=Path("heavy_fields_sample.jsonl"))
    parser.add_argument("--pilot-size", type=int, default=600, help="dev+test, см. A1_decisions.md")
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    print(f"Loading {args.model} (dtype={args.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device

    records = []
    for question, passages in EXAMPLE_QUESTIONS[: args.n_questions]:
        rec = run_one(model, tokenizer, question, passages, device)
        records.append(rec)
        print(f"  {question[:50]:50s} prompt={rec['n_prompt_tokens']:4d} tok, "
              f"generated={rec['n_generated_tokens']:3d} tok")

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    total_bytes = args.out.stat().st_size
    n = len(records)
    avg_bytes = total_bytes / n
    total_gen_tokens = sum(r["n_generated_tokens"] for r in records)
    avg_bytes_per_gen_token = total_bytes / total_gen_tokens

    print(f"\n=== Замер веса тяжёлых полей ({args.model}) ===")
    print(f"Записей: {n}, суммарно сгенерированных токенов: {total_gen_tokens}")
    print(f"Файл: {args.out} ({total_bytes / 1e6:.2f} МБ)")
    print(f"Среднее на запись: {avg_bytes / 1e6:.3f} МБ")
    print(f"Среднее на сгенерированный токен: {avg_bytes_per_gen_token / 1e3:.2f} КБ")
    print(f"\nЭкстраполяция на пилот ({args.pilot_size} вопросов, "
          f"это только rag-ветка; closed_book без hidden_states/attention по схеме):")
    print(f"  {avg_bytes * args.pilot_size / 1e9:.2f} ГБ")


if __name__ == "__main__":
    sys.exit(main())
