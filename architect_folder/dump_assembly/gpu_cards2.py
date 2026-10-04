"""
GPU-пачка №2 карточек банка B (researcher_folder/B2_cards.md), без новых
генераций ответов — поверх перефразов, уже собранных augment_perturbations:

  memory-strength-paraphrase (ACL 2025, «Memory Strength and Evidence Style»):
      сила параметрической памяти = согласованность closed-book ответов на
      исходный вопрос и его перефразы. Формализация — наша: NLI-кластеры
      ответов (как semantic entropy, dump_assembly/semantic_clusters.py) и
      дискретная энтропия кластеров. mem_entropy_greedy — жадные ответы
      (оригинал + перефразы), mem_entropy_all — плюс сэмплы.
  semantic-reformulation-entropy: то же для rag-ответов, у каждого перефраза
      свой ретривал (смесь «модель нестабильна» и «ретривер нестабилен» —
      это и есть перенос из карточки). Только short-form.
  context-sufficiency-self-eval: генератор отвечает на вопрос «достаточно ли
      пассажей для ответа» (Yes/No); сигнал — P(Yes) / (P(Yes) + P(No)) на
      первом токене ответа, один forward без генерации. Вторая часть
      карточки («какой пассаж использован») не реализована.

Кластеризация везде условлена на исходный вопрос (перефразы эквивалентны ему).

Запуск:
    python -m dump_assembly.gpu_cards2 extract --dump dump_pilot_v4.jsonl --out gpu_cards2_input.jsonl
    python -m dump_assembly.gpu_cards2 run --input gpu_cards2_input.jsonl --out gpu_cards2.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .semantic_clusters import DEFAULT_NLI, DebertaNLI, cluster_samples

LONGFORM = "franq_longform"
DEFAULT_GENERATOR = "meta-llama/Llama-3.1-8B-Instruct"
SUFFICIENCY_PROMPT = (
    "Passages:\n{passages}\n\nQuestion: {question}\n\n"
    "Do the passages above contain enough information to answer the question? Answer Yes or No."
)


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def cluster_entropy(ids: list[int]) -> float:
    if len(ids) < 2:
        return float("nan")
    c = np.array(list(Counter(ids).values()), dtype=float)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def answer_sets(rec: dict) -> dict[str, list[str]]:
    paras = rec["paraphrases"]
    cb_greedy = [rec["cb_answer"]] + [p["cb_answer"] for p in paras if p.get("cb_answer")]
    cb_all = cb_greedy + rec["cb_samples"] + [s for p in paras for s in p.get("cb_samples") or []]
    out = {"mem_greedy": cb_greedy, "mem_all": cb_all}
    if rec["source"] != LONGFORM:
        rag_greedy = [rec["rag_answer"]] + [p["rag_answer"] for p in paras if p.get("rag_answer")]
        rag_all = rag_greedy + rec["rag_samples"] + [s for p in paras for s in p.get("rag_samples") or []]
        out.update(sre_greedy=rag_greedy, sre_all=rag_all)
    return out


def paraphrase_signals(rec: dict, nli) -> dict[str, float]:
    sets = answer_sets(rec)
    names = {"mem_greedy": "mem_entropy_greedy", "mem_all": "mem_entropy_all",
             "sre_greedy": "sre_entropy_greedy", "sre_all": "sre_entropy_all"}
    out = {v: float("nan") for v in names.values()}
    for key, answers in sets.items():
        answers = [a for a in answers if a and a.strip()]
        if len(answers) >= 2:
            out[names[key]] = cluster_entropy(cluster_samples(rec["question"], answers, nli))
    return out


class Sufficiency:
    def __init__(self, model_name: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype),
                                                          device_map="auto").eval()
        self.yes = self._first_ids(["Yes", " Yes", "yes"])
        self.no = self._first_ids(["No", " No", "no"])

    def _first_ids(self, words: list[str]) -> list[int]:
        return sorted({self.tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in words})

    def __call__(self, question: str, passages: list[str]) -> float:
        text = SUFFICIENCY_PROMPT.format(
            passages="\n".join(f"Passage {i + 1}: {p}" for i, p in enumerate(passages)), question=question)
        rendered = self.tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                      add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")["input_ids"].to(self.model.device)
        with self.torch.no_grad():
            probs = self.model(input_ids=ids).logits[0, -1].float().softmax(-1)
        p_yes, p_no = float(probs[self.yes].sum()), float(probs[self.no].sum())
        return p_yes / (p_yes + p_no) if p_yes + p_no > 0 else float("nan")


def cmd_extract(args) -> None:
    n = 0
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.dump):
            paras = r.get("perturbations", {}).get("query_paraphrases", [])
            fout.write(json.dumps({
                "qid": r["qid"], "source": r["source"], "question": r["question"],
                "passages": [p["text"] for p in r.get("passages", [])],
                "rag_answer": r["rag"]["answer"], "rag_samples": r["rag"].get("samples", []),
                "cb_answer": r["closed_book"]["answer"], "cb_samples": r["closed_book"].get("samples", []),
                "paraphrases": [{k: p.get(k) for k in ("text", "cb_answer", "cb_samples", "rag_answer", "rag_samples")}
                                for p in paras],
            }) + "\n")
            n += 1
    print(f"Готово: {n} записей -> {args.out}")


def cmd_run(args) -> None:
    done = {j["qid"] for j in iter_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in iter_jsonl(args.input) if r["qid"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} уже посчитано, к обработке {len(todo)}")
    if not todo:
        return
    nli = DebertaNLI(args.nli_model, args.batch_size)
    suff = None if args.skip_sufficiency else Sufficiency(args.generator, args.dtype)
    with open(args.out, "a") as fout:
        for n, r in enumerate(todo):
            sig = paraphrase_signals(r, nli)
            sig["ctx_sufficiency"] = suff(r["question"], r["passages"]) if suff and r["passages"] else float("nan")
            fout.write(json.dumps({"qid": r["qid"], "models": {"nli": args.nli_model, "generator": args.generator},
                                   "signals": sig}) + "\n")
            if (n + 1) % 25 == 0:
                fout.flush()
                print(f"  {n + 1}/{len(todo)}")
    print(f"Готово -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("run")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nli-model", default=DEFAULT_NLI)
    p.add_argument("--generator", default=DEFAULT_GENERATOR)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-sufficiency", action="store_true", help="без генератора (только NLI-карточки)")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_run)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
