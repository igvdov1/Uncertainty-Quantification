"""
Перефразы вопросов (C9 §1) — отдельный шаг ДО ретрива, чтобы ретрив по
исходным вопросам и по перефразам шёл одним проходом по корпусу.

Старый generate_paraphrases (retrieval.py) подавал сырой текст без
chat-шаблона и без стоп-условия: Instruct-модель продолжала текст — в
перефраз попадали остатки инструкции, несколько вариантов подряд и сам
ответ («The final answer is: Vladivostok»). Здесь:
  - chat-шаблон + few-shot диалог, генерация до конца реплики (<|eot_id|>);
  - берётся первая строка;
  - фильтр clean_paraphrase (одна строка, «?» в конце, нет служебных слов,
    нет gold-ответа, которого нет в исходном вопросе, не совпадает с
    исходным, длина в разумных пределах);
  - кандидаты генерируются раундами по n_candidates, пока не наберётся
    n_keep разных прошедших фильтр (не больше max_rounds раундов).

Запуск (из architect_folder/):
    python -m dump_assembly.paraphrases generate --questions questions.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct --out paraphrases.jsonl
    python -m dump_assembly.paraphrases retrieval-input --questions questions.jsonl \
        --paraphrases paraphrases.jsonl --out retrieval_queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .labeling import normalize_answer_v3
from .resume import completed_qids

N_KEEP = 3
N_CANDIDATES = 6
MAX_ROUNDS = 3
MAX_NEW_TOKENS = 40

INSTRUCTION = ("Rewrite the question so that it asks for exactly the same thing in different words. "
               "Do not answer it. Output only the rewritten question on a single line.")
FEW_SHOT = [
    ("who wrote the novel war and peace", "Which author wrote the novel War and Peace?"),
    ("In which country is the city of Timbuktu?", "Timbuktu is a city located in which country?"),
    ("what is the capital of the state of oregon", "Which city serves as the capital of Oregon?"),
]

_BAD_WORDS = re.compile(r"\b(answer|answers|rewrite|rewritten|step|question|questions|option|options)\b", re.I)
_LIST_MARKERS = re.compile(r"\(or\)|(^|\s)[A-D][).:]\s|\b[0-9]\)\s|\|\|")


def chat_messages(question: str) -> list[dict]:
    msgs = []
    for q, p in FEW_SHOT:
        msgs.append({"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {q}"})
        msgs.append({"role": "assistant", "content": p})
    msgs.append({"role": "user", "content": f"{INSTRUCTION}\n\nQuestion: {question}"})
    return msgs


def clean_paraphrase(text: str, question: str, gold_answers: list[str]) -> tuple[str | None, str]:
    """(перефраз, "ok") или (None, причина отказа)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return None, "empty"
    p = lines[0].strip().strip('"').strip()
    if len(lines) > 1:
        return None, "multiline"
    if not p.endswith("?") or p.count("?") > 1:
        return None, "question_mark"
    if _BAD_WORDS.search(p) or _LIST_MARKERS.search(p):
        return None, "service_words"
    p_norm, q_norm = normalize_answer_v3(p), normalize_answer_v3(question)
    if p_norm == q_norm:
        return None, "same_as_question"
    n_p, n_q = len(p_norm.split()), max(len(q_norm.split()), 1)
    if not (0.5 <= n_p / n_q <= 2.5):
        return None, "length"
    padded_p, padded_q = f" {p_norm} ", f" {q_norm} "
    for g in gold_answers:
        g_norm = normalize_answer_v3(g)
        if len(g_norm) >= 2 and f" {g_norm} " in padded_p and f" {g_norm} " not in padded_q:
            return None, "gold_leak"
    return p, "ok"


def select_paraphrases(candidates: list[str], question: str, gold_answers: list[str],
                       n_keep: int = N_KEEP, kept: list[str] | None = None,
                       reasons: Counter | None = None) -> list[str]:
    kept = list(kept or [])
    reasons = reasons if reasons is not None else Counter()
    seen = {normalize_answer_v3(k) for k in kept}
    for c in candidates:
        p, why = clean_paraphrase(c, question, gold_answers)
        if p is not None and normalize_answer_v3(p) in seen:
            p, why = None, "duplicate"
        reasons[why] += 1
        if p is not None and len(kept) < n_keep:
            kept.append(p)
            seen.add(normalize_answer_v3(p))
    return kept


def generate_candidates(model, tokenizer, question: str, device, n: int = N_CANDIDATES) -> list[str]:
    import torch
    # tokenize=False + отдельная токенизация: в разных версиях transformers apply_chat_template
    # с return_tensors возвращает то тензор, то BatchEncoding
    text = tokenizer.apply_chat_template(chat_messages(question), add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    # конец реплики: <|eot_id|> у Llama-3, <|im_end|> у Qwen — что есть в словаре
    stop = {tokenizer.eos_token_id} | {tokenizer.convert_tokens_to_ids(t) for t in ("<|eot_id|>", "<|im_end|>")
                                       if t in tokenizer.get_vocab()}
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=True, temperature=1.0, top_p=0.95, num_return_sequences=n,
                             eos_token_id=sorted(stop), pad_token_id=tokenizer.eos_token_id)
    return [tokenizer.decode(o[ids.shape[1]:], skip_special_tokens=True) for o in out]


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def done_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in open(path):
        try:
            out.add(json.loads(line)["qid"])
        except (json.JSONDecodeError, KeyError):
            pass
    return out


def cmd_generate(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions = list(iter_jsonl(args.questions))[: args.limit]
    done = completed_qids(args.out)
    todo = [q for q in questions if q["qid"] not in done]
    print(f"Перефразы: {len(done)} уже есть, к обработке {len(todo)}")
    if not todo:
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto").eval()
    device = next(model.parameters()).device
    total, short = Counter(), 0
    with open(args.out, "a") as f:
        for i, q in enumerate(todo):
            kept, reasons = [], Counter()
            for _ in range(MAX_ROUNDS):
                cands = generate_candidates(model, tokenizer, q["question"], device)
                kept = select_paraphrases(cands, q["question"], q["gold_answers"], kept=kept, reasons=reasons)
                if len(kept) >= N_KEEP:
                    break
            short += len(kept) < N_KEEP
            total.update(reasons)
            f.write(json.dumps({"qid": q["qid"], "paraphrases": kept, "filter": dict(reasons)}, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                f.flush()
                print(f"  {i + 1}/{len(todo)}  меньше {N_KEEP} перефразов: {short}  фильтр: {dict(total)}")
    print(f"Готово -> {args.out}. Меньше {N_KEEP} перефразов у {short} вопросов. Фильтр: {dict(total)}")


def cmd_retrieval_input(args) -> None:
    """Запросы для ретрива одним проходом: исходный вопрос (q_id = qid) и
    перефразы (q_id = qid__para{i}); answers = gold исходного вопроса, чтобы
    has_answer считался и у пассажей перефразов."""
    paras = {r["qid"]: r["paraphrases"] for r in iter_jsonl(args.paraphrases)} if args.paraphrases else {}
    n = 0
    with open(args.out, "w") as f:
        for q in iter_jsonl(args.questions):
            rows = [(q["qid"], q["question"])] + [(f"{q['qid']}__para{i}", p) for i, p in enumerate(paras.get(q["qid"], []))]
            for qid, text in rows:
                f.write(json.dumps({"q_id": qid, "question": text, "answers": q["gold_answers"]}, ensure_ascii=False) + "\n")
                n += 1
    print(f"Готово: {n} запросов -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    g = sub.add_parser("generate")
    g.add_argument("--questions", required=True, type=Path)
    g.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--out", required=True, type=Path)
    g.set_defaults(fn=cmd_generate)
    r = sub.add_parser("retrieval-input")
    r.add_argument("--questions", required=True, type=Path)
    r.add_argument("--paraphrases", type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.set_defaults(fn=cmd_retrieval_input)
    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        sys.exit(1)
    args.fn(args)


if __name__ == "__main__":
    main()
