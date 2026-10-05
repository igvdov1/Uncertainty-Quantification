"""
Пересборка генерации дампа на промпте v2 (generation.PROMPT_VERSIONS) поверх
готового дампа v4 — без повторного ретривала.

Что берётся из v4 как есть: вопросы, сплиты, пассажи dense-ретривера и их
скоры, perturbations (тексты перефразов, их ретривал, BM25), long-form записи
целиком (их ответ — teacher-forced текст FRANQ с инструкцией v1, от версии
промпта short-form он не зависит). Что генерируется заново для short-form:
rag/closed_book жадный ответ и сэмплы со всеми полями, p_true, verbalized,
alignscore (NLI-сигнал), label_factual по gold. label_faithful = None —
ставится LLM-судьёй (relabel_faithfulness.py, промпт v3). Ответы на перефразы
обнуляются — их заново генерирует augment_perturbations generate.

Два шага (оба на сервере, из architect_folder/):

    # 1. gold-ответы — тем же кодом, что собирал пилот (нужны HF datasets и FRANQ)
    python -m dump_assembly.rebuild_v2 export-gold \\
        --franq-dataset-dir rag_uncertainty/claim_level/dataset --out gold_answers.jsonl

    # 2. генерация v2 (резюмируется по qid)
    python -m dump_assembly.rebuild_v2 rebuild --in dump_pilot_v4.jsonl --gold gold_answers.jsonl \\
        --model meta-llama/Llama-3.1-8B-Instruct --out dump_pilot_v5_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LONGFORM = "franq_longform"
ANSWER_KEYS = ("cb_answer", "cb_samples", "rag_answer", "rag_samples")


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def strip_paraphrase_answers(perturbations: dict) -> dict:
    """Ответы на перефразы сгенерированы промптом v1 — убираем, ретривал оставляем."""
    out = json.loads(json.dumps(perturbations))
    for p in out.get("query_paraphrases", []):
        for k in ANSWER_KEYS:
            p.pop(k, None)
    return out


def question_from_record(r: dict, gold: dict[str, list[str]]):
    from .questions import Question
    if r["qid"] not in gold:
        raise KeyError(f"{r['qid']}: нет gold-ответов — проверьте, что export-gold запущен с теми же "
                       f"--dev-size/--test-size/--seed, что и сборка пилота")
    return Question(qid=r["qid"], question=r["question"], source=r["source"], gold_answers=gold[r["qid"]],
                    passages=r["passages"], split=r["split"])


def cmd_export_gold(args) -> None:
    from .questions import build_pilot_question_set
    qs = build_pilot_question_set(args.franq_dataset_dir, args.dev_size, args.test_size, args.seed)
    n = 0
    with open(args.out, "w") as f:
        for q in qs:
            if q.source != LONGFORM:
                f.write(json.dumps({"qid": q.qid, "source": q.source, "question": q.question,
                                    "gold_answers": q.gold_answers}) + "\n")
                n += 1
    print(f"Готово: gold для {n} short-form вопросов -> {args.out}")


def cmd_rebuild(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import generation, labeling
    from .run_assembly import load_completed_qids_and_clean, process_one

    generation.PROMPT_VERSION = args.prompt_version
    # v4 собирался с --skip-attention-head; внимание по головам теперь считает mech_cards.py
    generation.INCLUDE_ATTENTION_HEAD = False
    gold = {}
    for g in iter_jsonl(args.gold):
        gold[g["qid"]] = g["gold_answers"]
    done = set() if args.no_resume else load_completed_qids_and_clean(args.out)
    print(f"Промпт {args.prompt_version}; gold для {len(gold)} вопросов; уже готово {len(done)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="auto",
                                                 attn_implementation="eager").eval()
    device = next(model.parameters()).device
    nli_scorer = labeling.load_nli_scorer(device=args.nli_device)

    n_new = 0
    with open(args.out, "a" if done else "w") as fout:
        for i, r in enumerate(iter_jsonl(args.in_path)):
            if r["qid"] in done:
                continue
            if r["source"] == LONGFORM:
                r["prompt_version"] = "v1"
                fout.write(json.dumps(r) + "\n")
                continue
            # сверка: тот же вопрос, что в gold (иначе qid разошлись между сборками)
            q = question_from_record(r, gold)
            try:
                rec = process_one(model, tokenizer, device, q, nli_scorer,
                                  perturbations_override=strip_paraphrase_answers(r.get("perturbations", {})))
            except Exception as e:  # как в run_assembly: падение одной записи не роняет прогон
                print(f"  [{i}] {r['qid']} FAILED: {e}", file=sys.stderr)
                continue
            # пассажи и их скоры — ровно из v4 (process_one берёт их из Question, но перестрахуемся)
            rec["passages_top20_scores"] = r.get("passages_top20_scores", rec.get("passages_top20_scores"))
            fout.write(json.dumps(rec) + "\n")
            n_new += 1
            if n_new % 10 == 0:
                fout.flush()
                print(f"  {n_new} short-form пересобрано")
    print(f"Готово -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export-gold")
    p.add_argument("--franq-dataset-dir", required=True, type=Path)
    p.add_argument("--dev-size", type=int, default=300)
    p.add_argument("--test-size", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(fn=cmd_export_gold)

    p = sub.add_parser("rebuild")
    p.add_argument("--in", dest="in_path", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--prompt-version", default="v2")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--nli-device", default="cuda")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--no-resume", action="store_true")
    p.set_defaults(fn=cmd_rebuild)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
