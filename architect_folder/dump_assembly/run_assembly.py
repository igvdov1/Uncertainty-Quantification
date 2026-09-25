"""
A4 — CLI сборки дампа. Склеивает questions.py + retrieval.py + generation.py
+ signals.py + labeling.py + assemble.py в JSONL по dump_schema_v1.md.

Компоненты без GPU (questions.py, часть labeling.py) протестированы
локально. Генерация/сигналы/NLI требуют GPU — полный прогон не тестирован
end-to-end (нет кластера в этой сессии) — см. cluster_runbook.md для
пошагового плана прогона и того, что стоит проверить на малом N сначала.

Запуск (после того как retrieval уже посчитан отдельным шагом, см. runbook):
    python -m dump_assembly.run_assembly \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --franq-dataset-dir /path/to/rag_uncertainty/claim_level/dataset \
        --retrieval-output /path/to/retrieval_dense.jsonl \
        --bm25-output /path/to/retrieval_bm25.jsonl \
        --dev-size 300 --test-size 300 \
        --out dump_pilot.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import generation, labeling, retrieval, signals
from .assemble import assemble_record
from .questions import Question, build_pilot_question_set


def load_completed_qids_and_clean(out_path: Path) -> set[str]:
    """Для resume: qid уже успешно записанных записей в существующем
    выводе. Если прошлый прогон упал посреди записи строки, в файле может
    остаться оборванный, невалидный JSON последней строкой — если потом
    просто открыть файл на дозапись (append), новая строка приклеится
    прямо к этому мусору без разделителя и испортит файл. Поэтому здесь
    файл сразу перезаписывается только валидными строками (аккуратный
    "compact"), прежде чем начинать append."""
    if not out_path.exists():
        return set()
    valid_lines, completed = [], set()
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qid = json.loads(line)["qid"]
            except (json.JSONDecodeError, KeyError):
                continue
            valid_lines.append(line)
            completed.add(qid)
    with open(out_path, "w") as f:
        for line in valid_lines:
            f.write(line + "\n")
    return completed


def process_one(model, tokenizer, device, q: Question, nli_scorer) -> dict | None:
    is_longform = q.source == "franq_longform"
    passage_texts = [p["text"] for p in (q.passages or [])]

    if is_longform:
        ex = q.franq_raw
        target_text = ex.get("output", ex.get("answer", "")).strip()
        if not target_text:
            return None
        rag_fields = generation.teacher_force(model, tokenizer, q.question, passage_texts, target_text, device)
        cb_fields = generation.teacher_force(model, tokenizer, q.question, None, target_text, device)
        claims = labeling.franq_longform_claims({**ex, "_qid": q.qid})
        label_faithful, label_factual = labeling.franq_longform_answer_labels(claims)
    else:
        rag_fields = generation.generate_greedy(model, tokenizer, q.question, passage_texts, device)
        cb_fields = generation.generate_greedy(model, tokenizer, q.question, None, device)
        claims = [{"cid": f"{q.qid}_c0", "text": rag_fields["answer"],
                   "label_faithful": None, "label_factual": None}]  # заполняется ниже
        label_factual = labeling.factuality_shortform(rag_fields["answer"], q.gold_answers)
        label_faithful = nli_scorer.label(rag_fields["answer"], passage_texts) if passage_texts else 0
        claims[0]["label_faithful"] = label_faithful
        claims[0]["label_factual"] = label_factual

    samples_rag = generation.sample(model, tokenizer, q.question, passage_texts, device)
    samples_cb = generation.sample(model, tokenizer, q.question, None, device)

    p_true_rag = signals.p_true(model, tokenizer, q.question, passage_texts, rag_fields["answer"], device)
    p_true_cb = signals.p_true(model, tokenizer, q.question, None, cb_fields["answer"], device)
    verbalized = signals.verbalized_conf(model, tokenizer, q.question, passage_texts, rag_fields["answer"], device)
    alignscore = nli_scorer.score(rag_fields["answer"], passage_texts) if passage_texts else float("nan")

    dump_signals = {
        "p_true_rag": p_true_rag, "p_true_cb": p_true_cb,
        "verbalized_conf_rag": verbalized,
        "alignscore": alignscore,  # V1: NLI-заменитель, см. signals.py докстринг
    }

    paraphrases_text = retrieval.generate_paraphrases(model, tokenizer, q.question, device, n=3)
    paraphrases = [{"text": p, "topk_doc_ids": [], "topk_scores": []} for p in paraphrases_text]
    # topk_doc_ids/scores для перефразов заполняются отдельным ретривал-проходом (runbook, шаг 2b)

    retriever_alt = {"bm25": {"topk_doc_ids": [], "topk_scores": [], "top20_scores": [], "scores_raw": True}}
    # заполняется из --bm25-output, см. runbook

    # long-form FRANQ пассажи не из ретривера (распарсены из текста) — у них нет score
    passages_top20_scores = [(p.get("score") or 0.0) for p in (q.passages or [])][:20]

    return assemble_record(
        q=q, rag_fields=rag_fields, cb_fields=cb_fields,
        samples_rag=samples_rag, samples_cb=samples_cb,
        paraphrases=paraphrases, retriever_alt=retriever_alt,
        passages_top20_scores=passages_top20_scores,
        tokenizer_name=tokenizer.name_or_path,
        label_faithful=label_faithful, label_factual=label_factual,
        claims=claims, signals=dump_signals,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--franq-dataset-dir", required=True, type=Path)
    parser.add_argument("--retrieval-output", type=Path, help="результат Contriever-ретривала для short-form")
    parser.add_argument("--bm25-output", type=Path, help="результат BM25-ретривала (retriever_alt)")
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="ограничить N вопросов — для смоука")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--nli-device", default="cpu")
    parser.add_argument("--no-resume", action="store_true",
                         help="не пропускать уже записанные qid в --out, перезаписать файл с нуля")
    parser.add_argument("--skip-attention-head", action="store_true",
                         help="не считать attention_by_head — самое тяжёлое поле по RAM/диску "
                              "(растёт с длиной последовательности, особенно на long-form teacher_force); "
                              "не нужно для гейта A5/бейзлайнов, понадобится позже для attention-based идей")
    args = parser.parse_args()

    if args.skip_attention_head:
        generation.INCLUDE_ATTENTION_HEAD = False
        print("attention_by_head отключён (--skip-attention-head)")

    print("Loading question set...")
    questions = build_pilot_question_set(args.franq_dataset_dir, args.dev_size, args.test_size, args.seed)
    if args.limit:
        questions = questions[: args.limit]

    already_done = set() if args.no_resume else load_completed_qids_and_clean(args.out)
    if already_done:
        n_before = len(questions)
        questions = [q for q in questions if q.qid not in already_done]
        print(f"Resume: {len(already_done)} qid уже есть в {args.out}, "
              f"пропускаю их ({n_before} -> {len(questions)} к обработке)")

    needs_retrieval = [q for q in questions if q.passages is None]
    if needs_retrieval and args.retrieval_output:
        retrieval.attach_retrieval_results(questions, args.retrieval_output)
    elif needs_retrieval:
        print(f"ВНИМАНИЕ: {len(needs_retrieval)} вопросов без retrieval-output — "
              f"нужен --retrieval-output, см. cluster_runbook.md шаг 2", file=sys.stderr)
        questions = [q for q in questions if q.passages is not None]

    if not questions:
        print("Все вопросы уже обработаны (resume) — нечего делать, модель не загружаю.")
        return

    print(f"Loading model {args.model}...")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device

    print("Loading NLI scorer for faithfulness...")
    nli_scorer = labeling.load_nli_scorer(device=args.nli_device)

    file_mode = "a" if already_done else "w"
    print(f"Assembling {len(questions)} records (mode={file_mode})...")
    with open(args.out, file_mode) as f:
        for i, q in enumerate(questions):
            try:
                record = process_one(model, tokenizer, device, q, nli_scorer)
            except Exception as e:
                print(f"  [{i}] {q.qid} FAILED: {e}", file=sys.stderr)
                continue
            if record is None:
                print(f"  [{i}] {q.qid} skipped (no target text)", file=sys.stderr)
                continue
            f.write(json.dumps(record) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(questions)} done")

    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
