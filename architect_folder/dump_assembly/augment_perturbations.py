"""
A5+ — дозаполнение perturbations в уже собранном дампе (dump_pilot_v2.jsonl)
без пересчёта остального. Обоснование и точный дифф схемы — dump_schema_v1.md,
раздел "Расширение A5+ (post-freeze)".

Нужно для карточек ветки-исследователя (researcher_folder/B2_cards.md):
paraphrase-rank-stability, retriever-disagreement, semantic-reformulation-entropy,
memory-strength-paraphrase.

Соответствие id пассажей: retriever_alt.bm25 использует прибилженный индекс
Pyserini "wikipedia-dpr" — он построен из того же файла psgs_w100.tsv, что и
наш Contriever-корпус (см. ragu_hpc_smoke.md), поэтому doc_id должны совпадать
с doc_id из dense-ретривала. Перед первым использованием сверить: взять любой
doc_id из dense-результатов основного вопроса, найти тот же id через
pyserini.search.lucene.LuceneSearcher.doc(doc_id) и сравнить текст.

Четыре шага, каждый — отдельный проход (не пересчитывает остальное):

  1. prepare-retrieval-input   — CPU. Пишет retrieval_input.jsonl для
     перефразов (только short-form) в формате passage_retrieval.py
     (self-rag/contriever), синтетический q_id = "{qid}__para{i}".
     Дальше вручную запустить ТОТ ЖЕ passage_retrieval.py CLI на этом
     файле (см. cluster_runbook.md шаг 2a/2b) — получить
     retrieval_paraphrases.jsonl.

  2. fill-bm25                 — CPU/Java (pyserini). Заполняет
     retriever_alt.bm25.* для ОРИГИНАЛЬНЫХ (не перефразированных)
     short-form вопросов через прибилженный индекс wikipedia-dpr.

  3. generate                  — GPU. Читает retrieval_paraphrases.jsonl
     (из шага 1 после ручного прогона passage_retrieval.py) + дамп,
     генерирует rag_answer/rag_samples (short-form, с топ-k пассажей
     самого перефраза) и cb_answer/cb_samples (все записи), пишет
     augmented dump.

Запуск (из architect_folder/):
    python -m dump_assembly.augment_perturbations prepare-retrieval-input \
        --in dump_pilot_v2.jsonl --out retrieval_input_paraphrases.jsonl

    # вручную: passage_retrieval.py --data retrieval_input_paraphrases.jsonl ...
    # -> retrieval_paraphrases.jsonl/retrieval_input_paraphrases.jsonl

    python -m dump_assembly.augment_perturbations fill-bm25 \
        --in dump_pilot_v2.jsonl --out dump_pilot_v3_bm25.jsonl \
        --pyserini-index wikipedia-dpr

    python -m dump_assembly.augment_perturbations generate \
        --in dump_pilot_v3_bm25.jsonl \
        --paraphrase-retrieval retrieval_paraphrases.jsonl/retrieval_input_paraphrases.jsonl \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --out dump_pilot_v3.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

N_SAMPLES_PARAPHRASE = 3  # компромисс по стоимости, см. dump_schema_v1.md
TOP_N_PASSAGES = 5  # тот же top_n, что и в основном пайплайне (retrieval.attach_retrieval_results)


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _paraphrase_qid(qid: str, i: int) -> str:
    return f"{qid}__para{i}"


def cmd_prepare_retrieval_input(args: argparse.Namespace) -> None:
    n_written = 0
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.in_path):
            if r["source"] == "franq_longform":
                continue  # у long-form нет query-based ретривала вообще
            paraphrases = r.get("perturbations", {}).get("query_paraphrases", [])
            for i, p in enumerate(paraphrases):
                fout.write(json.dumps({
                    "question": p["text"],
                    "answers": [],
                    "q_id": _paraphrase_qid(r["qid"], i),
                }) + "\n")
                n_written += 1
    print(f"Wrote {n_written} paraphrase queries to {args.out}")


def cmd_fill_bm25(args: argparse.Namespace) -> None:
    from pyserini.search.lucene import LuceneSearcher

    print(f"Loading pyserini prebuilt index '{args.pyserini_index}' (скачается при первом запуске)...")
    searcher = LuceneSearcher.from_prebuilt_index(args.pyserini_index)

    n_filled, n_skipped = 0, 0
    with open(args.in_path) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            if r["source"] == "franq_longform":
                fout.write(json.dumps(r) + "\n")
                continue
            hits = searcher.search(r["question"], k=args.top_n)
            if not hits:
                n_skipped += 1
                fout.write(json.dumps(r) + "\n")
                continue
            top20 = searcher.search(r["question"], k=20)
            r["perturbations"]["retriever_alt"]["bm25"] = {
                "topk_doc_ids": [h.docid for h in hits],
                "topk_scores": [float(h.score) for h in hits],
                "top20_scores": [float(h.score) for h in top20],
                "scores_raw": True,
            }
            n_filled += 1
            fout.write(json.dumps(r) + "\n")

    print(f"BM25 заполнен для {n_filled} записей, пропущено (нет хитов) {n_skipped}. Записано в {args.out}")


def cmd_generate(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from . import generation, retrieval

    print(f"Loading paraphrase retrieval results from {args.paraphrase_retrieval}...")
    para_retrieval = retrieval.read_retrieval_output(args.paraphrase_retrieval, top_n=TOP_N_PASSAGES)

    print(f"Loading model {args.model}...")
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto", attn_implementation="eager",
    )
    model.eval()
    device = next(model.parameters()).device

    already_done = set()
    if not args.no_resume and args.out.exists():
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        already_done.add(json.loads(line)["qid"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        print(f"Resume: {len(already_done)} qid уже готовы, пропускаю их")

    file_mode = "a" if already_done else "w"
    with open(args.out, file_mode) as fout:
        for i, r in enumerate(iter_jsonl(args.in_path)):
            if r["qid"] in already_done:
                continue
            is_longform = r["source"] == "franq_longform"
            paraphrases = r.get("perturbations", {}).get("query_paraphrases", [])
            # тот же промпт, что у основного ответа записи (дампы до v4 — без поля, это v1)
            pv = r.get("prompt_version", "v1")

            for pi, p in enumerate(paraphrases):
                para_text = p["text"]

                # closed-book — для всех источников (память модели, RAG не при чём)
                cb_out = generation.generate_greedy(model, tokenizer, para_text, None, device, prompt_version=pv)
                cb_samples = generation.sample(model, tokenizer, para_text, None, device,
                                                n_samples=N_SAMPLES_PARAPHRASE, prompt_version=pv)
                p["cb_answer"] = cb_out["answer"]
                p["cb_samples"] = cb_samples["samples"]

                # rag — только short-form, с топ-k пассажей САМОГО перефраза
                if not is_longform:
                    pqid = _paraphrase_qid(r["qid"], pi)
                    ctxs = para_retrieval.get(pqid)
                    if ctxs:
                        p["topk_doc_ids"] = [c["doc_id"] for c in ctxs]
                        p["topk_scores"] = [c["score"] for c in ctxs]
                        passage_texts = [c["text"] for c in ctxs]
                        rag_out = generation.generate_greedy(model, tokenizer, para_text, passage_texts, device,
                                                             prompt_version=pv)
                        rag_samples = generation.sample(model, tokenizer, para_text, passage_texts, device,
                                                         n_samples=N_SAMPLES_PARAPHRASE, prompt_version=pv)
                        p["rag_answer"] = rag_out["answer"]
                        p["rag_samples"] = rag_samples["samples"]
                    else:
                        print(f"  [{i}] {pqid}: нет результата ретривала для перефраза", file=sys.stderr)
                        p["rag_answer"] = None
                        p["rag_samples"] = []
                else:
                    p["rag_answer"] = None
                    p["rag_samples"] = []

            fout.write(json.dumps(r) + "\n")
            if (i + 1) % 10 == 0:
                print(f"  {i + 1} done")

    print(f"Done. Wrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("prepare-retrieval-input")
    p1.add_argument("--in", dest="in_path", required=True, type=Path)
    p1.add_argument("--out", required=True, type=Path)
    p1.set_defaults(func=cmd_prepare_retrieval_input)

    p2 = sub.add_parser("fill-bm25")
    p2.add_argument("--in", dest="in_path", required=True, type=Path)
    p2.add_argument("--out", required=True, type=Path)
    p2.add_argument("--pyserini-index", default="wikipedia-dpr")
    p2.add_argument("--top-n", type=int, default=5)
    p2.set_defaults(func=cmd_fill_bm25)

    p3 = sub.add_parser("generate")
    p3.add_argument("--in", dest="in_path", required=True, type=Path)
    p3.add_argument("--paraphrase-retrieval", required=True, type=Path)
    p3.add_argument("--model", required=True)
    p3.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p3.add_argument("--out", required=True, type=Path)
    p3.add_argument("--no-resume", action="store_true")
    p3.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
