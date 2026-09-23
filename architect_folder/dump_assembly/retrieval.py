"""
A4 — ретривал для short-form вопросов (long-form FRANQ уже несёт готовые
пассажи, см. questions.py::load_franq_longform).

Не переизобретаю FAISS-индексацию по Wikipedia с нуля — это ровно та же
инфраструктура, что уже задокументирована в ragu_hpc_smoke.md (Contriever
-MSMARCO + psgs_w100.tsv.gz + wikipedia_embeddings.tar, self-rag/contriever
passage_retrieval.py). Этот модуль — только формат-склейка: наши Question
-> их входной JSONL, их выходной JSONL -> наши Question.passages.

BM25 альтернативный ретривер (perturbations.retriever_alt) — через
готовый предпостроенный индекс Pyserini (wikipedia-dpr), не через
самодельную индексацию: см. cluster_runbook.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from .questions import Question


def write_retrieval_input(questions: list[Question], out_path: Path) -> None:
    """Формат self-rag/contriever passage_retrieval.py: {question, answers, q_id}."""
    with open(out_path, "w") as f:
        for q in questions:
            f.write(json.dumps({
                "question": q.question,
                "answers": q.gold_answers or [],
                "q_id": q.qid,
            }) + "\n")


def read_retrieval_output(path: Path, top_n: int = 20) -> dict[str, list[dict]]:
    """Читает выход passage_retrieval.py: augmented JSONL с полем "ctxs" —
    [{"id","title","text","score","hasanswer"}, ...]. Возвращает qid -> passages
    в формате dump_schema_v1 (rank, text, score, doc_id, is_golden)."""
    out: dict[str, list[dict]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            qid = row["q_id"]
            ctxs = row["ctxs"][:top_n]
            out[qid] = [
                {
                    "rank": i + 1,
                    "text": c["text"],
                    "score": float(c["score"]),
                    "doc_id": c["id"],
                    "is_golden": bool(c.get("hasanswer", False)),
                }
                for i, c in enumerate(ctxs)
            ]
    return out


def read_top20_scores(path: Path) -> dict[str, list[float]]:
    """passages_top20_scores — до нормализации, top-20 независимо от top_n,
    используемого в промпте (B0-A6: 'top-20 при top-k=3 — почти бесплатный
    источник сигнала')."""
    out: dict[str, list[float]] = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["q_id"]] = [float(c["score"]) for c in row["ctxs"][:20]]
    return out


def attach_retrieval_results(
    questions: list[Question],
    retrieval_output_path: Path,
    top_n: int = 5,
) -> None:
    """Мутирует questions[i].passages по месту для тех, у кого passages ещё
    не заданы (long-form FRANQ пропускается — у них passages уже есть)."""
    results = read_retrieval_output(retrieval_output_path, top_n=top_n)
    for q in questions:
        if q.passages is not None:
            continue
        if q.qid not in results:
            raise KeyError(f"{q.qid}: нет результата ретривала в {retrieval_output_path}")
        q.passages = results[q.qid]


def generate_paraphrases(model, tokenizer, question: str, device, n: int = 3) -> list[str]:
    """≥3 перефраза запроса (A1 §1.4) через саму генеративную модель.
    Дёшево (короткая генерация), не требует отдельной модели-парафразера."""
    prompt = (
        f"Rewrite the following question in a different way, keeping the same meaning. "
        f"Output only the rewritten question.\nQuestion: {question}\nRewritten:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **input_ids, max_new_tokens=40, do_sample=True, temperature=1.0,
        num_return_sequences=n, top_p=0.95,
    )
    prompt_len = input_ids["input_ids"].shape[1]
    paraphrases = [
        tokenizer.decode(out[i, prompt_len:], skip_special_tokens=True).strip()
        for i in range(n)
    ]
    return paraphrases


def _cli_prepare_input() -> None:
    """CLI для шага 2a cluster_runbook.md — готовит входной JSONL для
    passage_retrieval.py. Запускать из architect_folder/:
        python -m dump_assembly.retrieval --franq-dataset-dir <путь> \
            --dev-size 300 --test-size 300 --out retrieval_input.jsonl
    """
    import argparse

    from .questions import build_pilot_question_set

    parser = argparse.ArgumentParser()
    parser.add_argument("--franq-dataset-dir", required=True, type=Path)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    qs = build_pilot_question_set(args.franq_dataset_dir, args.dev_size, args.test_size, args.seed)
    shortform = [q for q in qs if q.passages is None]
    write_retrieval_input(shortform, args.out)
    print(f"Wrote {len(shortform)} short-form questions to {args.out}")


if __name__ == "__main__":
    _cli_prepare_input()
