"""
A4 — потоковая альтернатива Contriever passage_retrieval.py, для случая,
когда локального диска не хватает под psgs_w100.tsv (~13 ГБ, распакованный)
+ wikipedia_embeddings (~65 ГБ), но достаточно RAM (нужно ~90-110 ГБ
свободных). Стримит оба файла напрямую с dl.fbaipublicfiles.com в память,
не сохраняя их на диск ни в каком виде — ни архив, ни распакованное.

Формат подтверждён по исходнику facebookresearch/contriever, не
предположен:
  - passage_retrieval.py::index_encoded_data — каждый файл в
    wikipedia_embeddings.tar — pickle.dump((ids, embeddings)),
    embeddings: np.ndarray (N, 768)
  - src/data.py::load_passages — psgs_w100.tsv: TSV с заголовком
    (строка "id\\ttext\\ttitle"), затем id, text, title по столбцам
  - src/index.py::Indexer — faiss.IndexFlatIP(768), без n_subquantizers
  - src/contriever.py::Contriever.forward — mean pooling по
    attention_mask, БЕЗ нормализации (normalize=False по умолчанию)

Ограничение: индекс существует только в памяти процесса, который его
построил — нет промежуточного сохранения на диск. Один прогон должен
и построить индекс, и сразу выполнить весь поиск по всем вопросам
пилота (не сервис, разовая пакетная операция).

Запуск (из architect_folder/):
    python -m dump_assembly.streaming_retrieval \
        --franq-dataset-dir rag_uncertainty/claim_level/dataset \
        --dev-size 300 --test-size 300 \
        --out retrieval_dense.jsonl
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import pickle
import tarfile
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from transformers import AutoModel, AutoTokenizer

PSGS_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
EMBEDDINGS_URL = "https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar"
VECTOR_SIZE = 768


def stream_load_passages(url: str = PSGS_URL, log_every: int = 2_000_000) -> dict[str, dict]:
    passages: dict[str, dict] = {}
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with gzip.GzipFile(fileobj=r.raw) as gz:
            text_stream = io.TextIOWrapper(gz, encoding="utf-8")
            reader = csv.reader(text_stream, delimiter="\t")
            for i, row in enumerate(reader):
                if row[0] == "id":
                    continue
                passages[row[0]] = {"text": row[1], "title": row[2]}
                if log_every and (i + 1) % log_every == 0:
                    print(f"  passages loaded: {i + 1}")
    print(f"Total passages: {len(passages)}")
    return passages


def stream_build_index(url: str = EMBEDDINGS_URL, vector_size: int = VECTOR_SIZE):
    index = faiss.IndexFlatIP(vector_size)
    ids: list[str] = []

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                shard_ids, shard_embeddings = pickle.load(f)
                shard_embeddings = np.asarray(shard_embeddings, dtype="float32")
                index.add(shard_embeddings)
                ids.extend(str(x) for x in shard_ids)
                print(f"  shard {member.name}: +{len(shard_ids)} (total {len(ids)})")

    print(f"Total indexed: {len(ids)}")
    return index, ids


@torch.no_grad()
def embed_queries(model, tokenizer, queries: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        out = model(**encoded)
        mask = encoded["attention_mask"][..., None].bool()
        last_hidden = out.last_hidden_state.masked_fill(~mask, 0.0)
        emb = last_hidden.sum(dim=1) / encoded["attention_mask"].sum(dim=1)[..., None]
        embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def search(index, ids: list[str], query_embeddings: np.ndarray, top_k: int = 20):
    query_embeddings = query_embeddings.astype("float32")
    scores, indexes = index.search(query_embeddings, top_k)
    results = []
    for q in range(len(query_embeddings)):
        results.append([(ids[idx], float(scores[q][j])) for j, idx in enumerate(indexes[q])])
    return results


def main() -> None:
    import argparse

    from .questions import build_pilot_question_set

    parser = argparse.ArgumentParser()
    parser.add_argument("--franq-dataset-dir", required=True, type=Path)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print("Loading pilot question set...")
    qs = build_pilot_question_set(args.franq_dataset_dir, args.dev_size, args.test_size, args.seed)
    shortform = [q for q in qs if q.passages is None]
    print(f"{len(shortform)} short-form questions need retrieval")

    print("Loading Contriever-MSMARCO query encoder...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    model = AutoModel.from_pretrained("facebook/contriever-msmarco").to(args.device)
    model.eval()

    print("Streaming passages (psgs_w100.tsv.gz, ~13 ГБ распакованных, без записи на диск)...")
    passages = stream_load_passages()

    print("Streaming embeddings + building FAISS index (~65 ГБ в RAM, без записи на диск)...")
    index, ids = stream_build_index()

    print("Embedding queries...")
    query_texts = [q.question for q in shortform]
    query_embeddings = embed_queries(model, tokenizer, query_texts, args.device)

    print("Searching...")
    results = search(index, ids, query_embeddings, top_k=args.top_k)

    print(f"Writing {args.out}...")
    with open(args.out, "w") as f:
        for q, hits in zip(shortform, results):
            ctxs = [
                {"id": doc_id, "title": passages.get(doc_id, {}).get("title", ""),
                 "text": passages.get(doc_id, {}).get("text", ""), "score": score,
                 "hasanswer": None}
                for doc_id, score in hits
            ]
            f.write(json.dumps({"question": q.question, "answers": q.gold_answers,
                                 "q_id": q.qid, "ctxs": ctxs}) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
