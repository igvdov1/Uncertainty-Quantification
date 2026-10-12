"""
Плотный ретрив Contriever-MSMARCO по Wikipedia-2018 (psgs_w100) из
репозитория, без внешнего passage_retrieval.py (C9 §10).

Как устроено (минимум RAM и диска):
  - запросы (исходные вопросы и перефразы, файл paraphrases.py
    retrieval-input) эмбеддятся один раз;
  - файл эмбеддингов корпуса читается по шардам (pickle (ids, embeddings) в
    wikipedia_embeddings.tar) — с диска или потоком по URL; каждый шард
    скалярно умножается на все запросы на GPU и сливается с текущим top-k —
    в памяти одновременно один шард (~4 ГБ), а не весь индекс (~65 ГБ);
  - тексты пассажей (psgs_w100.tsv[.gz]) читаются потоком, в память
    попадают только найденные id;
  - has_answer — как в DPR (Karpukhin et al. 2020, qa_validation.has_answer):
    Unicode NFD, токены SimpleTokenizer без регистра, вхождение
    последовательности токенов ответа в текст пассажа.

Выход — тот же формат, что у contriever/passage_retrieval.py: на запрос
строка {q_id, question, answers, ctxs: [{id, title, text, score, hasanswer}]},
ctxs — top-k (по умолчанию 20) по убыванию скора.

Формат файлов подтверждён по исходнику facebookresearch/contriever:
  - passage_retrieval.py::index_encoded_data — pickle.dump((ids, embeddings));
  - src/data.py::load_passages — TSV с заголовком "id\\ttext\\ttitle";
  - src/contriever.py — mean pooling по attention_mask, без нормализации;
    скор = скалярное произведение (faiss.IndexFlatIP).

Запуск (из architect_folder/):
    python -m dump_assembly.streaming_retrieval \
        --queries retrieval_queries.jsonl \
        --embeddings contriever/wikipedia_embeddings   # или URL .tar
        --passages contriever/psgs_w100.tsv            # или .tsv.gz / URL
        --out retrieval.jsonl
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import pickle
import sys
import tarfile
import unicodedata
from pathlib import Path

import numpy as np

PSGS_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
EMBEDDINGS_URL = "https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar"
ENCODER = "facebook/contriever-msmarco"

csv.field_size_limit(sys.maxsize)


# ---- has_answer по DPR -------------------------------------------------------

try:
    import regex as _re_unicode
    _TOKEN_RE = _re_unicode.compile(r"[\p{L}\p{N}\p{M}]+|[^\p{Z}\p{C}]", flags=_re_unicode.IGNORECASE | _re_unicode.UNICODE)
except ImportError:  # regex — зависимость transformers, но на всякий случай
    import re as _re_unicode
    _TOKEN_RE = _re_unicode.compile(r"\w+|[^\w\s]", flags=_re_unicode.UNICODE)


def dpr_tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(unicodedata.normalize("NFD", text))]


def has_answer(answers: list[str], text: str) -> bool:
    words = dpr_tokens(text)
    for a in answers:
        a_tok = dpr_tokens(a)
        if not a_tok:
            continue
        n = len(a_tok)
        for i in range(len(words) - n + 1):
            if words[i:i + n] == a_tok:
                return True
    return False


# ---- top-k по шардам ---------------------------------------------------------

class RunningTopK:
    """Текущие top-k (скор, глобальный индекс) для каждого запроса."""

    def __init__(self, n_queries: int, k: int):
        self.k = k
        self.scores = np.full((n_queries, k), -np.inf, dtype=np.float32)
        self.index = np.full((n_queries, k), -1, dtype=np.int64)

    def update(self, scores: np.ndarray, index: np.ndarray) -> None:
        """scores, index: (n_queries, m) — кандидаты из очередного блока."""
        all_s = np.concatenate([self.scores, scores], axis=1)
        all_i = np.concatenate([self.index, index], axis=1)
        top = np.argsort(-all_s, axis=1, kind="stable")[:, : self.k]
        self.scores = np.take_along_axis(all_s, top, axis=1)
        self.index = np.take_along_axis(all_i, top, axis=1)


def search_shard(q_emb, shard_emb: np.ndarray, offset: int, topk: RunningTopK,
                 device: str = "cuda", chunk: int = 262_144, q_block: int = 4096) -> None:
    """Точный поиск по скалярному произведению. q_emb — torch (n_q, d) на device."""
    import torch
    k = topk.k
    for start in range(0, len(shard_emb), chunk):
        e = torch.from_numpy(np.ascontiguousarray(shard_emb[start:start + chunk])).to(device, dtype=q_emb.dtype)
        kk = min(k, e.shape[0])
        for qs in range(0, q_emb.shape[0], q_block):
            s = q_emb[qs:qs + q_block] @ e.T
            vals, idx = torch.topk(s.float(), kk, dim=1)
            sub = RunningTopK(0, k)
            sub.scores, sub.index = topk.scores[qs:qs + q_block], topk.index[qs:qs + q_block]
            sub.update(vals.cpu().numpy(), idx.cpu().numpy() + offset + start)
            topk.scores[qs:qs + q_block], topk.index[qs:qs + q_block] = sub.scores, sub.index
        del e


def iter_embedding_shards(source: str):
    """(ids, embeddings) по шардам: каталог/glob с pickle-файлами, локальный
    .tar или URL .tar (поток, без записи на диск)."""
    if source.startswith("http"):
        import requests
        with requests.get(source, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tarfile.open(fileobj=r.raw, mode="r|") as tar:
                yield from _iter_tar(tar)
    elif source.endswith(".tar"):
        with tarfile.open(source, mode="r|") as tar:
            yield from _iter_tar(tar)
    else:
        files = sorted(glob.glob(source if any(c in source for c in "*?[") else str(Path(source) / "*")))
        if not files:
            raise FileNotFoundError(f"нет файлов эмбеддингов по {source}")
        for fn in files:
            with open(fn, "rb") as f:
                ids, emb = pickle.load(f)
            print(f"  шард {fn}: {len(ids)}")
            yield ids, emb


def _iter_tar(tar):
    for member in tar:
        if member.isfile():
            ids, emb = pickle.load(tar.extractfile(member))
            print(f"  шард {member.name}: {len(ids)}")
            yield ids, emb


def iter_passages(source: str):
    """(id, text, title) из psgs_w100.tsv / .tsv.gz / URL .tsv.gz."""
    if source.startswith("http"):
        import requests
        r = requests.get(source, stream=True, timeout=120)
        r.raise_for_status()
        stream = io.TextIOWrapper(gzip.GzipFile(fileobj=r.raw), encoding="utf-8")
    elif source.endswith(".gz"):
        stream = io.TextIOWrapper(gzip.open(source), encoding="utf-8")
    else:
        stream = open(source, encoding="utf-8")
    with stream:
        for row in csv.reader(stream, delimiter="\t"):
            if row and row[0] != "id":
                yield row[0], row[1], row[2]


def embed_queries(texts: list[str], device: str, batch_size: int = 128):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = AutoModel.from_pretrained(ENCODER).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"][..., None].to(h.dtype)
            out.append((h * m).sum(1) / m.sum(1))
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path, help="JSONL {q_id, question, answers}")
    parser.add_argument("--embeddings", default=EMBEDDINGS_URL, help="каталог/glob шардов, .tar или URL")
    parser.add_argument("--passages", default=PSGS_URL, help="psgs_w100.tsv, .tsv.gz или URL")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    queries = [json.loads(l) for l in open(args.queries) if l.strip()]
    print(f"{len(queries)} запросов; эмбеддинг ({ENCODER})...")
    import torch
    q_emb = embed_queries([q["question"] for q in queries], args.device).to(torch.float16)

    topk = RunningTopK(len(queries), args.top_k)
    shard_ids: list[np.ndarray] = []
    offset = 0
    print(f"Поиск по шардам из {args.embeddings}...")
    for ids, emb in iter_embedding_shards(args.embeddings):
        search_shard(q_emb, np.asarray(emb), offset, topk, args.device)
        shard_ids.append(np.asarray([str(x) for x in ids]))
        offset += len(ids)
    all_ids = np.concatenate(shard_ids)
    print(f"Проиндексировано {offset} пассажей")

    hit_ids = all_ids[topk.index]
    needed = set(hit_ids.ravel().tolist())
    print(f"Чтение текстов {len(needed)} найденных пассажей из {args.passages}...")
    texts: dict[str, tuple[str, str]] = {}
    for pid, text, title in iter_passages(args.passages):
        if pid in needed:
            texts[pid] = (text, title)
            if len(texts) == len(needed):
                break
    missing = needed - texts.keys()
    if missing:
        print(f"ВНИМАНИЕ: не найдено текстов для {len(missing)} id", file=sys.stderr)

    with open(args.out, "w") as f:
        for qi, q in enumerate(queries):
            ctxs = []
            for j in range(args.top_k):
                pid = str(hit_ids[qi, j])
                text, title = texts.get(pid, ("", ""))
                ctxs.append({"id": pid, "title": title, "text": text, "score": float(topk.scores[qi, j]),
                             "hasanswer": has_answer(q.get("answers") or [], text)})
            f.write(json.dumps({**q, "ctxs": ctxs}, ensure_ascii=False) + "\n")
    print(f"Готово -> {args.out}")


if __name__ == "__main__":
    main()
