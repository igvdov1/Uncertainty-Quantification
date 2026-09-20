"""
Генератор синтетических записей дампа (формат — dump_schema_v1.md) для
гейта A3: скринер должен посчитать весь реестр и матрицу, ни разу не
увидев настоящий дамп.

Не претендует на реалистичность текста — только на то, чтобы формы,
типы и диапазоны полей совпадали со схемой, и чтобы сигналы не были
чистым шумом (у AUROC должна быть возможность оказаться не 0.5 —
иначе тест не отличит "функция сломана" от "функция считает шум").
Для этого вводится скрытая переменная quality на вопрос, которая
управляет и метками, и распределениями токен-логпробов/ретривала.
"""
from __future__ import annotations

import numpy as np

VOCAB_SIZE = 32000
TOPK = 20


def _make_token_logprobs(rng: np.random.Generator, n_tokens: int, quality: float) -> list[float]:
    # выше quality -> логпробы ближе к 0 (модель увереннее в своём токене)
    mean = -0.15 - 1.2 * (1 - quality)
    lp = rng.normal(loc=mean, scale=0.3, size=n_tokens)
    return np.minimum(lp, -1e-4).tolist()


def _make_token_topk(rng: np.random.Generator, n_tokens: int, quality: float) -> list[list[float]]:
    out = []
    for _ in range(n_tokens):
        logits = rng.normal(loc=0.0, scale=1.0 + 2.0 * (1 - quality), size=TOPK)
        logits[0] += 3.0 + 4.0 * quality  # выделяем "выбранный" токен
        out.append(logits.tolist())
    return out


def _make_token_entropy(rng: np.random.Generator, n_tokens: int, quality: float) -> list[float]:
    base = 2.5 * (1 - quality)
    return np.clip(rng.normal(loc=base, scale=0.3, size=n_tokens), 0.0, None).tolist()


def _make_samples(rng: np.random.Generator, base_answer: str, quality: float, n: int = 10) -> tuple[list[str], list[list[float]]]:
    samples, logprobs = [], []
    for i in range(n):
        if rng.random() < quality:
            samples.append(base_answer)
        else:
            samples.append(f"{base_answer} (variant {rng.integers(0, 1000)})")
        n_tok = rng.integers(5, 15)
        logprobs.append(_make_token_logprobs(rng, n_tok, quality))
    return samples, logprobs


def _make_passages(rng: np.random.Generator, quality: float, k: int = 5):
    top_score = 0.5 + 0.4 * quality + rng.normal(0, 0.05)
    scores = sorted(
        [top_score] + list(top_score - np.abs(rng.normal(0.05, 0.05, size=k - 1)).cumsum()),
        reverse=True,
    )
    passages = [
        {
            "rank": i + 1,
            "text": f"passage text {i} for a synthetic question",
            "score": float(scores[i]),
            "doc_id": f"doc_{rng.integers(0, 100000)}",
            "is_golden": (i == 0 and quality > 0.5),
        }
        for i in range(k)
    ]
    top20 = sorted(np.clip(scores + list(rng.normal(0.2, 0.1, size=20 - k)), 0, 1).tolist(), reverse=True)
    return passages, top20


def make_record(rng: np.random.Generator, qid: str, split: str) -> dict:
    quality = float(rng.beta(2, 2))  # смещено к средним значениям, но с разбросом
    label_faithful = int(rng.random() < (0.15 + 0.75 * quality))
    label_factual = int(rng.random() < (0.10 + 0.75 * quality) and label_faithful or rng.random() < 0.05)

    n_tokens = int(rng.integers(15, 40))
    answer = f"synthetic answer {qid}"

    passages, top20_scores = _make_passages(rng, quality)

    rag_lp = _make_token_logprobs(rng, n_tokens, quality)
    rag_topk = _make_token_topk(rng, n_tokens, quality)
    rag_entropy = _make_token_entropy(rng, n_tokens, quality)
    rag_samples, rag_sample_lp = _make_samples(rng, answer, quality)

    cb_quality = float(np.clip(quality - 0.2 + rng.normal(0, 0.1), 0.0, 1.0))  # closed-book обычно хуже
    n_tokens_cb = int(rng.integers(15, 40))
    cb_lp = _make_token_logprobs(rng, n_tokens_cb, cb_quality)
    cb_topk = _make_token_topk(rng, n_tokens_cb, cb_quality)
    cb_entropy = _make_token_entropy(rng, n_tokens_cb, cb_quality)
    cb_samples, cb_sample_lp = _make_samples(rng, answer + " (closed-book)", cb_quality)

    n_claims = int(rng.integers(1, 4))
    claims = []
    pos = 0
    for i in range(n_claims):
        span_len = int(rng.integers(3, 8))
        claim_quality = float(np.clip(quality + rng.normal(0, 0.15), 0.0, 1.0))
        claims.append({
            "cid": f"{qid}_c{i}",
            "text": f"claim {i} of {qid}",
            "token_span": [pos, min(pos + span_len, n_tokens)],
            "label_faithful": int(rng.random() < (0.15 + 0.75 * claim_quality)),
            "label_factual": int(rng.random() < (0.10 + 0.75 * claim_quality)),
        })
        pos += span_len

    record = {
        "qid": qid,
        "question": f"synthetic question {qid}?",
        "split": split,

        "passages": passages,
        "passages_top20_scores": top20_scores,
        "passage_embeddings": rng.normal(size=(len(passages), 16)).tolist(),

        "rag": {
            "answer": answer,
            "prompt_raw": f"context ... question: synthetic question {qid}?",
            "prompt_spans": {"instruction": [0, 10], "passages": [[10, 60]], "question": [60, 70]},
            "tokenizer_name": "synthetic-tokenizer",
            "greedy_tokens": list(range(n_tokens)),
            "token_logprobs": rag_lp,
            "token_topk": rag_topk,
            "token_entropy": rag_entropy,
            "hidden_states_last": rng.normal(size=(n_tokens, 8)).tolist(),
            "attention_by_group": rng.dirichlet(np.ones(4), size=n_tokens).tolist(),
            "attention_by_passage": rng.dirichlet(np.ones(len(passages)), size=n_tokens).tolist(),
            "samples": rag_samples,
            "sample_tokens": [list(range(len(s))) for s in rag_samples],
            "sample_logprobs": rag_sample_lp,
        },

        "closed_book": {
            "answer": answer + " (closed-book)",
            "tokenizer_name": "synthetic-tokenizer",
            "greedy_tokens": list(range(n_tokens_cb)),
            "token_logprobs": cb_lp,
            "token_topk": cb_topk,
            "token_entropy": cb_entropy,
            "samples": cb_samples,
            "sample_tokens": [list(range(len(s))) for s in cb_samples],
            "sample_logprobs": cb_sample_lp,
        },

        "signals": {
            "p_true_rag": float(np.clip(0.1 + 0.85 * quality + rng.normal(0, 0.05), 0, 1)),
            "p_true_cb": float(np.clip(0.1 + 0.85 * cb_quality + rng.normal(0, 0.05), 0, 1)),
            "verbalized_conf_rag": float(np.clip(0.2 + 0.7 * quality + rng.normal(0, 0.1), 0, 1)),
            "alignscore": float(np.clip(0.1 + 0.85 * quality + rng.normal(0, 0.05), 0, 1)),
            "semantic_entropy_rag": float(2.0 * (1 - quality)),
            "semantic_entropy_cb": float(2.0 * (1 - cb_quality)),
        },

        "perturbations": {
            "query_paraphrases": [
                {
                    "text": f"paraphrase {i} of synthetic question {qid}?",
                    "topk_doc_ids": [p["doc_id"] for p in passages],
                    "topk_scores": [p["score"] for p in passages],
                }
                for i in range(3)
            ],
            "retriever_alt": {
                "bm25": {
                    "topk_doc_ids": [p["doc_id"] for p in passages],
                    "topk_scores": [p["score"] * 0.9 for p in passages],
                    "top20_scores": top20_scores,
                    "scores_raw": True,
                }
            },
        },

        "claims": claims,

        "label_faithful": label_faithful,
        "label_factual": label_factual,
    }
    return record


def generate_synthetic_dataset(n: int = 200, seed: int = 0, split: str = "dev") -> list[dict]:
    rng = np.random.default_rng(seed)
    return [make_record(rng, qid=f"synthq_{i:05d}", split=split) for i in range(n)]
