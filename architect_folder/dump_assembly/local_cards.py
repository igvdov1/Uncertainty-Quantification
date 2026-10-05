"""
Бесплатные карточки банка B, которым нужна подгонка на dev (не на метках —
на самих hidden states / semantic entropy), локально на CPU за один проход
по дампу. Результат — sidecar со скалярами ТОЛЬКО для test-сплита short-form:
подгонка на dev, оценка на test, скринер видит NaN на dev и сравнивает с
бейзлайнами на тех же 300 записях.

Карточки:
  density-based-dev-embeddings — расстояние Махаланобиса (как MD в
      LM-Polygraph: эмбеддинг ответа = среднее hidden_states_last по
      сгенерированным токенам) до распределения dev. 4096 измерений на 224
      dev-записях — ковариация вырождена, поэтому StandardScaler -> PCA(50)
      -> Ledoit-Wolf. Отдельно rag и closed_book, плюс разность (карточка:
      «разница между ними как сигнал»).
  semantic-entropy-probes — Kossen et al. 2024 (arXiv 2406.15927): SE на dev
      бинаризуется порогом, минимизирующим внутриклассовую дисперсию (их
      «best split»), логистическая регрессия по hidden state второго с конца
      сгенерированного токена (их SLT) предсказывает «высокая SE». Выход —
      вероятность. Отличие от статьи: у нас только последний слой (в дампе
      нет других), и перед регрессией тот же StandardScaler -> PCA(50) —
      224 примера на 4096 признаков.

Запуск (из architect_folder/):
    python -m dump_assembly.local_cards --dump dump_pilot_v4.jsonl \
        --sidecar semantic_clusters.jsonl --out local_cards.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from screener import registry, run_screener

LONGFORM = "franq_longform"
N_PCA = 50
MODES = (("rag", "rag"), ("closed_book", "cb"))


# ---- формулы (без дампа) ----------------------------------------------------

def best_split_threshold(values: np.ndarray) -> float:
    """Порог бинаризации SE как у Kossen et al.: минимизирует сумму
    внутриклассовых квадратичных отклонений двух групп."""
    v = np.sort(np.asarray(values, dtype=float))
    best_t, best_cost = v[0], np.inf
    for i in range(1, len(v)):
        lo, hi = v[:i], v[i:]
        cost = ((lo - lo.mean()) ** 2).sum() + ((hi - hi.mean()) ** 2).sum()
        if cost < best_cost:
            best_cost, best_t = cost, (v[i - 1] + v[i]) / 2
    return float(best_t)


class Mahalanobis:
    def __init__(self, n_components: int = N_PCA):
        self.n_components = n_components

    def fit(self, x: np.ndarray) -> "Mahalanobis":
        k = min(self.n_components, x.shape[0] - 1, x.shape[1])
        self.proj = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0)).fit(x)
        z = self.proj.transform(x)
        self.cov = LedoitWolf().fit(z)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        return np.sqrt(self.cov.mahalanobis(self.proj.transform(x)))


def fit_sep(x: np.ndarray, se: np.ndarray) -> tuple:
    t = best_split_threshold(se)
    y = (se > t).astype(int)
    if len(set(y)) < 2:
        raise ValueError("SE на dev не разделяется на два класса")
    k = min(N_PCA, x.shape[0] - 1, x.shape[1])
    clf = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                        LogisticRegression(max_iter=5000)).fit(x, y)
    return clf, t


# ---- сбор признаков из дампа ------------------------------------------------

def record_features(r: dict) -> dict:
    lab = r.get("label_factual")
    out = {"qid": r["qid"], "split": r["split"], "source": r["source"],
           "label_factual": None if lab is None else int(lab),
           "len_norm_rag": registry.len_norm_nll(r, "rag")}
    for mode, suf in MODES:
        h = np.asarray(r[mode]["hidden_states_last"], dtype=np.float32)
        out[f"mean_{suf}"] = h.mean(axis=0)
        out[f"slt_{suf}"] = h[-2] if len(h) >= 2 else h[-1]
        try:
            out[f"se_{suf}"] = registry.semantic_entropy(r, mode)
        except (KeyError, IndexError):
            out[f"se_{suf}"] = float("nan")
    return out


def compute(features: list[dict]) -> tuple[list[dict], dict]:
    short = [f for f in features if f["source"] != LONGFORM]
    dev = [f for f in short if f["split"] == "dev"]
    test = [f for f in short if f["split"] == "test"]
    sig = {f["qid"]: {} for f in test}
    report = {"n_dev": len(dev), "n_test": len(test)}

    for _, suf in MODES:
        md = Mahalanobis().fit(np.stack([f[f"mean_{suf}"] for f in dev]))
        for f, v in zip(test, md.score(np.stack([f[f"mean_{suf}"] for f in test]))):
            sig[f["qid"]][f"md_{suf}"] = float(v)

        dev_ok = [f for f in dev if not np.isnan(f[f"se_{suf}"])]
        clf, t = fit_sep(np.stack([f[f"slt_{suf}"] for f in dev_ok]), np.array([f[f"se_{suf}"] for f in dev_ok]))
        probs = clf.predict_proba(np.stack([f[f"slt_{suf}"] for f in test]))[:, 1]
        for f, p in zip(test, probs):
            sig[f["qid"]][f"sep_{suf}"] = float(p)
        se_test = np.array([f[f"se_{suf}"] for f in test])
        ok = ~np.isnan(se_test)
        # критерий убийства карточки: корреляция пробы с полной SE на test < 0.7
        report[f"sep_{suf}_spearman_with_se_test"] = float(spearmanr(probs[ok], se_test[ok]).correlation)
        report[f"sep_{suf}_threshold"] = t
        report[f"sep_{suf}_dev_pos_rate"] = float(np.mean([f[f"se_{suf}"] > t for f in dev_ok]))

    for q in sig:
        sig[q]["md_diff"] = sig[q]["md_rag"] - sig[q]["md_cb"]
    rows = [{"qid": q, "signals": s} for q, s in sig.items()]
    return rows, report


def cp_internal_report(features: list[dict], alpha: float = 0.3, seed: int = 0) -> dict:
    """cp-internal-representations: конформный отбор ответов с нонконформностью
    по внутренним представлениям (Махаланобис среднего hidden state) против
    нонконформности по логпробам (NLL). Трёхчастный сплит: dev short-form
    делится пополам — A строит плотность hidden states, B калибрует порог
    (Learn-then-Test, screener/calibration.py); оценка на test.
    Критерий убийства карточки: покрытие (доля отвеченных) и ошибка среди
    отвеченных не отличаются между двумя нонконформностями."""
    from screener.calibration import ltt_threshold, selective
    short = [f for f in features if f["source"] != LONGFORM and f["label_factual"] is not None]
    dev = [f for f in short if f["split"] == "dev"]
    test = [f for f in short if f["split"] == "test"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dev))
    part_a = [dev[i] for i in idx[: len(dev) // 2]]
    part_b = [dev[i] for i in idx[len(dev) // 2:]]
    md = Mahalanobis().fit(np.stack([f["mean_rag"] for f in part_a]))
    err_b = np.array([1 - f["label_factual"] for f in part_b])
    err_t = np.array([1 - f["label_factual"] for f in test])
    scores = {
        "hidden (Махаланобис)": (md.score(np.stack([f["mean_rag"] for f in part_b])),
                                 md.score(np.stack([f["mean_rag"] for f in test]))),
        "logprob (NLL)": (np.array([f["len_norm_rag"] for f in part_b]), np.array([f["len_norm_rag"] for f in test])),
    }
    rep = {"alpha": alpha, "n": {"A": len(part_a), "B": len(part_b), "test": len(test)}}
    for name, (sb, st) in scores.items():
        tau = ltt_threshold(sb, err_b, alpha)
        rep[name] = selective(st, err_t, tau)
    return rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path, action="append", default=[],
                        help="semantic_clusters.jsonl — таргет для semantic-entropy-probes")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    records = run_screener.iter_jsonl(args.dump)
    if args.sidecar:
        records = run_screener.attach_sidecars(records, run_screener.load_sidecars(args.sidecar))
    features = [record_features(r) for r in records]
    rows, report = compute(features)
    report["cp_internal"] = cp_internal_report(features)
    with open(args.out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Готово: {len(rows)} test-записей -> {args.out}")


if __name__ == "__main__":
    main()
