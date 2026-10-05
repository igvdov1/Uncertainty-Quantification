"""
Локальная (CPU) подгонка и сигналы для признаков mech_cards.py -> sidecar
для скринера ({"qid", "signals"}).

Без подгонки (на всех записях — годятся и для синтеза):
  intrygue_minmax_k5, intrygue_mean_k5 — дефолт статьи: top-5 induction heads;
      min-max: max энтропии · min SinkRate; mean: mean · mean.
  lumina — 0.5·IPR − 0.5·MMD (lam=0.5 эталона); lumina_ipr, lumina_mmd.
  ats_nll, ats_max_nll, ats_entropy — после адаптивной температуры.
  src_entropy, src_given_sem — энтропия источников сэмплов и условная
      энтропия источника при смысловом кластере (source-clustering-entropy).

С подгонкой на dev short-form (значения только на test, как local_cards):
  intrygue_tuned — n голов подбирается по AUROC на dev (fit_hyperparameters эталона).
  redeep — головы/слои сортируются по AUROC на dev, top_n/top_k/β — перебор
      по сетке (в эталоне Optuna, 1000 испытаний), min-max шкалирование по dev.
      Отличие: шкалируем средние по записи, а не токен-уровневые значения.
  tad — ridge-регрессия токен-уровня (stage 1: внимание + вероятности;
      stage 2: + предсказание stage 1 для предыдущего токена), таргет —
      метка записи (у Vazhentsev — sim(y, y*) записи); U = 1 − среднее C.
  hack_dontknow — линейный SVM на hidden state слоя 15 (closed-book промпт):
      HK+ (жадный и 5 сэмплов closed-book все неверны по gold) против
      «стабильно верно» (все 6 верны), как в HACK; сигнал — decision function.

Запуск:
    python -m dump_assembly.mech_fit --dump dump_pilot_v5.jsonl --gold gold_answers.jsonl \\
        --mech-dir mech_out --sidecar semantic_clusters.jsonl --out mech_signals.jsonl
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LONGFORM = "franq_longform"
N_HACK_SAMPLES = 5   # HACK: жадный + 5 сэмплов (у них T=0.5; у нас сэмплы дампа при T=1)


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _entropy(ids) -> float:
    ids = [i for i in ids if i is not None and i >= 0]
    if len(ids) < 2:
        return float("nan")
    p = np.array(list(Counter(ids).values()), float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum())


def conditional_entropy(src: list[int], sem: list[int]) -> float:
    """H(источник | смысловой кластер) по сэмплам, у которых есть оба."""
    pairs = [(s, c) for s, c in zip(src, sem) if s is not None and s >= 0]
    if len(pairs) < 2:
        return float("nan")
    by_c: dict[int, list[int]] = {}
    for s, c in pairs:
        by_c.setdefault(c, []).append(s)
    n = len(pairs)
    return float(sum(len(v) / n * (0.0 if len(v) < 2 else _entropy(v)) for v in by_c.values()))


def intrygue_score(entropy: np.ndarray, sinks: np.ndarray, n: int, variant: str) -> np.ndarray:
    """entropy: (R, 2) [max, mean]; sinks: (R, H) в порядке убывания InductionScore."""
    if variant == "minmax":
        return entropy[:, 0] * sinks[:, :n].min(1)
    return entropy[:, 1] * sinks[:, :n].mean(1)


def fit_intrygue_n(entropy, sinks, err, variant: str) -> int:
    best = max(range(1, sinks.shape[1] + 1),
               key=lambda n: roc_auc_score(err, intrygue_score(entropy, sinks, n, variant)))
    return best


def fit_redeep(ecs, pks, err, betas=np.arange(0.1, 2.01, 0.1)) -> dict:
    """ecs: (R, Hc), pks: (R, L) — средние по токенам ответа. err: 1 = ошибка.
    Как ReDeEP.fit_hyperparameters эталона: сортировка голов по AUROC(−ECS),
    слоёв по AUROC(PKS), затем перебор top_n, top_k, β."""
    head_order = np.argsort([-roc_auc_score(err, -ecs[:, h]) for h in range(ecs.shape[1])], kind="stable")
    layer_order = np.argsort([-roc_auc_score(err, pks[:, l]) for l in range(pks.shape[1])], kind="stable")
    es_cum = np.cumsum(ecs[:, head_order], 1)
    kd_cum = np.cumsum(pks[:, layer_order], 1)
    best = (-1.0, None)
    for n, k in itertools.product(range(1, ecs.shape[1] + 1), range(1, pks.shape[1] + 1)):
        es, kd = es_cum[:, n - 1], kd_cum[:, k - 1]
        es_s = (es - es.min()) / (np.ptp(es) or 1.0)
        kd_s = (kd - kd.min()) / (np.ptp(kd) or 1.0)
        for b in betas:
            auc = roc_auc_score(err, kd_s - b * es_s)
            if auc > best[0]:
                best = (auc, dict(top_n=n, top_k=k, beta=float(b), es_min=es.min(), es_ptp=np.ptp(es) or 1.0,
                                  kd_min=kd.min(), kd_ptp=np.ptp(kd) or 1.0))
    p = best[1]
    p.update(head_order=head_order, layer_order=layer_order, dev_auroc=best[0])
    return p


def redeep_score(ecs, pks, p) -> np.ndarray:
    es = ecs[:, p["head_order"][:p["top_n"]]].sum(1)
    kd = pks[:, p["layer_order"][:p["top_k"]]].sum(1)
    return (kd - p["kd_min"]) / p["kd_ptp"] - p["beta"] * (es - p["es_min"]) / p["es_ptp"]


def tad_token_matrix(att: np.ndarray, logp: np.ndarray) -> np.ndarray:
    lp = np.nan_to_num(logp, nan=0.0)
    missing = np.isnan(logp[:, 1:]).astype(np.float32)
    return np.concatenate([att.astype(np.float32), lp, missing], 1)


def fit_tad(dev_feats: list[tuple[np.ndarray, np.ndarray]], dev_conf: np.ndarray, alphas=(1.0, 10.0, 100.0, 1000.0)):
    """Двухстадийный TAD. dev_feats: (att (T, F), logp (T, 1+N)) по записям; dev_conf: 1 = верный ответ."""
    X = [tad_token_matrix(a, l) for a, l in dev_feats]
    groups = np.concatenate([np.full(len(x), i) for i, x in enumerate(X)])
    y = np.concatenate([np.full(len(x), c, float) for x, c in zip(X, dev_conf)])
    Xs = np.concatenate(X)

    def cv_alpha(Xm):
        n_splits = min(5, len(X))
        best = None
        for a in alphas:
            pred = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits).split(Xm, y, groups):
                pred[te] = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(Xm[tr], y[tr]).predict(Xm[te])
            rec = np.array([1 - pred[groups == g].mean() for g in range(len(X))])
            auc = roc_auc_score(1 - dev_conf, rec)
            if best is None or auc > best[0]:
                best = (auc, a, pred)
        return best

    _, a1, oof1 = cv_alpha(Xs)
    m1 = make_pipeline(StandardScaler(), Ridge(alpha=a1)).fit(Xs, y)
    prev1 = np.concatenate([np.concatenate([[np.nan], oof1[groups == g][:-1]]) for g in range(len(X))])
    Xs2 = np.concatenate([Xs, np.nan_to_num(prev1, nan=0.5)[:, None]], 1)
    _, a2, _ = cv_alpha(Xs2)
    m2 = make_pipeline(StandardScaler(), Ridge(alpha=a2)).fit(Xs2, y)
    return m1, m2


def tad_score(feats: tuple[np.ndarray, np.ndarray], m1, m2) -> float:
    x = tad_token_matrix(*feats)
    c1 = m1.predict(x)
    prev = np.concatenate([[0.5], c1[:-1]])
    c2 = m2.predict(np.concatenate([x, prev[:, None]], 1))
    return float(1 - np.mean(c2))


def run(meta: dict[str, dict], feats: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """meta[qid]: split, source, label_factual, sem_ids, cb_correct (list[bool] | None)."""
    out = {q: {} for q in feats}
    report = {}
    qids = sorted(feats)
    # без подгонки
    for q in qids:
        f, s = feats[q], out[q]
        ent, sink = f["intrygue_entropy"], f["intrygue_sink"]
        s["intrygue_minmax_k5"] = float(ent[0] * sink[:5].min())
        s["intrygue_mean_k5"] = float(ent[1] * sink[:5].mean())
        ipr, mmd = f["lumina"]
        s["lumina_ipr"], s["lumina_mmd"] = float(ipr), float(mmd)
        s["lumina"] = float(0.5 * ipr - 0.5 * mmd) if np.isfinite(mmd) else float("nan")
        if "ats" in f:
            s["ats_nll"], s["ats_max_nll"], s["ats_entropy"] = (float(v) for v in f["ats"][:3])
        if "source_ids" in f:
            src = list(f["source_ids"])
            sem = meta[q].get("sem_ids") or []
            s["src_entropy"] = _entropy(src)
            s["src_given_sem"] = conditional_entropy(src, sem[:len(src)]) if sem else float("nan")

    # с подгонкой: dev short-form -> test short-form
    dev = [q for q in qids if meta[q]["source"] != LONGFORM and meta[q]["split"] == "dev"]
    test = [q for q in qids if meta[q]["source"] != LONGFORM and meta[q]["split"] == "test"]
    if len(dev) < 30 or len(test) < 30:
        report["note"] = f"мало dev/test для подгонки ({len(dev)}/{len(test)})"
        return out, report
    err_dev = np.array([1 - meta[q]["label_factual"] for q in dev])

    E = lambda qs: np.stack([feats[q]["intrygue_entropy"] for q in qs])
    S = lambda qs: np.stack([feats[q]["intrygue_sink"] for q in qs])
    n_best = fit_intrygue_n(E(dev), S(dev), err_dev, "minmax")
    for q, v in zip(test, intrygue_score(E(test), S(test), n_best, "minmax")):
        out[q]["intrygue_tuned"] = float(v)
    report["intrygue_n_heads"] = n_best

    ecs = lambda qs: np.stack([feats[q]["redeep_ecs"] for q in qs])
    pks = lambda qs: np.stack([feats[q]["redeep_pks"] for q in qs])
    p = fit_redeep(ecs(dev), pks(dev), err_dev)
    for q, v in zip(test, redeep_score(ecs(test), pks(test), p)):
        out[q]["redeep"] = float(v)
    report["redeep"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()
                        if k not in ("head_order", "layer_order")}
    report["redeep"]["top_heads"] = p["head_order"][:p["top_n"]].tolist()
    report["redeep"]["top_layers"] = p["layer_order"][:p["top_k"]].tolist()

    m1, m2 = fit_tad([(feats[q]["tad_att"], feats[q]["tad_logp"]) for q in dev], 1 - err_dev)
    for q in test:
        out[q]["tad"] = tad_score((feats[q]["tad_att"], feats[q]["tad_logp"]), m1, m2)

    hk = [(q, meta[q]["cb_correct"]) for q in dev if meta[q].get("cb_correct")]
    pos = [q for q, c in hk if not any(c)]          # HK+: ни одного верного
    neg = [q for q, c in hk if all(c)]              # стабильно верно
    report["hack_dev_counts"] = {"hk_plus": len(pos), "consistently_correct": len(neg)}
    if len(pos) >= 10 and len(neg) >= 10:
        X = np.stack([feats[q]["hack_h15"].astype(np.float32) for q in pos + neg])
        y = np.array([1] * len(pos) + [0] * len(neg))
        svm = make_pipeline(StandardScaler(), LinearSVC(C=0.01, max_iter=20000)).fit(X, y)
        dec = svm.decision_function(np.stack([feats[q]["hack_h15"].astype(np.float32) for q in test]))
        for q, v in zip(test, dec):
            out[q]["hack_dontknow"] = float(v)
    return out, report


def load_meta(dump: Path, gold: Path | None, sidecars: list[Path]) -> dict[str, dict]:
    from .labeling import factuality_shortform
    gold_map = {g["qid"]: g["gold_answers"] for g in iter_jsonl(gold)} if gold else {}
    sem = {}
    for p in sidecars:
        for row in iter_jsonl(p):
            if "rag" in row and "cluster_ids" in row["rag"]:
                sem[row["qid"]] = row["rag"]["cluster_ids"]
    meta = {}
    for r in iter_jsonl(dump):
        m = {"split": r["split"], "source": r["source"], "label_factual": r.get("label_factual"),
             "sem_ids": sem.get(r["qid"])}
        g = gold_map.get(r["qid"])
        if g:
            cb = [r["closed_book"]["answer"]] + r["closed_book"].get("samples", [])[:N_HACK_SAMPLES]
            m["cb_correct"] = [bool(factuality_shortform(a, g)) for a in cb]
        meta[r["qid"]] = m
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, type=Path)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--mech-dir", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path, action="append", default=[])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    meta = load_meta(args.dump, args.gold, args.sidecar)
    feats = {}
    for f in sorted(args.mech_dir.glob("*.npz")):
        with np.load(f) as z:
            feats[f.stem] = {k: z[k] for k in z.files}
    feats = {q: v for q, v in feats.items() if q in meta}
    out, report = run(meta, feats)
    with open(args.out, "w") as f:
        for q, s in out.items():
            f.write(json.dumps({"qid": q, "signals": s}) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))
    print(f"Готово: {len(out)} записей -> {args.out}")


if __name__ == "__main__":
    main()
