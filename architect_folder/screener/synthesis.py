"""
Синтез (этап после скрининга карточек): ансамбли сигналов поверх простого
NLL, с честной оценкой. Наборы признаков зафиксированы ДО просмотра
результатов синтеза — по итогам скрининга и добавочной ценности
(runner_folder/C3, C4); подбор признаков допускается только внутри CV (F).

Только short-form. Модель — логистическая регрессия на квантильно
преобразованных риск-скорах (QuantileTransformer подгоняется на трейн-фолде).
Оценка — OOF-предсказания по StratifiedKFold 5 x N_REPEATS (усреднение по
повторам): AUROC, Brier, доля ошибок среди 50% самых уверенных; Δ AUROC к
A (NLL) и B (NLL+длина) — парный bootstrap по записям. Дополнительно — dev->test
для фиксированных наборов (обучение на dev 224, оценка на test 300).

Стоимость наборов (в генерациях сверх одного жадного RAG-ответа):
  NLL, длина жадного ответа — 0;  ctx_sufficiency, p_true — +1 forward;
  длина сэмплов / SE / NCP по rag — 10 сэмплов;  closed-book сигналы —
  + жадный closed-book ответ (+10 сэмплов для ncp/mem);  CCP — +teacher-forcing.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer

from . import bootstrap, metrics

N_SPLITS, N_REPEATS = 5, 10

CONFLICT = ["ncp_cross", "ncp_cb", "ccp_diff", "mem_entropy_all", "uplift_len_norm"]
SETS: dict[str, list[str]] = {
    "A: NLL": ["len_norm_rag"],
    "B: NLL+длина": ["len_norm_rag", "mean_sample_len_rag"],
    "C: B+ctx_sufficiency": ["len_norm_rag", "mean_sample_len_rag", "ctx_sufficiency"],
    "D: C+конфликт знаний": ["len_norm_rag", "mean_sample_len_rag", "ctx_sufficiency"] + CONFLICT,
    "бюджет: NLL+длина ответа+ctx_sufficiency": ["len_norm_rag", "answer_len_rag", "ctx_sufficiency"],
}
# в «всё сразу» и отбор не берём: оракулы/дубли и подогнанные на dev (у них нет dev-значений)
EXCLUDE_FROM_POOL = {"mean_nll_rag", "mean_nll_cb", "uplift_mean_nll"}


def model() -> object:
    return make_pipeline(QuantileTransformer(n_quantiles=100, output_distribution="normal"),
                         LogisticRegression(C=1.0, max_iter=2000))


def oof(x: np.ndarray, y_err: np.ndarray, seed: int = 0, n_repeats: int = N_REPEATS, select=None) -> tuple:
    """OOF P(ошибка). select(x_tr, y_tr) -> индексы столбцов (отбор внутри фолда)."""
    out = np.zeros(len(y_err))
    chosen: list[tuple] = []
    for rep in range(n_repeats):
        for tr, te in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed + rep).split(x, y_err):
            cols = select(x[tr], y_err[tr]) if select else list(range(x.shape[1]))
            chosen.append(tuple(cols))
            out[te] += model().fit(x[tr][:, cols], y_err[tr]).predict_proba(x[te][:, cols])[:, 1]
    return out / n_repeats, chosen


def forward_select(names: list[str], max_k: int = 6, min_gain: float = 0.003, start: tuple = (0,)):
    """Жадный отбор по внутренней 3-fold CV AUROC, начиная с NLL (столбец 0)."""
    def select(x, y):
        cols = list(start)
        best = _inner_auc(x[:, cols], y)
        while len(cols) < max_k:
            gains = [(_inner_auc(x[:, cols + [j]], y), j) for j in range(x.shape[1]) if j not in cols]
            if not gains:
                break
            auc, j = max(gains)
            if auc - best < min_gain:
                break
            cols.append(j)
            best = auc
        return cols
    return select


def _inner_auc(x: np.ndarray, y: np.ndarray) -> float:
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(3, shuffle=True, random_state=123).split(x, y):
        p[te] = model().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
    return metrics.auroc(p, 1 - y)


def summarize(p_err: np.ndarray, y_err: np.ndarray) -> dict:
    order = np.argsort(p_err)
    half = order[: len(order) // 2]
    return {"auroc": metrics.auroc(p_err, 1 - y_err), "brier": float(np.mean((p_err - y_err) ** 2)),
            "err_at_50": float(y_err[half].mean())}


def run_synthesis(risk: dict[str, np.ndarray], labels: np.ndarray, aux: dict, n_boot: int = 1000,
                  seed: int = 0) -> dict:
    short = (aux["form"] == "short") & (labels >= 0)
    pool = sorted(n for n, v in risk.items()
                  if n not in EXCLUDE_FROM_POOL and not np.isnan(v[short]).any() and np.std(v[short]) > 0)
    y = (1 - labels[short]).astype(int)
    split = aux["split"][short]
    res: dict = {"n": int(short.sum()), "pool": pool, "sets": {}}

    preds = {}
    for name, cols in SETS.items():
        if any(c not in pool for c in cols):
            res["sets"][name] = {"missing": [c for c in cols if c not in pool]}
            continue
        x = np.column_stack([risk[c][short] for c in cols])
        preds[name], _ = oof(x, y, seed)
        res["sets"][name] = summarize(preds[name], y)
        # dev -> test
        d, t = split == "dev", split == "test"
        if d.sum() < 50 or t.sum() < 50:
            continue
        p_t = model().fit(x[d], y[d]).predict_proba(x[t])[:, 1]
        base_t = model().fit(x[d][:, :1], y[d]).predict_proba(x[t][:, :1])[:, 1]
        res["sets"][name]["dev_test_auroc"] = metrics.auroc(p_t, 1 - y[t])
        res["sets"][name]["dev_test_nll_auroc"] = metrics.auroc(base_t, 1 - y[t])

    x_all = np.column_stack([risk[c][short] for c in pool])
    preds["E: всё сразу"], _ = oof(x_all, y, seed)
    res["sets"]["E: всё сразу"] = summarize(preds["E: всё сразу"], y) | {"k": len(pool)}

    nll_first = ["len_norm_rag"] + [c for c in pool if c != "len_norm_rag"]
    x_sel = np.column_stack([risk[c][short] for c in nll_first])
    preds["F: отбор внутри CV"], chosen = oof(x_sel, y, seed, n_repeats=2, select=forward_select(nll_first))
    freq: dict[str, int] = {}
    for cols in chosen:
        for j in cols:
            freq[nll_first[j]] = freq.get(nll_first[j], 0) + 1
    res["sets"]["F: отбор внутри CV"] = summarize(preds["F: отбор внутри CV"], y) | {
        "selected_freq": {k: v / len(chosen) for k, v in sorted(freq.items(), key=lambda kv: -kv[1])},
        "mean_k": float(np.mean([len(c) for c in chosen]))}

    for ref in ("A: NLL", "B: NLL+длина"):
        if ref not in preds:
            continue
        for name, p in preds.items():
            if name == ref:
                continue
            bs = bootstrap.paired_bootstrap(p, preds[ref], 1 - y, metrics.auroc, n_boot=n_boot, seed=seed)
            res["sets"][name][f"vs {ref}"] = (bs["point_diff"], bs["ci_low"], bs["ci_high"])
    return res


def print_synthesis(target: str, res: dict) -> None:
    print(f"\n=== Синтез, target={target} (short-form, n={res['n']}; OOF {N_SPLITS}x{N_REPEATS}) ===")
    print(f"  пул для E/F: {len(res['pool'])} сигналов")
    for name, s in res["sets"].items():
        if "missing" in s:
            print(f"  {name:42s} нет сигналов: {s['missing']}")
            continue
        line = f"  {name:42s} AUROC={s['auroc']:.3f}  Brier={s['brier']:.3f}  ошибка@50%={s['err_at_50']:.3f}"
        for ref in ("A: NLL", "B: NLL+длина"):
            if f"vs {ref}" in s:
                d, lo, hi = s[f"vs {ref}"]
                line += f"  Δ к {ref.split(':')[0]}={d:+.3f} [{lo:+.3f},{hi:+.3f}]"
        if "dev_test_auroc" in s:
            line += f"  | dev->test {s['dev_test_auroc']:.3f} (NLL {s['dev_test_nll_auroc']:.3f})"
        print(line)
        if "selected_freq" in s:
            top = ", ".join(f"{k} {v:.0%}" for k, v in list(s["selected_freq"].items())[:10])
            print(f"      отбор (среднее k={s['mean_k']:.1f}): {top}")
