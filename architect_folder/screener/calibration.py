"""
Этап калибровки (решение 2026-10-05): у каждого сигнала две версии — «как
есть» и откалиброванная, плюс конформные гарантии. Закрывает карточки
conformal-scalar-wrapper, conformal-abstention-guarantees,
conformal-rag-group-conditional и retrieval-часть traq-conformal-passage-set
(researcher_folder/B2_cards.md) не как отдельные сигналы, а как надстройку над
всеми сигналами сразу.

Протокол: только short-form; подгонка на dev (224), оценка на test (300).
Сигналы, подогнанные на dev (density, SE-probes из local_cards), здесь не
участвуют — у них нет dev-значений (для них нужен трёхчастный сплит).

Для каждого сигнала (risk: выше = вероятнее ошибка):
  - raw: AUROC на test; для сигналов-вероятностей (p_true, verbalized) ещё
    ECE/Brier как есть;
  - Platt и изотоническая регрессия на dev -> P(ошибка) на test: ECE (10
    бинов), Brier, AUROC (у изотонической AUROC может просесть из-за плато);
  - conformal-scalar-wrapper / conformal-abstention-guarantees: порог
    «отвечаем, если risk <= τ» подбирается на dev Learn-then-Test'ом
    (Angelopoulos et al. 2021): кандидаты τ от строгого к мягкому, для
    каждого — верхняя граница Клоппера-Пирсона (δ=0.1) на долю ошибок среди
    принятых, fixed-sequence: берём последний τ, у которого граница <= α.
    На test: доля ошибок среди принятых (должна быть <= α; критерий убийства
    карточки — превышение больше 5 п.п.), доля принятых (participation), и
    abstention отдельно на вопросах с golden-пассажем в топ-k и без
    (критерий conformal-abstention-guarantees: если одинаково — гарантия не
    про «в контексте нет ответа»);
  - conformal-rag-group-conditional: то же, но τ подбирается отдельно в
    каждой группе source (Mondrian) — сравнение худшей группы на test для
    общего и группового порога.
traq (retrieval-часть): нонконформность = скор лучшего golden-пассажа (is_golden
= DPR hasanswer); порог на dev для покрытия golden-пассажа >= 1-α; размер
множества пассажей выше порога на test как сигнал. В дампе только топ-5, так
что достижимое покрытие ограничено долей вопросов с golden в топ-5.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from . import metrics

ALPHAS = (0.2, 0.3, 0.4)
DELTA = 0.1
N_BINS = 10
PROB_SIGNALS = {"p_true_rag": "confidence", "p_true_cb": "confidence", "verbalized_conf_rag": "confidence"}


def ece(p_err: np.ndarray, err: np.ndarray, n_bins: int = N_BINS) -> float:
    bins = np.clip((p_err * n_bins).astype(int), 0, n_bins - 1)
    return float(sum(abs(p_err[bins == b].mean() - err[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def brier(p_err: np.ndarray, err: np.ndarray) -> float:
    return float(np.mean((p_err - err) ** 2))


def clopper_pearson_upper(k: int, n: int, delta: float = DELTA) -> float:
    return 1.0 if n == 0 else float(beta.ppf(1 - delta, k + 1, n - k)) if k < n else 1.0


COVERAGE_GRID = np.clip(np.arange(0.10, 1.0001, 0.05), 0.0, 1.0)


def ltt_threshold(risk: np.ndarray, err: np.ndarray, alpha: float, delta: float = DELTA) -> float:
    """Learn-then-Test, fixed sequence от строгого τ к мягкому: последний τ,
    для которого UCB доли ошибок среди принятых <= α. -inf — не отвечаем никогда.
    Кандидаты — квантили риска по сетке долей принятых (от 10%): с самых
    строгих порогов (1-2 принятых) граница Клоппера-Пирсона огромна, и
    fixed-sequence остановился бы на первом же шаге."""
    tau = -np.inf
    for t in np.quantile(risk, COVERAGE_GRID):
        acc = risk <= t
        if clopper_pearson_upper(int(err[acc].sum()), int(acc.sum()), delta) <= alpha:
            tau = t
        else:
            break
    return tau


def selective(risk: np.ndarray, err: np.ndarray, tau: float) -> dict:
    acc = risk <= tau
    return {"coverage": float(acc.mean()),
            "sel_error": float(err[acc].mean()) if acc.any() else float("nan")}


def calibrate_signal(r_dev, err_dev, r_test, err_test, golden_test, group_dev, group_test,
                     raw_prob_err_test=None) -> dict:
    out = {"auroc_raw": metrics.auroc(r_test, 1 - err_test)}
    if raw_prob_err_test is not None:
        out["ece_raw"], out["brier_raw"] = ece(raw_prob_err_test, err_test), brier(raw_prob_err_test, err_test)

    platt = LogisticRegression().fit(r_dev.reshape(-1, 1), err_dev)
    p = platt.predict_proba(r_test.reshape(-1, 1))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(r_dev, err_dev)
    q = iso.predict(r_test)
    out.update(ece_platt=ece(p, err_test), brier_platt=brier(p, err_test),
               ece_iso=ece(q, err_test), brier_iso=brier(q, err_test),
               auroc_iso=metrics.auroc(q, 1 - err_test))

    out["conformal"] = {}
    for a in ALPHAS:
        tau = ltt_threshold(r_dev, err_dev, a)
        s = selective(r_test, err_test, tau)
        acc = r_test <= tau
        s["abstain_with_golden"] = float((~acc[golden_test]).mean()) if golden_test.any() else float("nan")
        s["abstain_without_golden"] = float((~acc[~golden_test]).mean()) if (~golden_test).any() else float("nan")
        # Mondrian: порог отдельно по группам
        worst_marg, worst_mond, cov_mond = -np.inf, -np.inf, 0
        for g in np.unique(group_test):
            mt, md = group_test == g, group_dev == g
            sm = selective(r_test[mt], err_test[mt], tau)["sel_error"]
            tau_g = ltt_threshold(r_dev[md], err_dev[md], a) if md.any() else -np.inf
            sg = selective(r_test[mt], err_test[mt], tau_g)
            worst_marg = max(worst_marg, np.nan_to_num(sm, nan=-np.inf))
            worst_mond = max(worst_mond, np.nan_to_num(sg["sel_error"], nan=-np.inf))
            cov_mond += sg["coverage"] * mt.sum()
        s.update(worst_group_err_marginal=float(worst_marg), worst_group_err_mondrian=float(worst_mond),
                 coverage_mondrian=float(cov_mond / len(group_test)))
        out["conformal"][a] = s
    return out


def traq_retrieval(scores_dev, golden_dev, scores_test, golden_test, err_test, alpha: float) -> dict:
    """scores_*: список массивов скоров топ-k по вопросам; golden_*: списки флагов."""
    best_golden = np.array([max((s for s, g in zip(sc, gl) if g), default=-np.inf)
                            for sc, gl in zip(scores_dev, golden_dev)])
    n = len(best_golden)
    k = int(np.floor(alpha * (n + 1)))  # допускаем k непокрытых (с поправкой n+1)
    tau = np.sort(best_golden)[k - 1] if k >= 1 else -np.inf
    if not np.isfinite(tau):
        return {"alpha": alpha, "tau": float("-inf"), "note": "α меньше доли вопросов без golden в топ-k на dev"}
    size = np.array([int(np.sum(np.asarray(sc) >= tau)) for sc in scores_test])
    covered = np.array([any(s >= tau and g for s, g in zip(sc, gl)) for sc, gl in zip(scores_test, golden_test)])
    return {"alpha": alpha, "tau": float(tau), "coverage_test": float(covered.mean()),
            "mean_set_size": float(size.mean()),
            # больше пассажей над порогом = ретривал увереннее -> confidence
            "auroc_set_size": metrics.auroc(-size.astype(float), 1 - err_test)}


def run_calibration(risk: dict[str, np.ndarray], labels: np.ndarray, aux: dict, polarity: dict[str, str]) -> dict:
    short = aux["form"] == "short"
    dev, test = short & (aux["split"] == "dev"), short & (aux["split"] == "test")
    ok_label = labels >= 0
    dev &= ok_label
    test &= ok_label
    err = 1 - labels
    out = {"signals": {}, "n_dev": int(dev.sum()), "n_test": int(test.sum())}
    for name, r in sorted(risk.items()):
        if np.isnan(r[dev]).mean() > 0.1 or np.isnan(r[test]).mean() > 0.1 or np.nanstd(r[dev]) == 0:
            continue
        d, t = dev & ~np.isnan(r), test & ~np.isnan(r)
        raw_p = None
        if name in PROB_SIGNALS:
            raw_p = 1 + r[t]  # risk = -confidence -> P(ошибка) = 1 - confidence
        out["signals"][name] = calibrate_signal(r[d], err[d], r[t], err[t], aux["golden_any"][t],
                                                aux["source"][d], aux["source"][t], raw_p)
    sd = [aux["passage_scores"][i] for i in np.where(dev)[0]]
    gd = [aux["golden_flags"][i] for i in np.where(dev)[0]]
    st = [aux["passage_scores"][i] for i in np.where(test)[0]]
    gt = [aux["golden_flags"][i] for i in np.where(test)[0]]
    out["traq"] = [traq_retrieval(sd, gd, st, gt, err[test], a) for a in (0.4, 0.5)]
    return out


def _f(x: float, fmt: str = ".2f") -> str:
    return "—" if x is None or not np.isfinite(x) else format(x, fmt)


def print_calibration(target: str, res: dict) -> None:
    print(f"\n=== Калибровка, target={target} (short-form: dev {res['n_dev']} -> test {res['n_test']}) ===")
    print("  Конформный порог (LTT, δ=0.1): «доля ответов (ошибка среди отвеченных)» на test при α=0.2/0.3/0.4;\n"
          "  при α=0.3 ещё доля отказов на вопросах с golden-пассажем / без и худшая группа source: общий / Mondrian")
    print(f"  {'сигнал':30s} AUROC | ECE Platt/iso | Brier | α=0.2       α=0.3       α=0.4       | отказ g/без | худш. гр.")
    for name, s in sorted(res["signals"].items(), key=lambda kv: -kv[1]["auroc_raw"]):
        cs = "  ".join(f"{_f(s['conformal'][a]['coverage'])} ({_f(s['conformal'][a]['sel_error'])})" for a in ALPHAS)
        c3 = s["conformal"][0.3]
        raw = f"  [как есть: ECE {s['ece_raw']:.3f}]" if "ece_raw" in s else ""
        print(f"  {name:30s} {s['auroc_raw']:.3f} | {s['ece_platt']:.3f}/{s['ece_iso']:.3f}   | {s['brier_platt']:.3f} | "
              f"{cs} | {_f(c3['abstain_with_golden'])}/{_f(c3['abstain_without_golden'])}   | "
              f"{_f(c3['worst_group_err_marginal'])}/{_f(c3['worst_group_err_mondrian'])}{raw}")
    for tr in res["traq"]:
        if "coverage_test" in tr:
            print(f"  traq retrieval α={tr['alpha']}: покрытие golden на test {tr['coverage_test']:.3f} "
                  f"(цель {1 - tr['alpha']:.2f}), средний размер множества {tr['mean_set_size']:.2f}, "
                  f"AUROC размера множества {tr['auroc_set_size']:.3f}")
        else:
            print(f"  traq retrieval α={tr['alpha']}: {tr['note']}")
