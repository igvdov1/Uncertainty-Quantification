"""
Каркас скринера (A3): читает записи дампа (или синтетику), считает весь
реестр дешёвых бейзлайнов, таблицу метрик в обоих таргетах, корреляционную
матрицу, парный bootstrap для топ-2 методов и внутри-инстансное
ранжирование на уровне клеймов. Ничего не знает о происхождении записи —
синтетика и настоящий дамп проходят один и тот же путь.

CLI:
    python -m screener.run_screener                        # синтетика, 200 вопросов
    python -m screener.run_screener --input dump.jsonl      # настоящий дамп
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import bootstrap, correlation, metrics, registry, schema

TARGETS = ("faithful", "factual")


def iter_jsonl(path: Path):
    """Генератор, не список — записи с тяжёлыми полями (hidden_states,
    attention) не накапливаются в памяти все разом. На реальном дампе
    (600 записей, несколько ГБ на диске уже даже без attention_by_head)
    полный список в памяти раздувается в разы против веса на диске
    из-за оверхеда Python-объектов — на этом падал OOM."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_signal_table_and_claims(records_iter):
    """Один проход по records_iter (не список — генератор или список,
    без разницы, но каждую запись отпускаем сразу после извлечения
    нужного). Возвращает и таблицу сигналов, и лёгкие данные для
    claim_level_kendall (token_logprobs + claims, без тяжёлых полей) —
    отдельного второго прохода по тяжёлым записям больше нет."""
    rows, qids, claim_data = [], [], []
    labels_lists: dict[str, list[int]] = {t: [] for t in TARGETS}

    for r in records_iter:
        schema.validate_record(r)
        rows.append(registry.compute_all_signals(r))
        qids.append(r["qid"])
        for t in TARGETS:
            labels_lists[t].append(schema.label(r, t))
        claim_data.append((r["rag"]["token_logprobs"], r.get("claims", [])))
        # r больше нигде не удерживается — после этой итерации сборщик
        # мусора может забрать hidden_states/attention/token_topk и т.д.

    names = sorted(rows[0].keys())
    table = {name: np.array([row[name] for row in rows], dtype=float) for name in names}
    labels_by_target = {t: np.array(v) for t, v in labels_lists.items()}
    return table, qids, labels_by_target, claim_data


def to_risk_scores(table: dict[str, np.ndarray], polarity: dict[str, str]) -> dict[str, np.ndarray]:
    """Инвертирует confidence-сигналы, чтобы везде дальше был риск-скор:
    выше = вероятнее ошибка."""
    risk = {}
    for name, values in table.items():
        pol = polarity.get(name, "uncertainty")
        risk[name] = -values if pol == "confidence" else values
    return risk


def compute_metrics_table(risk: dict[str, np.ndarray], labels_by_target: dict[str, np.ndarray]) -> dict:
    results: dict[str, dict] = {}
    for target, labels in labels_by_target.items():
        results[target] = {}
        for name, scores in risk.items():
            mask = ~np.isnan(scores)
            if mask.sum() < 2 or len(set(labels[mask])) < 2:
                continue
            s, l = scores[mask], labels[mask]
            results[target][name] = {
                "auroc": metrics.auroc(s, l),
                "aurc": metrics.aurc(s, l),
                "prr": metrics.prr(s, l),
                "coverage_at_5pct_risk": metrics.coverage_at_risk(s, l, 0.05),
                "n": int(mask.sum()),
            }
    return results


def claim_level_kendall(claim_data: list[tuple[list[float], list[dict]]], target: str = "factual") -> float:
    """Демонстрация внутри-инстансного ранжирования (бриф, раздел 6):
    risk-скор на клейм = средний NLL токенов его спана в rag-ветке.
    claim_data — лёгкие (token_logprobs, claims) пары из
    build_signal_table_and_claims, не полные записи с тяжёлыми полями."""
    taus = []
    for lp, cl in claim_data:
        if len(cl) < 2:
            continue
        scores, labels = [], []
        for c in cl:
            # не у всех источников клеймы выровнены на token_span (напр.
            # long-form FRANQ переиспользует их разметку без спанов на
            # наших токенах, см. cluster_runbook.md "Известные ограничения") —
            # такие клеймы пропускаем, не роняем весь прогон
            span = c.get("token_span")
            if not span:
                continue
            a, b = span
            span_lp = lp[a:b] if b > a else lp[a:min(a + 1, len(lp))]
            if not span_lp:
                continue
            scores.append(-float(np.mean(span_lp)))
            labels.append(schema.claim_label(c, target))
        tau = metrics.per_question_kendall_tau(scores, labels)
        if not np.isnan(tau):
            taus.append(tau)
    return float(np.mean(taus)) if taus else float("nan")


def run(records_iter, n_boot: int = 1000, seed: int = 0) -> dict:
    table, qids, labels_by_target, claim_data = build_signal_table_and_claims(records_iter)
    polarity = registry.signal_polarity()
    risk = to_risk_scores(table, polarity)
    metrics_table = compute_metrics_table(risk, labels_by_target)
    names, corr = correlation.spearman_matrix(risk)

    target = "factual"
    ranked = sorted(
        metrics_table[target].items(),
        key=lambda kv: kv[1]["auroc"] if not np.isnan(kv[1]["auroc"]) else -1.0,
        reverse=True,
    )
    boot_result = None
    if len(ranked) >= 2:
        name_a, name_b = ranked[0][0], ranked[1][0]
        mask = ~(np.isnan(risk[name_a]) | np.isnan(risk[name_b]))
        boot_result = {
            "pair": (name_a, name_b),
            **bootstrap.paired_bootstrap(
                risk[name_a][mask], risk[name_b][mask], labels_by_target[target][mask],
                metrics.auroc, n_boot=n_boot, seed=seed,
            ),
        }

    return {
        "n_records": len(qids),
        "metrics_table": metrics_table,
        "correlation": {"names": names, "matrix": corr},
        "paired_bootstrap_top2_factual": boot_result,
        "per_question_kendall_tau_factual_mean": claim_level_kendall(claim_data, target="factual"),
    }


def _print_report(result: dict) -> None:
    for target, table in result["metrics_table"].items():
        print(f"\n=== target: {target} ({result['n_records']} записей) ===")
        ranked = sorted(table.items(), key=lambda kv: kv[1]["auroc"] if not np.isnan(kv[1]["auroc"]) else -1.0, reverse=True)
        for name, row in ranked:
            print(f"  {name:24s} auroc={row['auroc']:.3f}  aurc={row['aurc']:.3f}  "
                  f"prr={row['prr']:.3f}  cov@5%risk={row['coverage_at_5pct_risk']:.3f}  n={row['n']}")

    bp = result["paired_bootstrap_top2_factual"]
    if bp:
        print(f"\nПарный bootstrap, target=factual, {bp['pair'][0]} vs {bp['pair'][1]}:")
        print(f"  diff(AUROC)={bp['point_diff']:+.3f}  95% CI=[{bp['ci_low']:+.3f}, {bp['ci_high']:+.3f}]  "
              f"P(A лучше B)={bp['p_a_better']:.2f}  n_boot={bp['n_boot_valid']}")

    tau = result["per_question_kendall_tau_factual_mean"]
    print(f"\nВнутри-инстансное ранжирование клеймов (mean Kendall tau, factual): {tau:.3f}")
    print(f"\nКорреляционная матрица: {len(result['correlation']['names'])} сигналов x "
          f"{len(result['correlation']['names'])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="JSONL дамп; если не задан — синтетика")
    parser.add_argument("--n-synthetic", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    if args.input:
        records_iter = iter_jsonl(args.input)  # генератор — не грузит всё в память разом
    else:
        from . import synthetic
        records_iter = synthetic.generate_synthetic_dataset(n=args.n_synthetic, seed=args.seed)

    result = run(records_iter, n_boot=args.n_boot, seed=args.seed)
    _print_report(result)


if __name__ == "__main__":
    main()
