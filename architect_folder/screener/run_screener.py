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

from . import bootstrap, card_report, correlation, metrics, refusal, registry, schema

TARGETS = ("faithful", "factual")
EXCLUDED = -1  # метка исключена из оценки (см. --faithful-exclude-refusals)


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


def load_sidecars(paths: list[Path]) -> dict[str, dict]:
    """qid -> {имя_sidecar: строка}. Виды (по полям строки): semantic_clusters
    (dump_assembly/semantic_clusters.py) и signals — готовые скаляры карточек
    (dump_assembly/gpu_cards.py)."""
    by_qid: dict[str, dict] = {}
    for path in paths:
        for row in iter_jsonl(path):
            if "rag" in row and "cluster_ids" in row["rag"]:
                by_qid.setdefault(row["qid"], {})["semantic_clusters"] = row
            elif "signals" in row:
                # скаляры карточек (dump_assembly/gpu_cards.py); несколько файлов сливаются
                by_qid.setdefault(row["qid"], {}).setdefault("signals", {}).update(row["signals"])
            else:
                raise ValueError(f"{path}: неизвестный формат sidecar (qid={row.get('qid')})")
    return by_qid


def attach_sidecars(records_iter, by_qid: dict[str, dict]):
    """Подмешивает sidecar-данные в record["derived"] на лету, не трогая дамп."""
    for r in records_iter:
        extra = by_qid.get(r["qid"])
        if extra:
            r.setdefault("derived", {}).update(extra)
        yield r


def build_signal_table_and_claims(records_iter, faithful_exclude_refusals: bool = False):
    """Один проход по records_iter (не список — генератор или список,
    без разницы, но каждую запись отпускаем сразу после извлечения
    нужного). Возвращает и таблицу сигналов, и лёгкие данные для
    claim_level_kendall (token_logprobs + claims, без тяжёлых полей) —
    отдельного второго прохода по тяжёлым записям больше нет."""
    rows, qids, claim_data, sources = [], [], [], []
    labels_lists: dict[str, list[int]] = {t: [] for t in TARGETS}

    for r in records_iter:
        schema.validate_record(r)
        rows.append(registry.compute_all_signals(r))
        qids.append(r["qid"])
        sources.append(r.get("source", "?"))
        for t in TARGETS:
            if (t == "faithful" and faithful_exclude_refusals and r.get("source") != "franq_longform"
                    and refusal.is_refusal(r["rag"]["answer"])):
                labels_lists[t].append(EXCLUDED)
            else:
                labels_lists[t].append(schema.label(r, t))
        claim_data.append((r["rag"]["token_logprobs"], r.get("claims", [])))
        # r больше нигде не удерживается — после этой итерации сборщик
        # мусора может забрать hidden_states/attention/token_topk и т.д.

    names = sorted(rows[0].keys())
    table = {name: np.array([row[name] for row in rows], dtype=float) for name in names}
    labels_by_target = {t: np.array(v) for t, v in labels_lists.items()}
    return table, qids, labels_by_target, claim_data, np.array(sources)


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
            mask = ~np.isnan(scores) & (labels != EXCLUDED)
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


LEAKAGE_AUROC = 0.99


LEAKAGE_MIN_GROUP = 30


def leakage_suspects(metrics_table: dict) -> list[tuple[str, str, float]]:
    """Сигналы с AUROC >= 0.99 (или <= 0.01 — идеальный с перевёрнутой
    полярностью). На реальных данных UQ-сигнал так не разделяет —
    почти наверняка метка посчитана из этого же сигнала. Так выглядела
    утечка alignscore -> label_faithful (runner_folder/C1_faithful_label_leakage.md)."""
    out = []
    for target, table in metrics_table.items():
        for name, row in table.items():
            a = row["auroc"]
            if not np.isnan(a) and (a >= LEAKAGE_AUROC or a <= 1 - LEAKAGE_AUROC):
                out.append((target, name, a))
    return out


def leakage_suspects_by_group(risk: dict[str, np.ndarray], labels_by_target: dict[str, np.ndarray],
                              groups: np.ndarray) -> list[tuple[str, str, float]]:
    """То же внутри каждой группы (source). Утечка alignscore была видна
    только так: на short-form AUROC=1.000, а на всём дампе 0.859 — общий
    порог её не ловил."""
    out = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < LEAKAGE_MIN_GROUP:
            continue
        table = compute_metrics_table({k: v[m] for k, v in risk.items()},
                                      {t: l[m] for t, l in labels_by_target.items()})
        out.extend((f"{t}@{g}", name, a) for t, name, a in leakage_suspects(table))
    return out


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


def run(records_iter, n_boot: int = 1000, seed: int = 0, faithful_exclude_refusals: bool = False,
        cards: bool = False) -> dict:
    table, qids, labels_by_target, claim_data, sources = build_signal_table_and_claims(
        records_iter, faithful_exclude_refusals)
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
        mask = ~(np.isnan(risk[name_a]) | np.isnan(risk[name_b])) & (labels_by_target[target] != EXCLUDED)
        boot_result = {
            "pair": (name_a, name_b),
            **bootstrap.paired_bootstrap(
                risk[name_a][mask], risk[name_b][mask], labels_by_target[target][mask],
                metrics.auroc, n_boot=n_boot, seed=seed,
            ),
        }

    form = np.where(sources == "franq_longform", "long", "short")
    by_form = {
        g: compute_metrics_table({k: v[form == g] for k, v in risk.items()},
                                 {t: l[form == g] for t, l in labels_by_target.items()})
        for g in ("short", "long") if (form == g).sum() >= LEAKAGE_MIN_GROUP
    }

    cards_result = card_report.card_report(risk, labels_by_target, form == "short", n_boot=n_boot, seed=seed) \
        if cards else None

    return {
        "n_records": len(qids),
        "metrics_by_form": by_form,
        "card_report": cards_result,
        "metrics_table": metrics_table,
        "correlation": {"names": names, "matrix": corr},
        "paired_bootstrap_top2_factual": boot_result,
        "per_question_kendall_tau_factual_mean": claim_level_kendall(claim_data, target="factual"),
        "leakage_suspects": leakage_suspects(metrics_table)
                            + leakage_suspects_by_group(risk, labels_by_target, sources),
    }


def _print_report(result: dict) -> None:
    for target, table in result["metrics_table"].items():
        print(f"\n=== target: {target} ({result['n_records']} записей) ===")
        ranked = sorted(table.items(), key=lambda kv: kv[1]["auroc"] if not np.isnan(kv[1]["auroc"]) else -1.0, reverse=True)
        for name, row in ranked:
            print(f"  {name:24s} auroc={row['auroc']:.3f}  aurc={row['aurc']:.3f}  "
                  f"prr={row['prr']:.3f}  cov@5%risk={row['coverage_at_5pct_risk']:.3f}  n={row['n']}")

    # По формам отдельно: long-form (76 FRANQ) и short-form различаются и по
    # длине/NLL, и по базовой частоте меток — AUROC на смеси награждает
    # сигнал за то, что он отличает long от short.
    for g, gtable in result.get("metrics_by_form", {}).items():
        for target, table in gtable.items():
            ranked = sorted(table.items(), key=lambda kv: kv[1]["auroc"] if not np.isnan(kv[1]["auroc"]) else -1.0,
                            reverse=True)
            n = next(iter(table.values()))["n"] if table else 0
            print(f"\n--- {g}-form, target: {target} (n={n}), топ-12 ---")
            for name, row in ranked[:12]:
                print(f"  {name:30s} auroc={row['auroc']:.3f}")

    bp = result["paired_bootstrap_top2_factual"]
    if bp:
        print(f"\nПарный bootstrap, target=factual, {bp['pair'][0]} vs {bp['pair'][1]}:")
        print(f"  diff(AUROC)={bp['point_diff']:+.3f}  95% CI=[{bp['ci_low']:+.3f}, {bp['ci_high']:+.3f}]  "
              f"P(A лучше B)={bp['p_a_better']:.2f}  n_boot={bp['n_boot_valid']}")

    for target, name, a in result["leakage_suspects"]:
        print(f"\n!!! ПОДОЗРЕНИЕ НА УТЕЧКУ МЕТКИ: target={target}, {name} auroc={a:.3f} — "
              f"проверьте, не посчитана ли метка из этого сигнала")

    if result.get("card_report"):
        card_report.print_card_report(result["card_report"])

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
    parser.add_argument("--faithful-exclude-refusals", action="store_true",
                        help="исключить short-form отказы («в пассажах нет информации») из оценки faithful — "
                             "для dump_pilot_v4, см. screener/refusal.py")
    parser.add_argument("--cards", action="store_true",
                        help="вердикты по карточкам банка B (screener/card_report.py), short-form")
    parser.add_argument("--sidecar", type=Path, action="append", default=[],
                        help="доп. посчитанные поля по qid (напр. semantic_clusters.jsonl); можно несколько раз")
    args = parser.parse_args()

    if args.input:
        records_iter = iter_jsonl(args.input)  # генератор — не грузит всё в память разом
    else:
        from . import synthetic
        records_iter = synthetic.generate_synthetic_dataset(n=args.n_synthetic, seed=args.seed)

    if args.sidecar:
        records_iter = attach_sidecars(records_iter, load_sidecars(args.sidecar))

    result = run(records_iter, n_boot=args.n_boot, seed=args.seed,
                 faithful_exclude_refusals=args.faithful_exclude_refusals, cards=args.cards)
    _print_report(result)


if __name__ == "__main__":
    main()
