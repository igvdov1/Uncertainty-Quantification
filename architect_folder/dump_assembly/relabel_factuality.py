"""
Пересчёт label_factual для уже собранного дампа после фикса
factuality_shortform (см. labeling.py). Не трогает генерацию — только
перечитывает уже сгенерированный rag.answer и пересчитывает метку по
исправленной логике. GPU не нужен.

Запуск (из architect_folder/):
    python -m dump_assembly.relabel_factuality \
        --franq-dataset-dir rag_uncertainty/claim_level/dataset \
        --dev-size 300 --test-size 300 \
        --in dump_pilot.jsonl --out dump_pilot_fixed.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .labeling import factuality_shortform
from .questions import build_pilot_question_set


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--franq-dataset-dir", required=True, type=Path)
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in", dest="in_path", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    questions = build_pilot_question_set(args.franq_dataset_dir, args.dev_size, args.test_size, args.seed)
    gold_by_qid = {q.qid: q.gold_answers for q in questions}

    n_changed = 0
    n_total = 0
    with open(args.in_path) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            n_total += 1
            if r["source"] != "franq_longform":
                gold = gold_by_qid.get(r["qid"])
                if gold is None:
                    raise KeyError(f"{r['qid']}: нет gold_answers — проверьте dev-size/test-size/seed "
                                    f"совпадают с теми, что использовались при сборке дампа")
                new_label = factuality_shortform(r["rag"]["answer"], gold)
                if new_label != r["label_factual"]:
                    n_changed += 1
                r["label_factual"] = new_label
                if r.get("claims"):
                    r["claims"][0]["label_factual"] = new_label
            fout.write(json.dumps(r) + "\n")

    print(f"Готово. {n_total} записей, у {n_changed} изменился label_factual. Записано в {args.out}")


if __name__ == "__main__":
    main()
