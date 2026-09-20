"""
Пересчёт части Table 2 статьи FRANQ (arXiv 2505.21072) из уже сохранённых
в репозитории stat-ml/rag_uncertainty precomputed-полей, без модели и без GPU.

Цель: A2-смоук — «кодовая база выдаёт число, расхождение с публикацией
в пределах шума либо объяснено» — для той части реестра, которая уже
лежит в датасете как unc_* поля.
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DATASET_DIR = Path("/tmp/rag_uncertainty/claim_level/dataset")

LABEL_RE = re.compile(r'\("([^"]+)",\s*"(True|False)"\)')


def parse_factuality(auto_label: str):
    matches = LABEL_RE.findall(auto_label)
    if not matches:
        return None
    _, factual_str = matches[-1]  # берём последнее совпадение (после рассуждений модели)
    return factual_str == "True"


def load_claims(model_file: str):
    with open(DATASET_DIR / model_file) as f:
        data = json.load(f)
    rows = []
    for key in sorted(k for k in data.keys() if k.isdigit()):
        ex = data[key]
        n = len(ex["claims"])
        for i in range(n):
            factual = parse_factuality(ex["auto_labels"][i])
            if factual is None:
                continue
            rows.append({
                "qid": key,
                "factual": factual,
                "neg_log_mp": ex["unc_neg_log_MP"][i],
                "perplexity": ex["unc_perplexity"][i],
                "entropy": ex["unc_entropy"][i],
                "neg_ccp": ex["unc_neg_CCP"][i],
            })
    return rows


def auroc_false_is_positive(scores, factual_flags):
    # Table 2: "false claims are designated as the positive class"
    y = [0 if f else 1 for f in factual_flags]  # positive (1) = false claim
    return roc_auc_score(y, scores)


PAPER_TABLE2 = {
    "Llama-3.2-3B-Instruct.json": {
        "Max Claim Probability": 0.497,
        "Perplexity": 0.477,
        "Mean Token Entropy": 0.562,
        "CCP": 0.587,
    },
    "Falcon3-3B-Base.json": {
        "Max Claim Probability": 0.678,
        "Perplexity": 0.636,
        "Mean Token Entropy": 0.646,
        "CCP": 0.641,
    },
}

for model_file in ["Llama-3.2-3B-Instruct.json", "Falcon3-3B-Base.json"]:
    rows = load_claims(model_file)
    n_total = sum(len(json.load(open(DATASET_DIR / model_file))[k]["claims"])
                  for k in json.load(open(DATASET_DIR / model_file)).keys() if k.isdigit())
    print(f"\n=== {model_file} ===")
    print(f"claims total={n_total}, claims with parseable factuality label={len(rows)}")

    factual = [r["factual"] for r in rows]
    n_false = sum(1 for f in factual if not f)
    print(f"factual=True: {len(rows) - n_false}, factual=False: {n_false}")

    checks = {
        "Max Claim Probability": [r["neg_log_mp"] for r in rows],   # higher neg_log_mp = less confident = more likely false
        "Perplexity": [r["perplexity"] for r in rows],
        "Mean Token Entropy": [r["entropy"] for r in rows],
        "CCP": [r["neg_ccp"] for r in rows],  # neg_CCP: higher = more uncertain = more likely false
    }

    for name, scores in checks.items():
        our_auc = auroc_false_is_positive(scores, factual)
        paper_auc = PAPER_TABLE2[model_file][name]
        diff = our_auc - paper_auc
        print(f"  {name:22s}  ours={our_auc:.3f}  paper={paper_auc:.3f}  diff={diff:+.3f}")
