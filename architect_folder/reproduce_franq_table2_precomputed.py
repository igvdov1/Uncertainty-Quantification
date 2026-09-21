"""
Пересчёт части Table 2 статьи FRANQ (arXiv 2505.21072) из уже сохранённых
в репозитории stat-ml/rag_uncertainty precomputed-полей, без модели и без GPU.

v2: воспроизводит ТОЧНЫЙ train/test сплит из кода авторов, а не наивный
подсчёт по всем клеймам разом (v1, см. A2_findings.md §3, дал системное
расхождение: у Llama выше публикации, у Falcon — ниже, с переворотом
порядка методов).

Сплит восстановлен по коду репозитория (не по прозе статьи, которая
формулирует это неточно — "500 claims for training", тогда как в коде
`--test-size 500` буквально значит размер TEST, а не train):

  claim_level/run_train.py::shuffle()
    — перемешивает ВОПРОСЫ (не отдельные клеймы), np.random.seed(42),
      блоками по числу клеймов на вопрос (естественно вытекает из формата
      per-model JSON: {"0": {...}, "1": {...}, ...}).
  claim_level/plot_utils.py::calc_metrics()
    — test = ПОСЛЕДНИЕ test_size=500 клеймов после перемешивания;
      ВСЕ методы (включая нетренируемые baseline'ы вроде MCP/Perplexity/
      CCP) считаются только на этом test-срезе, не на всём датасете;
      NaN-метки не выбрасываются до сплита — маскируются после среза,
      чтобы не сдвинуть позиции внутри перемешанного блока.
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

DATASET_DIR = Path("/tmp/rag_uncertainty/claim_level/dataset")
TEST_SIZE = 500
SHUFFLE_SEED = 42

LABEL_RE = re.compile(r'\("([^"]+)",\s*"(True|False)"\)')


def parse_factuality(auto_label: str) -> float:
    """True->0.0, False->1.0 (та же конвенция, что tgt в calc_metrics:
    0 = True/факт верен, 1 = False/факт неверен = положительный класс
    для AUROC). Неразобранное -> NaN, как в оригинале (не выбрасывается,
    чтобы не сдвинуть позиции клеймов относительно перемешивания)."""
    matches = LABEL_RE.findall(auto_label)
    if not matches:
        return float("nan")
    _, factual_str = matches[-1]
    return 0.0 if factual_str == "True" else 1.0


def load_flat_arrays(model_file: str):
    """Возвращает (n_claims_per_question, tgt, scores_dict) в ИСХОДНОМ
    порядке вопросов 0..75 — как в per-model JSON, до перемешивания."""
    with open(DATASET_DIR / model_file) as f:
        data = json.load(f)
    keys = sorted((k for k in data.keys() if k.isdigit()), key=int)

    n_claims_per_q = []
    tgt, neg_log_mp, perplexity, entropy, neg_ccp = [], [], [], [], []
    for key in keys:
        ex = data[key]
        n = len(ex["claims"])
        n_claims_per_q.append(n)
        for i in range(n):
            tgt.append(parse_factuality(ex["auto_labels"][i]))
            neg_log_mp.append(ex["unc_neg_log_MP"][i])
            perplexity.append(ex["unc_perplexity"][i])
            entropy.append(ex["unc_entropy"][i])
            neg_ccp.append(ex["unc_neg_CCP"][i])

    scores = {
        "Max Claim Probability": np.array(neg_log_mp),
        "Perplexity": np.array(perplexity),
        "Mean Token Entropy": np.array(entropy),
        "CCP": np.array(neg_ccp),
    }
    return np.array(n_claims_per_q), np.array(tgt), scores


def shuffle_like_franq(n_claims_per_q: np.ndarray, tgt: np.ndarray, scores: dict, seed: int = SHUFFLE_SEED):
    """Точная копия claim_level/run_train.py::shuffle() — перемешивает
    вопросы, а не отдельные клеймы, блоками по n_claims_per_q."""
    n_q = len(n_claims_per_q)
    n_claims_cum = np.cumsum(n_claims_per_q)

    np.random.seed(seed)  # тот же вызов, что в оригинале (глобальный RandomState, не Generator)
    order = np.arange(n_q)
    np.random.shuffle(order)

    def reorder(flat: np.ndarray) -> np.ndarray:
        return np.concatenate([
            flat[n_claims_cum[i] - n_claims_per_q[i]: n_claims_cum[i]]
            for i in order
        ])

    tgt_shuffled = reorder(tgt)
    scores_shuffled = {name: reorder(arr) for name, arr in scores.items()}
    return tgt_shuffled, scores_shuffled


def auroc_on_test_split(model_file: str, test_size: int = TEST_SIZE):
    n_claims_per_q, tgt, scores = load_flat_arrays(model_file)
    tgt_s, scores_s = shuffle_like_franq(n_claims_per_q, tgt, scores)

    total = len(tgt_s)
    train_size = total - test_size
    test_tgt = tgt_s[train_size:]
    test_mask = ~np.isnan(test_tgt)

    print(f"  train_size={train_size}, test_size={test_size}, "
          f"test с валидной меткой={test_mask.sum()}/{test_size}")

    results = {}
    for name, arr in scores_s.items():
        test_scores = arr[train_size:][test_mask]
        results[name] = roc_auc_score(test_tgt[test_mask], test_scores)
    return results


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

if __name__ == "__main__":
    for model_file in ["Llama-3.2-3B-Instruct.json", "Falcon3-3B-Base.json"]:
        print(f"\n=== {model_file} ===")
        ours = auroc_on_test_split(model_file)
        for name, our_auc in ours.items():
            paper_auc = PAPER_TABLE2[model_file][name]
            diff = our_auc - paper_auc
            flag = "" if abs(diff) < 0.02 else "  <-- вне шума"
            print(f"  {name:22s}  ours={our_auc:.3f}  paper={paper_auc:.3f}  diff={diff:+.3f}{flag}")
