"""
A2 — подготовка 20-вопросного смоук-сабсета для claim-level трека FRANQ
(github.com/stat-ml/rag_uncertainty, ветка main, папка claim_level).

Загрузчик датасета (claim_level/utils.py: load_dataset) ожидает файл
<dataset_dir>/<model_basename>.json со словарём {"meta_info": {...}, "0": {...}, "1": {...}, ...}.
Скрипт берёт первые N вопросов (по умолчанию 20) по возрастанию qid и
пишет тот же формат в отдельную директорию, готовую к передаче в
--dataset-dir run_lmpoly_baselines.py / run_rag_baselines.py / run_train.py.

Важная оговорка (см. A2_findings.md): оригинальный сплит статьи — 500
claims на train, остаток на test, отбор наверняка не по первым N вопросам.
На 20 вопросах это разбиение не воспроизводится, поэтому смоук на этом
сабсете подтверждает работоспособность кода и порядок величины AUROC,
но не точное число из Table 2. Точное число — на полных 76 вопросах,
это отдельный прогон, см. раздел 3 A2_findings.md.

Запуск:
    python franq_subsample_20.py \
        --dataset-dir /path/to/rag_uncertainty/claim_level/dataset \
        --model Falcon3-3B-Base \
        --n 20 \
        --out ./smoke20
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--model", required=True,
                         help="basename без .json, напр. Falcon3-3B-Base или Llama-3.2-3B-Instruct")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0, help="для случайного отбора; None = первые N по qid")
    args = parser.parse_args()

    src_path = args.dataset_dir / f"{args.model}.json"
    with open(src_path) as f:
        data = json.load(f)

    qids = sorted((k for k in data.keys() if k.isdigit()), key=int)
    if args.seed is not None:
        import random
        rng = random.Random(args.seed)
        qids = rng.sample(qids, min(args.n, len(qids)))
        qids = sorted(qids, key=int)
    else:
        qids = qids[:args.n]

    out_data = {"meta_info": data["meta_info"]}
    for k in qids:
        out_data[k] = data[k]

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f)

    n_claims = sum(len(out_data[k]["claims"]) for k in qids)
    print(f"Wrote {len(qids)} questions ({n_claims} claims) to {out_path}")
    print(f"qids: {qids}")


if __name__ == "__main__":
    main()
