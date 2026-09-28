#!/usr/bin/env bash
# Одна GPU-сессия ветки C: NLI-кластеры для semantic entropy + LLM-судья faithful.
# Запуск из architect_folder/, рядом должны лежать se_input.jsonl и judge_input.jsonl
# (делаются локально командами extract, дамп на кластер не нужен).
#
#   bash scripts/run_c1_gpu.sh              # судья на HF-бэкенде, нужна GPU 80 ГБ
#   JUDGE_BACKEND=vllm JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ bash scripts/run_c1_gpu.sh   # GPU 48 ГБ
#
# Всё резюмируется: при обрыве просто запустить ещё раз.
# Вернуть ветке C: semantic_clusters.jsonl, judge_labels.jsonl, c1_calibration.txt
set -euo pipefail

JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
JUDGE_BACKEND="${JUDGE_BACKEND:-hf}"

for f in se_input.jsonl judge_input.jsonl; do
    [[ -f "$f" ]] || { echo "нет $f — скопируйте его в $(pwd)"; exit 1; }
done

echo "=== 1/4 semantic entropy: NLI-кластеры (deberta-v2-xlarge-mnli) ==="
python -m dump_assembly.semantic_clusters cluster --input se_input.jsonl --out semantic_clusters.jsonl

echo "=== 2/4 судья: клеймы FRANQ (калибровка) ==="
python -m dump_assembly.relabel_faithfulness judge --input judge_input.jsonl --out judge_labels.jsonl \
    --only-kind claim --model "$JUDGE_MODEL" --backend "$JUDGE_BACKEND"

echo "=== 3/4 калибровка ==="
python -m dump_assembly.relabel_faithfulness calibrate --input judge_input.jsonl --labels judge_labels.jsonl \
    | tee c1_calibration.txt

echo "=== 4/4 судья: short-form ответы ==="
# идёт независимо от калибровки: применять ли метки — решаем по c1_calibration.txt,
# а второй раз GPU поднимать не придётся
python -m dump_assembly.relabel_faithfulness judge --input judge_input.jsonl --out judge_labels.jsonl \
    --only-kind answer --model "$JUDGE_MODEL" --backend "$JUDGE_BACKEND"

echo "Готово. Вернуть: semantic_clusters.jsonl judge_labels.jsonl c1_calibration.txt"
