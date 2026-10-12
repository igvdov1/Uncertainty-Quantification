#!/usr/bin/env bash
# LLM-судья faithful (промпт v3 с вердиктом abstain) для дампа из run_c4_full.sh. Запуск из architect_folder/
# ПОСЛЕ run_c4_full.sh и pip install vllm (vllm может заменить torch/transformers).
# Калибровка судьи — на 1453 клеймах FRANQ (data/judge_claims_franq.jsonl: полный текст клеймов и экспертные
# метки; long-form в новом дампе нет, поэтому клеймы лежат отдельным файлом). Гейт: kappa >= 0.4.
# Судья — тот же Qwen2.5-32B-Instruct-AWQ, что в v4/v5 (сравнимость калибровки). Резюмируется по input_hash.
set -euo pipefail
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
for f in judge_answers.jsonl data/judge_claims_franq.jsonl; do
    [[ -s "$f" ]] || { echo "нет $f"; exit 1; }
done
cat data/judge_claims_franq.jsonl judge_answers.jsonl > judge_input.jsonl

echo "=== $(date '+%F %T') судья: клеймы FRANQ (калибровка) ===" | tee -a c4_judge_log.txt
python -m dump_assembly.relabel_faithfulness judge --input judge_input.jsonl --out judge_labels.jsonl \
    --only-kind claim --model "$JUDGE_MODEL" --backend vllm 2>&1 | tee -a c4_judge_log.txt
python -m dump_assembly.relabel_faithfulness calibrate --input judge_input.jsonl --labels judge_labels.jsonl \
    | tee c4_calibration.txt
echo "=== $(date '+%F %T') судья: short-form ответы ===" | tee -a c4_judge_log.txt
python -m dump_assembly.relabel_faithfulness judge --input judge_input.jsonl --out judge_labels.jsonl \
    --only-kind answer --model "$JUDGE_MODEL" --backend vllm 2>&1 | tee -a c4_judge_log.txt
echo "Готово. Дальше (на поде или локально): bash scripts/run_c4_local.sh"
