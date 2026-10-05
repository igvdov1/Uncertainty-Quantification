#!/usr/bin/env bash
# LLM-судья faithful для dump v5 (промпт v3 с вердиктом abstain). Запускать ПОСЛЕ run_c3_v2_gpu.sh:
# установка vllm может заменить torch/transformers.
#   pip install vllm
# Рядом: judge_answers_v5.jsonl (из run_c3_v2_gpu.sh) и judge_claims_v4.jsonl (клеймы FRANQ для калибровки,
# собраны локально: grep '"kind": "claim"' judge_input.jsonl > judge_claims_v4.jsonl).
# Судья — тот же Qwen2.5-32B-Instruct-AWQ, что в прогонах v4 (для сравнимости калибровки).
set -euo pipefail
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
for f in judge_answers_v5.jsonl judge_claims_v4.jsonl; do
    [[ -f "$f" ]] || { echo "нет $f"; exit 1; }
done
cat judge_claims_v4.jsonl judge_answers_v5.jsonl > judge_input_v5.jsonl

echo "=== судья: клеймы FRANQ (калибровка) ===" | tee -a c3_judge_log.txt
python -m dump_assembly.relabel_faithfulness judge --input judge_input_v5.jsonl --out judge_labels_v5.jsonl \
    --only-kind claim --model "$JUDGE_MODEL" --backend vllm 2>&1 | tee -a c3_judge_log.txt
python -m dump_assembly.relabel_faithfulness calibrate --input judge_input_v5.jsonl --labels judge_labels_v5.jsonl \
    | tee c3_calibration.txt
echo "=== судья: short-form ответы ===" | tee -a c3_judge_log.txt
python -m dump_assembly.relabel_faithfulness judge --input judge_input_v5.jsonl --out judge_labels_v5.jsonl \
    --only-kind answer --model "$JUDGE_MODEL" --backend vllm 2>&1 | tee -a c3_judge_log.txt
echo "Вернуть: judge_input_v5.jsonl judge_labels_v5.jsonl c3_calibration.txt c3_judge_log.txt"
