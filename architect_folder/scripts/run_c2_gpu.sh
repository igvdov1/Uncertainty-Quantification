#!/usr/bin/env bash
# GPU-пачка ветки C №2: CCP (ccp-faithful-substitution) поверх dump_pilot_v4.
# Запуск из architect_folder/, рядом должен лежать ccp_input.jsonl (делается
# локально: python -m dump_assembly.ccp extract --dump dump_pilot_v4.jsonl --out ccp_input.jsonl).
# Нужно: доступ к meta-llama/Llama-3.1-8B-Instruct на HF (huggingface-cli login), ~16 ГБ VRAM.
#   pip install torch transformers accelerate sentencepiece protobuf
# Резюмируется: при обрыве запустить ещё раз. Вернуть ветке C: ccp.jsonl и лог
# (в конце — доля «жадный токен = top-1», должна быть > 0.95).
set -euo pipefail
[[ -f ccp_input.jsonl ]] || { echo "нет ccp_input.jsonl — скопируйте его в $(pwd)"; exit 1; }

echo "=== CCP: short-form ==="
python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl --only-short 2>&1 | tee ccp_log.txt
echo "=== CCP: long-form (длинные ответы, дольше) ==="
python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl 2>&1 | tee -a ccp_log.txt
echo "Готово. Вернуть: ccp.jsonl ccp_log.txt"
