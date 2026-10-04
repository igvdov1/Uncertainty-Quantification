#!/usr/bin/env bash
# GPU-пачка ветки C №2 поверх dump_pilot_v4 (ничего не генерирует заново):
#   1) memory-strength-paraphrase, semantic-reformulation-entropy (NLI по ответам на перефразы),
#      context-sufficiency-self-eval (один forward Llama на вопрос)   -> gpu_cards2.jsonl
#   2) CCP (teacher forcing готовых ответов через Llama + NLI)       -> ccp.jsonl
# Запуск из architect_folder/; рядом должны лежать gpu_cards2_input.jsonl и ccp_input.jsonl
# (собираются локально командами extract, дамп на сервер не нужен).
# Нужно: доступ к meta-llama/Llama-3.1-8B-Instruct на HF (huggingface-cli login), ~16 ГБ VRAM (A4500 хватит).
#   pip install torch transformers accelerate sentencepiece protobuf
# Всё резюмируется: при обрыве запустить ещё раз.
# Вернуть ветке C: gpu_cards2.jsonl ccp.jsonl c2_log.txt
set -euo pipefail
for f in gpu_cards2_input.jsonl ccp_input.jsonl; do
    [[ -f "$f" ]] || { echo "нет $f — скопируйте его в $(pwd)"; exit 1; }
done

echo "=== 1/3 перефразы + достаточность контекста ===" | tee c2_log.txt
python -m dump_assembly.gpu_cards2 run --input gpu_cards2_input.jsonl --out gpu_cards2.jsonl 2>&1 | tee -a c2_log.txt
echo "=== 2/3 CCP: short-form ===" | tee -a c2_log.txt
python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl --only-short 2>&1 | tee -a c2_log.txt
echo "=== 3/3 CCP: long-form (длинные ответы, дольше всего) ===" | tee -a c2_log.txt
python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl 2>&1 | tee -a c2_log.txt
echo "Готово. Вернуть: gpu_cards2.jsonl ccp.jsonl c2_log.txt"
