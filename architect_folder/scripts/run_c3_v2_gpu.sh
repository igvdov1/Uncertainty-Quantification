#!/usr/bin/env bash
# Пересборка дампа на промпте v2 + все GPU-шаги ветки C (кроме LLM-судьи — он в run_c3_judge.sh).
# Запуск из architect_folder/. Рядом должны лежать (собираются локально, см. runner_folder/C7_runbook_v2.md):
#   rebuild_input.jsonl   — short-form вопросы/пассажи/перефразы из v4 (rebuild_v2 extract-input)
# Нужно: GPU >= 48 ГБ (attention по всем головам; 80 ГБ спокойнее), доступ к Llama-3.1-8B-Instruct на HF.
#   pip install torch transformers accelerate sentencepiece protobuf datasets lettucedetect scikit-learn
# Опционально: PARA_RETRIEVAL=путь/к/retrieval_paraphrases.jsonl — тогда перегенерируются и rag-ответы на перефразы.
# Всё резюмируется: при обрыве запустить ещё раз. Лог: c3_log.txt
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
log() { echo "=== $* ===" | tee -a c3_log.txt; }
run() { "$@" 2>&1 | tee -a c3_log.txt; }

[[ -f rebuild_input.jsonl ]] || { echo "нет rebuild_input.jsonl — скопируйте в $(pwd)"; exit 1; }
[[ -d rag_uncertainty ]] || git clone https://github.com/stat-ml/rag_uncertainty.git

log "1/10 gold-ответы"
[[ -f gold_answers.jsonl ]] || run python -m dump_assembly.rebuild_v2 export-gold \
    --franq-dataset-dir rag_uncertainty/claim_level/dataset --out gold_answers.jsonl

log "2/10 пересборка short-form на промпте v2"
run python -m dump_assembly.rebuild_v2 rebuild --in rebuild_input.jsonl --gold gold_answers.jsonl \
    --model "$MODEL" --prompt-version v2 --out dump_v5_short_raw.jsonl

log "3/10 ответы на перефразы (closed-book; rag — если задан PARA_RETRIEVAL)"
PARA_ARGS=()
[[ -n "${PARA_RETRIEVAL:-}" ]] && PARA_ARGS=(--paraphrase-retrieval "$PARA_RETRIEVAL")
run python -m dump_assembly.augment_perturbations generate --in dump_v5_short_raw.jsonl \
    --model "$MODEL" --out dump_v5_short.jsonl "${PARA_ARGS[@]}"

log "4/10 входы для карточек"
run python -m dump_assembly.semantic_clusters extract --dump dump_v5_short.jsonl --out se_input_v5.jsonl
run python -m dump_assembly.gpu_cards extract --dump dump_v5_short.jsonl --out gpu_cards_input_v5.jsonl
run python -m dump_assembly.gpu_cards2 extract --dump dump_v5_short.jsonl --out gpu_cards2_input_v5.jsonl
run python -m dump_assembly.ccp extract --dump dump_v5_short.jsonl --out ccp_input_v5.jsonl
run python -m dump_assembly.mech_cards extract --dump dump_v5_short.jsonl --out mech_input_v5.jsonl
run python -m dump_assembly.relabel_faithfulness extract --dump dump_v5_short.jsonl --out judge_answers_v5.jsonl

log "5/10 semantic entropy: NLI-кластеры"
run python -m dump_assembly.semantic_clusters cluster --input se_input_v5.jsonl --out semantic_clusters_v5.jsonl
log "6/10 SelfCheck / NCP / LettuceDetect"
run python -m dump_assembly.gpu_cards run --input gpu_cards_input_v5.jsonl --out gpu_cards_v5.jsonl
log "7/10 перефразы + достаточность контекста"
run python -m dump_assembly.gpu_cards2 run --input gpu_cards2_input_v5.jsonl --out gpu_cards2_v5.jsonl --generator "$MODEL"
log "8/10 CCP"
run python -m dump_assembly.ccp run --input ccp_input_v5.jsonl --out ccp_v5.jsonl --only-short --generator "$MODEL"
log "9/10 ATS: обучение линейной головы"
[[ -f ats_head.pt ]] || run python -m dump_assembly.ats train --model "$MODEL" --n-examples 5000 --out ats_head.pt
log "10/10 INTRYGUE / ReDeEP / LUMINA / TAD / HACK / source-clustering"
run python -m dump_assembly.mech_cards run --input mech_input_v5.jsonl --out-dir mech_out_v5 --only-short \
    --ats-head ats_head.pt --generator "$MODEL"
tar czf mech_out_v5.tgz mech_out_v5

log "Готово. Дальше: bash scripts/run_c3_judge.sh (после pip install vllm)"
echo "Вернуть: dump_v5_short.jsonl gold_answers.jsonl semantic_clusters_v5.jsonl gpu_cards_v5.jsonl gpu_cards2_v5.jsonl ccp_v5.jsonl mech_out_v5.tgz ats_head.pt c3_log.txt"
