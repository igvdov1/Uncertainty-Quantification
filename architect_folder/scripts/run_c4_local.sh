#!/usr/bin/env bash
# CPU-часть после run_c4_full.sh и run_c4_judge.sh: метка faithful, подгоняемые на dev карточки, скринер
# в двух постановках — все ответы (отказ — одна из ошибок) и только содержательные ответы (--exclude-refusals).
# Запуск из architect_folder/ (на поде или локально; нужны файлы, перечисленные в конце run_c4_full.sh,
# плюс judge_input.jsonl и judge_labels.jsonl).
set -euo pipefail
log() { echo "=== $(date '+%F %T') $* ===" | tee -a c4_local_log.txt; }
run() { "$@" 2>&1 | tee -a c4_local_log.txt; }
[[ -d mech_out ]] || tar xzf mech_out.tgz

log "1/5 метка faithful из вердиктов судьи"
run python -m dump_assembly.relabel_faithfulness apply --dump dump_full.jsonl --input judge_input.jsonl \
    --labels judge_labels.jsonl --out dump_final.jsonl
log "2/5 density / SE-probes / cp-internal (подгонка на dev)"
run python -m dump_assembly.local_cards --dump dump_final.jsonl --sidecar semantic_clusters.jsonl --out local_cards.jsonl
log "3/5 INTRYGUE-tuned / ReDeEP / TAD / HACK (подгонка на dev)"
run python -m dump_assembly.mech_fit --dump dump_final.jsonl --gold questions.jsonl --mech-dir mech_out \
    --sidecar semantic_clusters.jsonl --out mech_cards.jsonl

SIDECARS=(--sidecar semantic_clusters.jsonl --sidecar gpu_cards.jsonl --sidecar gpu_cards2.jsonl --sidecar ccp.jsonl
          --sidecar local_cards.jsonl --sidecar mech_cards.jsonl)
log "4/5 скринер: все ответы"
python -m screener.run_screener --input dump_final.jsonl "${SIDECARS[@]}" --cards --calibration --synthesis \
    --task-dependent --cache-dir screener_cache > screener_c4_all.txt 2>&1
log "5/5 скринер: только содержательные ответы (без отказов)"
python -m screener.run_screener --input dump_final.jsonl "${SIDECARS[@]}" --exclude-refusals --cards --calibration \
    --synthesis --task-dependent --cache-dir screener_cache > screener_c4_answered.txt 2>&1
log "Готово: screener_c4_all.txt, screener_c4_answered.txt"
