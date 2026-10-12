#!/usr/bin/env bash
# Сборка расширенного дампа с нуля на одной GPU (C9 §0: ~2000 вопросов на источник), все шаги подряд,
# кроме LLM-судьи (scripts/run_c4_judge.sh — после pip install vllm, он может заменить torch).
# Запуск из architect_folder/. Схема шагов — pipeline_map/pipeline.md.
#
# Нужно:
#   GPU >= 32 ГБ; доступ к meta-llama/Llama-3.1-8B-Instruct на HF (huggingface-cli login или HF_TOKEN);
#   диск: корпус Contriever ~70 ГБ (если RETRIEVAL_SOURCE=disk) + дампы ~15 ГБ на 8000 вопросов;
#   pip install torch transformers accelerate sentencepiece protobuf datasets lettucedetect scikit-learn regex requests
#   для BM25 (карточка retriever-disagreement): pip install pyserini + Java 21 (индекс wikipedia-dpr ~10 ГБ
#   скачается сам); без pyserini шаг пропускается с предупреждением.
#
# Параметры (переменные окружения):
#   PER_SOURCE=2000 DEV_FRAC=0.4   — размер выборки на источник и доля dev
#   LIMIT=                         — смоук: только первые N вопросов на всех шагах (например LIMIT=40)
#   RETRIEVAL_SOURCE=disk|stream   — disk: скачать корпус в CORPUS_DIR (wget -c, докачивается);
#                                    stream: читать по сети без записи на диск (обрыв = начать заново)
#   CORPUS_DIR=contriever_corpus   BM25=auto|0   CLEAN_CORPUS=0 (1 — удалить корпус после ретрива, −70 ГБ)
#
# Всё резюмируется: при обрыве запустить ещё раз — готовые шаги пропускаются, долгие продолжают с места.
# Лог: c4_log.txt
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PER_SOURCE="${PER_SOURCE:-2000}"
DEV_FRAC="${DEV_FRAC:-0.4}"
RETRIEVAL_SOURCE="${RETRIEVAL_SOURCE:-disk}"
CORPUS_DIR="${CORPUS_DIR:-contriever_corpus}"
BM25="${BM25:-auto}"
CLEAN_CORPUS="${CLEAN_CORPUS:-0}"
LIMIT_ARGS=()
[[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$LIMIT")
PSGS_URL=https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
EMB_URL=https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar

log() { echo "=== $(date '+%F %T') $* ===" | tee -a c4_log.txt; }
run() { "$@" 2>&1 | tee -a c4_log.txt; }
done_step() { [[ -s "$1" ]] && { echo "  есть $1 — пропускаю" | tee -a c4_log.txt; }; }

log "1/16 отбор вопросов"
if ! done_step questions_full.jsonl; then
    run python -m dump_assembly.questions --per-source "$PER_SOURCE" --dev-frac "$DEV_FRAC" --out questions_full.jsonl
fi
if [[ -n "${LIMIT:-}" ]]; then head -n "$LIMIT" questions_full.jsonl > questions.jsonl; else cp questions_full.jsonl questions.jsonl; fi

log "2/16 перефразы (chat-шаблон + фильтр)"
run python -m dump_assembly.paraphrases generate --questions questions.jsonl --model "$MODEL" --out paraphrases.jsonl

log "3/16 запросы для ретрива (вопросы + перефразы)"
run python -m dump_assembly.paraphrases retrieval-input --questions questions.jsonl --paraphrases paraphrases.jsonl \
    --out retrieval_queries.jsonl

log "4/16 плотный ретрив Contriever-MSMARCO, top-20"
if ! done_step retrieval.jsonl || [[ $(wc -l < retrieval.jsonl) -ne $(wc -l < retrieval_queries.jsonl) ]]; then
    if [[ "$RETRIEVAL_SOURCE" == "disk" ]]; then
        mkdir -p "$CORPUS_DIR"
        run wget -c -q --show-progress -O "$CORPUS_DIR/psgs_w100.tsv.gz" "$PSGS_URL"
        run wget -c -q --show-progress -O "$CORPUS_DIR/wikipedia_embeddings.tar" "$EMB_URL"
        EMB="$CORPUS_DIR/wikipedia_embeddings.tar"; PSGS="$CORPUS_DIR/psgs_w100.tsv.gz"
    else
        EMB="$EMB_URL"; PSGS="$PSGS_URL"
    fi
    run python -m dump_assembly.streaming_retrieval --queries retrieval_queries.jsonl \
        --embeddings "$EMB" --passages "$PSGS" --out retrieval.jsonl
    [[ "$CLEAN_CORPUS" == "1" ]] && rm -rf "$CORPUS_DIR"
fi

log "5/16 основная генерация (промпт v2): ответы RAG/CB, сэмплы, p_true, verbalized, метки factual"
run python -m dump_assembly.run_assembly --model "$MODEL" --questions questions.jsonl \
    --retrieval-output retrieval.jsonl --paraphrases paraphrases.jsonl --prompt-version v2 \
    --skip-attention-head --out dump_base.jsonl

DUMP_IN=dump_base.jsonl
if [[ "$BM25" == "auto" ]] && ! python -c "import pyserini" 2>/dev/null; then
    echo "  ВНИМАНИЕ: pyserini не установлен — BM25 пропущен (карточка retriever-disagreement будет пустой)" \
        | tee -a c4_log.txt
    BM25=0
fi
if [[ "$BM25" != "0" ]]; then
    log "6/16 BM25 (retriever-disagreement)"
    if ! done_step dump_bm25.jsonl || [[ $(wc -l < dump_bm25.jsonl) -ne $(wc -l < dump_base.jsonl) ]]; then
        run python -m dump_assembly.augment_perturbations fill-bm25 --in dump_base.jsonl --out dump_bm25.jsonl \
            --pyserini-index wikipedia-dpr
    fi
    DUMP_IN=dump_bm25.jsonl
fi

log "7/16 ответы на перефразы + ячейки «перефраз × пассажи» (H2)"
run python -m dump_assembly.augment_perturbations generate --in "$DUMP_IN" \
    --paraphrase-retrieval retrieval.jsonl --swap-cells --model "$MODEL" --out dump_full.jsonl

log "8/16 входы для карточек"
for m in semantic_clusters:se_input gpu_cards:gpu_cards_input gpu_cards2:gpu_cards2_input ccp:ccp_input \
         mech_cards:mech_input relabel_faithfulness:judge_answers; do
    mod="${m%%:*}"; out="${m##*:}.jsonl"
    if [[ ! -s "$out" || "$out" -ot dump_full.jsonl ]]; then
        run python -m "dump_assembly.$mod" extract --dump dump_full.jsonl --out "$out"
    fi
done

log "9/16 semantic entropy: NLI-кластеры"
run python -m dump_assembly.semantic_clusters cluster --input se_input.jsonl --out semantic_clusters.jsonl
log "10/16 SelfCheck / NCP (с вопросом и без) / LettuceDetect"
run python -m dump_assembly.gpu_cards run --input gpu_cards_input.jsonl --out gpu_cards.jsonl
log "11/16 memory-strength / SRE / context-sufficiency"
run python -m dump_assembly.gpu_cards2 run --input gpu_cards2_input.jsonl --out gpu_cards2.jsonl --generator "$MODEL"
log "12/16 CCP (по токенам и по словам)"
run python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl --only-short --word-level --generator "$MODEL"
log "13/16 ATS: обучение линейной головы"
done_step ats_head.pt || run python -m dump_assembly.ats train --model "$MODEL" --n-examples 5000 --out ats_head.pt
log "14/16 INTRYGUE / ReDeEP / LUMINA / TAD / HACK / source-clustering"
run python -m dump_assembly.mech_cards run --input mech_input.jsonl --out-dir mech_out --only-short \
    --ats-head ats_head.pt --generator "$MODEL"

log "15/16 упаковка"
tar czf mech_out.tgz mech_out
log "16/16 готово. Дальше: pip install vllm && bash scripts/run_c4_judge.sh"
echo "Вернуть (или оставить на поде для run_c4_local.sh): questions.jsonl paraphrases.jsonl retrieval.jsonl dump_full.jsonl \
semantic_clusters.jsonl gpu_cards.jsonl gpu_cards2.jsonl ccp.jsonl mech_out.tgz ats_head.pt c4_log.txt" | tee -a c4_log.txt
