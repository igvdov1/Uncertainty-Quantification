#!/usr/bin/env bash
# Сборка расширенного дампа с нуля на одной GPU (C9 §0: ~2000 вопросов на источник), все шаги подряд,
# кроме LLM-судьи (scripts/run_c4_judge.sh — после pip install vllm, он может заменить torch).
# Запуск из architect_folder/. Схема шагов — pipeline_map/pipeline.md.
#
# Нужно:
#   GPU >= 32 ГБ; доступ к meta-llama/Llama-3.1-8B-Instruct на HF (huggingface-cli login или HF_TOKEN);
#   диск: корпус Contriever ~70 ГБ (если RETRIEVAL_SOURCE=disk) + дампы ~15 ГБ на 8000 вопросов;
#   pip install torch transformers accelerate sentencepiece protobuf datasets lettucedetect scikit-learn regex requests
#   для BM25 (карточка retriever-disagreement): pyserini + Java 21 — в текущем окружении или в отдельном
#   (PYSERINI_PY=путь/к/python; по умолчанию ищется ../../pyserini_env рядом с репозиторием); индекс
#   wikipedia-dpr скачается сам; без pyserini шаг пропускается с предупреждением.
#
# Параметры (переменные окружения):
#   PER_SOURCE=2000 DEV_FRAC=0.4   — размер выборки на источник и доля dev
#   LIMIT=                         — смоук: только первые N вопросов на всех шагах (например LIMIT=40)
#   RETRIEVAL_SOURCE=disk|stream   — disk: корпус в CORPUS_DIR — уже распакованный (psgs_w100.tsv и
#                                    wikipedia_embeddings/, как после старого runbook) или скачивается туда
#                                    архивами (wget -c, докачивается);
#                                    stream: читать по сети без записи на диск (обрыв = начать заново)
#   CORPUS_DIR=contriever   BM25=auto|0   CLEAN_CORPUS=0 (1 — удалить корпус после ретрива, −70 ГБ)
#
# Всё резюмируется: при обрыве запустить ещё раз — готовые шаги пропускаются, долгие продолжают с места.
# Лог: c4_log.txt
set -euo pipefail
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PER_SOURCE="${PER_SOURCE:-2000}"
DEV_FRAC="${DEV_FRAC:-0.4}"
RETRIEVAL_SOURCE="${RETRIEVAL_SOURCE:-disk}"
CORPUS_DIR="${CORPUS_DIR:-contriever}"
BM25="${BM25:-auto}"
CLEAN_CORPUS="${CLEAN_CORPUS:-0}"
LIMIT_ARGS=()
[[ -n "${LIMIT:-}" ]] && LIMIT_ARGS=(--limit "$LIMIT")
PSGS_URL=https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
EMB_URL=https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar

log() { echo "=== $(date '+%F %T') $* ===" | tee -a c4_log.txt; }
run() { "$@" 2>&1 | tee -a c4_log.txt; }
# долгие шаги резюмируются по qid — если процесс убит (OOM-kill, код 137), перезапускаем до 5 раз
retry() {
    local a
    for a in 1 2 3 4 5; do
        run "$@" && return 0
        echo "  попытка $a упала (код $?), перезапуск через 30 с — готовое не пересчитывается" | tee -a c4_log.txt
        sleep 30
    done
    return 1
}
done_step() { [[ -s "$1" ]] && { echo "  есть $1 — пропускаю" | tee -a c4_log.txt; }; }

log "1/16 отбор вопросов"
if ! done_step questions_full.jsonl; then
    run python -m dump_assembly.questions --per-source "$PER_SOURCE" --dev-frac "$DEV_FRAC" --out questions_full.jsonl
fi
if [[ -n "${LIMIT:-}" ]]; then head -n "$LIMIT" questions_full.jsonl > questions.jsonl; else cp questions_full.jsonl questions.jsonl; fi

log "2/16 перефразы (chat-шаблон + фильтр)"
retry python -m dump_assembly.paraphrases generate --questions questions.jsonl --model "$MODEL" --out paraphrases.jsonl

log "3/16 запросы для ретрива (вопросы + перефразы)"
run python -m dump_assembly.paraphrases retrieval-input --questions questions.jsonl --paraphrases paraphrases.jsonl \
    --out retrieval_queries.jsonl

log "4/16 плотный ретрив Contriever-MSMARCO, top-20"
if ! done_step retrieval.jsonl || [[ $(wc -l < retrieval.jsonl) -ne $(wc -l < retrieval_queries.jsonl) ]]; then
    if [[ "$RETRIEVAL_SOURCE" == "disk" ]]; then
        mkdir -p "$CORPUS_DIR"
        if [[ -s "$CORPUS_DIR/psgs_w100.tsv" ]]; then
            PSGS="$CORPUS_DIR/psgs_w100.tsv"
        else
            run wget -c -q --show-progress -O "$CORPUS_DIR/psgs_w100.tsv.gz" "$PSGS_URL"
            PSGS="$CORPUS_DIR/psgs_w100.tsv.gz"
        fi
        if [[ -d "$CORPUS_DIR/wikipedia_embeddings" ]] && [[ -n "$(ls -A "$CORPUS_DIR/wikipedia_embeddings")" ]]; then
            EMB="$CORPUS_DIR/wikipedia_embeddings"
        else
            run wget -c -q --show-progress -O "$CORPUS_DIR/wikipedia_embeddings.tar" "$EMB_URL"
            EMB="$CORPUS_DIR/wikipedia_embeddings.tar"
        fi
        echo "  корпус: $PSGS, эмбеддинги: $EMB" | tee -a c4_log.txt
    else
        EMB="$EMB_URL"; PSGS="$PSGS_URL"
    fi
    run python -m dump_assembly.streaming_retrieval --queries retrieval_queries.jsonl \
        --embeddings "$EMB" --passages "$PSGS" --out retrieval.jsonl
    [[ "$CLEAN_CORPUS" == "1" ]] && rm -rf "$CORPUS_DIR/psgs_w100.tsv" "$CORPUS_DIR/psgs_w100.tsv.gz" \
        "$CORPUS_DIR/wikipedia_embeddings" "$CORPUS_DIR/wikipedia_embeddings.tar"
fi

log "5/16 основная генерация (промпт v2): ответы RAG/CB, сэмплы, p_true, verbalized, метки factual"
retry python -m dump_assembly.run_assembly --model "$MODEL" --questions questions.jsonl \
    --retrieval-output retrieval.jsonl --paraphrases paraphrases.jsonl --prompt-version v2 \
    --skip-attention-head --out dump_base.jsonl

DUMP_IN=dump_base.jsonl
# питон с pyserini: явный PYSERINI_PY, иначе отдельное окружение pyserini_env рядом с репозиторием, иначе текущий
if [[ -z "${PYSERINI_PY:-}" ]]; then
    PYSERINI_PY=python
    for env in ../../pyserini_env ../pyserini_env; do
        [[ -x "$env/bin/python" ]] && { PYSERINI_PY="$(cd "$env" && pwd)/bin/python"; break; }
    done
fi
# JDK для pyserini (нужен javac): JAVA_HOME, если задан; иначе окружение pyserini (conda с openjdk),
# JDK рядом с репозиторием (/workspace/jdk*), системные /usr/lib/jvm/*, javac из PATH
if [[ -z "${JAVA_HOME:-}" || ! -x "${JAVA_HOME:-}/bin/javac" ]]; then
    PY_ENV_DIR="$(dirname "$(dirname "$(command -v "$PYSERINI_PY")")")"
    for jdk in "$PY_ENV_DIR" ../../jdk* ../jdk* /workspace/jdk* /opt/jdk* /usr/lib/jvm/*; do
        if [[ -x "$jdk/bin/javac" ]]; then export JAVA_HOME="$(cd "$jdk" && pwd)"; break; fi
    done
    if [[ -z "${JAVA_HOME:-}" ]] && command -v javac >/dev/null; then
        export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
    fi
fi
if [[ "$BM25" == "auto" ]] && ! "$PYSERINI_PY" -c "import pyserini" 2>/dev/null; then
    echo "  ВНИМАНИЕ: pyserini не найден ($PYSERINI_PY) — BM25 пропущен (карточка retriever-disagreement будет пустой)" \
        | tee -a c4_log.txt
    BM25=0
fi
if [[ "$BM25" != "0" ]]; then
    log "6/16 BM25 (retriever-disagreement)"
    if ! done_step dump_bm25.jsonl || [[ $(wc -l < dump_bm25.jsonl) -ne $(wc -l < dump_base.jsonl) ]]; then
        echo "  pyserini: $PYSERINI_PY, JAVA_HOME=${JAVA_HOME:-не найден}" | tee -a c4_log.txt
        # BM25 нужен одной карточке — его падение не должно останавливать прогон
        if ! run "$PYSERINI_PY" -m dump_assembly.augment_perturbations fill-bm25 --in dump_base.jsonl \
                --out dump_bm25.jsonl --pyserini-index wikipedia-dpr; then
            echo "  ВНИМАНИЕ: BM25 упал (см. выше) — продолжаю без него" | tee -a c4_log.txt
            rm -f dump_bm25.jsonl
        fi
    fi
    [[ -s dump_bm25.jsonl ]] && DUMP_IN=dump_bm25.jsonl
fi

log "7/16 ответы на перефразы + ячейки «перефраз × пассажи» (H2)"
retry python -m dump_assembly.augment_perturbations generate --in "$DUMP_IN" \
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
retry python -m dump_assembly.semantic_clusters cluster --input se_input.jsonl --out semantic_clusters.jsonl
log "10/16 SelfCheck / NCP (с вопросом и без) / LettuceDetect"
retry python -m dump_assembly.gpu_cards run --input gpu_cards_input.jsonl --out gpu_cards.jsonl
log "11/16 memory-strength / SRE / context-sufficiency"
retry python -m dump_assembly.gpu_cards2 run --input gpu_cards2_input.jsonl --out gpu_cards2.jsonl --generator "$MODEL"
log "12/16 CCP (по токенам и по словам)"
retry python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl --only-short --word-level --generator "$MODEL"
log "13/16 ATS: обучение линейной головы"
done_step ats_head.pt || run python -m dump_assembly.ats train --model "$MODEL" --n-examples 5000 --out ats_head.pt
log "14/16 INTRYGUE / ReDeEP / LUMINA / TAD / HACK / source-clustering"
retry python -m dump_assembly.mech_cards run --input mech_input.jsonl --out-dir mech_out --only-short \
    --ats-head ats_head.pt --generator "$MODEL"

log "15/16 упаковка"
tar czf mech_out.tgz mech_out
log "16/16 готово. Дальше: pip install vllm && bash scripts/run_c4_judge.sh"
echo "Вернуть (или оставить на поде для run_c4_local.sh): questions.jsonl paraphrases.jsonl retrieval.jsonl dump_full.jsonl \
semantic_clusters.jsonl gpu_cards.jsonl gpu_cards2.jsonl ccp.jsonl mech_out.tgz ats_head.pt c4_log.txt" | tee -a c4_log.txt
