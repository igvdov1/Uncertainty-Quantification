"""
Переразметка label_faithful LLM-судьёй поверх уже собранного дампа.

Зачем: у short-form label_faithful был порогом той же NLI-модели, что
пишет в дамп сигнал alignscore — метка = сигнал, AUROC=1.0 по построению
(runner_folder/C1_faithful_label_leakage.md). Здесь метку ставит
независимый судья (по умолчанию Qwen2.5-32B-Instruct — не генератор
дампа, чтобы модель не судила сама себя) в формате FRANQ:
faithful / unfaithful-contra / unfaithful-neutral.

Четыре шага, дамп (ГБ) между машинами не возим:

    # 1. локально, CPU: из дампа -> маленький файл с тем, что нужно судье.
    #    --franq-dataset-dir: подставить полный текст клеймов FRANQ вместо
    #    обрывков decoded_claims, которые лежат в дампах до v4
    python -m dump_assembly.relabel_faithfulness extract \
        --dump dump_pilot_v3.jsonl --out judge_input.jsonl \
        --franq-dataset-dir rag_uncertainty/claim_level/dataset

    # 2. на GPU: судья. Resume по item_id — перезапуск продолжает с места обрыва
    python -m dump_assembly.relabel_faithfulness judge \
        --input judge_input.jsonl --out judge_labels.jsonl \
        --model Qwen/Qwen2.5-32B-Instruct

    # 3. локально: согласие судьи с per-claim разметкой FRANQ (гейт перед apply)
    python -m dump_assembly.relabel_faithfulness calibrate \
        --input judge_input.jsonl --labels judge_labels.jsonl

    # 4. локально: вписать метки в дамп
    python -m dump_assembly.relabel_faithfulness apply \
        --dump dump_pilot_v3.jsonl --input judge_input.jsonl --labels judge_labels.jsonl \
        --out dump_pilot_v4.jsonl

Что судится:
  - short-form: rag.answer целиком против его пассажей (kind="answer");
    это и есть новая label_faithful записи.
  - long-form FRANQ: каждый клейм против пассажей вопроса (kind="claim") —
    ТОЛЬКО для калибровки судьи против их разметки. Метка long-form записи
    остаётся из разметки FRANQ, apply лишь переагрегирует её мягко
    (labeling.franq_longform_answer_labels, доля faithful-клеймов >= 0.8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .labeling import LONGFORM_FAITHFUL_MIN_FRAC, franq_longform_answer_labels

LONGFORM_SOURCE = "franq_longform"
VERDICTS = ("faithful", "unfaithful-contra", "unfaithful-neutral")
_VERDICT_RE = re.compile(r"VERDICT:\s*\**\s*(faithful|unfaithful-contra|unfaithful-neutral)\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are a careful annotator checking whether a statement is grounded in retrieved passages. "
    "You judge only against the passages, never against your own knowledge of the world."
)

USER_TEMPLATE = """Question: {question}

Retrieved passages:
{passages}

Statement to check:
{statement}

Classify the statement with respect to the passages:
- faithful: every factual assertion in the statement is supported by the passages.
- unfaithful-contra: some assertion in the statement contradicts the passages.
- unfaithful-neutral: some assertion is neither supported nor contradicted by the passages (not verifiable from them).

Notes:
- The statement answers the question above. Read a short answer as the full claim it makes in response to the question (e.g. "Paris." to "What is the capital of France?" asserts "The capital of France is Paris").
- Judge only against the passages, even if you know the statement is true or false in reality.
- Saying that the passages do not contain the information is not an assertion; judge only the facts the statement actually asserts.
- The statement may be cut off mid-sentence; ignore an incomplete trailing fragment.
- Do not penalize a statement for being incomplete, brief or omitting details: if what it does assert is supported by the passages, it is faithful.

Give one or two sentences of reasoning, then a final line exactly in the form:
VERDICT: <faithful|unfaithful-contra|unfaithful-neutral>"""


# ---- общие утилиты ---------------------------------------------------

def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_messages(item: dict) -> list[dict]:
    passages = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(item["passages"]))
    user = USER_TEMPLATE.format(question=item["question"], passages=passages, statement=item["text"])
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def input_hash(item: dict) -> str:
    """Хэш промпта: resume считает item готовым, только если вердикт получен
    на ровно этом промпте (сменили текст клейма / шаблон — перепросит)."""
    return hashlib.sha1(json.dumps(build_messages(item), sort_keys=True).encode()).hexdigest()[:16]


def parse_verdict(raw: str) -> str | None:
    """Последний VERDICT: в ответе (судья может упомянуть метки в рассуждении)."""
    matches = _VERDICT_RE.findall(raw)
    return matches[-1].lower() if matches else None


def verdict_to_label(verdict: str | None) -> int | None:
    if verdict is None:
        return None
    return int(verdict == "faithful")


# ---- 1. extract ------------------------------------------------------

def load_franq_full_claims(dataset_dir: Path, model_file: str) -> dict[str, list[str]]:
    """qid (franq_lf_{key}) -> полные тексты клеймов FRANQ по индексу;
    та же нумерация, что в questions.load_franq_longform / cid _c{i}."""
    with open(Path(dataset_dir) / model_file) as f:
        data = json.load(f)
    return {f"franq_lf_{k}": v.get("claims", []) for k, v in data.items() if k.isdigit()}


def extract_items(record: dict, full_claims: dict[str, list[str]] | None = None) -> list[dict]:
    passages = [p["text"] for p in record.get("passages", [])]
    base = {"qid": record["qid"], "source": record["source"], "question": record["question"], "passages": passages}
    if record["source"] == LONGFORM_SOURCE:
        texts = (full_claims or {}).get(record["qid"])
        items = []
        for c in record.get("claims", []):
            text = c["text"]
            if texts is not None:
                text = texts[int(c["cid"].rsplit("_c", 1)[1])]
            items.append({**base, "item_id": c["cid"], "kind": "claim", "text": text,
                          "ref_label_faithful": c["label_faithful"]})
        return items
    return [{**base, "item_id": f"{record['qid']}_answer", "kind": "answer", "text": record["rag"]["answer"],
             "ref_label_faithful": None}]


def cmd_extract(args) -> None:
    full_claims = None
    if args.franq_dataset_dir:
        full_claims = load_franq_full_claims(args.franq_dataset_dir, args.franq_model_file)
    else:
        print("ВНИМАНИЕ: без --franq-dataset-dir клеймы long-form берутся из дампа как есть "
              "(в дампах до v4 — обрывки decoded_claims)", file=sys.stderr)
    kinds = Counter()
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.dump):
            for item in extract_items(r, full_claims):
                if not item["passages"]:
                    # судить не с чем; short-form без пассажей в дампе быть не должно
                    print(f"  {item['item_id']}: нет пассажей, пропускаю", file=sys.stderr)
                    continue
                kinds[item["kind"]] += 1
                fout.write(json.dumps(item) + "\n")
    print(f"Готово: {dict(kinds)} -> {args.out}")


# ---- 2. judge --------------------------------------------------------

class HFJudge:
    def __init__(self, model_name: str, dtype: str, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=getattr(torch, dtype), device_map="auto",
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, batch_messages: list[list[dict]]) -> list[str]:
        prompts = [self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in batch_messages]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class VLLMJudge:
    def __init__(self, model_name: str, dtype: str, max_new_tokens: int, max_model_len: int):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=model_name, dtype=dtype, max_model_len=max_model_len)
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)

    def __call__(self, batch_messages: list[list[dict]]) -> list[str]:
        outs = self.llm.chat(batch_messages, self.params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def run_judge(items: list[dict], judge, out_path: Path, batch_size: int, file_mode: str = "a") -> None:
    """judge: callable(list[messages]) -> list[str]. Пишет построчно с flush —
    обрыв прогона теряет максимум один батч."""
    n_unparsed = 0
    with open(out_path, file_mode) as fout:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            raws = judge([build_messages(it) for it in batch])
            for it, raw in zip(batch, raws):
                verdict = parse_verdict(raw)
                n_unparsed += verdict is None
                fout.write(json.dumps({
                    "item_id": it["item_id"], "qid": it["qid"], "kind": it["kind"],
                    "input_hash": input_hash(it), "verdict": verdict, "label_faithful": verdict_to_label(verdict), "raw": raw,
                }) + "\n")
            fout.flush()
            print(f"  {min(start + batch_size, len(items))}/{len(items)}  (без вердикта: {n_unparsed})")


def cmd_judge(args) -> None:
    items = list(iter_jsonl(args.input))
    if args.only_kind:
        items = [it for it in items if it["kind"] == args.only_kind]
    if args.limit:
        items = items[:args.limit]
    done = set()
    if args.out.exists():
        # готово = есть вердикт на ровно этом промпте; строки без вердикта или
        # от старого промпта/текста перепрашиваются
        done = {(j["item_id"], j.get("input_hash")) for j in iter_jsonl(args.out) if j["verdict"] is not None}
    todo = [it for it in items if (it["item_id"], input_hash(it)) not in done]
    print(f"{len(items)} items, {len(done)} уже размечено, к разметке {len(todo)}")
    if not todo:
        return
    if args.backend == "vllm":
        judge = VLLMJudge(args.model, args.dtype, args.max_new_tokens, args.max_model_len)
    else:
        judge = HFJudge(args.model, args.dtype, args.max_new_tokens)
    run_judge(todo, judge, args.out, args.batch_size)
    print(f"Готово -> {args.out}")


# ---- чтение меток судьи ----------------------------------------------

def load_judge_labels(path: Path, items: list[dict] | None = None) -> dict[str, dict]:
    """item_id -> последняя строка с вердиктом (resume может дописать
    повторную попытку для item, у которого раньше вердикта не было).
    С items — берутся только строки, посчитанные на текущем промпте этих
    items (в файле могут остаться вердикты от старого промпта)."""
    want = {it["item_id"]: input_hash(it) for it in items} if items is not None else None
    by_id: dict[str, dict] = {}
    for j in iter_jsonl(path):
        if want is not None and want.get(j["item_id"]) != j.get("input_hash"):
            continue
        if j["verdict"] is not None or j["item_id"] not in by_id:
            by_id[j["item_id"]] = j
    return by_id


# ---- 3. calibrate ----------------------------------------------------

def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in pairs) / n
    pa1 = sum(a for a, _ in pairs) / n
    pb1 = sum(b for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def calibration_report(items: list[dict], judged: dict[str, dict],
                       min_frac: float = LONGFORM_FAITHFUL_MIN_FRAC) -> dict:
    """Согласие судьи с per-claim разметкой FRANQ + с record-level мягким
    агрегатом на long-form. pairs — (ref, judge), 1 = faithful."""
    claim_items = [it for it in items if it["kind"] == "claim"]
    pairs, missing = [], 0
    by_qid: dict[str, list[tuple[int, int]]] = {}
    verdicts_when_ref0 = Counter()
    for it in claim_items:
        j = judged.get(it["item_id"])
        if j is None or j["label_faithful"] is None:
            missing += 1
            continue
        pair = (int(it["ref_label_faithful"]), int(j["label_faithful"]))
        pairs.append(pair)
        by_qid.setdefault(it["qid"], []).append(pair)
        if pair[0] == 0:
            verdicts_when_ref0[j["verdict"]] += 1

    confusion = Counter(pairs)
    n = len(pairs)
    rec_pairs = []
    for qpairs in by_qid.values():
        ref = franq_longform_answer_labels([{"label_faithful": r, "label_factual": 1} for r, _ in qpairs], min_frac)[0]
        jud = franq_longform_answer_labels([{"label_faithful": j, "label_factual": 1} for _, j in qpairs], min_frac)[0]
        rec_pairs.append((ref, jud))

    tp, tn = confusion[(1, 1)], confusion[(0, 0)]
    return {
        "n_claims": n,
        "n_claims_missing_verdict": missing,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "cohen_kappa": cohen_kappa(pairs),
        "confusion_ref_x_judge": {f"ref={a},judge={b}": confusion[(a, b)] for a in (1, 0) for b in (1, 0)},
        "recall_faithful": tp / (tp + confusion[(1, 0)]) if tp + confusion[(1, 0)] else float("nan"),
        "recall_unfaithful": tn / (tn + confusion[(0, 1)]) if tn + confusion[(0, 1)] else float("nan"),
        "judge_verdicts_on_ref_unfaithful": dict(verdicts_when_ref0),
        "n_records": len(rec_pairs),
        "record_accuracy": sum(a == b for a, b in rec_pairs) / len(rec_pairs) if rec_pairs else float("nan"),
        "record_kappa": cohen_kappa(rec_pairs),
        "record_ref_positive": sum(a for a, _ in rec_pairs),
        "record_judge_positive": sum(b for _, b in rec_pairs),
    }


def cmd_calibrate(args) -> None:
    items = list(iter_jsonl(args.input))
    rep = calibration_report(items, load_judge_labels(args.labels, items))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    kappa = rep["cohen_kappa"]
    if not kappa == kappa or kappa < args.min_kappa:
        print(f"\nВНИМАНИЕ: kappa={kappa:.3f} < {args.min_kappa} — судья плохо согласуется с разметкой FRANQ, "
              f"метки short-form от него применять не стоит, сначала разобраться (промпт/модель).")
    else:
        print(f"\nOK: kappa={kappa:.3f} >= {args.min_kappa}")


# ---- 4. apply --------------------------------------------------------

def apply_to_record(record: dict, judged: dict[str, dict], min_frac: float) -> tuple[int | None, int | None]:
    """Меняет record на месте. Возвращает (старая, новая) label_faithful."""
    old = record.get("label_faithful")
    if record["source"] == LONGFORM_SOURCE:
        # метка — из разметки FRANQ, судья её не заменяет; только мягкая агрегация
        new, _ = franq_longform_answer_labels(record.get("claims", []), min_frac)
    else:
        j = judged.get(f"{record['qid']}_answer")
        if j is None or j["label_faithful"] is None:
            raise KeyError(f"{record['qid']}: нет вердикта судьи — прогоните judge до конца (resume) "
                           f"или проверьте, что judge_input собран из этого же дампа")
        new = j["label_faithful"]
        if record.get("claims"):
            record["claims"][0]["label_faithful"] = new
    record["label_faithful"] = new
    return old, new


def cmd_apply(args) -> None:
    judged = load_judge_labels(args.labels, list(iter_jsonl(args.input)))
    stats = Counter()
    align, labels_short = [], []
    with open(args.dump) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            kind = "long" if r["source"] == LONGFORM_SOURCE else "short"
            old, new = apply_to_record(r, judged, args.faithful_min_frac)
            stats[f"{kind}_n"] += 1
            stats[f"{kind}_pos"] += new
            stats[f"{kind}_changed"] += int(old != new)
            if kind == "short":
                align.append(r.get("signals", {}).get("alignscore", float("nan")))
                labels_short.append(new)
            fout.write(json.dumps(r) + "\n")

    for kind in ("short", "long"):
        print(f"{kind}-form: {stats[kind + '_n']} записей, faithful=1 у {stats[kind + '_pos']}, "
              f"метка изменилась у {stats[kind + '_changed']}")
    # санити: alignscore против новой метки больше не должен давать ~1.0
    pairs = [(a, l) for a, l in zip(align, labels_short) if a == a]
    pos = [a for a, l in pairs if l == 1]
    neg = [a for a, l in pairs if l == 0]
    if pos and neg:
        auc = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
        print(f"alignscore -> новая label_faithful, short-form AUROC = {auc:.3f} (было 1.000 из-за утечки)")
    print(f"Записано в {args.out}")


# ---- CLI -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract")
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--franq-dataset-dir", type=Path, default=None,
                   help="rag_uncertainty/claim_level/dataset — полный текст клеймов long-form")
    p.add_argument("--franq-model-file", default="Falcon3-3B-Base.json",
                   help="тот же файл, что в questions.load_franq_longform")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("judge")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--max-model-len", type=int, default=8192, help="только vllm")
    p.add_argument("--only-kind", choices=["answer", "claim"], default=None,
                   help="claim — сначала только калибровка на FRANQ, answer — только short-form")
    p.add_argument("--limit", type=int, default=None, help="смоук: первые N items")
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("calibrate")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--min-kappa", type=float, default=0.4)
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("apply")
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path, help="judge_input.jsonl, по которому судили")
    p.add_argument("--labels", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--faithful-min-frac", type=float, default=LONGFORM_FAITHFUL_MIN_FRAC)
    p.set_defaults(fn=cmd_apply)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
