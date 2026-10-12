"""
Claim-Conditioned Probability (CCP; Fadeeva et al. 2024, "Fact-Checking the
Output of Large Language Models via Token-Level Uncertainty Quantification")
для карточки ccp-faithful-substitution (researcher_folder/B2_cards.md).

В дампе token_topk хранит только значения логитов, без id токенов, — какими
были альтернативы, неизвестно. Здесь они восстанавливаются без новой
генерации: уже сгенерированный жадный ответ прогоняется через генератор
одним forward-проходом (teacher forcing) с тем же промптом, что при сборке,
и в каждой позиции берутся top-k токенов с вероятностями.

Формула (их уравнение для CCP_word):
    CCP_word(x_j) = Σ_{k: NLI(x_j^k, x_j) = entail} P(x_j^k | x_<j)
                    / Σ_{k: NLI(x_j^k, x_j) ∈ {entail, contra}} P(x_j^k | x_<j)
x_j^k — top-k альтернатив в позиции j, сам жадный токен входит (с собой —
entail). Нейтральные альтернативы не входят ни в числитель, ни в знаменатель.
CCP утверждения = Π_j CCP_word(x_j); сигнал неопределённости = 1 − CCP.
Для short-form утверждение — весь ответ (клеймы с token_span в дампе только
у long-form, и те без выравнивания на наши токены).

Наши решения (в статье/LM-Polygraph могут отличаться):
  - NLI-вход: «вопрос + префикс ответа + токен» — premise с альтернативой,
    hypothesis с исходным токеном; токены не склеиваются в слова;
  - k=10, альтернативы с P < 1e-3 отбрасываются (в сумме почти ничего не
    весят, а NLI-вызовов экономят большую часть);
  - префикс ответа обрезается до последних 300 символов.

Вариант по словам (C9 §18), пишется рядом с прежним (ccp_word_*, ccp_first_*):
  - жадный ответ делится на слова по токенам Llama (новое слово — токен с
    ведущим пробелом «Ġ»); токен конца ответа и слова из одной пунктуации
    в произведение не входят;
  - альтернативы — top-k первого токена слова с P ≥ WORD_MIN_PROB; каждая
    дописывается жадно до конца слова (одна короткая пакетная генерация на
    слово); вес альтернативы = P(её первого токена), как и у жадного слова;
  - NLI: «вопрос + префикс + слово-альтернатива» против «вопрос + префикс +
    жадное слово»;
  - ccp_first_* — только первое слово ответа (гипотеза H4: неопределённость
    на ответной сущности).

Запуск:
    # локально, CPU
    python -m dump_assembly.ccp extract --dump dump_pilot_v4.jsonl --out ccp_input.jsonl
    # GPU: нужен доступ к meta-llama/Llama-3.1-8B-Instruct на HF (~16 ГБ в bf16)
    python -m dump_assembly.ccp run --input ccp_input.jsonl --out ccp.jsonl
    # скринер
    python -m screener.run_screener ... --sidecar ccp.jsonl --cards
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .generation import build_prompt_ids

LONGFORM = "franq_longform"
MODES = (("rag", "rag"), ("closed_book", "cb"))
DEFAULT_GENERATOR = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_NLI = "microsoft/deberta-large-mnli"
TOP_K, MIN_PROB, PREFIX_CHARS = 10, 1e-3, 300
WORD_MIN_PROB, WORD_MAX_TOKENS = 1e-2, 8
CONTRA, NEUTRAL, ENTAIL = 0, 1, 2


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- формула (без моделей) ---------------------------------------------

def ccp_word(p_greedy: float, alt_probs: list[float], alt_labels: list[int]) -> float:
    num = p_greedy + sum(p for p, l in zip(alt_probs, alt_labels) if l == ENTAIL)
    den = p_greedy + sum(p for p, l in zip(alt_probs, alt_labels) if l in (ENTAIL, CONTRA))
    return num / den if den > 0 else 1.0


def ccp_scores(words: list[float]) -> dict[str, float]:
    """1 − Π CCP_word (как у них) и 1 − среднее (не схлопывается к 1 на длинных ответах)."""
    if not words:
        return {"ccp": float("nan"), "ccp_mean": float("nan")}
    w = np.clip(np.asarray(words, dtype=float), 1e-12, 1.0)
    return {"ccp": float(1 - np.exp(np.log(w).sum())), "ccp_mean": float(1 - w.mean())}


def answer_ccp(question: str, greedy: list[int], step_topk: list[tuple[list[int], list[float]]],
               p_greedy: list[float], decode, nli) -> list[float]:
    """step_topk[j] = (id альтернатив, их вероятности) в позиции j; nli(pairs) -> метки."""
    pairs, owners = [], []
    alts_per_pos: list[tuple[list[float], list[int]]] = []
    for j, (ids, probs) in enumerate(step_topk):
        prefix = decode(greedy[:j])[-PREFIX_CHARS:]
        orig = f"{question} {prefix}{decode([greedy[j]])}"
        keep = [(i, p) for i, p in zip(ids, probs) if i != greedy[j] and p >= MIN_PROB]
        alts_per_pos.append(([p for _, p in keep], []))
        for i, _ in keep:
            pairs.append((f"{question} {prefix}{decode([i])}", orig))
            owners.append(j)
    labels = nli(pairs) if pairs else []
    for j, l in zip(owners, labels):
        alts_per_pos[j][1].append(l)
    return [ccp_word(p_greedy[j], probs, labs) for j, (probs, labs) in enumerate(alts_per_pos)]


def word_spans(token_strs: list[str], is_special: list[bool], is_punct: list[bool]) -> list[tuple[int, int]]:
    """Слова жадного ответа как полуинтервалы позиций токенов [start, end).
    Новое слово начинается с токена с ведущим пробелом/переводом строки
    (BPE Llama: «Ġ», «Ċ») или после пунктуации; спецтокены и слова только из
    пунктуации отбрасываются."""
    spans, start = [], None
    for j, (t, sp, pu) in enumerate(zip(token_strs, is_special, is_punct)):
        boundary = sp or pu or start is None or t.startswith(("Ġ", "Ċ", "▁")) or is_punct[j - 1]
        if boundary and start is not None:
            spans.append((start, j))
            start = None
        if not sp and not pu and start is None:
            start = j
    if start is not None:
        spans.append((start, len(token_strs)))
    return spans


def answer_ccp_words(question: str, greedy: list[int], spans: list[tuple[int, int]],
                     step_topk: list[tuple[list[int], list[float]]], p_greedy: list[float],
                     decode, complete_word, nli) -> list[float]:
    """CCP_word по словам; complete_word(prefix_ids, alt_ids) -> тексты слов-альтернатив."""
    out = []
    for start, end in spans:
        prefix = decode(greedy[:start])[-PREFIX_CHARS:]
        orig_word = decode(greedy[start:end]).strip()
        ids, probs = step_topk[start]
        alts = [(i, p) for i, p in zip(ids, probs) if i != greedy[start] and p >= WORD_MIN_PROB]
        if not alts:
            out.append(1.0)
            continue
        words = complete_word(greedy[:start], [i for i, _ in alts])
        pairs = [(f"{question} {prefix}{' ' if prefix and not prefix.endswith(' ') else ''}{w.strip()}",
                  f"{question} {prefix}{' ' if prefix and not prefix.endswith(' ') else ''}{orig_word}") for w in words]
        labels = nli(pairs)
        # слово-альтернатива, совпавшая с жадным словом после дописывания, — то же слово
        labels = [ENTAIL if w.strip().lower() == orig_word.lower() else l for w, l in zip(words, labels)]
        out.append(ccp_word(p_greedy[start], [p for _, p in alts], labels))
    return out


# ---- модели ---------------------------------------------------------------

class Generator:
    def __init__(self, model_name: str, dtype: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype),
                                                          device_map="auto").eval()

    def topk(self, prompt_ids: list[int], greedy: list[int]) -> tuple[list, list[float]]:
        """Teacher forcing жадного ответа: top-k (ids, probs) и P(жадного токена) по позициям."""
        ids = self.torch.tensor([prompt_ids + greedy], device=self.model.device)
        with self.torch.no_grad():
            logits = self.model(input_ids=ids).logits[0, len(prompt_ids) - 1:-1].float()
        probs = logits.softmax(dim=-1)
        top = probs.topk(TOP_K, dim=-1)
        g = probs[self.torch.arange(len(greedy)), self.torch.tensor(greedy, device=probs.device)]
        steps = [(i.tolist(), p.tolist()) for i, p in zip(top.indices, top.values)]
        return steps, g.tolist()

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_info(self, ids: list[int]) -> tuple[list[str], list[bool], list[bool]]:
        import string
        strs = self.tokenizer.convert_ids_to_tokens(ids)
        special = set(self.tokenizer.all_special_ids)
        texts = [self.tokenizer.decode([i]).strip() for i in ids]
        punct = [bool(t) and all(c in string.punctuation or c in "–—…“”‘’«»" for c in t) for t in texts]
        return strs, [i in special for i in ids], punct

    def complete_words(self, prompt_ids: list[int], prefix: list[int], alt_ids: list[int]) -> list[str]:
        """Жадно дописывает каждую альтернативу первого токена до конца слова.
        Все входы одной длины — пакет без паддинга."""
        torch = self.torch
        base = prompt_ids + prefix
        ids = torch.tensor([base + [a] for a in alt_ids], device=self.model.device)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      max_new_tokens=WORD_MAX_TOKENS, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        words = []
        for row in out[:, len(base):].tolist():
            strs, special, punct = self.token_info(row)
            end = 1
            while end < len(row) and not (special[end] or punct[end] or strs[end].startswith(("Ġ", "Ċ", "▁"))):
                end += 1
            words.append(self.decode(row[:end]))
        return words


class NLI:
    def __init__(self, model_name: str, batch_size: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        names = {i: l.lower() for i, l in self.model.config.id2label.items()}
        ours = {"contradiction": CONTRA, "neutral": NEUTRAL, "entailment": ENTAIL}
        self.map = {i: ours[n] for i, n in names.items()}
        self.batch_size = batch_size

    def __call__(self, pairs: list[tuple[str, str]]) -> list[int]:
        out: list[int] = []
        for s in range(0, len(pairs), self.batch_size):
            chunk = pairs[s:s + self.batch_size]
            enc = self.tokenizer([a for a, _ in chunk], [b for _, b in chunk], return_tensors="pt",
                                 padding=True, truncation=True, max_length=256).to(self.device)
            with self.torch.no_grad():
                pred = self.model(**enc).logits.argmax(dim=-1).tolist()
            out.extend(self.map[i] for i in pred)
        return out


# ---- команды ------------------------------------------------------------

def cmd_extract(args) -> None:
    n = 0
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.dump):
            fout.write(json.dumps({
                "qid": r["qid"], "source": r["source"], "question": r["question"],
                "passages": [p["text"] for p in r.get("passages", [])],
                # дампы до v4 включительно — без поля, это v1; long-form всегда v1
                "prompt_version": "v1" if r["source"] == LONGFORM else r.get("prompt_version", "v1"),
                **{m: {"greedy_tokens": r[m]["greedy_tokens"]} for m, _ in MODES},
            }) + "\n")
            n += 1
    print(f"Готово: {n} записей -> {args.out}")


def cmd_run(args) -> None:
    done = {j["qid"] for j in iter_jsonl(args.out)} if args.out.exists() else set()
    todo = [r for r in iter_jsonl(args.input) if r["qid"] not in done]
    if args.only_short:
        todo = [r for r in todo if r["source"] != LONGFORM]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} уже посчитано, к обработке {len(todo)}")
    if not todo:
        return
    gen = Generator(args.generator, args.dtype)
    nli = NLI(args.nli_model, args.batch_size)
    # санити: short-form ответ генерировался argmax-ом, значит при точно
    # воспроизведённом промпте жадный токен почти всегда top-1. Низкая доля —
    # промпт/модель не те, что при сборке дампа; CCP тогда не считать.
    match, total = 0, 0
    with open(args.out, "a") as fout:
        for n, r in enumerate(todo):
            sig, row_match = {}, {}
            for mode, suf in MODES:
                greedy = r[mode]["greedy_tokens"]
                passages = r["passages"] if mode == "rag" else None
                prompt_ids, _ = build_prompt_ids(gen.tokenizer, r["question"], passages, r["prompt_version"])
                steps, p_greedy = gen.topk(prompt_ids, greedy)
                m = sum(ids[0] == g for (ids, _), g in zip(steps, greedy))
                row_match[suf] = m / max(len(greedy), 1)
                if r["source"] != LONGFORM:
                    match, total = match + m, total + len(greedy)
                words = answer_ccp(r["question"], greedy, steps, p_greedy, gen.decode, nli)
                for k, v in ccp_scores(words).items():
                    sig[f"{k}_{suf}"] = v
                if args.word_level and r["source"] != LONGFORM:
                    spans = word_spans(*gen.token_info(greedy))
                    ww = answer_ccp_words(r["question"], greedy, spans, steps, p_greedy, gen.decode,
                                          lambda pre, alts: gen.complete_words(prompt_ids, pre, alts), nli)
                    sc = ccp_scores(ww)
                    sig[f"ccp_word_{suf}"], sig[f"ccp_word_mean_{suf}"] = sc["ccp"], sc["ccp_mean"]
                    sig[f"ccp_first_{suf}"] = 1 - ww[0] if ww else float("nan")
            sig["ccp_diff"] = sig["ccp_rag"] - sig["ccp_cb"]
            if "ccp_word_rag" in sig:
                sig["ccp_word_diff"] = sig["ccp_word_rag"] - sig["ccp_word_cb"]
            fout.write(json.dumps({"qid": r["qid"], "models": {"generator": args.generator, "nli": args.nli_model},
                                   "top1_match": row_match, "signals": sig}) + "\n")
            if (n + 1) % 25 == 0:
                fout.flush()
                print(f"  {n + 1}/{len(todo)}  жадный = top-1 (short-form): {match / max(total, 1):.3f}")
    if total == 0:
        # только long-form: их текст teacher-forced, а не argmax модели — совпадения и не ждём
        print(f"Готово -> {args.out}\nShort-form в этом прогоне не было — проверка top-1 не применима")
    else:
        rate = match / total
        print(f"Готово -> {args.out}\nЖадный токен = top-1 на short-form: {rate:.3f}"
              + ("" if rate > 0.95 else "  <-- НИЗКО: промпт или модель не совпадают со сборкой дампа"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("run")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--generator", default=DEFAULT_GENERATOR, help="тот же, что собирал дамп")
    p.add_argument("--nli-model", default=DEFAULT_NLI)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--only-short", action="store_true",
                   help="без long-form: их ответы ~400 токенов, это большая часть NLI-вызовов")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--word-level", action="store_true", help="ещё CCP по словам (ccp_word_*, ccp_first_*), C9 §18")
    p.set_defaults(fn=cmd_run)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
