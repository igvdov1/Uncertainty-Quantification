"""
A4 — генерация и извлечение полей дампа (dump_schema_v1.md, rag/closed_book).

Два режима:
  - generate_greedy(...) — автогенерация (нужна для short-form: NQ/TriviaQA/
    PopQA/WebQuestions, у которых нет готового ответа-текста).
  - teacher_force(...) — прогон УЖЕ ГОТОВОГО текста через нашу модель одним
    forward-проходом (нужен для long-form FRANQ: 76 вопросов переиспользуют
    оригинальный текст ответа и разметку, но logprobs/attention/hidden_states
    должны быть от НАШЕГО генератора, а не позаимствованы из чужого прогона).

Логика извлечения тяжёлых полей унаследована от measure_heavy_fields_weight.py
(уже провалидирована на tiny-random-Llama и прогнана на реальном
Llama-3.1-8B-Instruct, см. A1_decisions.md §1.2) и расширена:
  - token_topk / token_entropy — полные, не top-5 (B0-A1)
  - attention_by_head — узкий срез по головам на 1-2 слоях (B3 ветки B,
    одобрено 2026-09-22), в дополнение к attention_by_group/attention_by_passage
  - sample_* — сэмплирование с логпробами для sampling-based сигналов
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

MAX_NEW_TOKENS = 50
N_SAMPLES = 10
TOPK = 20
HEAD_LAYERS = [16, 31]  # placeholder, см. dump_schema_v1.md — уточнить под INTRYGUE/ReDeEP


@dataclass
class PromptSpans:
    instruction: tuple[int, int]
    passages: list[tuple[int, int]]
    question: tuple[int, int]

    def as_dict(self) -> dict:
        return {
            "instruction": list(self.instruction),
            "passages": [list(p) for p in self.passages],
            "question": list(self.question),
        }


INSTRUCTION_RAG = "Using the context passages provided below, answer the question concisely."
INSTRUCTION_CB = "Answer the question concisely, using your own knowledge."


_CHAT_SENTINEL = "\x00"


def _chat_prefix_suffix_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Токены до и после контента user-сообщения в chat-template модели.
    Нужно, чтобы обернуть существующую ручную токенизацию по частям (она
    уже трекает офсеты instruction/passages/question для attention-полей)
    в правильный chat-формат, не переписывая эту логику с нуля.

    Без этого Instruct-модель (Llama-3.1/Qwen2.5) получает сырой текст без
    <|start_header_id|>/<|im_start|> разметки, на которой её дообучали
    распознавать конец ответа — и не останавливается на EOS, а генерирует
    до MAX_NEW_TOKENS почти во всех случаях (найдено на реальном дампе:
    514/524 closed_book и 520/524 rag ответов short-form упирались ровно
    в лимit, обрывая ответ на середине слова)."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _CHAT_SENTINEL}],
        add_generation_prompt=True, tokenize=False,
    )
    before, after = rendered.split(_CHAT_SENTINEL)
    prefix_ids = tokenizer(before, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(after, add_special_tokens=False)["input_ids"]
    return prefix_ids, suffix_ids


def build_prompt_ids(tokenizer, question: str, passages: list[str] | None) -> tuple[list[int], PromptSpans]:
    def toks(text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    prefix_ids, suffix_ids = _chat_prefix_suffix_ids(tokenizer)
    ids: list[int] = list(prefix_ids)

    instruction = INSTRUCTION_RAG if passages else INSTRUCTION_CB
    a = len(ids)
    ids += toks(f"Instruction: {instruction}\n\n")
    b = len(ids)

    passage_spans = []
    for i, p in enumerate(passages or []):
        c = len(ids)
        ids += toks(f"Passage {i + 1}: {p}\n")
        d = len(ids)
        passage_spans.append((c, d))

    e = len(ids)
    ids += toks(f"\nQuestion: {question}\nAnswer:")
    f = len(ids)
    ids += suffix_ids

    return ids, PromptSpans(instruction=(a, b), passages=passage_spans, question=(e, f))


def _group_for_position(pos: int, prompt_len: int, spans: PromptSpans) -> int:
    if pos >= prompt_len:
        return 3
    a, b = spans.instruction
    if a <= pos < b:
        return 0
    e, f = spans.question
    if e <= pos < f:
        return 2
    return 1


def _passage_index_for_position(pos: int, spans: PromptSpans):
    for i, (c, d) in enumerate(spans.passages):
        if c <= pos < d:
            return i
    return None


def _entropy_from_logits(logits: torch.Tensor) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum())


def _extract_attention_fields(attn_step_layers: tuple, prompt_len: int, spans: PromptSpans, n_passages: int):
    """attn_step_layers: tuple по слоям, каждый (batch, heads, query_len, key_len).
    Возвращает (attention_by_group[4], attention_by_passage[n_passages], attention_by_head для HEAD_LAYERS)."""
    last_layer = attn_step_layers[-1][0].mean(dim=0)  # (query_len, key_len), усреднено по головам
    row = last_layer[-1, :].float().cpu().numpy()
    key_len = row.shape[0]

    group_sums = [0.0, 0.0, 0.0, 0.0]
    passage_sums = [0.0] * n_passages
    for pos in range(key_len):
        g = _group_for_position(pos, prompt_len, spans)
        group_sums[g] += float(row[pos])
        if g == 1:
            pidx = _passage_index_for_position(pos, spans)
            if pidx is not None:
                passage_sums[pidx] += float(row[pos])

    head_values = []
    for layer_idx in HEAD_LAYERS:
        layer_idx = min(layer_idx, len(attn_step_layers) - 1)
        layer_attn = attn_step_layers[layer_idx][0]  # (heads, query_len, key_len)
        per_head_row = layer_attn[:, -1, :].float().cpu().numpy()  # (heads, key_len)
        head_values.append(per_head_row.tolist())

    return group_sums, passage_sums, head_values


@torch.no_grad()
def generate_greedy(model, tokenizer, question: str, passages: list[str] | None, device) -> dict:
    """Автогенерация с полным извлечением полей. passages=None -> closed-book."""
    prompt_ids, spans = build_prompt_ids(tokenizer, question, passages)
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)
    n_passages = len(spans.passages)

    out = model.generate(
        input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        output_attentions=bool(passages), output_hidden_states=True,
        output_scores=True, return_dict_in_generate=True,
    )
    gen_ids = out.sequences[0, prompt_len:]
    n_gen = gen_ids.shape[0]
    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    token_logprobs, token_topk, token_entropy = [], [], []
    for t in range(n_gen):
        logits = out.scores[t][0]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs.append(float(log_probs[gen_ids[t]]))
        topk = torch.topk(logits.float(), k=TOPK)
        token_topk.append(topk.values.cpu().tolist())
        token_entropy.append(_entropy_from_logits(logits))

    hidden_states_last = [
        out.hidden_states[t][-1][0, -1, :].float().cpu().numpy().tolist() for t in range(n_gen)
    ]

    result = {
        "answer": answer_text,
        "greedy_tokens": gen_ids.cpu().tolist(),
        "token_logprobs": token_logprobs,
        "token_topk": token_topk,
        "token_entropy": token_entropy,
        "hidden_states_last": hidden_states_last,
        "prompt_raw": tokenizer.decode(prompt_ids),
        "prompt_spans": spans.as_dict(),
    }

    if passages:
        attn_by_group, attn_by_passage, attn_by_head = [], [], []
        for t in range(n_gen):
            g, p, h = _extract_attention_fields(out.attentions[t], prompt_len, spans, n_passages)
            attn_by_group.append(g)
            attn_by_passage.append(p)
            attn_by_head.append(h)
        result["attention_by_group"] = attn_by_group
        result["attention_by_passage"] = attn_by_passage
        result["attention_by_head"] = {"layers": HEAD_LAYERS, "values": attn_by_head}

    return result


@torch.no_grad()
def teacher_force(model, tokenizer, question: str, passages: list[str] | None, target_text: str, device) -> dict:
    """Прогоняет ГОТОВЫЙ target_text через модель одним forward-проходом.
    Даёт те же поля, что generate_greedy, но без вызова .generate() — для
    long-form FRANQ, где ответ и разметка переиспользуются, а logprobs/
    attention должны быть от нашей модели (см. докстринг модуля)."""
    prompt_ids, spans = build_prompt_ids(tokenizer, question, passages)
    prompt_len = len(prompt_ids)
    n_passages = len(spans.passages)

    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + target_ids
    input_ids = torch.tensor([full_ids], device=device)
    n_gen = len(target_ids)

    out = model(
        input_ids=input_ids, output_attentions=bool(passages),
        output_hidden_states=True, use_cache=False,
    )
    logits_all = out.logits[0]  # (seq_len, vocab)

    token_logprobs, token_topk, token_entropy = [], [], []
    for t in range(n_gen):
        pos = prompt_len + t - 1  # логиты на позиции pos предсказывают токен pos+1
        logits = logits_all[pos]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_logprobs.append(float(log_probs[target_ids[t]]))
        topk = torch.topk(logits.float(), k=TOPK)
        token_topk.append(topk.values.cpu().tolist())
        token_entropy.append(_entropy_from_logits(logits))

    hidden_states_last = [
        out.hidden_states[-1][0, prompt_len + t - 1, :].float().cpu().numpy().tolist()
        for t in range(n_gen)
    ]

    result = {
        "answer": target_text,
        "greedy_tokens": target_ids,
        "token_logprobs": token_logprobs,
        "token_topk": token_topk,
        "token_entropy": token_entropy,
        "hidden_states_last": hidden_states_last,
        "prompt_raw": tokenizer.decode(prompt_ids),
        "prompt_spans": spans.as_dict(),
    }

    if passages and out.attentions is not None:
        attn_by_group, attn_by_passage, attn_by_head = [], [], []
        for t in range(n_gen):
            pos = prompt_len + t - 1
            # для forward-прохода (не generate) attentions — полные (query_len=seq_len);
            # берём срез слоёв на нужной query-позиции вручную
            layers_at_pos = tuple(layer[:, :, pos: pos + 1, : pos + 1] for layer in out.attentions)
            g, p, h = _extract_attention_fields(layers_at_pos, prompt_len, spans, n_passages)
            attn_by_group.append(g)
            attn_by_passage.append(p)
            attn_by_head.append(h)
        result["attention_by_group"] = attn_by_group
        result["attention_by_passage"] = attn_by_passage
        result["attention_by_head"] = {"layers": HEAD_LAYERS, "values": attn_by_head}

    return result


@torch.no_grad()
def sample(model, tokenizer, question: str, passages: list[str] | None, device,
           n_samples: int = N_SAMPLES, temperature: float = 1.0) -> dict:
    """≥10 сэмплов с логпробами и id токенов (A1 §1.4 / B0-A5)."""
    prompt_ids, _ = build_prompt_ids(tokenizer, question, passages)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    out = model.generate(
        input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
        temperature=temperature, num_return_sequences=n_samples,
        output_scores=True, return_dict_in_generate=True,
    )

    samples_text, sample_tokens, sample_logprobs = [], [], []
    for i in range(n_samples):
        seq = out.sequences[i, prompt_len:]
        eos_pos = (seq == tokenizer.eos_token_id).nonzero()
        cut = int(eos_pos[0]) if len(eos_pos) else len(seq)
        seq = seq[:cut]
        samples_text.append(tokenizer.decode(seq, skip_special_tokens=True))
        sample_tokens.append(seq.cpu().tolist())
        lp = []
        for t in range(len(seq)):
            step_logits = out.scores[t][i]
            log_probs = torch.log_softmax(step_logits.float(), dim=-1)
            lp.append(float(log_probs[seq[t]]))
        sample_logprobs.append(lp)

    return {"samples": samples_text, "sample_tokens": sample_tokens, "sample_logprobs": sample_logprobs}
