"""
A4 — блок `signals` дампа: p_true, verbalized_conf. Стандартные self-eval
техники (Kadavath et al. 2022 / Tian et al. 2023), не отдельные модели —
один короткий доп. forward/generate на нашем же генераторе.

alignscore — НЕ реализован здесь. Настоящий AlignScore (Zha et al. 2023) —
отдельный репозиторий, RoBERTa, дообученная специально под faithfulness.
V1 использует NLI-faithfulness (labeling.NLIFaithfulnessScorer) как
временную замену — это честная замена, не то же самое, см. cluster_runbook.md
для плана интеграции настоящего AlignScore.
"""
from __future__ import annotations

import re

import torch

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@torch.no_grad()
def p_true(model, tokenizer, question: str, passages: list[str] | None, answer: str, device) -> float:
    """P(True) — Kadavath et al. 2022: модель сама оценивает свой ответ.
    Промпт заканчивается на 'True or False? Answer:', берём P(токен 'True')
    против P(токен 'False') на первом сгенерированном токене."""
    context = ""
    if passages:
        context = "Context:\n" + "\n".join(passages) + "\n\n"
    prompt = (
        f"{context}Question: {question}\nProposed answer: {answer}\n"
        f"Is the proposed answer true or false? Answer with exactly one word, True or False.\nAnswer:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**input_ids)
    logits = out.logits[0, -1, :].float()

    true_ids = tokenizer(" True", add_special_tokens=False)["input_ids"][:1]
    false_ids = tokenizer(" False", add_special_tokens=False)["input_ids"][:1]
    if not true_ids or not false_ids:
        return float("nan")

    two_way = torch.tensor([logits[true_ids[0]], logits[false_ids[0]]])
    probs = torch.softmax(two_way, dim=0)
    return float(probs[0])


@torch.no_grad()
def verbalized_conf(model, tokenizer, question: str, passages: list[str] | None, answer: str, device) -> float:
    """Verbalized confidence (Tian et al. 2023) — модель прямо называет
    число. Менее надёжно, чем p_true (нужен парсинг текста), но дёшево и
    стандартно как второй, независимый self-eval сигнал."""
    context = ""
    if passages:
        context = "Context:\n" + "\n".join(passages) + "\n\n"
    prompt = (
        f"{context}Question: {question}\nProposed answer: {answer}\n"
        f"On a scale from 0 to 100, how confident are you that this answer is correct? "
        f"Respond with only the number.\nConfidence:"
    )
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**input_ids, max_new_tokens=6, do_sample=False)
    text = tokenizer.decode(out[0, input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
    match = _NUMBER_RE.search(text)
    if not match:
        return float("nan")
    value = float(match.group())
    return max(0.0, min(1.0, value / 100.0))
