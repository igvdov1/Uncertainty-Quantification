"""
Adaptive Temperature Scaling (Xie et al., EMNLP 2024, arXiv 2409.19817) —
карточка adaptive-temperature-scaling, упрощённый вариант (решение 2026-10-06):
голова — ЛИНЕЙНЫЙ слой (в статье — causal transformer layer; по их абляции
линейная чуть хуже), обучение на подвыборке Alpaca-GPT4 (в статье — весь
датасет, 2 эпохи, ~6 GPU-часов L40 для 7B). Бэкбон заморожен.

По статье:
  τ_t = head(h_t),  h_t — последний hidden state (после финальной нормы);
  q̂_t = z_t · exp(τ_t)                 (z — логиты модели);
  selective smoothing loss, α = 0.5:
     если argmax z_t = y_t:   −(1−α) · log softmax(q̂_t)_{y_t}
     иначе:                   −(α/|V|) · Σ_v log softmax(q̂_t)_v

Обучение (на сервере):
    python -m dump_assembly.ats train --model meta-llama/Llama-3.1-8B-Instruct \\
        --n-examples 5000 --out ats_head.pt
Применение к нашим ответам — в mech_cards.py (ats_token_stats).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

ALPHA = 0.5


def selective_smoothing_loss(logits: torch.Tensor, tau: torch.Tensor, targets: torch.Tensor,
                             alpha: float = ALPHA) -> torch.Tensor:
    """logits: (T, V), tau: (T,), targets: (T,). Среднее по токенам."""
    scaled = logits * tau.exp().unsqueeze(-1)
    logp = F.log_softmax(scaled.float(), dim=-1)
    correct = logits.argmax(-1) == targets
    nll_correct = -(1 - alpha) * logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    smooth_wrong = -(alpha / logits.shape[-1]) * logp.sum(-1)
    return torch.where(correct, nll_correct, smooth_wrong).mean()


def ats_token_stats(logits: torch.Tensor, hidden: torch.Tensor, targets: torch.Tensor, head) -> dict[str, float]:
    """Сигналы после ATS: длина-нормированный NLL, max NLL и средняя энтропия
    распределения softmax(z · e^τ) на позициях, предсказывающих токены ответа."""
    with torch.no_grad():
        tau = head(hidden.float()).squeeze(-1)
        logp = F.log_softmax(logits.float() * tau.exp().unsqueeze(-1), dim=-1)
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1)
    return {"ats_nll": float(nll.mean()), "ats_max_nll": float(nll.max()), "ats_entropy": float(ent.mean()),
            "ats_mean_tau": float(tau.mean())}


def make_head(hidden_size: int) -> torch.nn.Linear:
    head = torch.nn.Linear(hidden_size, 1)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)  # τ = 0 -> исходная температура 1
    return head


def load_head(path: Path, device) -> torch.nn.Linear:
    state = torch.load(path, map_location=device)
    head = make_head(state["weight"].shape[1]).to(device)
    head.load_state_dict({"weight": state["weight"], "bias": state["bias"]})
    return head.eval()


def alpaca_examples(n: int, seed: int):
    from datasets import load_dataset
    ds = load_dataset("vicgalle/alpaca-gpt4", split="train").shuffle(seed=seed).select(range(n))
    for ex in ds:
        user = ex["instruction"] + (f"\n\n{ex['input']}" if ex.get("input") else "")
        yield user, ex["output"]


def cmd_train(args) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="auto").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device
    head = make_head(model.config.hidden_size).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.0)
    total = args.n_examples
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    running, seen = 0.0, 0
    for step, (user, answer) in enumerate(alpaca_examples(args.n_examples, args.seed)):
        prompt = tok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(answer, add_special_tokens=False)["input_ids"][: args.max_answer_tokens] + [tok.eos_token_id]
        ids = torch.tensor([p_ids + a_ids], device=device)
        with torch.no_grad():
            out = model(input_ids=ids, output_hidden_states=True)
        sl = slice(len(p_ids) - 1, len(p_ids) + len(a_ids) - 1)  # позиции, предсказывающие токены ответа
        logits = out.logits[0, sl].float()
        hidden = out.hidden_states[-1][0, sl].float()
        targets = torch.tensor(a_ids, device=device)
        loss = selective_smoothing_loss(logits, head(hidden).squeeze(-1), targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()
        seen += 1
        if (step + 1) % 200 == 0:
            print(f"  {step + 1}/{args.n_examples}  loss={running / seen:.4f}  "
                  f"|w|={head.weight.norm().item():.4f} b={head.bias.item():+.4f}")
            running, seen = 0.0, 0
    torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu(), "model": args.model,
                "n_examples": args.n_examples, "alpha": ALPHA}, args.out)
    print(f"Готово -> {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n-examples", type=int, default=5000)
    p.add_argument("--max-answer-tokens", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("ats_head.pt"))
    p.set_defaults(fn=cmd_train)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
