"""
A1 — проверка кандидата на роль основного генератора.

Гейт: HF transformers действительно отдаёт (а) top-k логиты/логпробы на каждом
шаге генерации и (б) attention-карты по слоям и головам, для выбранной модели
и выбранной прецизии/квантизации.

Прогоняется на HPC (нужен GPU) — из чата у архитектора нет доступа к кластеру.
Результат — гейт перед A2: если attention недоступен на выбранной версии
модели, схема отката в A1_decisions.md (раздел 1.2) вступает в силу.

Запуск:
    python verify_generator_backend.py --model meta-llama/Llama-3.1-8B-Instruct
    python verify_generator_backend.py --model Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = (
    "Context: The Eiffel Tower is located in Paris, France. It was completed in 1889.\n"
    "Question: In which city is the Eiffel Tower located?\n"
    "Answer:"
)

TOP_K = 20


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"[1/5] Loading tokenizer + model: {args.model} (dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",  # required: sdpa/flash-attn do not return attentions
    )
    model.eval()

    print("[2/5] Tokenizing prompt")
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    print("[3/5] Generating with output_scores=True, output_attentions=True")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    generated_ids = out.sequences[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"    generated: {generated_text!r}")

    print("[4/5] Checking logits / top-k access")
    n_steps = len(out.scores)
    assert n_steps == generated_ids.shape[0], (
        f"expected {generated_ids.shape[0]} score steps, got {n_steps}"
    )
    step0_logits = out.scores[0][0]  # (vocab,)
    vocab_size = step0_logits.shape[-1]
    topk = torch.topk(step0_logits, k=TOP_K)
    print(f"    steps={n_steps}, vocab_size={vocab_size}, top-{TOP_K} logits OK "
          f"(example top1 id={topk.indices[0].item()})")

    print("[5/5] Checking attention access")
    assert out.attentions is not None, "model.generate returned no attentions " \
        "(check attn_implementation='eager' and that the model architecture supports it)"
    n_attn_steps = len(out.attentions)
    step0_attn = out.attentions[0]  # tuple over layers
    n_layers = len(step0_attn)
    layer0 = step0_attn[0]  # (batch, n_heads, seq_q, seq_k)
    n_heads = layer0.shape[1]
    seq_k = layer0.shape[-1]
    print(f"    attn_steps={n_attn_steps}, n_layers={n_layers}, n_heads={n_heads}, "
          f"seq_k(step0)={seq_k}")

    all_finite = torch.isfinite(layer0).all().item()
    non_degenerate = (layer0.std() > 1e-6).item()
    print(f"    values finite={all_finite}, non-degenerate (std>1e-6)={non_degenerate}")

    print("\n=== VERDICT ===")
    ok = (
        n_steps == generated_ids.shape[0]
        and out.attentions is not None
        and n_layers > 0
        and n_heads > 0
        and all_finite
        and non_degenerate
    )
    print("PASS — model exposes logprobs and real attention maps via HF transformers"
          if ok else "FAIL — see checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
