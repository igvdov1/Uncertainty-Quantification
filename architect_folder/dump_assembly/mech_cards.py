"""
Механистические и обучаемые карточки банка B через teacher forcing готовых
ответов дампа — без новой генерации. Один forward генератора на запись
(+1 для LUMINA с чужим контекстом, +1 короткий для HACK, +N_SAMPLES для
source-clustering) с attention по всем головам и hidden states по слоям.
Здесь только извлечение признаков; подгонка на dev и сигналы — mech_fit.py.

Методы и источники (эталонный код — github.com/kuleshovaai/intrygue,
src/haldetect/method/*; списки голов для Llama-3.1-8B-Instruct — оттуда же,
config/{induction,copy}_heads.yaml):

  INTRYGUE (arXiv 2603.21607): SinkRate(A, n) = max_j Σ_{i>N−n} α_ij / w_j,
      w_j = число строк ответа, видящих позицию j; по каждой induction head.
      Энтропия по словарю на позициях, предсказывающих токены ответа
      (без двух последних, как в эталоне). Скор = агрегат(энтропия) ·
      агрегат(SinkRate по top-n голов) — в mech_fit.
  ReDeEP (arXiv 2410.11414): ECS — для copy head на каждом токене ответа
      top-10% позиций промпта по вниманию, среднее их последнего hidden
      state, косинус с hidden state токена; PKS — JSD между logit-lens
      распределениями остаточного потока до FFN и после FFN слоя (×1e6,
      среднее по словарю, как в эталоне). Храним средние по токенам ответа
      для каждой головы / слоя; отбор и веса — в mech_fit.
  LUMINA (arXiv 2509.21875): IPR — по logit lens выходов слоёв: энтропия и
      вероятность финального argmax-токена на каждом слое (формула эталона);
      MMD — между следующими-токенными распределениями с исходным и с чужим
      контекстом (промпт следующей записи + тот же ответ), top-100 токенов,
      косинусное ядро на входных эмбеддингах. Скор = 0.5·IPR − 0.5·MMD.
  ATS (arXiv 2409.19817, линейная голова) — dump_assembly/ats.py.
  TAD (Vazhentsev et al., arXiv 2408.10692): признаки токена — внимание на
      N_PREV предыдущих токенов по всем головам и слоям + условные
      вероятности текущего и N_PREV предыдущих токенов.
  HACK (arXiv 2510.24222): hidden state слоя 15 на последнем токене
      closed-book промпта (линейный SVM — в mech_fit).
  source-clustering-entropy (собственный синтез B2): для каждого rag-сэмпла
      пассаж с наибольшей массой внимания (последний слой, среднее по
      головам — как attention_by_passage дампа).

Запуск:
    python -m dump_assembly.mech_cards extract --dump dump_pilot_v5.jsonl --out mech_input.jsonl
    python -m dump_assembly.mech_cards run --input mech_input.jsonl --out-dir mech_out --ats-head ats_head.pt
Выход: mech_out/{qid}.npz — все признаки записи; resume по готовым файлам.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LONGFORM = "franq_longform"
DEFAULT_GENERATOR = "meta-llama/Llama-3.1-8B-Instruct"
N_PREV = 3            # TAD: внимание на 3 предыдущих токена
HACK_LAYER = 15       # HACK: «15th layer»
MMD_TOPK = 100        # LUMINA: top-100 токенов
ECS_TOP_FRAC = 0.10   # ReDeEP: top-10% позиций промпта
N_SOURCE_SAMPLES = 10

# kuleshovaai/intrygue, src/haldetect/config/*.yaml, ключ Llama-3.1-8B-Instruct (порядок = убывание скора)
HEADS = {
    "Llama-3.1-8B-Instruct": {
        "induction": [[15, 30], [8, 1], [2, 22], [16, 20], [15, 1], [5, 8], [10, 14], [20, 14], [5, 11], [20, 1],
                      [24, 27], [26, 15], [19, 3], [13, 6], [2, 20], [27, 7], [27, 6], [16, 1]],
        "copy": [[31, 12], [28, 25], [23, 0], [29, 7], [30, 23], [29, 18], [27, 15], [31, 1], [28, 24], [28, 7],
                 [29, 5], [9, 7], [30, 4], [31, 0], [30, 15], [22, 27], [29, 23], [19, 5], [27, 29], [31, 29],
                 [25, 23], [28, 13], [29, 13], [27, 30], [31, 28], [19, 16], [30, 6], [22, 25], [29, 20], [24, 0],
                 [27, 23], [25, 20]],
    },
}


def iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---- формулы (numpy/torch, без модели) -----------------------------------

def sink_rate(attn: "np.ndarray", response_len: int) -> float:
    """attn: (N, N) карта одной головы по всей последовательности.
    Как INTRYGUE.sink_rate эталона: столбцовые суммы по последним n строкам,
    нормированные на число строк ответа, которые видят столбец; максимум."""
    n_total = attn.shape[-1]
    norm = np.arange(1, n_total + 1)[::-1].astype(float)
    norm[:-response_len] = response_len
    cols = attn[-response_len:].sum(axis=0) / norm
    return float(cols.max())


def jsd_ref(logits_a, logits_b):
    """JSD как calculate_dist в redeep_wrapper эталона: KL усредняются по
    словарю (mean(-1)), результат ×1e6. Вход: (T, V) torch."""
    import torch.nn.functional as F
    pa, pb = F.softmax(logits_a.float(), -1), F.softmax(logits_b.float(), -1)
    m = 0.5 * (pa + pb)
    kl1 = F.kl_div(F.log_softmax(logits_a.float(), -1), m, reduction="none").mean(-1)
    kl2 = F.kl_div(F.log_softmax(logits_b.float(), -1), m, reduction="none").mean(-1)
    return 0.5 * (kl1 + kl2) * 1e6


def lumina_ipr(layer_entropy, layer_greedy_probs, final_greedy_probs, final_token_probs, eps: float = 1e-8):
    """Формула LUMINA._compute_ipr эталона. layer_*: (L, T); final_*: (T,)."""
    import torch
    n_layers = layer_entropy.shape[0]
    weights = 1.0 / (layer_entropy + eps)
    idx = torch.arange(1, n_layers + 1, device=layer_entropy.device, dtype=layer_entropy.dtype).unsqueeze(1)
    ratios = 1 - torch.clamp(layer_greedy_probs / final_greedy_probs.unsqueeze(0), max=1.0)
    return (ratios * idx).sum(0) / (idx * weights).sum(0) * (final_token_probs / final_greedy_probs)


def lumina_mmd(p_probs, q_probs, embedding, k: int = MMD_TOPK):
    """MMD с косинусным ядром на эмбеддингах top-k токенов (эталон LUMINA._compute_mmd)."""
    import torch
    import torch.nn.functional as F

    def topk(probs):
        v, i = torch.topk(probs, k, dim=-1)
        return v.float(), embedding(i).float()

    def kern(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (1 + x @ y.transpose(-1, -2)) / 2

    pv, pe = topk(p_probs)
    qv, qe = topk(q_probs)
    return (torch.einsum("ti,tij,tj->t", pv, kern(pe, pe), pv) + torch.einsum("ti,tij,tj->t", qv, kern(qe, qe), qv)
            - 2 * torch.einsum("ti,tij,tj->t", pv, kern(pe, qe), qv))


# ---- извлечение на модели ------------------------------------------------

class Extractor:
    def __init__(self, model_name: str, dtype: str, ats_head: Path | None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype), device_map="auto",
                                                          attn_implementation="eager").eval()
        self.device = next(self.model.parameters()).device
        key = model_name.split("/")[-1]
        if key not in HEADS:
            raise ValueError(f"нет списков induction/copy heads для {key}; есть: {list(HEADS)}")
        self.induction, self.copy = HEADS[key]["induction"], HEADS[key]["copy"]
        self.layers = self.model.model.layers
        self.norm, self.lm_head = self.model.model.norm, self.model.lm_head
        self.embedding = self.model.get_input_embeddings()
        self.ats = None
        if ats_head is not None:
            from .ats import load_head
            self.ats = load_head(ats_head, self.device)

    def lens(self, h):
        return self.lm_head(self.norm(h))

    def forward(self, ids: list[int], keep: slice, attentions: bool = True, hidden: bool = True):
        """Forward с перехватом остаточного потока до FFN (вход post_attention_layernorm)
        и выхода каждого слоя на позициях keep."""
        torch = self.torch
        anchors, matures, hooks = [], [], []
        for layer in self.layers:
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(
                lambda m, args: anchors.append(args[0][0, keep].detach())))
            hooks.append(layer.register_forward_hook(
                lambda m, args, out: matures.append((out[0] if isinstance(out, tuple) else out)[0, keep].detach())))
        try:
            with torch.no_grad():
                out = self.model(input_ids=torch.tensor([ids], device=self.device),
                                 output_attentions=attentions, output_hidden_states=hidden)
        finally:
            for h in hooks:
                h.remove()
        return out, anchors, matures

    def record(self, r: dict, other_prompt: dict | None) -> dict[str, np.ndarray]:
        torch = self.torch
        from .generation import build_prompt_ids
        pv = "v1" if r["source"] == LONGFORM else r.get("prompt_version", "v1")
        prompt, spans = build_prompt_ids(self.tok, r["question"], r["passages"] or None, pv)
        y = r["rag_tokens"]
        P, T = len(prompt), len(y)
        ids = prompt + y
        res: dict[str, np.ndarray] = {}
        # позиции P-1..P+T-1: [0:T] предсказывают токены ответа, [1:T+1] — сами токены ответа
        out, anchors, matures = self.forward(ids, slice(P - 1, P + T))
        logits = out.logits[0, P - 1:P + T - 1].float()                    # (T, V)
        logp = torch.log_softmax(logits, -1)
        probs = logp.exp()
        y_t = torch.tensor(y, device=logits.device)
        tok_logp = logp.gather(-1, y_t.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * logp).sum(-1)
        att = out.attentions                                                # L x (1, H, S, S)

        # INTRYGUE
        ent = entropy[: max(T - 1, 1)] if T > 1 else entropy               # эталон: без двух последних позиций
        res["intrygue_entropy"] = np.array([float(ent.max()), float(ent.mean())])
        res["intrygue_sink"] = np.array([sink_rate(att[l][0, h].float().cpu().numpy(), T) for l, h in self.induction])

        # ReDeEP: ECS по copy heads, PKS по слоям
        last_h = out.hidden_states[-1][0].float()
        k = max(1, int(P * ECS_TOP_FRAC))
        ecs = []
        for l, h in self.copy:
            a = att[l][0, h, P:P + T, :P].float()                           # (T, P)
            top = torch.topk(a, k, dim=-1).indices                           # (T, k)
            attended = last_h[:P][top].mean(1)                              # (T, D)
            ecs.append(float(torch.nn.functional.cosine_similarity(attended, last_h[P:P + T], dim=-1).mean()))
        res["redeep_ecs"] = np.array(ecs)
        res["redeep_pks"] = np.array([float(jsd_ref(self.lens(m[1:]), self.lens(a_[1:])).mean())
                                      for a_, m in zip(anchors, matures)])

        # LUMINA: IPR по logit lens выходов слоёв на позициях, предсказывающих ответ
        greedy_ids = logits.argmax(-1)
        final_greedy = probs.max(-1).values
        lay_ent, lay_gp = [], []
        for m in matures:
            lp = torch.log_softmax(self.lens(m[:T]).float(), -1)
            lay_ent.append(-(lp.exp() * lp).sum(-1))
            lay_gp.append(lp.exp().gather(-1, greedy_ids.unsqueeze(-1)).squeeze(-1))
        ipr = lumina_ipr(torch.stack(lay_ent), torch.stack(lay_gp), final_greedy, tok_logp.exp())
        mmd = float("nan")
        if other_prompt is not None:
            op, _ = build_prompt_ids(self.tok, other_prompt["question"], other_prompt["passages"] or None,
                                     other_prompt["prompt_version"])
            with torch.no_grad():
                o = self.model(input_ids=torch.tensor([op + y], device=self.device))
            q = torch.softmax(o.logits[0, len(op) - 1:len(op) + T - 1].float(), -1)
            mmd = float(lumina_mmd(probs, q, self.embedding).mean())
        res["lumina"] = np.array([float(ipr.mean()), mmd])

        # ATS
        if self.ats is not None:
            from .ats import ats_token_stats
            s = ats_token_stats(logits, out.hidden_states[-1][0, P - 1:P + T - 1], y_t, self.ats)
            res["ats"] = np.array([s["ats_nll"], s["ats_max_nll"], s["ats_entropy"], s["ats_mean_tau"]])

        # TAD: внимание токена ответа (позиция P+i) на N_PREV предыдущих, все слои и головы
        rows = torch.arange(P, P + T, device=att[0].device)
        feats = []
        for lag in range(1, N_PREV + 1):
            cols = rows - lag
            feats.append(torch.stack([a_[0][:, rows, cols] for a_ in att]).float())   # (L, H, T)
        tad_att = torch.stack(feats).permute(3, 1, 2, 0).reshape(T, -1)              # (T, L*H*N_PREV)
        lp_np = tok_logp.cpu().numpy()
        prev = np.stack([np.concatenate([np.full(lag, np.nan), lp_np[:-lag]])[:T] for lag in range(1, N_PREV + 1)], 1)
        res["tad_att"] = tad_att.cpu().numpy().astype(np.float16)
        res["tad_logp"] = np.concatenate([lp_np[:, None], prev], 1).astype(np.float32)   # (T, 1+N_PREV)
        del out, att, anchors, matures

        # HACK: слой 15 на последнем токене closed-book промпта
        cb_prompt, _ = build_prompt_ids(self.tok, r["question"], None, pv)
        with torch.no_grad():
            hs = self.model(input_ids=torch.tensor([cb_prompt], device=self.device), output_hidden_states=True).hidden_states
        res["hack_h15"] = hs[HACK_LAYER][0, -1].float().cpu().numpy().astype(np.float16)

        # source-clustering: пассаж с максимальной массой внимания для каждого rag-сэмпла
        if r["passages"] and r.get("rag_sample_tokens"):
            src = []
            for s_tok in r["rag_sample_tokens"][:N_SOURCE_SAMPLES]:
                if not s_tok:
                    src.append(-1)
                    continue
                with torch.no_grad():
                    a_last = self.model(input_ids=torch.tensor([prompt + s_tok], device=self.device),
                                        output_attentions=True).attentions[-1][0].float().mean(0)   # (S, S)
                rows_s = a_last[P:P + len(s_tok)]
                mass = [float(rows_s[:, a:b].sum()) for a, b in spans.passages]
                src.append(int(np.argmax(mass)))
            res["source_ids"] = np.array(src)
        return res


def cmd_extract(args) -> None:
    n = 0
    with open(args.out, "w") as fout:
        for r in iter_jsonl(args.dump):
            fout.write(json.dumps({
                "qid": r["qid"], "source": r["source"], "split": r["split"], "question": r["question"],
                "passages": [p["text"] for p in r.get("passages", [])],
                "prompt_version": "v1" if r["source"] == LONGFORM else r.get("prompt_version", "v1"),
                "rag_tokens": r["rag"]["greedy_tokens"],
                "rag_sample_tokens": r["rag"].get("sample_tokens", []),
            }) + "\n")
            n += 1
    print(f"Готово: {n} записей -> {args.out}")


def cmd_run(args) -> None:
    rows = list(iter_jsonl(args.input))
    if args.only_short:
        rows = [r for r in rows if r["source"] != LONGFORM]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [i for i, r in enumerate(rows) if not (args.out_dir / f"{r['qid']}.npz").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows) - len(todo)} уже есть, к обработке {len(todo)}")
    if not todo:
        return
    ex = Extractor(args.generator, args.dtype, args.ats_head)
    same_form = {}
    for i, r in enumerate(rows):
        same_form.setdefault(r["source"] == LONGFORM, []).append(i)
    for n, i in enumerate(todo):
        r = rows[i]
        group = same_form[r["source"] == LONGFORM]
        other = rows[group[(group.index(i) + 1) % len(group)]]   # LUMINA: контекст следующей записи той же формы
        with ex.torch.no_grad():   # logit lens / lm_head вне no_grad строили граф — лишняя память на GPU
            feats = ex.record(r, other)
        np.savez(args.out_dir / f"{r['qid']}.npz", **feats)
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(todo)}")
    print(f"Готово -> {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--dump", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("run")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--generator", default=DEFAULT_GENERATOR)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--ats-head", type=Path, default=None)
    p.add_argument("--only-short", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_run)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
