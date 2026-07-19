#!/usr/bin/env python3
"""E1 — router-only fine-tune of OLMoE with an entropy-concentration objective.

Freezes everything except the per-layer router gates (model.layers.*.mlp.gate.weight,
~2M params) and trains  L = L_lm + alpha * H(router)  where H is the mean per-token
routing entropy across layers. Concentrating the router raises streaming-cache hit
rates; the frozen experts cannot lose calibration (the renorm lesson). Design and
pre-registered success criteria: docs/design/finetune-track.md (E1).

usage: train_router.py --model <hf_dir> --alpha 0.05 --steps 300 --out gates_a05.safetensors
"""
import argparse, math, time

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--alpha", type=float, required=True)
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch", type=int, default=2)
p.add_argument("--seq", type=int, default=512)
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--out", required=True)
p.add_argument("--corpus", choices=["wikitext", "fineweb"], default="fineweb")
p.add_argument("--objective", choices=["token", "usage"], default="usage",
               help="'usage' = minimize aggregate-usage entropy (anti-load-balancing; the "
                    "cache-relevant concentration). 'token' = round-1 per-token objective.")
a = p.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable()
model.train()

gate_params = []
for name, param in model.named_parameters():
    if ".mlp.gate." in name:
        param.requires_grad_(True); gate_params.append((name, param))
    else:
        param.requires_grad_(False)
print(f"trainable: {len(gate_params)} gate tensors, "
      f"{sum(q.numel() for _, q in gate_params):,} params")

opt = torch.optim.AdamW([q for _, q in gate_params], lr=a.lr)

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
text = "\n\n".join(t for t in ds["text"] if t.strip())
ids = tok(text, return_tensors="pt").input_ids[0]
n_blocks = ids.numel() // a.seq
blocks = ids[: n_blocks * a.seq].view(n_blocks, a.seq)
print(f"corpus: {ids.numel():,} tokens -> {n_blocks} blocks of {a.seq}")

step, t0 = 0, time.time()
while step < a.steps:
    for i in range(0, n_blocks - a.batch, a.batch):
        batch = blocks[i : i + a.batch].cuda()
        out = model(input_ids=batch, labels=batch, output_router_logits=True)
        # Two concentrations, only one of which the cache cares about (E1 round-1 lesson):
        # per-token entropy sharpens each token's mix but lets tokens diverge; the cache
        # wants low entropy of the AGGREGATE usage — the anti-load-balancing loss.
        tok_ents, use_ents = [], []
        for rl in out.router_logits:
            p = torch.softmax(rl.float(), dim=-1)
            tok_ents.append(torch.distributions.Categorical(probs=p).entropy().mean())
            u = p.mean(0)
            use_ents.append(-(u * (u + 1e-9).log()).sum())
        ent = torch.stack(tok_ents).mean()
        use_ent = torch.stack(use_ents).mean()
        loss = out.loss + a.alpha * (use_ent if a.objective == "usage" else ent)
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 25 == 0 or step == 1:
            print(f"step {step:4d}  lm {out.loss.item():.4f}  Htok {ent.item():.4f} "
                  f"Huse {use_ent.item():.4f} (eff use {math.exp(use_ent.item()):.1f})  "
                  f"{step / (time.time() - t0):.2f} it/s", flush=True)
        if step >= a.steps:
            break

save_file({name: q.detach().to(torch.bfloat16).cpu() for name, q in gate_params}, a.out)
print(f"saved {len(gate_params)} tuned gates -> {a.out}")
