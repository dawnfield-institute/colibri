# The fine-tune track — routing-skew amplification (design, pre-registered)

**Author:** Peter Groom, Dawn Field Institute · 2026-07-19
**Status:** DESIGN — for review before any training run.

## Why this is the keystone

The optimization ledger's phase diagram puts four levers in their *inactive* regions
pending routing skew: mixed precision (cold tail must be genuinely cold), the VRAM hot
tier's catch rate, domain-affinity co-batching, and trained truncation (the +37–103%
speed that naive truncation couldn't afford). One training-time intervention — amplifying
the router's natural concentration — moves all four boundaries at once. GLM-5.2 already
routes skewed (top-26% of seen experts → 69.5% of routes); the task is *amplification of
existing structure*, not creation — a materially safer training objective.

## Objective function (the SEC-shaped part)

`L = L_lm + α·L_concentrate − β·L_diversity + γ·L_coherence`

- `L_concentrate`: mean per-token routing entropy over the top-M mass (drive collapse
  toward fewer effective experts/token).
- `L_diversity`: an anti-collapse floor — penalize any expert whose batch-level usage
  falls below ε (the classic MoE collapse guard; we keep it *weak* deliberately — we WANT
  skew, just not death).
- `L_coherence`: temporal term rewarding routed-set overlap between adjacent tokens
  (raises the measured 42.7%/1.9× lag-1 locality that prefetch feeds on).

The α/β tension is the collapse-vs-concentration balance — structurally the ∂S/∂t =
α∇I − β∇H trade the DFT program studies; the constants are tuned empirically, the FORM
is the transplant.

## Testbed and sequencing (OLMoE on the 3090 — everything fits)

**Router-only training first**: freeze all weights except `mlp.gate` matrices
(16 × 2048×64 ≈ 2M params — trains in minutes, cannot damage expert calibration, and the
renorm catastrophe taught us the residual stream's scale relationships are untouchable).

| # | experiment | success criterion (pre-registered) |
|---|---|---|
| E1 | Router-only, `L_concentrate` only, α sweep | +15pp hit rate at cap 16/64 for ≤2% ppl cost (428-token 5-domain eval, PPL=1 meter) |
| E2 | + `L_diversity` floor | no expert below 0.2× uniform share; E1's gains retained |
| E3 | + `L_coherence` | lag-1 overlap 0.43 → ≥0.55 at ≤1% additional ppl |
| E4 | Trained top-6 truncation (K=6 during training) | ≥ naive top-6 speed (+37%) at ≤5% ppl cost (naive cost: +16%) |
| E5 | Mixed precision on the E1-skewed model (hot-int8/cold-int2) | ppl ≤ 1.5× baseline (naive on flat usage: 19.6×) |

Cache-impact evaluation is **offline**: replay routing traces from the tuned model
through a cache simulator (analyze-tools already compute hit-at-cap from traces) — no
engine changes needed to measure the thing we care about.

## GLM-5.2 router-only feasibility (the interesting question)

GLM's routers are tiny relative to the model (per-layer `hidden × n_experts` + bias;
all 75 MoE layers ≈ low hundreds of MB) — they *fit* on the 3090 trivially. The blocker
is training signal: full-model forwards at 0.3 tok/s cannot generate gradient data.

**Proposed route — offline router distillation from serving traffic:**
1. Extend ROUTE_TRACE to also dump the router's *input activation* per (token, layer)
   (~5KB/layer/token; a day of serve traffic = a real dataset, collected as a byproduct).
2. Train replacement routers offline on the 3090: inputs = recorded activations, target =
   a *sharpened* version of the recorded routing distribution (temperature-scaled toward
   concentration, coherence term across recorded adjacent tokens).
3. Ship as a drop-in: routers are f32 in the container ("small and sensitive"); swap the
   `mlp.gate` tensors, keep everything else byte-identical. A/B with the loss meter +
   task battery before adoption.
This needs no training cluster, no model forwards during training, and is fully
reversible (keep the original router shards).

Risk register: distribution shift (routers trained on our traffic's activation
distribution — fine for a personal lane, flagged for generality); silent quality drift
(mandatory eval battery per the #194 lesson: `[gMASK]<sop>` prefix, multi-domain ppl,
n≥400 tokens); expert starvation on rare domains (keep original routers for fallback).

## What this is not

Not full fine-tuning, not quantization-aware training, not architecture change. Smallest
intervention, biggest boundary-mover, every step measured against pre-registered criteria
with the loss meter that already caught one catastrophic "improvement."
