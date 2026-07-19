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

---

## E1 log (running record)

**Round 1 (2026-07-19): FAIL, informative.** Per-token entropy objective, α∈{0.02,0.05,0.2},
300 steps, wikitext-2. All three worsened both axes (ppl 6.77–7.20 vs bar ≤6.51; hit
51.8–52.6% vs bar ≥68.9%; baseline 6.38/53.9%). Two causes identified: (1) domain shift —
gates adapted to wikitext (lm 2.70→2.22) and moved away from the eval distribution;
(2) **objective flaw — per-token entropy is the wrong concentration.** Sharpening each
token's mix doesn't make tokens *share* experts; aggregate usage stays flat and the cache
gains nothing. The cache-relevant quantity is the entropy of the **batch-mean routing
distribution** — exactly what the standard MoE load-balancing aux loss maximizes. The
corrected objective is its sign-flip: **the anti-load-balancing loss.**

**Round 2 (launched): usage-entropy objective** (α∈{0.5, 2.0}) + **α=0 control** to price
the pure domain-shift cost. Trainer logs both entropies (Htok, Huse) to keep the
distinction measurable.

**Round 2 (2026-07-19): MECHANISM PROVEN, window not yet found.** Usage objective at
α=2.0: batch-usage entropy 4.16→3.47 and **hit@16 53.9%→64.5% (+10.6pp)** — the first
direct demonstration that streaming-cache hit rate is trainable. Cost at that dose: ppl
6.38→41.7 (unacceptable). α=0.5 = dead zone (ppl 9.29, no concentration). The α=0
control prices pure wikitext domain-shift at +0.66 ppl — a recoverable corpus artifact,
not an objective cost. Round 3: fineweb corpus (eval-adjacent distribution) ×
{α=1.0/lr 5e-4, α=2.0/lr 2e-4} — dose retained, gradient gentler, corpus fixed.
