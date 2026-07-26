# Pre-registration: is the "physics-loadable MoE" thesis worth chasing? (Phase 0, OLMoE)

**Author:** Peter Groom, Dawn Field Institute · 2026-07-26
**Committed BEFORE running any analysis** (register-invariants discipline).

## What we're testing

The "modular power supply" thesis: one coherently-trained MoE, loaded sparsely per task —
a universal substrate always resident (the chassis), task-specific experts paged in per job
(the modular cables). Phase 0 tests the three **engineering preconditions** on OLMoE, using
the 6 existing single-prompt-per-domain routing traces (prose/code/physics/dialog/factual/
math; 16 layers × 64 experts, top-8). Test 0.3 adds a small fresh multi-prompt OLMoE run.

**Substrate caveat (pinned):** OLMoE is aux-loss-*balanced* — training flattens usage
(measured: pooled rank-frequency ratio 0.966 vs φ's 0.618). So this is the **pessimistic**
case for separability, and the DFT φ-signature is **expected NULL here** and is NOT tested
in Phase 0 (reserved for aux-free Laguna, Phase 1). A positive Phase-0 result is therefore a
*lower bound*; a negative one moves the work to the fine-tune, it does not kill the thesis.

## Hypotheses, metrics, thresholds, nulls

### H1 — Separability (Test 0.1)
Tasks route to distinguishable expert sets.
- **Metric:** per-domain, per-layer normalized expert-usage vectors; pairwise cosine
  similarity averaged over layers → 6×6 matrix. **Separability index** S = mean(within the
  STEM triad code/math/physics) − mean(cross-cluster pairs).
- **Null:** token-level label permutation — repartition all tokens into 6 random pseudo-
  domains of the same sizes, recompute S; 1000 shuffles → p = P(S_perm ≥ S_obs).
- **Success (pre-registered):** S_obs > 0 with p < 0.01, AND the STEM triad is the tightest
  cluster in agglomerative clustering of the 6×6 matrix.
- **Falsifier:** S_obs ≈ 0 or p ≥ 0.05 → domains are not separable even given the labels.

### H2 — Loadability (Test 0.2)
A task is covered by [universal ∪ its-cluster] experts, not the whole set.
- **Metric:** classify each (layer, expert) by breadth = # domains where its usage share
  exceeds 0.5× uniform (0.5/64). **universal** = breadth 6; **cluster-specific** = breadth
  1–3 confined to one cluster; **tail** = low everywhere. Report % of experts and % of
  routed traffic per class. Then per task: coverage of that task's routing by
  [universal ∪ its-cluster] experts, and that set's size as a fraction of all experts.
- **Success (pre-registered):** for the coding task, [universal ∪ STEM-cluster] covers
  ≥ 90% of coding routing while being ≤ 75% of experts (i.e., ≥ 25% of experts are safely
  *not loaded* for coding). Universal-class traffic share is materially < 100%.
- **Falsifier:** every task needs ~all experts (load-set ≈ 100% for ≥90% coverage, or
  universal traffic ≈ 100%) → no "don't-load-the-rest" headroom.

### H3 — Predictability (Test 0.3, needs the fresh multi-prompt corpus)
A task's expert set is predictable from task *type* (enables preload, not just reactive
streaming).
- **Metric:** build a domain hot-set (experts covering 90% of routing) from prompts 1..k of
  that domain; measure its coverage of a **held-out same-domain** prompt k+1 vs its coverage
  of **cross-domain** prompts. Report the gap.
- **Success (pre-registered):** same-domain held-out coverage − mean cross-domain coverage
  ≥ 15 percentage points (held-out same-domain ≥ 80%).
- **Falsifier:** gap < 5 pp → not predictable from type; must stream reactively, no preload win.

## Go / no-go decision rule (pinned)

- **GO** (chase Phase 1 on aux-free Laguna): H1 success AND H2 success AND H3 success.
- **PARTIAL** (chase, but the fine-tune is load-bearing): H1 success, H2 or H3 marginal.
- **INFORMATIVE NO-GO** (structure must be *created* by the SEC fine-tune, elevates Phase 2):
  H1 falsified on OLMoE — expected-possible given the flat-training substrate; not a kill.

## Confounds acknowledged in advance

Small n (6 domains, 1 prompt each in 0.1/0.2) → permutation nulls, directional not
definitive; 0.3 adds prompts. Layer heterogeneity → all metrics reported per-layer, not only
pooled. Cluster definition (STEM = code/math/physics) is pre-declared here to avoid
post-hoc cluster fitting; agglomerative clustering is reported as an independent check.
