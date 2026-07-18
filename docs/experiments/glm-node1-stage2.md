# GLM-5.2 on a 12900K / 64GB / 990 Pro — full Stage-2 characterization

**Author:** Peter Groom, Dawn Field Institute · 2026-07-18
**Box:** i9-12900K (8P+8E, AVX2+AVX-VNNI, no AVX-512) · 64GB DDR4-3200 · Samsung 990 Pro 2TB
(ext4, dedicated) · RTX 3090 24GB · Proxmox host. Model: `mateogrgic` int4 + int8 MTP heads.

## Community-table row

| disk (iobench 19MB×64×8T) | config | measured |
|---|---|---|
| 6.44 GB/s O_DIRECT · 5.78 buffered cold | cap 21–26/layer auto · **MTP=0** · PILOT off | **0.34–0.36 tok/s** · hit 53–55% · RSS 45.6GB |
| 〃 | default (MTP on) | 0.31–0.33 · hit 40–42% · MTP acceptance 43–44% · RSS 47.9GB |

Profile (decode, default config): 48% expert-disk wait / 34% expert-matmul / 12% attention.
~9.9GB disk read per token at 40% hit. NVMe utilization during single-stream decode:
**24–28%** (bursts ~2.5GB/s) — large headroom for concurrent streams.

## Finding 1: MTP speculation is a wash below residency

MTP-on loads **1,062 experts/token** (draft + absorb + verify batch-union) vs **600**
with MTP=0 — +77% expert traffic to save 2.29× forwards. On a disk-bound box these cancel:
0.34 tok/s without MTP ≥ 0.31–0.33 with, and the leaner working set caches better
(55% vs 40% hits). **Speculation trades bandwidth for latency; when bandwidth is the
constraint, it buys nothing.** The "~2× MTP lever" is real only once the hot set is
resident (RAM/VRAM). Suggested guidance: default MTP=0 when the resource plan reports
"limit: disk expert misses".

## Finding 2: the CUDA expert tier adds +0–6% here — and the mechanism is Amdahl, not ISA

The community null (#101: "AVX-512 CPU matches the 5090") replicates on this AVX2-only
CPU, so CPU ISA isn't the mechanism. The GPU demonstrably accelerates expert-matmul
(0.67s/token vs ~1.0+), but that's the ~33% slice of a ~50%-disk workload — wall-clock
barely moves and the box becomes more disk-bound. GPU+MTP=0: 0.35–0.36 tok/s (best
config). First GPU run pays a one-time VRAM-population + cold-pin tax (0.25 tok/s).

## Finding 3: GLM routing is strongly skewed (aux-loss-free training preserves skew)

From ROUTE_TRACE (86 positions, 1 prompt — short-trace caveat): top 26% of seen experts
catch **69.5%** of routes; lag-1 routed-set overlap **0.325 vs 0.172** random baseline
(1.9×). Contrast OLMoE (aux-loss-balanced training): 47.6% and 1.46× on the same
methodology — **load-balancing losses flatten routing; aux-free (noaux_tc) preserves the
natural skew that streaming caches need.** Cross-model, same engine, same analysis.

## Operating gotchas (cost us hours)

- `coli run` overrides `NGEN` from its resource plan — generation is EOS-bound.
- `timeout` around `coli` orphans the `glm` child (open pipe hangs the pipeline).
- glibc ≥2.41 breaks CUDA 12.9 builds (sinpi/cospi exception specs); CUDA 13.3 headers fix
  it — see `docs/BUILD-cuda-glibc241.md`.

Tools used for all analyses: `c/tools/analyze_glm_trace.py` (routing stats),
`c/tools/make_fakequant.py` (quant-depth curves), `c/tools/analyze_shards.py`
(multi-GPU partition simulation), plus the OLMoE testbed instrumentation on this branch.
