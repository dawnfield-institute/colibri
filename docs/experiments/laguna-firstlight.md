# Laguna-S-2.1 first light on node1 (2026-07-19)

**Runtime:** poolside llama.cpp fork, `laguna` branch (mainline: 0/145 arch support;
upstream PR #25165 pending). Built after fixing their gcc-14 bug — `std::isfinite`
without `#include <cmath>` in `common/speculative.cpp` (one-line PR back to poolside).

## As-shipped architecture (from the Q4_K_M GGUF header, authoritative)

| field | value | note |
|---|---|---|
| experts | **256 routed + 1 shared**, top-**10** | very fine-grained (GLM: far fewer, bigger) |
| expert FFN | 1024 (on 3072 embedding) | ~9.4M params/expert → ~5MB int4 grain (GLM ~19MB) |
| gating func | 2 = **softplus/sigmoid** | aux-loss-free family → expect SKEW (cache-friendly) |
| weights norm / scale | True / 2.5 | |
| blocks | 48, **1 leading dense** | block 0 dense, 1–47 MoE |
| attention | 48 heads / **8 KV** (GQA 6:1), dim 128 | |
| sliding window | **512** (SWA on 36 layers, 12 global) | bounds long-context KV — big for coding |
| context (GGUF) | **262144** (yarn ×32 of 8192; card's 1M is the ceiling) | default KV alloc is huge → OOM if uncapped |
| embedding | 3072 | small hidden → fast matmul |
| vocab | 100352 | |
| active params | 8B (card) | vs GLM 40B → the 10× decode speed |

## Measured (CPU, 20 threads, Q4_K_M = 70GiB)

- **llama-bench: pp512 27.86 t/s · tg128 3.21 t/s** — 10× GLM-5.2's 0.32.
- Interactive gen held **3.1 t/s** even under swap thrash — decode rate is robust; the
  thrash cost is load + host responsiveness, not throughput.
- Output: coherent, on-task, **native reasoning** (`[Start thinking]` block, correct
  problem decomposition before coding).

## The fit problem — precise, and it's the model, not the context

The 70GiB model alone exceeds node1's 62GB RAM by ~8GiB, so on 64GB it swaps regardless
of context (capping `-c` fixes the KV blowup — ~13GB at 256K, mostly the 12 global layers
since SWA bounds the other 36 — but not the base 70GiB). Consequences:
- **On 64GB today: unusable interactively** (swap death — host-wide, `ls` times out).
  Bench survives via minimal-footprint mmap. Must cap `-c` AND accept it's swap-limited.
- **On 128GB (ordered): fully resident** — 70GiB + ~13GB KV + working set fits with room;
  decode climbs toward the CPU-compute ceiling.
- **Stopgap before RAM:** a smaller GGUF quant (Q3/IQ2 ≈ 45–55GB) would fit and run
  fully-resident/fast today — poolside ships only Q4_K_M + Q8_0, but `llama-quantize`
  from F16 could make one. Trade quality for fit-now.

## Strategic read: Laguna is a BETTER substrate for our program than GLM

256 fine-grained experts + top-10 + sigmoid gating is the ideal shape for the whole
colibrì optimization stack: finer streaming grains, higher cache-cap resolution, more
routing-skew headroom for the VRAM hot-tier, and 256 router knobs/layer (vs 64) for the
fine-tune keystone. If the routing is skewed like GLM's, every lever in the phase diagram
applies here with *better* numbers — and this is the model we actually want (agentic
coding), not a generalist testbed.
