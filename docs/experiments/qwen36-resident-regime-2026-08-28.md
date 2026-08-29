# Qwen3.6-35B-A3B in the resident regime — first measurements

node1: i9-12900K (8P+8E, avx-vnni, no AVX-512), 64 GB DDR4-3200, Samsung 990 Pro
2 TB ext4, Debian trixie. Engine: fork `sync/upstream-v1.9.0` (upstream `main`
@ 184e052, v1.9.0) plus the routing telemetry wired into `qwen36.c` in this
branch. Container `Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64` (22 GB, 46 files).
Every run under `systemd-run --scope -p MemoryMax=40G -p MemorySwapMax=0`.

Prompt A (used to build the history): "hello, briefly: what is a
mixture-of-experts model?". Prompt B (held-out): "Write a Python function that
merges two sorted lists, and explain its complexity." 48 new tokens each.

## First light

    2.22 tok/s · expert hit 78.5% · RSS 16.97 GB · TTFT 7.84 s · cap 256/layer

Coherent, correct output. For scale, GLM-5.2 on this same box measured 0.31–0.46
tok/s in July: **roughly 7×, on a model that is actually usable for coding.**
This is the regime the July phase diagram predicted and never got to measure —
the container fits in RAM, so a cache miss is a page-cache memcpy rather than a
disk read, and every conclusion that was true when misses cost 10 ms of NVMe has
to be re-derived.

## The telemetry works on the real model

Identical generated text and **identical cache accounting (hit=15321 miss=4199)**
with `ROUTE_TRACE` and `COLI_USAGE` set and unset; 2.20 vs 2.22 tok/s is run
noise. The trace covers all 40 MoE layers, 2440 rows of 8 pairs, and
`.coli_usage` writes a valid header (`-1 40 256`) carrying 19520 selections =
2440 × 8 exactly. Seeding it back pins from history: "[HOT] Pinned 1280 experts
(top-32/layer)", which is 40 × 32.

## Learned pinning is monotonically HARMFUL here

Upstream's first open hypothesis asks whether routing history places experts
better than plain LRU, and warns it can overfit a prompt. On this model, on this
workload, it does not merely fail to help.

cap 32/layer, single runs:

| pins | same prompt as history | held-out prompt |
|---|---|---|
| `HOT=0` (pure LRU) | **65.0% hit · 1.83 tok/s** | **64.0% · 1.74** |
| `HOT=8` | 64.0% · 1.83 | 63.7% · 1.75 |
| `HOT=16` | 63.3% · 1.74 | 63.0% · 1.71 |
| `HOT=32` (= cap) | 60.3% · 1.70 | — |

A clean dose-response: more pinning, monotonically worse, on both the prompt the
history came from and a held-out one. Repeated three times at the endpoints, with
non-overlapping ranges:

| | run 1 | run 2 | run 3 |
|---|---|---|---|
| `HOT=0` | 1.88 | 1.86 | 1.82 |
| `HOT=32` | 1.65 | 1.66 | 1.71 |

**~10% slower with pins.** This is the opposite sign to the GLM-5.2 result on the
same machine in July, where six rounds of `REPIN=1` converged in ~3 rounds and
took aggregate throughput from 0.404 to 0.461.

At cap 128 and 256 the difference vanishes (78.2% hit either way) because the
cache holds most of the routed set regardless, so there is nothing for placement
to decide.

### Two explanations tested and refuted

**"The whole cache is pinned, so LRU has no room to adapt."** Refuted by the
dose-response: `HOT=8` at cap 32 pins only a quarter of the cache and is already
(slightly) worse, and the degradation is smooth rather than a cliff at
`HOT`=cap.

**"Qwen3.6's routing is flat, so there is no hot set to pin."** Refuted by
measuring it — which this branch's `ROUTE_TRACE` wiring is what makes possible.
Per-layer, averaged over 40 layers and 19520 routes:

| | top-10% of experts | top-26% | top-50% |
|---|---|---|---|
| share of routes | **62.8%** | **90.0%** | 99.2% |

with 105 of 256 experts firing per layer on average. For comparison, the July
numbers: OLMoE 47.6% at top-26% (flat — aux-loss load balancing), GLM-5.2
69.5%. **Qwen3.6 is the most skewed of the three.** A hot set very much exists.

### What is left, stated as a hypothesis and not a result

The skew is the reason pinning cannot win, not a reason it should. cap 32 of 256
is 12.5% of the experts, and the top ~12% already catch ~65% of routes — which
is precisely the set an LRU converges on by itself. Pinning then freezes the set
LRU would have found anyway, and pays for it by giving up the ability to track
drift. Pinning should only help where LRU *cannot* find the hot set: when the
working set churns faster than LRU adapts, or when a miss is expensive enough
that a slightly better hit rate outweighs lost adaptivity. July's GLM box was the
second case — misses cost NVMe reads. Here they cost a memcpy.

That is a hypothesis with an obvious test (vary miss cost by forcing the
container out of page cache, and vary workload churn with a rotating prompt mix),
and it has not been run. What is measured above is the null and the skew; the
explanation is not yet earned.

## Caveats

One prompt of 14 tokens, 48 generated, per cell; the endpoint comparison is
n=3 and the intermediate pin counts are n=1. The workload is far smaller and far
less varied than anything upstream's hypothesis is really asking about, and the
skew figures come from a single 2440-row trace on prompt A. What these support is
that pinning does not help *here*, and that the usual explanations for such a
null do not apply. They do not support a general claim about `PIN=auto`.
