#!/usr/bin/env python3
"""Routing concentration from a ROUTE_TRACE, corrected for how short the trace is.

The obvious way to state routing skew -- "the top 26% of experts catch X% of the
routes" -- is not comparable across models, and is not even comparable across two
runs of the same model, because it is biased upward by thin sampling. With fewer
observed routes than experts, most experts have simply not been seen yet, so the
ones that have been seen hold all the mass. A perfectly uniform router looks
extremely concentrated under a short trace.

The bias is not small. Measured on a 2440-row Qwen3.6 trace (40 layers, 256
experts, top-8 -- 1.9 routes per expert per layer):

    measured top-26% share   90.0%
    uniform router, same shape of sample   50.1%

So most of a headline "90%" was the trace length. Comparing that against a number
taken from a model with 64 experts and 30 routes per expert per layer measures
sampling density, not routing.

What this prints instead:

  - sample density (routes per expert per layer), the axis any two numbers must
    be matched on before they can be compared;
  - the measured concentration;
  - the same statistic for a uniform router drawn at the SAME density, which is
    the null, not zero;
  - the excess of one over the other, which is the part that is about routing;
  - and the above over prefixes of the trace, so it is visible whether the number
    has converged or is still climbing -- if it is still climbing, the trace is
    too short to quote at all.

Usage:
    python3 analyze_route_skew.py TRACE N_EXPERTS [--frac 0.26] [--label NAME]
"""
import argparse
import collections
import random
import statistics


def load(path):
    rows = []
    for line in open(path):
        f = line.split()
        if len(f) < 4:
            continue
        rows.append((int(f[2]), [int(t.split(":")[0]) for t in f[3:]]))
    if not rows:
        raise SystemExit(f"{path}: no trace rows "
                         "(expected '<call> <row> <layer> <id>:<gate> ...')")
    return rows


def concentration(sample, n_experts, frac):
    """Mean over layers of the share of routes held by that layer's top frac."""
    per = collections.defaultdict(collections.Counter)
    for layer, ids in sample:
        for e in ids:
            if e >= 0:
                per[layer][e] += 1
    shares = []
    for c in per.values():
        k = max(1, int(n_experts * frac))
        shares.append(sum(v for _, v in c.most_common(k)) / sum(c.values()))
    return statistics.mean(shares) if shares else 0.0


def uniform_null(n_rows, layers, n_experts, topk, frac, trials=3):
    """Same number of rows over the same layers, routed uniformly at random."""
    vals = []
    for seed in range(trials):
        rnd = random.Random(seed)
        fake = []
        per_layer = max(1, n_rows // len(layers))
        for layer in layers:
            for _ in range(per_layer):
                fake.append((layer, rnd.sample(range(n_experts), topk)))
        vals.append(concentration(fake, n_experts, frac))
    return statistics.mean(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("n_experts", type=int)
    ap.add_argument("--frac", type=float, default=0.26,
                    help="head fraction of experts to measure (default 0.26)")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    rows = load(args.trace)
    topk = len(rows[0][1])
    layers = sorted({l for l, _ in rows})
    label = args.label or args.trace

    print(f"\n{label}: {len(rows)} rows · {len(layers)} layers · "
          f"{args.n_experts} experts · top-{topk} · head={args.frac:.0%}")
    print(f"{'rows':>8} {'routes/exp/layer':>17} {'measured':>10} "
          f"{'uniform null':>13} {'excess':>9}")
    for frac in (0.05, 0.1, 0.25, 0.5, 1.0):
        n = max(len(layers), int(len(rows) * frac))
        sample = rows[:n]
        density = (n * topk) / (len(layers) * args.n_experts)
        m = concentration(sample, args.n_experts, args.frac)
        u = uniform_null(n, layers, args.n_experts, topk, args.frac)
        print(f"{n:>8} {density:>17.2f} {m*100:>9.1f}% {u*100:>12.1f}% "
              f"{(m-u)*100:>+8.1f}pp")

    density = (len(rows) * topk) / (len(layers) * args.n_experts)
    if density < 10:
        print(f"\n  density {density:.2f} routes/expert/layer is thin. Quote the "
              f"excess, not the measured\n  share, and only compare it against "
              f"another trace at a similar density.")


if __name__ == "__main__":
    main()
