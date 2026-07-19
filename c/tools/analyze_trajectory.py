#!/usr/bin/env python3
"""Trajectory + co-firing structure from existing ROUTE_TRACE csvs.

 1. Lag-k routed-set overlap: how far does routing memory extend?
    overlap(k) = |S_t ∩ S_{t+k}| / 8, averaged. Decay shape tells us whether a
    trailing-window pin captures a drifting working set (slow decay) or only
    adjacent-token reuse exists (fast decay to the flat-usage baseline).
 2. Trailing-window union coverage: fraction of token t's routed set already
    present in the union of the last W tokens' routed sets — the direct upper
    bound on what a "keep the trajectory" cache policy can hit (per layer, per W).
 3. Co-firing structure within (layer, token): pairwise expert co-occurrence;
    greedy clustering; report fraction of routed traffic that lands inside
    clusters — the disk-layout coalescing potential.
"""
import csv, glob, math
import numpy as np

def load(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(',')
            if len(p) < 3: continue
            tok, layer = int(p[0]), int(p[1])
            pairs = [(int(p[i]), float(p[i+1])) for i in range(3, len(p)-1, 2)]
            rows.append((tok, layer, frozenset(e for e, w in pairs)))
    return rows

traces = {p[len('trace_'):-len('.csv')]: load(p) for p in sorted(glob.glob('trace_*.csv'))}
n_layers = 1 + max(r[1] for t in traces.values() for r in t)
E, K = 64, 8

# random-baseline overlap for flat-ish usage: E[|A∩B|]/K with both ~usage dist.
# empirical baseline: overlap of routed sets from RANDOM token pairs (same layer).
rng = np.random.default_rng(0)

# ---------- 1: lag-k overlap ----------
print('== lag-k routed-set overlap (mean fraction of top-8 shared) ==')
lags = [1, 2, 4, 8, 16, 32, 64]
seqs = {}   # (dom, layer) -> ordered list of sets
for dom, t in traces.items():
    by = {}
    for tok, layer, s in t: by.setdefault(layer, []).append((tok, s))
    for layer, lst in by.items():
        lst.sort(); seqs[(dom, layer)] = [s for _, s in lst]
print('lag: ' + '  '.join(f'{k:>5}' for k in lags) + '   rand-pair baseline')
vals = {k: [] for k in lags}; base = []
for (dom, layer), lst in seqs.items():
    n = len(lst)
    for k in lags:
        if n > k + 10:
            vals[k].extend(len(lst[i] & lst[i+k]) / K for i in range(n - k))
    idx = rng.integers(0, n, size=(200, 2))
    base.extend(len(lst[a] & lst[b]) / K for a, b in idx if abs(a - b) > 64)
print('     ' + '  '.join(f'{np.mean(vals[k]):5.3f}' for k in lags) + f'   {np.mean(base):5.3f}')

# ---------- 2: trailing-window union coverage ----------
print('\n== trailing-window coverage: P(expert of token t in union of last W tokens) ==')
print('   (upper bound for a trajectory-keep policy; union size shown = RAM cost in experts)')
for W in [1, 2, 4, 8, 16, 32]:
    cov, usz = [], []
    for (dom, layer), lst in seqs.items():
        for i in range(W, len(lst)):
            u = set().union(*lst[i-W:i])
            cov.append(len(lst[i] & u) / K)
            usz.append(len(u))
    print(f'W={W:3d}: coverage={np.mean(cov):.3f}   mean union size={np.mean(usz):5.1f} experts/layer')

# ---------- 3: co-firing clusters ----------
print('\n== co-firing structure (pooled domains, per layer) ==')
print('layer  top-pair-lift  clustered-traffic@8x8')
for L in range(n_layers):
    cnt = np.zeros(E); co = np.zeros((E, E)); n_ev = 0
    for dom in traces:
        for s in seqs[(dom, L)]:
            ids = sorted(s); n_ev += 1
            for i in ids: cnt[i] += 1
            for a in range(len(ids)):
                for b in range(a+1, len(ids)):
                    co[ids[a], ids[b]] += 1; co[ids[b], ids[a]] += 1
    p = cnt / max(n_ev, 1)
    lift = np.zeros_like(co)
    nz = np.outer(p, p) > 0
    lift[nz] = (co[nz] / max(n_ev, 1)) / np.outer(p, p)[nz]
    np.fill_diagonal(lift, 0)
    # greedy: 8 clusters of 8 by co-count
    unassigned = set(range(E)); clusters = []
    while unassigned and len(clusters) < 8:
        seed = max(unassigned, key=lambda e: cnt[e])
        cl = {seed}; unassigned.discard(seed)
        while len(cl) < 8 and unassigned:
            best = max(unassigned, key=lambda e: sum(co[e, c] for c in cl))
            cl.add(best); unassigned.discard(best)
        clusters.append(cl)
    memb = {}
    for ci, cl in enumerate(clusters):
        for e in cl: memb[e] = ci
    inside = tot = 0
    for dom in traces:
        for s in seqs[(dom, L)]:
            ids = list(s)
            for a in range(len(ids)):
                for b in range(a+1, len(ids)):
                    tot += 1
                    if memb.get(ids[a], -1) == memb.get(ids[b], -2): inside += 1
    top_lift = np.sort(lift.flatten())[::-1][:20].mean()
    print(f'{L:5d}  {top_lift:12.2f}  {inside/max(tot,1):20.3f}')
print('\n(random assignment baseline for clustered-traffic@8x8 = 7/63 = 0.111)')
