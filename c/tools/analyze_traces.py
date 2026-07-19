#!/usr/bin/env python3
"""Analyze olmoe ROUTE_TRACE csvs: usage structure, entropy phases, predictability.

Questions (in order of what they inform):
 1. Rank-frequency of expert usage per layer: power-law vs geometric fit; if geometric,
    is the ratio near 1/phi = 0.618? (PAC hypothesis -> closed-form cache sizing)
 2. Working set: how many experts cover 50/90/99% of routed accesses (cache sizing).
 3. Routing entropy distribution + per-layer profile (SEC phase structure?).
 4. Does entropy PREDICT temporal reuse? corr(entropy at token t, routed-set overlap
    t -> t+1). If yes: cheap online signal to gate prefetch/pin promotion.
 5. Cross-domain expert overlap (Jaccard of top-20 per layer): are hot sets shared
    across workloads (global pin ok) or domain-specific (per-workload pinning pays)?
"""
import csv, glob, math, sys
import numpy as np

PHI_INV = 0.6180339887

def load(path):
    rows = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            p = line.strip().split(',')
            if len(p) < 3: continue
            tok, layer, H = int(p[0]), int(p[1]), float(p[2])
            pairs = [(int(p[i]), float(p[i+1])) for i in range(3, len(p)-1, 2)]
            rows.append((tok, layer, H, pairs))
    return rows

traces = {}
for path in sorted(glob.glob('trace_*.csv')):
    name = path[len('trace_'):-len('.csv')]
    traces[name] = load(path)
    print(f'loaded {name}: {len(traces[name])} (token,layer) records')

n_layers = 1 + max(r[1] for t in traces.values() for r in t)
E = 64

# ---------- 1+2: usage counts per layer (all domains pooled) ----------
counts = np.zeros((n_layers, E))
for t in traces.values():
    for tok, layer, H, pairs in t:
        for eid, w in pairs:
            counts[layer, eid] += 1

print('\n== rank-frequency fits per layer (pooled, all domains) ==')
print('layer  gini   50%   90%   99%   pow_slope  R2pow   geo_ratio  R2geo')
geo_ratios, pow_slopes = [], []
for L in range(n_layers):
    c = np.sort(counts[L])[::-1]
    tot = c.sum()
    if tot == 0: continue
    cum = np.cumsum(c) / tot
    n50 = int(np.searchsorted(cum, 0.5) + 1)
    n90 = int(np.searchsorted(cum, 0.9) + 1)
    n99 = int(np.searchsorted(cum, 0.99) + 1)
    sorted_c = np.sort(c); n = len(c)
    gini = (2*np.sum((np.arange(1,n+1))*sorted_c)/(n*sorted_c.sum()) - (n+1)/n)
    nz = c[c > 0]; r = np.arange(1, len(nz)+1)
    # power: log f = a - s*log r ; geometric: log f = a - b*r  (ratio = exp(-b))
    ps, pa = np.polyfit(np.log(r), np.log(nz), 1)[0], None
    pfit = np.polyfit(np.log(r), np.log(nz), 1)
    gfit = np.polyfit(r, np.log(nz), 1)
    def r2(x, y, fit):
        pred = np.polyval(fit, x); ss = np.sum((y-pred)**2); tt = np.sum((y-y.mean())**2)
        return 1 - ss/tt if tt > 0 else 0
    r2p = r2(np.log(r), np.log(nz), pfit)
    r2g = r2(r, np.log(nz), gfit)
    ratio = math.exp(gfit[0])
    geo_ratios.append(ratio); pow_slopes.append(pfit[0])
    print(f'{L:5d}  {gini:.3f}  {n50:4d}  {n90:4d}  {n99:4d}   {pfit[0]:8.3f}  {r2p:.3f}   {ratio:8.4f}  {r2g:.3f}')
print(f'\nmean geometric ratio: {np.mean(geo_ratios):.4f}  (1/phi = {PHI_INV:.4f})')
print(f'mean power slope:     {np.mean(pow_slopes):.3f}')

# ---------- 3: entropy distribution ----------
print('\n== routing entropy (nats; uniform max = ln(64) = 4.159) ==')
allH = np.array([r[2] for t in traces.values() for r in t])
print(f'global: mean={allH.mean():.3f} std={allH.std():.3f} p5={np.percentile(allH,5):.3f} p95={np.percentile(allH,95):.3f}')
perL = {L: [] for L in range(n_layers)}
for t in traces.values():
    for tok, layer, H, pairs in t: perL[layer].append(H)
prof = '  '.join(f'L{L}:{np.mean(perL[L]):.2f}' for L in range(n_layers))
print(f'per-layer mean: {prof}')
hist, edges = np.histogram(allH, bins=20)
print('histogram (20 bins):', ' '.join(str(h) for h in hist))

# ---------- 4: entropy -> next-token reuse ----------
print('\n== does entropy predict reuse? corr(H_t, |S_t ∩ S_t+1)|/K per layer ==')
cors = []
for name, t in traces.items():
    bylayer = {}
    for tok, layer, H, pairs in t:
        bylayer.setdefault(layer, []).append((tok, H, frozenset(e for e, w in pairs)))
    for layer, seq in bylayer.items():
        seq.sort()
        Hs, ov = [], []
        for i in range(len(seq)-1):
            if seq[i+1][0] != seq[i][0] + 1: continue
            Hs.append(seq[i][1])
            ov.append(len(seq[i][2] & seq[i+1][2]) / max(len(seq[i][2]), 1))
        if len(Hs) > 20:
            c = np.corrcoef(Hs, ov)[0, 1]
            cors.append(c)
cors = np.array(cors)
print(f'per-(domain,layer) Pearson r: mean={cors.mean():.3f} std={cors.std():.3f} '
      f'frac negative={np.mean(cors < 0):.2f} (negative = high entropy -> less reuse)')

# mean overlap overall = upper bound for what a prev-token prefetcher gets for free
ovs = []
for name, t in traces.items():
    bylayer = {}
    for tok, layer, H, pairs in t:
        bylayer.setdefault(layer, []).append((tok, frozenset(e for e, w in pairs)))
    for layer, seq in bylayer.items():
        seq.sort()
        for i in range(len(seq)-1):
            if seq[i+1][0] == seq[i][0] + 1:
                ovs.append(len(seq[i][1] & seq[i+1][1]) / 8.0)
print(f'mean consecutive-token routed-set overlap: {np.mean(ovs):.3f} (fraction of top-8 reused next token)')

# ---------- 5: cross-domain overlap ----------
print('\n== cross-domain top-20 expert overlap (Jaccard, averaged over layers) ==')
names = sorted(traces.keys())
top = {}
for name in names:
    cc = np.zeros((n_layers, E))
    for tok, layer, H, pairs in traces[name]:
        for eid, w in pairs: cc[layer, eid] += 1
    top[name] = [set(np.argsort(cc[L])[::-1][:20]) for L in range(n_layers)]
print('        ' + '  '.join(f'{n[:7]:>7}' for n in names))
for a in names:
    row = []
    for b in names:
        js = [len(top[a][L] & top[b][L]) / len(top[a][L] | top[b][L]) for L in range(n_layers)]
        row.append(f'{np.mean(js):7.3f}')
    print(f'{a[:7]:>7} ' + '  '.join(row))
