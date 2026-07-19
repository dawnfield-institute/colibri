'''GLM ROUTE_TRACE analysis: the day-one routing statistics for the real target.
Format per line: call pos layer eid:w eid:w ...  (glm.c g_route_fp)
Answers: rank-frequency shape (flatter or more skewed than OLMoE?), hot-set coverage
(the multi-GPU VRAM-tier prize on GLM), lag-1 routed-set overlap (prefetch fuel),
per-layer expert counts. Compare against OLMoE: geo ratio 0.966, top-26% -> 47.6%,
lag-1 0.427 vs 0.292 baseline.'''
import sys, math
import numpy as np
path = sys.argv[1] if len(sys.argv) > 1 else 'glm_trace_cold.txt'
recs = {}   # (call,pos,layer) -> set(eids)
cnt  = {}   # layer -> {eid: count}
for line in open(path):
    p = line.split()
    if len(p) < 4: continue
    try: call, pos, layer = int(p[0]), int(p[1]), int(p[2])
    except ValueError: continue
    ids = set()
    for tok in p[3:]:
        e = int(tok.split(':')[0]); ids.add(e)
        cnt.setdefault(layer, {}).setdefault(e, 0); cnt[layer][e] += 1
    recs[(call, pos, layer)] = ids
layers = sorted(cnt)
print(f'{len(recs)} (call,pos,layer) records | {len(layers)} layers | '
      f'experts seen/layer: min={min(len(cnt[L]) for L in layers)} max={max(len(cnt[L]) for L in layers)}')
print()
print('== rank-frequency + hot-set coverage (per layer avg) ==')
ratios, covs = [], {p: [] for p in (12.5, 25, 26.5, 50)}
for L in layers:
    c = np.sort(np.array(list(cnt[L].values()), dtype=float))[::-1]
    E = len(c)
    nz = c[c > 0]; r = np.arange(1, len(nz) + 1)
    if len(nz) > 4:
        g = np.polyfit(r, np.log(nz), 1); ratios.append(math.exp(g[0]))
    for pcts in covs:
        k = max(1, round(E * pcts / 100))
        covs[pcts].append(c[:k].sum() / max(c.sum(), 1))
print(f'mean geometric ratio: {np.mean(ratios):.4f}  (OLMoE was 0.9663; 1/phi=0.618)')
for pcts in covs:
    print(f'top {pcts:4.1f}% of seen experts -> {np.mean(covs[pcts])*100:5.1f}% of routes')
print()
print('== lag-1 routed-set overlap (decode positions, per layer) ==')
ov, base = [], []
bylayer = {}
for (call, pos, layer), ids in recs.items():
    bylayer.setdefault(layer, []).append((call, pos, ids))
rng = np.random.default_rng(0)
for L, lst in bylayer.items():
    lst.sort()
    seq = [ids for _, _, ids in lst]
    K = np.mean([len(s) for s in seq])
    for i in range(len(seq) - 1):
        ov.append(len(seq[i] & seq[i+1]) / max(len(seq[i]), 1))
    if len(seq) > 20:
        idx = rng.integers(0, len(seq), size=(200, 2))
        base.extend(len(seq[a] & seq[b]) / max(len(seq[a]), 1) for a, b in idx if abs(a - b) > 16)
print(f'lag-1 overlap: {np.mean(ov):.3f} | random-pair baseline: {np.mean(base):.3f}  (OLMoE: 0.427 vs 0.292)')
