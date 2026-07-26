#!/usr/bin/env python3
"""Phase-0 loadable-MoE test: separability (H1) + loadability (H2) on OLMoE traces.

Pre-registered in docs/experiments/loadable-moe-preregistration.md (committed first).
Reads trace_<domain>.csv (schema: tok,layer,entropy,eid,w,eid,w,...), OLMoE 16L x 64E top-8.

usage: analyze_loadability.py <dir-with-trace_*.csv>
"""
import csv, glob, os, sys
import numpy as np

DIR = sys.argv[1] if len(sys.argv) > 1 else "."
STEM = {"code", "math", "physics"}          # pre-declared cluster (see pre-registration)
UNIFORM_MULT = 0.5                            # "fires" = usage share > 0.5x uniform
COVER = 0.90

def load(path):
    """-> dict[layer] -> list of routed-expert-id lists (one per token)."""
    rows = {}
    with open(path) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 4: continue
            layer = int(p[1])
            eids = [int(p[i]) for i in range(3, len(p) - 1, 2)]
            rows.setdefault(layer, []).append(eids)
    return rows

traces = {os.path.basename(p)[len("trace_"):-len(".csv")]: load(p)
          for p in sorted(glob.glob(os.path.join(DIR, "trace_*.csv")))}
domains = sorted(traces)
LAYERS = sorted({L for t in traces.values() for L in t})
E = 1 + max(e for t in traces.values() for toks in t.values() for eids in toks for e in eids)
print(f"domains: {domains}")
print(f"layers: {len(LAYERS)}  experts: {E}\n")

def usage_matrix(trace):
    """domain -> [L, E] normalized usage (per layer sums to 1)."""
    M = np.zeros((len(LAYERS), E))
    for li, L in enumerate(LAYERS):
        for eids in trace.get(L, []):
            for e in eids: M[li, e] += 1
    s = M.sum(1, keepdims=True); s[s == 0] = 1
    return M / s

U = {d: usage_matrix(traces[d]) for d in domains}

# ---------- H1: separability ----------
def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0

def domain_sim(ua, ub):
    return float(np.mean([cos(ua[li], ub[li]) for li in range(len(LAYERS))]))

print("== H1: domain-domain similarity (mean cosine over layers) ==")
S = np.zeros((len(domains), len(domains)))
for i, a in enumerate(domains):
    for j, b in enumerate(domains):
        S[i, j] = domain_sim(U[a], U[b])
print("        " + "  ".join(f"{d[:6]:>6}" for d in domains))
for i, a in enumerate(domains):
    print(f"{a[:7]:>7} " + "  ".join(f"{S[i,j]:6.3f}" for j in range(len(domains))))

def sep_index(sim, doms):
    stem = [i for i, d in enumerate(doms) if d in STEM]
    within = [sim[i, j] for a in range(len(stem)) for b in range(a+1, len(stem))
              for i, j in [(stem[a], stem[b])]]
    cross = [sim[i, j] for i in range(len(doms)) for j in range(i+1, len(doms))
             if (doms[i] in STEM) != (doms[j] in STEM)]
    return np.mean(within) - np.mean(cross), np.mean(within), np.mean(cross)

s_obs, w_obs, c_obs = sep_index(S, domains)
print(f"\nseparability index S = within-STEM {w_obs:.3f} - cross {c_obs:.3f} = {s_obs:.3f}")

# permutation null: repartition all (layer, token routed-set) into random pseudo-domains
rng = np.random.default_rng(0)
pool = []  # (layer, eids, domain_idx) — but for the null we ignore true domain
for di, d in enumerate(domains):
    for L in LAYERS:
        for eids in traces[d].get(L, []):
            pool.append((LAYERS.index(L), eids))
sizes = [sum(len(traces[d].get(L, [])) for L in LAYERS) for d in domains]
def perm_S():
    idx = rng.permutation(len(pool))
    fake = [np.zeros((len(LAYERS), E)) for _ in domains]
    pos = 0
    for di, n in enumerate(sizes):
        for k in idx[pos:pos+n]:
            li, eids = pool[k]
            for e in eids: fake[di][li, e] += 1
        pos += n
    for m in fake:
        ss = m.sum(1, keepdims=True); ss[ss == 0] = 1; m /= ss
    sim = np.zeros((len(domains), len(domains)))
    for i in range(len(domains)):
        for j in range(len(domains)):
            sim[i, j] = float(np.mean([cos(fake[i][li], fake[j][li]) for li in range(len(LAYERS))]))
    return sep_index(sim, domains)[0]
NPERM = 1000
null = np.array([perm_S() for _ in range(NPERM)])
p = float((null >= s_obs).mean())
print(f"permutation null (N={NPERM}): mean {null.mean():.3f} sd {null.std():.3f} | "
      f"p(S_perm >= S_obs) = {p:.4f}")

# agglomerative-ish independent check: nearest-neighbor of each domain
print("\nnearest neighbour of each domain (1 - sim):")
for i, d in enumerate(domains):
    order = sorted([(1 - S[i, j], domains[j]) for j in range(len(domains)) if j != i])
    print(f"  {d:>8} -> {order[0][1]} ({order[0][0]:.3f}), {order[1][1]} ({order[1][0]:.3f})")

# ---------- H2: loadability decomposition ----------
print("\n== H2: expert breadth (fires = usage share > %.2fx uniform) ==" % UNIFORM_MULT)
thr = UNIFORM_MULT / E
# breadth[layer, expert] = # domains where usage share > thr
fires = {d: (U[d] > thr) for d in domains}          # [L, E] bool
breadth = np.sum([fires[d] for d in domains], axis=0)  # [L, E] in 0..6
universal = breadth == len(domains)
tail = breadth == 0
# cluster-specific: fires only within STEM, or only within non-STEM
stem_fire = np.sum([fires[d] for d in domains if d in STEM], axis=0)
conv_fire = np.sum([fires[d] for d in domains if d not in STEM], axis=0)
stem_only = (stem_fire > 0) & (conv_fire == 0) & (~universal)
conv_only = (conv_fire > 0) & (stem_fire == 0) & (~universal)
mixed = (~universal) & (~tail) & (~stem_only) & (~conv_only)

def frac_experts(mask): return mask.sum() / mask.size
print(f"  universal (all 6 domains): {frac_experts(universal)*100:5.1f}% of (layer,expert) slots")
print(f"  STEM-only:                 {frac_experts(stem_only)*100:5.1f}%")
print(f"  CONV-only:                 {frac_experts(conv_only)*100:5.1f}%")
print(f"  mixed (cross-cluster, not universal): {frac_experts(mixed)*100:5.1f}%")
print(f"  tail (fires nowhere > thr): {frac_experts(tail)*100:5.1f}%")

# per-task load set = universal OR fires-for-its-cluster; coverage of that task's traffic
print("\n== H2: per-task loadability (load = universal + own-cluster experts) ==")
print("task       load-set size   coverage of task routing   'don't-load' fraction")
tot_route = {d: U[d].sum() for d in domains}   # = len(LAYERS) after norm; use raw counts instead
# use raw routed counts for coverage
raw = {}
for d in domains:
    M = np.zeros((len(LAYERS), E))
    for li, L in enumerate(LAYERS):
        for eids in traces[d].get(L, []):
            for e in eids: M[li, e] += 1
    raw[d] = M
for d in domains:
    own_cluster = stem_only if d in STEM else conv_only
    load = universal | own_cluster                      # [L,E] bool
    covered = raw[d][load].sum()
    total = raw[d].sum()
    cov = covered / total if total else 0
    size = load.sum() / load.size
    print(f"{d:>8}   {size*100:8.1f}%      {cov*100:8.1f}%              {(1-size)*100:6.1f}%")

# per-layer universal fraction (early vs late specialization profile)
print("\n== H2: per-layer universal-expert fraction (specialization profile) ==")
ul = universal.mean(1)
print("  " + " ".join(f"L{li}:{ul[li]*100:.0f}%" for li in range(len(LAYERS))))
