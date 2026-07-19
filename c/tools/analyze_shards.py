'''Experiment F + 5.4 hot-set viability, from existing ROUTE_TRACE csvs.
1. Hot-set coverage: fraction of routed accesses served by top-X% of experts
   (the multi-GPU VRAM-tier prize: does 26% of experts catch 70-80% of routes?)
2. Cross-shard simulation: partition each layer's 64 experts into N shards
   (usage-balanced greedy vs co-fire-clustered); per (token,layer) measure
   shards touched (sync cost) and load imbalance (slowest-card gating).'''
import csv, glob
import numpy as np
rows=[]
for p in sorted(glob.glob('trace_*.csv')):
    for line in open(p):
        q=line.strip().split(',')
        if not q or not q[0].isdigit(): continue
        rows.append((int(q[0]),int(q[1]),[int(q[i]) for i in range(3,len(q)-1,2)]))
n_layers=1+max(r[1] for r in rows); E=64
cnt=np.zeros((n_layers,E))
for _,L,ids in rows:
    for e in ids: cnt[L,e]+=1
print('== hot-set coverage (pooled 6 domains; per-layer avg) ==')
for pct in (12.5,25,26.5,37.5,50):
    k=max(1,round(E*pct/100)); covs=[]
    for L in range(n_layers):
        top=np.sort(cnt[L])[::-1]; covs.append(top[:k].sum()/max(top.sum(),1))
    print(f'top {pct:4.1f}% of experts ({k:2d}/64) -> {np.mean(covs)*100:5.1f}% of routed accesses')
print()
print('== cross-shard simulation (per layer partition of 64 experts) ==')
for N in (2,4):
    for strat in ('balanced','cofire'):
        shard=np.zeros((n_layers,E),dtype=int)
        for L in range(n_layers):
            if strat=='balanced':
                order=np.argsort(cnt[L])[::-1]
                loads=[0.0]*N; asg=[0]*E
                for e in order:
                    s=int(np.argmin(loads)); asg[e]=s; loads[s]+=cnt[L,e]
                shard[L]=asg
            else:
                co=np.zeros((E,E))
                for _,LL,ids in rows:
                    if LL!=L: continue
                    for a in range(len(ids)):
                        for b in range(a+1,len(ids)): co[ids[a],ids[b]]+=1; co[ids[b],ids[a]]+=1
                un=set(range(E)); asg=[0]*E; per=E//N
                for s in range(N):
                    seed=max(un,key=lambda e:cnt[L,e]); cl={seed}; un.discard(seed)
                    while len(cl)<per and un:
                        b=max(un,key=lambda e:sum(co[e,c] for c in cl)); cl.add(b); un.discard(b)
                    for e in cl: asg[e]=s
                shard[L]=asg
        touched=[]; imb=[]
        for _,L,ids in rows:
            ss=[shard[L,e] for e in ids]
            touched.append(len(set(ss)))
            per=np.bincount(ss,minlength=N)
            imb.append(per.max()/max(per.mean(),1e-9))
        print(f'N={N} {strat:9s}: shards-touched/token = {np.mean(touched):.2f}/{N} | load imbalance (max/mean per token) = {np.mean(imb):.2f}x')
