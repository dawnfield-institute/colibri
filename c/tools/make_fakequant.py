'''Fake-quant transform on the int8 OLMoE snapshot: dequant experts to f32,
re-quantize to 2^(bits-1)-1 levels row-wise, store back as int8+new scales.
Quality effect of int-b expert weights, zero engine changes (bytes unchanged —
this measures the QUALITY axis only, not the speed axis).
MIXED mode: per-layer hot set (from routing-trace usage) keeps 8-bit, cold get cold_bits.
usage: python make_fakequant.py <src> <dst> <bits>
       python make_fakequant.py <src> <dst> mixed <hot_n> <cold_bits>'''
import sys, glob, shutil, csv
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
mode = sys.argv[3]
dst.mkdir(parents=True, exist_ok=True)
shutil.copy2(src/'config.json', dst/'config.json')

hot = {}
if mode == 'mixed':
    hot_n, cold_bits = int(sys.argv[4]), int(sys.argv[5])
    counts = {}
    for f in glob.glob('/bulk-pool/scratch/colibri/c/trace_*.csv'):
        for line in open(f):
            p = line.strip().split(',')
            if not p[0].isdigit(): continue
            L = int(p[1])
            for i in range(3, len(p)-1, 2):
                counts.setdefault(L, {}).setdefault(int(p[i]), 0)
                counts[L][int(p[i])] += 1
    for L, cc in counts.items():
        hot[L] = set(sorted(cc, key=cc.get, reverse=True)[:hot_n])
    print(f'mixed: hot {hot_n}/64 experts per layer stay 8-bit, cold -> int{cold_bits}')

def requant(q, scale, bits):
    w = q.float() * scale.unsqueeze(1)                      # dequant
    qmax = 2**(bits-1) - 1
    s = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / qmax
    nq = (w / s).round().clamp(-qmax-1, qmax).to(torch.int8) # coarse levels in int8 container
    return nq, s.squeeze(1)

n_req = 0
for shard in sorted(src.glob('*.safetensors')):
    t = load_file(str(shard)); out = {}
    for name, ten in t.items():
        if name.endswith('.qs') or ten.dtype != torch.int8: out[name] = ten; continue
        # expert int8 weight: model.layers.L.mlp.experts.E.xxx.weight
        parts = name.split('.')
        L, E = int(parts[2]), int(parts[5])
        bits = 8 if (mode=='mixed' and E in hot.get(L,set())) else (cold_bits if mode=='mixed' else int(mode))
        if bits >= 8: out[name] = ten; continue
        nq, ns = requant(ten, t[name+'.qs'], bits)
        out[name] = nq; out[name+'.qs'] = ns; n_req += 1
    for name in list(out):
        if name.endswith('.qs') and name[:-3] in out and out[name[:-3]].dtype==torch.int8: pass
    save_file(out, str(dst/shard.name))
    print(f'{shard.name}: ok')
print(f'requantized {n_req} expert tensors -> {dst}')
