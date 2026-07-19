#!/usr/bin/env python3
"""Swap tuned router gates into a colibri OLMoE snapshot (E1 eval side).

Copies the snapshot shard-by-shard, replacing only model.layers.*.mlp.gate.weight
tensors with the tuned versions; every other byte is preserved. Reversible by
construction (the source snapshot is untouched).

usage: patch_gates.py --src /scratch/olmoe_i8 --gates gates_a05.safetensors --out /scratch/olmoe_a05
"""
import argparse, shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--gates", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

src, out = Path(a.src), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
gates = load_file(a.gates)
print(f"{len(gates)} tuned gate tensors loaded")

replaced = 0
for f in sorted(src.iterdir()):
    if f.suffix != ".safetensors":
        shutil.copy2(f, out / f.name)
        continue
    t = load_file(str(f))
    hit = [k for k in t if k in gates]
    for k in hit:
        t[k] = gates[k].to(t[k].dtype)
    replaced += len(hit)
    save_file(t, str(out / f.name))
    print(f"{f.name}: {'replaced ' + str(len(hit)) if hit else 'copied'}")
print(f"total gates replaced: {replaced} -> {out}")
