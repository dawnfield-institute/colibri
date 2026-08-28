#!/usr/bin/env python3
"""Repack a pre-merge OLMoE int8 container into the merged_weight layout.

convert_olmoe.py used to emit one tensor per projection:

    model.layers.L.mlp.experts.E.gate_proj.weight      I8  [inter, hidden]
    model.layers.L.mlp.experts.E.gate_proj.weight.qs   F32 [inter]
    ... same for up_proj and down_proj

olmoe.c now loads a single slab per expert (load_expert_merged), because one
pread per expert beats three:

    model.layers.L.mlp.experts.E.merged_weight   I8  [2*inter*hidden + hidden*inter]
    model.layers.L.mlp.experts.E.qs              F32 [inter + inter + hidden]

and it refuses anything whose size does not match exactly -- deliberately, since
a short read there is a heap overflow, not a bad answer. So an old container does
not load at all; it fails at the first expert with "expert weight is -1 bytes".

The two layouts hold the SAME BYTES in the same order. gate, up, down
concatenated is what the merged slab is, and the scale arrays concatenate the
same way. So this is a repack, not a conversion: no dequantization, no
requantization, no model download.

That matters for more than convenience. Re-running convert_olmoe_merged.py on the
original checkpoint produces a container quantized by today's code, which is not
necessarily bit-identical to one quantized months ago -- and any measurement taken
against the old container stops being comparable. Repacking keeps the weights
that were actually measured.

Reads and writes safetensors directly (8-byte little-endian header length, JSON
header, data blob; offsets relative to the blob). Standard library only.

Usage:
    python3 repack_olmoe_merged.py --in OLD_DIR --out NEW_DIR [--shard-gb 2.5]

Verify afterwards with --check, which re-reads the output and compares every
merged slab against the three source tensors it came from.
"""
import argparse
import json
import os
import shutil
import struct
import sys

HDR_LEN = 8


def read_header(path):
    """(header dict, byte offset where the data blob starts)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(HDR_LEN))[0]
        hdr = json.loads(f.read(n))
    return hdr, HDR_LEN + n


def build_index(src_dir):
    """name -> (path, absolute start, absolute end, dtype, shape), across all shards."""
    index = {}
    shards = sorted(p for p in os.listdir(src_dir) if p.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors files in {src_dir}")
    for shard in shards:
        path = os.path.join(src_dir, shard)
        hdr, base = read_header(path)
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            index[name] = (path, base + start, base + end, meta["dtype"], meta["shape"])
    return index, shards


def read_tensor(index, name):
    path, start, end, _, _ = index[name]
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(end - start)


def expert_keys(index):
    """{(layer, expert)} for every expert with all six source tensors present."""
    found = {}
    for name in index:
        parts = name.split(".")
        # model.layers.<L>.mlp.experts.<E>.<proj>.weight[.qs]
        if len(parts) < 7 or parts[0] != "model" or parts[1] != "layers":
            continue
        if parts[3] != "mlp" or parts[4] != "experts":
            continue
        try:
            layer, expert = int(parts[2]), int(parts[5])
        except ValueError:
            continue
        found.setdefault((layer, expert), set()).add(name)
    complete = {}
    for key, names in found.items():
        layer, expert = key
        stem = f"model.layers.{layer}.mlp.experts.{expert}"
        want = [f"{stem}.{p}_proj.weight" for p in ("gate", "up", "down")]
        want += [w + ".qs" for w in want]
        if all(w in names for w in want):
            complete[key] = want
    return complete


def merged_for(index, layer, expert):
    """(merged int8 bytes, merged f32 scale bytes, expected element counts)."""
    stem = f"model.layers.{layer}.mlp.experts.{expert}"
    w, s = b"", b""
    for proj in ("gate", "up", "down"):
        w += read_tensor(index, f"{stem}.{proj}_proj.weight")
        s += read_tensor(index, f"{stem}.{proj}_proj.weight.qs")
    return w, s


def write_shard(path, tensors, metadata):
    """tensors: list of (name, dtype, shape, payload bytes)."""
    header, off = {}, 0
    for name, dtype, shape, payload in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [off, off + len(payload)]}
        off += len(payload)
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, payload in tensors:
            f.write(payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--shard-gb", type=float, default=2.5)
    ap.add_argument("--check", action="store_true",
                    help="re-read the output and compare every slab to its sources")
    args = ap.parse_args()

    index, _ = build_index(args.src)
    experts = expert_keys(index)
    if not experts:
        sys.exit("no pre-merge expert tensors found -- is this container already merged?")

    consumed = {n for names in experts.values() for n in names}
    passthrough = [n for n in index if n not in consumed]
    layers = sorted({l for l, _ in experts})
    print(f"{len(index)} tensors in, {len(experts)} experts over {len(layers)} layers, "
          f"{len(passthrough)} passthrough")

    os.makedirs(args.dst, exist_ok=True)
    for extra in os.listdir(args.src):
        if not extra.endswith(".safetensors"):
            shutil.copy2(os.path.join(args.src, extra), os.path.join(args.dst, extra))

    limit = int(args.shard_gb * (1 << 30))
    pending, size, shard_paths = [], 0, []

    def flush():
        nonlocal pending, size
        if not pending:
            return
        path = os.path.join(args.dst, f"model-{len(shard_paths):05d}.safetensors")
        write_shard(path, pending, {"format": "pt"})
        shard_paths.append(path)
        print(f"  wrote {os.path.basename(path)}: {len(pending)} tensors, {size/2**30:.2f} GiB")
        pending, size = [], 0

    for name in sorted(passthrough):
        _, _, _, dtype, shape = index[name]
        payload = read_tensor(index, name)
        pending.append((name, dtype, shape, payload))
        size += len(payload)
        if size >= limit:
            flush()

    for layer, expert in sorted(experts):
        stem = f"model.layers.{layer}.mlp.experts.{expert}"
        w, s = merged_for(index, layer, expert)
        pending.append((f"{stem}.merged_weight", "I8", [len(w)], w))
        pending.append((f"{stem}.qs", "F32", [len(s) // 4], s))
        size += len(w) + len(s)
        if size >= limit:
            flush()
    flush()

    total = sum(os.path.getsize(p) for p in shard_paths)
    print(f"{len(shard_paths)} shards, {total/2**30:.2f} GiB total")

    if args.check:
        out_index, _ = build_index(args.dst)
        bad = 0
        for layer, expert in sorted(experts):
            stem = f"model.layers.{layer}.mlp.experts.{expert}"
            w, s = merged_for(index, layer, expert)
            if read_tensor(out_index, f"{stem}.merged_weight") != w:
                print(f"MISMATCH {stem}.merged_weight"); bad += 1
            if read_tensor(out_index, f"{stem}.qs") != s:
                print(f"MISMATCH {stem}.qs"); bad += 1
        for name in sorted(passthrough):
            if read_tensor(out_index, name) != read_tensor(index, name):
                print(f"MISMATCH {name}"); bad += 1
        if bad:
            sys.exit(f"{bad} tensors differ")
        print(f"check: {len(experts)*2 + len(passthrough)} tensors byte-identical to source")


if __name__ == "__main__":
    main()
