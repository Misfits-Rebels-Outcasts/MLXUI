#!/usr/bin/env python3
"""Stdlib-only safetensors inventory for the ACE-Step 1.5 weight set.

Parses the safetensors header (8-byte little-endian uint64 JSON length, then the
JSON header) without torch/numpy/safetensors installed.  Emits:

  * a compact per-file / per-prefix table (count, params, dtypes, 3-D tensor shapes)
  * a full tensor-name list to a sibling .txt file
  * layout-conversion annotations (what an MLX/Swift loader must do per tensor)

Usage:
  python3 -I inventory_safetensors.py [--out DIR] FILE [FILE ...]
"""

from __future__ import annotations

import json
import os
import struct
import sys
from collections import defaultdict

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}


def read_header(path):
    """Return (metadata_dict|None, {name: (dtype, shape, data_offsets)})."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short for a safetensors header")
        (n,) = struct.unpack("<Q", raw)
        if n <= 0 or 8 + n > size:
            raise ValueError(f"{path}: implausible header length {n} (file {size} B) -> incomplete download?")
        hdr = json.loads(f.read(n))

    meta = hdr.pop("__metadata__", None)
    tensors = {}
    for name, ent in hdr.items():
        dtype = ent["dtype"]
        shape = list(ent["shape"])
        start, end = ent["data_offsets"]
        nbytes = end - start
        expect = DTYPE_BYTES.get(dtype)
        if expect is None:
            raise ValueError(f"{path}:{name}: unknown dtype {dtype}")
        n_elem = 1
        for d in shape:
            n_elem *= d
        if n_elem * expect != nbytes:
            raise ValueError(
                f"{path}:{name}: offsets say {nbytes} B but {dtype} {shape} needs "
                f"{n_elem * expect} B"
            )
        tensors[name] = (dtype, shape, (start, end))
    return meta, tensors


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def group_of(name):
    """Module group = first path component (state_dict prefix)."""
    return name.split(".", 1)[0]


def classify(name, dtype, shape):
    """Annotate the layout conversion an MLX/Swift loader must perform."""
    notes = []
    base = name.rsplit(".", 1)[0]
    leaf = name.rsplit(".", 1)[-1]

    if leaf in ("weight_g", "weight_v"):
        notes.append("WEIGHT_NORM fuse g*v/||v|| -> .weight")
    if name.endswith((".weight_g", ".weight_v")) and base.rsplit(".", 1)[-1] == "conv_t1":
        notes.append("ConvTranspose1d: [in,out,K]->[out,K,in]")
    if leaf == "weight" and len(shape) == 3:
        # Conv1d vs ConvTranspose1d can both be 3-D; use the module name.
        parent = name.rsplit(".", 2)[-2]
        if parent in ("conv_t1",):
            notes.append("ConvTranspose1d: [in,out,K]->[out,K,in]")
        elif parent in ("proj_in", "proj_out"):
            # DiT Sequential(Lambda, Conv{}, Lambda) -> index remap
            if "proj_in" in name:
                notes.append("Seq idx remap proj_in.1.* -> proj_in.*; Conv1d [out,in,K]->[out,K,in]")
            else:
                notes.append("Seq idx remap proj_out.1.* -> proj_out.*; ConvTranspose1d [in,out,K]->[out,K,in]")
        elif leaf == "weight" and shape[1] == 1 and shape[2] == 1:
            notes.append("1x1 conv")
        else:
            notes.append("Conv1d [out,in,K]->[out,K,in]")
    if leaf in ("alpha", "beta"):
        notes.append("Snake1d [1,C,1]->[C]")
    if leaf == "bias" and len(shape) == 1:
        notes.append("bias: unchanged")
    if "rotary" in name or "position_ids" in name or "inv_freq" in name:
        notes.append("STATIC BUFFER: recomputed, may be absent")
    if name.endswith("scale_shift_table") and not notes:
        notes.append("plain param")
    return "; ".join(notes)


def normalize(name):
    """Replace purely numeric path components with N, so layer stacks collapse."""
    return [c if not c.isdigit() else "N" for c in name.split(".")]


def module_key(name):
    """Coarse module key: collapse layer indices but keep the child module name."""
    norm = normalize(name)
    key = ".".join(norm[:3])
    if len(norm) > 3 and norm[2] == "N":
        key += "." + norm[3]
    return key


def prefix_table(tensors):
    """Group by collapsed module path: count, params, dtypes, 3-D shapes."""
    out = defaultdict(lambda: {"n": 0, "params": 0, "dtypes": defaultdict(int),
                               "shapes3d": set(), "example": None})
    for name, (dtype, shape, _off) in tensors.items():
        key = module_key(name)
        g = out[key]
        g["n"] += 1
        g["params"] += numel(shape)
        g["dtypes"][dtype] += numel(shape)
        if len(shape) == 3:
            g["shapes3d"].add(tuple(shape))
        if g["example"] is None or len(name) < len(g["example"]):
            g["example"] = name
    return out


def tensor_shape_map(tensors):
    """Normalized path -> {leaf: (dtype, shape)} for one representative stack member."""
    per_path = defaultdict(dict)
    for name, (dtype, shape, _off) in tensors.items():
        norm = normalize(name)
        # keep the full normalized path (all layers share one entry)
        key = ".".join(norm)
        per_path[key][name.rsplit(".", 1)[-1]] = (dtype, tuple(shape))
    return per_path


def fmt_params(n):
    if n >= 1_000_000_000:
        return f"{n / 1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def main(argv):
    files = [a for a in argv[1:] if not a.startswith("--")]
    out_dir = "."
    if "--out" in argv:
        out_dir = argv[argv.index("--out") + 1]
        files = [f for f in files if f != out_dir]
    os.makedirs(out_dir, exist_ok=True)

    grand_total = 0
    for path in files:
        if not os.path.exists(path):
            print(f"!! MISSING: {path}", file=sys.stderr)
            continue
        meta, tensors = read_header(path)
        stem = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
        print("=" * 100)
        print(f"FILE  {path}")
        print(f"      size {os.path.getsize(path):,} B   tensors {len(tensors)}   "
              f"params {fmt_params(sum(numel(s) for _d, s, _o in tensors.values()))}")
        if meta:
            print(f"      __metadata__ {json.dumps(meta, sort_keys=True)}")
        total = sum(numel(s) for _d, s, _o in tensors.values())
        grand_total += total
        dt = defaultdict(int)
        dbytes = defaultdict(int)
        for _n, (dtype, shape, _o) in tensors.items():
            dt[dtype] += numel(shape)
            dbytes[dtype] += numel(shape) * DTYPE_BYTES[dtype]
        print("      dtypes: " + ", ".join(
            f"{k} {fmt_params(v)} ({dbytes[k] / 1e9:.3f} GB)" for k, v in sorted(dt.items())))

        # collapsed module table
        table = prefix_table(tensors)
        print(f"\n  {'module (layer idx -> N)':<52} {'n':>5} {'params':>10}  {'dtypes':<18} 3-D shapes")
        for key in sorted(table):
            g = table[key]
            dts = ",".join(sorted(g["dtypes"]))
            shapes = " ".join("x".join(map(str, s)) for s in sorted(g["shapes3d"])[:3])
            print(f"  {key:<52} {g['n']:>5} {fmt_params(g['params']):>10}  {dts:<18} {shapes}")

        # exact shapes for one representative of every normalized path
        paths = tensor_shape_map(tensors)
        mod_txt = os.path.join(out_dir, f"modules_{stem}.txt")
        with open(mod_txt, "w") as f:
            f.write(f"# module shapes: {path}\n")
            f.write("# one representative instance per normalized (layer-index-collapsed) path\n")
            for key in sorted(paths):
                leaves = paths[key]
                f.write(f"\n{key}\n")
                for leaf in sorted(leaves):
                    dtype, shape = leaves[leaf]
                    f.write(f"    {leaf:<32} {dtype:<6} {shape}  numel={numel(shape)}\n")
        print(f"  -> module shape map: {mod_txt}")

        # full name list
        txt = os.path.join(out_dir, f"names_{stem}.txt")
        with open(txt, "w") as f:
            f.write(f"# {path}\n# size {os.path.getsize(path)} B, {len(tensors)} tensors\n")
            f.write(f"# {'name':<70} {'dtype':<6} {'shape':<26} {'numel':>10}  conversion\n")
            for name in sorted(tensors):
                dtype, shape, _o = tensors[name]
                f.write(f"{name:<70} {dtype:<6} {str(shape):<26} {numel(shape):>10}  "
                        f"{classify(name, dtype, shape)}\n")
        print(f"  -> full tensor list: {txt}")

        # conversion buckets
        buckets = defaultdict(list)
        for name in sorted(tensors):
            dtype, shape, _o = tensors[name]
            note = classify(name, dtype, shape)
            if note:
                buckets[note].append(name)
        print("\n  LAYOUT-CONVERSION BUCKETS")
        for note in sorted(buckets, key=lambda k: -len(buckets[k])):
            names = buckets[note]
            print(f"   [{len(names):>4}] {note}")
            for n in names[:3]:
                print(f"          e.g. {n}  {tensors[n][1]}")
            if len(names) > 3:
                print(f"          ... (+{len(names) - 3} more)")
        print()

    print(f"GRAND TOTAL params across files: {fmt_params(grand_total)} ({grand_total:,})")


if __name__ == "__main__":
    main(sys.argv)
