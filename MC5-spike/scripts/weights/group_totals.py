#!/usr/bin/env python3
"""Top-level group totals per safetensors file (stdlib only)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inventory_safetensors import read_header, numel, fmt_params
from collections import defaultdict

for path in sys.argv[1:]:
    _m, t = read_header(path)
    g = defaultdict(lambda: [0, 0, defaultdict(int)])
    for name, (dtype, shape, _o) in t.items():
        key = name.split(".")[0]
        e = g[key]; e[0] += 1; e[1] += numel(shape); e[2][dtype] += numel(shape)
    print(f"\n{path}")
    print(f"  {'group':<24} {'tensors':>8} {'params':>10}  {'% of file':>9}  dtypes")
    tot = sum(e[1] for e in g.values())
    for k in sorted(g, key=lambda k: -g[k][1]):
        n, p, d = g[k]
        print(f"  {k:<24} {n:>8} {fmt_params(p):>10}  {100.0*p/tot:>8.2f}%  {','.join(sorted(d))}")
    print(f"  {'TOTAL':<24} {sum(e[0] for e in g.values()):>8} {fmt_params(tot):>10}")
