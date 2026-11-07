import re, sys

def parse(path):
    d = {}
    for line in open(path):
        line = line.strip()
        if '=' in line and not line.startswith('###'):
            k, v = line.split('=', 1)
            vals = []
            ok = True
            for part in v.split(','):
                try: vals.append(float(part))
                except ValueError: ok = False; break
            if ok and vals: d[k] = vals
    return d

py = {}
for f in ['python-out.txt', 'python-out2.txt', 'python-rope-hand.txt']:
    py.update(parse(f))
sw = parse('swift-out.txt')
names = [k for k in py if k in sw and len(py[k]) == len(sw[k])]
print(f"compared {len(names)} numeric arrays")
worst = []
for k in names:
    a, b = py[k], sw[k]
    diffs = [abs(x-y) for x,y in zip(a,b) if abs(x-y) == abs(x-y)]
    mx = max(diffs) if diffs else 0.0
    scale = max(1e-6, min(1.0, max(abs(x) for x in a + b)))
    worst.append((mx/scale, mx, k))
worst.sort(reverse=True)
for rel, mx, k in worst[:18]:
    print(f"{k}: max_abs_diff={mx:.6g} rel={rel:.3g}")
# constants (non-numeric) side-by-side
for k in ['sdpa_bf16_dtype','sdpa_f16_dtype','quant_wq_dtype','dequant_dtype','qlinear_weight_shape','conv1d_module_weight_shape','convT_module_weight_shape']:
    pass
