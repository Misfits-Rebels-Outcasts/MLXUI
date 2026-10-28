"""One-time generator for MLXUITests/Fixtures/RIFE (RIFE-3b). Runs the rife-mlx reference (xocialize/rife-mlx, MIT) on one
synthetic frame pair at t=0.5 and writes the inputs (uint8) and outputs (float32, before 8-bit quantisation) as raw
little-endian files. Usage: python3 -I make_rife_golden.py <weights_dir> <out_dir> <rife-mlx clone>"""
import sys, json, hashlib
sys.path.insert(0, sys.argv[3])   # the rife-mlx clone (the reference being measured against)
import numpy as np
import mlx.core as mx
from rife_mlx.model.RIFE_HDv3 import Model
from rife_mlx.config import VERSIONS

weights_dir, out = sys.argv[1], sys.argv[2]
H, W = 96, 130   # H pads 96 -> 128 only at scale<1; W 130 needs padding to 192 (exercises pad + crop)

rng = np.random.default_rng(425)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def scene(dx, dy):
    r = 0.5 + 0.5 * np.sin((xx - dx) * 0.11 + (yy - dy) * 0.05)
    g = 0.5 + 0.5 * np.cos((xx - dx) * 0.07 - (yy - dy) * 0.13)
    disc = (((xx - (45 + dx)) ** 2 + (yy - (48 + dy)) ** 2) < 18 ** 2).astype(np.float32)
    b = 0.25 + 0.6 * disc
    img = np.stack([r, g, b], axis=-1)
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)
a = (scene(0, 0) * 255).round().astype(np.uint8)
b = (scene(6, 3) * 255).round().astype(np.uint8)   # a disc moving right+down, texture shifting

cfg = VERSIONS["4.25"]
weights = dict(mx.load(weights_dir + "/model.safetensors").items())
model = Model(scale_list=cfg.scale_list, pad_to=cfg.pad_to)
model.flownet.load_weights(list(weights.items()), strict=True)
mx.eval(model.flownet.parameters())

def run(t, scale):
    x0 = mx.array(a.astype(np.float32)[None] / 255.0)
    x1 = mx.array(b.astype(np.float32)[None] / 255.0)
    y = model.inference(x0, x1, t, scale)
    mx.eval(y)
    return np.array(y)[0].astype(np.float32)

cases = {"t050_s100": (0.5, 1.0), "t025_s100": (0.25, 1.0), "t050_s050": (0.5, 0.5)}
meta = {"height": H, "width": W, "seed": 425, "reference": "xocialize/rife-mlx", "mlx": mx.__version__ if hasattr(mx, "__version__") else None,
        "weights_sha256": hashlib.sha256(open(weights_dir + "/model.safetensors", "rb").read()).hexdigest(), "cases": {}}
a.tofile(out + "/frame0.u8"); b.tofile(out + "/frame1.u8")
for name, (t, s) in cases.items():
    o = run(t, s)
    o.astype("<f4").tofile(out + f"/{name}.f32")
    meta["cases"][name] = {"timestep": t, "scale": s, "mean": float(o.mean()), "min": float(o.min()), "max": float(o.max())}
json.dump(meta, open(out + "/golden.json", "w"), indent=2, sort_keys=True)
print(json.dumps(meta, indent=2))
