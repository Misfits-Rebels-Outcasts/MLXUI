#!/usr/bin/env python3
"""MC5-2 -> MC5-3c oracle: tiny-tier reference for the weight-loading conversions.

Reads a handful of REAL tensors from the pinned checkpoint (bf16 -> exact f32 upcast), slices dim 0 where the
tensor is big (slicing dim 0 commutes with weight-norm fuse, which takes the norm per dim-0 row), and runs the
UPSTREAM converters unchanged (ca1e85f: acestep/models/mlx/dit_convert.convert_decoder_weights and
vae_convert.convert_vae_weights, via minimal fake state_dict holders). Writes inputs (`in.<name>`, original PyTorch
layout) and upstream's outputs (`out.<name>`, MLX layout, new names) into one safetensors file, f32 throughout.

Usage: oracle/ACE-Step-1.5/.venv/bin/python -I scripts/dump_convert_tiny.py <Ace-Step1.5 dir> <out.safetensors>
"""
import json, sys, os, types
import numpy as np, torch
from safetensors import safe_open
from safetensors.numpy import save_file

root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "oracle", "ACE-Step-1.5"))
from acestep.models.mlx.dit_convert import convert_decoder_weights
from acestep.models.mlx.vae_convert import convert_vae_weights

def grab(path, names, slice0=None):
    d = {}
    with safe_open(path, framework="pt") as f:
        for n in names:
            t = f.get_tensor(n)
            if slice0 and n in slice0:
                t = t[: slice0[n]]
            d[n] = t.float().contiguous()
    return d

# ---- DiT (keys relative to `decoder.`, as upstream's state_dict has them)
dit_path = os.path.join(root, "acestep-v15-turbo/model.safetensors")
dit_names = ["decoder.proj_in.1.weight", "decoder.proj_in.1.bias", "decoder.proj_out.1.weight",
             "decoder.proj_out.1.bias", "decoder.norm_out.weight", "decoder.time_embed.linear_1.weight"]
dit = grab(dit_path, dit_names, {"decoder.proj_in.1.weight": 16, "decoder.proj_out.1.weight": 16})
sd = {k[len("decoder."):]: v for k, v in dit.items()}
fake = types.SimpleNamespace(decoder=types.SimpleNamespace(state_dict=lambda: sd))
dit_out = {k: np.array(v) for k, v in convert_decoder_weights(fake)}

# ---- VAE decoder
vae_path = os.path.join(root, "vae/diffusion_pytorch_model.safetensors")
vn = []
for base in ["decoder.block.0.conv_t1", "decoder.block.4.conv_t1", "decoder.block.4.res_unit1.conv1",
             "decoder.block.4.res_unit1.conv2", "decoder.conv1", "decoder.conv2"]:
    vn += [base + ".weight_g", base + ".weight_v"]
    if base != "decoder.conv2":
        vn.append(base + ".bias")
for s in ["decoder.block.4.res_unit1.snake1", "decoder.snake1"]:
    vn += [s + ".alpha", s + ".beta"]
vae = grab(vae_path, vn, {"decoder.block.0.conv_t1.weight_g": 8, "decoder.block.0.conv_t1.weight_v": 8,
                          "decoder.block.0.conv_t1.bias": 10**9,
                          "decoder.conv1.weight_g": 16, "decoder.conv1.weight_v": 16})
class SD:  # holds a state_dict whose values mimic torch params (.detach().cpu().float().numpy())
    def __init__(self, d): self.d = d
    def state_dict(self): return self.d
vae_out = {k: np.array(v) for k, v in convert_vae_weights(SD(vae))}

tensors, meta = {}, {"pinned_upstream": "ca1e85fe9430179831e6bc6be790c332190a3866",
                     "pinned_weights": "ACE-Step/Ace-Step1.5@19671f406d603126926c1b7e2adc169acbcade22",
                     "torch": torch.__version__, "slices_dim0": json.dumps({"decoder.proj_in.1.weight": 16,
                     "decoder.proj_out.1.weight": 16, "decoder.block.0.conv_t1.weight_g/v": 8, "decoder.conv1.weight_g/v": 16})}
for k, v in {**dit, **vae}.items():
    tensors["in." + k] = v.numpy().astype(np.float32)
for k, v in dit_out.items():
    tensors["out.dit." + k] = np.ascontiguousarray(v.astype(np.float32))
for k, v in vae_out.items():
    tensors["out.vae." + k] = np.ascontiguousarray(v.astype(np.float32))
save_file(tensors, out, metadata=meta)
print(len(tensors), "tensors", os.path.getsize(out), "bytes")
for k in sorted(tensors): print(k, tensors[k].shape)
