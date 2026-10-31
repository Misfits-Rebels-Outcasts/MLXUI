#!/usr/bin/env python3
"""MP7-3d: tiny-config fp32 goldens for the Klein MMDiT front end (g3a_transformer.safetensors has no temb / rope / embedder / modulation tensors).

Builds the SAME tiny Flux2Transformer2DModel as tiny_goldens.py (same seed, same constructor, same randomize), so the weights are the ones in
g3a_transformer.safetensors; the script asserts that and writes NO weights, only inputs and outputs. Everything is produced by diffusers' own
submodules (time_guidance_embed, double/single modulation, x/context embedders, norm_out, pos_embed, apply_rotary_emb).

  .venv/bin/python embedders_goldens.py --out outputs6      (run from klein-ref/; copy the one output file next to this script)

Keys: in.* inputs, out.* outputs. rope.*: cos/sin of pos_embed(ids) for the g3a layout and for a same-length alternate layout (2x8 vs 4x4 grid).
"""
import argparse, hashlib, json, os
import numpy as np, torch
from safetensors.torch import save_file, load_file

SEED = 20261031
TINY_TF = dict(patch_size=1, in_channels=16, out_channels=None, num_layers=1, num_single_layers=2,
               attention_head_dim=16, num_attention_heads=4, joint_attention_dim=48, timestep_guidance_channels=32,
               mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, eps=1e-6, guidance_embeds=False)


def randomize(model):   # identical to tiny_goldens.py
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g) if ("norm" in n or "scale" in n) else 0.05 * torch.randn(p.shape, generator=g))


def ids_image(h, w):
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([torch.zeros(h * w), hh.reshape(-1), ww.reshape(-1), torch.zeros(h * w)], dim=-1).float()


def ids_text(n):
    z = torch.zeros(n)
    return torch.stack([z, z, z, torch.arange(n).float()], dim=-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs6")
    ap.add_argument("--g3a", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "g3a_transformer.safetensors"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from diffusers import Flux2Transformer2DModel
    from diffusers.models.embeddings import apply_rotary_emb
    from diffusers.models.transformers.transformer_flux2 import Flux2Modulation
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    # tiny_goldens.py draws the Qwen3 model (and its randomize) from the same global seed BEFORE the transformer; the transformer is built after
    # a fresh seeded() call, so seeding here reproduces it exactly. The weight assertion below proves it.
    torch.manual_seed(SEED); np.random.seed(SEED)
    tf = Flux2Transformer2DModel(**TINY_TF).float().eval(); randomize(tf)
    g3a = load_file(a.g3a)
    for k, v in tf.state_dict().items():
        assert torch.equal(v.float(), g3a["w." + k]), f"weight {k} differs from g3a_transformer.safetensors"
    assert set(tf.state_dict()) == set(k[2:] for k in g3a if k.startswith("w.")), "weight key sets differ"

    inner = TINY_TF["attention_head_dim"] * TINY_TF["num_attention_heads"]
    g = torch.Generator().manual_seed(SEED + 3)
    ts = g3a["in.timestep"]                      # [0.7] sigma, multiplied by 1000 inside the transformer
    img = g3a["in.hidden_states"]; txt = g3a["in.encoder_hidden_states"]
    img_ids = g3a["in.img_ids"]; txt_ids = g3a["in.txt_ids"]
    norm_in = torch.randn(1, 6, inner, generator=g)
    out = {"in.norm_hidden": norm_in}
    with torch.no_grad():
        t1000 = ts.to(img.dtype) * 1000
        out["out.time_proj"] = tf.time_guidance_embed.time_proj(t1000)
        temb = tf.time_guidance_embed(t1000, None)
        out["out.temb"] = temb
        mods = {"img": (tf.double_stream_modulation_img, 2), "txt": (tf.double_stream_modulation_txt, 2), "single": (tf.single_stream_modulation, 1)}
        for name, (m, sets) in mods.items():
            raw = m(temb); out[f"out.mod_{name}"] = raw
            for si, trip in enumerate(Flux2Modulation.split(raw, sets)):
                for part, t in zip(("shift", "scale", "gate"), trip):
                    out[f"out.mod_{name}_{si}_{part}"] = t.reshape(1, -1).clone()
        out["out.x_embedder"] = tf.x_embedder(img)
        out["out.context_embedder"] = tf.context_embedder(txt)
        out["out.norm_out"] = tf.norm_out(norm_in, temb)
        out["out.proj_out"] = tf.proj_out(out["out.norm_out"])
        # rope: g3a layout (4x4 image grid + text ids) and an alternate layout of the same length (2x8 grid)
        i0, t0 = img_ids[0], txt_ids[0]
        i1 = ids_image(2, 8)
        assert i0.shape[0] == i1.shape[0]
        for name, ids in (("img", i0), ("txt", t0), ("img_alt", i1)):
            c, s = tf.pos_embed(ids)
            out[f"rope.{name}_ids"] = ids; out[f"rope.{name}_cos"] = c; out[f"rope.{name}_sin"] = s
        # rotary application on [B, S, H, D] exactly as the attention processors call it (sequence_dim=1)
        q = torch.randn(1, i0.shape[0], TINY_TF["num_attention_heads"], TINY_TF["attention_head_dim"], generator=g)
        k = torch.randn(1, i0.shape[0], TINY_TF["num_attention_heads"], TINY_TF["attention_head_dim"], generator=g)
        c, s = tf.pos_embed(i0)
        out["in.q"] = q; out["in.k"] = k
        out["out.q_rot"] = apply_rotary_emb(q, (c, s), sequence_dim=1)
        out["out.k_rot"] = apply_rotary_emb(k, (c, s), sequence_dim=1)
        c1, s1 = tf.pos_embed(i1)
        out["out.q_rot_alt"] = apply_rotary_emb(q, (c1, s1), sequence_dim=1)
    path = os.path.join(a.out, "g3d_embedders.safetensors")
    # one metadata key only (safetensors writes metadata from a hash map; 2+ keys vary the header bytes)
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in out.items()}, path,
              metadata={"meta": json.dumps({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TINY_TF.items()}, "seed": SEED}, sort_keys=True)})
    print(hashlib.sha256(open(path, "rb").read()).hexdigest(), os.path.getsize(path), path)


if __name__ == "__main__":
    main()
