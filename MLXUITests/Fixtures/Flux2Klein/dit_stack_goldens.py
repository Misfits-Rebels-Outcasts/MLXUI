#!/usr/bin/env python3
"""MP7-3f: tiny-config fp32 golden for the WHOLE Klein MMDiT stack: 3 double + 3 single blocks, every block output, the final prediction.

g3a_transformer.safetensors (MP7-3a) has one double and two single blocks, which cannot show stacking (residual order between blocks, the shared modulation reused by every
block, the text/image hand-off between the double and single stage). This is a separate, much smaller config so the fixture folder stays under 2 MB.

  .venv/bin/python dit_stack_goldens.py --out outputs8      (run from klein-ref/; the one output file is g3f_dit_stack.safetensors)

Everything comes from diffusers 0.37.1 `Flux2Transformer2DModel`, built with a fixed seed. Weights are drawn here (N(0, 1/fan_in) for matrices, 1 + 0.1 N(0,1) for the 1-D
norm weights) so every residual branch has real magnitude; the default torch init leaves the gates near zero and would make the blocks nearly the identity.
Keys: w.* weights (state_dict names), in.* inputs, mid.double_N_txt / mid.double_N_img (outputs of each double block) and mid.single_N (the joint [text, image] stream
after each single block), out.pred; the same set for a second sigma under alt.* (alt.timestep, alt.mid.*, alt.pred). The intermediates are captured with forward hooks on the real blocks and cross-checked below (hand-off and final layers recomputed).
Tiny: heads 2 x head_dim 8 = inner 16, in_channels 8, joint_attention_dim 12, timestep channels 16, axes (2, 2, 2, 2), text 4 rows, image grid 2 x 3.
"""
import argparse, hashlib, json, os
import numpy as np, torch
from safetensors.torch import save_file

SEED = 20261031
TF = dict(patch_size=1, in_channels=8, out_channels=None, num_layers=3, num_single_layers=3, attention_head_dim=8, num_attention_heads=2,
          joint_attention_dim=12, timestep_guidance_channels=16, mlp_ratio=3.0, axes_dims_rope=(2, 2, 2, 2), rope_theta=2000, eps=1e-6,
          guidance_embeds=False)
TEXT_ROWS, GRID_H, GRID_W = 4, 2, 3


def ids_image(h, w):
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([torch.zeros(h * w), hh.reshape(-1), ww.reshape(-1), torch.zeros(h * w)], dim=-1).float()


def ids_text(n):
    z = torch.zeros(n)
    return torch.stack([z, z, z, torch.arange(n).float()], dim=-1)


def randomize(model):
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / (p.shape[1] ** 0.5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs8"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import diffusers
    from diffusers import Flux2Transformer2DModel
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    tf = Flux2Transformer2DModel(**TF).float().eval(); randomize(tf)
    assert len(tf.transformer_blocks) == 3 and len(tf.single_transformer_blocks) == 3

    g = torch.Generator().manual_seed(SEED + 2)
    n_img = GRID_H * GRID_W
    hidden = torch.randn(1, n_img, TF["in_channels"], generator=g)
    encoder = torch.randn(1, TEXT_ROWS, TF["joint_attention_dim"], generator=g)
    img_ids = ids_image(GRID_H, GRID_W)[None]; txt_ids = ids_text(TEXT_ROWS)[None]

    def forward(ts):
        """One forward with hooks on the real blocks; returns (mids, pred) after cross-checking the hand-off and the final layers."""
        mid, single_in, hooks = {}, {}, []
        for i, blk in enumerate(tf.transformer_blocks):
            hooks.append(blk.register_forward_hook(lambda m, inp, out, i=i: mid.update({f"double_{i}_txt": out[0].detach().clone(), f"double_{i}_img": out[1].detach().clone()})))
        for i, blk in enumerate(tf.single_transformer_blocks):
            hooks.append(blk.register_forward_hook(lambda m, inp, out, i=i: mid.update({f"single_{i}": (out if torch.is_tensor(out) else out[0]).detach().clone()})))
        hooks.append(tf.single_transformer_blocks[0].register_forward_pre_hook(lambda m, args, kwargs: single_in.update({"x": kwargs["hidden_states"].detach().clone()}), with_kwargs=True))
        with torch.no_grad():
            pred = tf(hidden_states=hidden, encoder_hidden_states=encoder, timestep=ts, img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)[0]
            for h in hooks: h.remove()
            # The hooks saw the real module outputs: (1) the single stage starts from cat([double_2 txt, double_2 img]); (2) the last single block's image rows through
            # norm_out + proj_out give the prediction.
            assert torch.equal(single_in["x"], torch.cat([mid["double_2_txt"], mid["double_2_img"]], dim=1)), "single stage input is not [txt, img] of the last double block"
            temb = tf.time_guidance_embed(ts * 1000, None)
            again = tf.proj_out(tf.norm_out(mid["single_2"][:, TEXT_ROWS:], temb))
            assert torch.equal(again, pred), "pred is not proj_out(norm_out(single_2[:, text:]))"
            # every stage actually changes the stream (the stack is not degenerate)
            prev = None
            for k in [f"double_{i}_img" for i in range(3)] + [f"single_{i}" for i in range(3)]:
                if prev is not None and prev.shape == mid[k].shape:
                    assert (mid[k] - prev).abs().max() > 1e-2, f"{k} barely differs from the previous stage"
                prev = mid[k]
        return mid, pred

    # Primary run: sigma 0.005 (timestep 5). The sinusoid argument is sigma * 1000 in float32; at sigma 0.7 (angles up to 700 rad) the sin/cos of the same float32
    # argument already differ between torch and MLX by ~1e-5 and the stack passes ~half of that through to the output (measured: nudging sigma 0.7 -> 0.7000001 moves
    # time_proj by 1e-4 and the prediction by 5e-5). A small sigma keeps the block parity measurement about the blocks. The second run (alt.*) keeps the realistic sigma.
    ts = torch.tensor([0.005]); ts_alt = torch.tensor([0.7])
    mid, pred = forward(ts)
    mid_alt, pred_alt = forward(ts_alt)
    t = {"w." + k: v for k, v in tf.state_dict().items()}
    t.update({"in.hidden_states": hidden, "in.encoder_hidden_states": encoder, "in.timestep": ts, "in.img_ids": img_ids, "in.txt_ids": txt_ids, "out.pred": pred})
    t.update({"mid." + k: v for k, v in mid.items()})
    t.update({"alt.timestep": ts_alt, "alt.pred": pred_alt})
    t.update({"alt.mid." + k: v for k, v in mid_alt.items()})
    path = os.path.join(a.out, "g3f_dit_stack.safetensors")
    # one metadata key only (safetensors writes metadata from a hash map; 2+ keys vary the header bytes)
    save_file({k: v.detach().to(torch.float32).contiguous() for k, v in t.items()}, path,
              metadata={"meta": json.dumps({"config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in TF.items()}, "seed": SEED,
                                            "diffusers": diffusers.__version__, "torch": torch.__version__}, sort_keys=True)})
    print(hashlib.sha256(open(path, "rb").read()).hexdigest(), os.path.getsize(path), path)
    for k in sorted(mid): print(k, tuple(mid[k].shape), float(mid[k].norm()))
    print("pred", tuple(pred.shape), float(pred.norm()))


if __name__ == "__main__":
    main()
