#!/usr/bin/env python3
"""Tiny-config parity goldens for the FLUX.2 Klein hand-port (MP7-3a..3g). Scratch tool, not part of the app.

Builds random-init fp32 reference modules from the Hugging Face diffusers / transformers classes at TINY sizes,
with fixed seeds, runs each one, and dumps weights + inputs + intermediates to outputs/*.safetensors so each Swift
slice can load the same tensors and compare. Takes NO real weights and makes NO network calls.

  python tiny_goldens.py [--out outputs]

Targets
  3a  transformer   : Flux2Transformer2DModel, tiny dims, 1 double + 2 single blocks, per-block intermediates
  3b  tokens        : fixed token ids (the real tokenizer files are not in hf/, so ids are synthetic; see README)
  3c  text encoder  : Qwen3Model tiny, hidden states at the tap layers stacked as the pipeline does
  3d  vae           : AutoencoderKLFlux2 tiny, encode mean + decode, plus the BN denorm / unpatchify glue
  3e  rope ids      : latent and text position ids
  3f  scheduler     : FlowMatchEulerDiscreteScheduler sigmas/timesteps for dynamic shifting + one Euler step
  3g  glue          : pack / unpack latents, mu (empirical), full 2-step denoise loop on the tiny modules
Every tensor is fp32. Total output is asserted <= 2 MB. The sha256 of every written file is printed.
"""
import argparse, hashlib, json, os, sys
import numpy as np
import torch
from safetensors.torch import save_file

SEED = 20261031
OUT_BUDGET = 2 * 1024 * 1024

# Tiny config. Shapes keep every code path (double + single blocks, 4-axis rope, GQA, BN stats) alive.
TINY_TF = dict(patch_size=1, in_channels=16, out_channels=None, num_layers=1, num_single_layers=2,
               attention_head_dim=16, num_attention_heads=4, joint_attention_dim=48, timestep_guidance_channels=32,
               mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, eps=1e-6, guidance_embeds=False)
TINY_QWEN = dict(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, rms_norm_eps=1e-6, rope_theta=1000000.0, max_position_embeddings=64,
                 tie_word_embeddings=False)
TAP_LAYERS = (1, 2, 3)  # real model: (9, 18, 27) of 36; joint_attention_dim = 3 * hidden_size = 48 here
TINY_VAE = dict(block_out_channels=(8, 8, 16, 16), down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4, in_channels=3, out_channels=3, latent_channels=4,
                layers_per_block=1, norm_num_groups=4, patch_size=(2, 2), use_quant_conv=True, use_post_quant_conv=True,
                mid_block_add_attention=True, batch_norm_eps=1e-4, batch_norm_momentum=0.1, sample_size=64, force_upcast=True)
SCHED = dict(base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=4096, max_shift=1.15,
             num_train_timesteps=1000, shift=3.0, shift_terminal=None, time_shift_type="exponential",
             use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False,
             stochastic_sampling=False)

written = {}


def dump(out, name, tensors, meta=None):
    clean = {k: v.detach().to(torch.float32).contiguous().cpu() for k, v in tensors.items()}
    path = os.path.join(out, name + ".safetensors")
    # one metadata key only: safetensors writes metadata from a hash map, so 2+ keys make the header bytes (and the sha256) vary run to run
    save_file(clean, path, metadata={"meta": json.dumps(meta or {}, sort_keys=True, default=str)})
    written[name + ".safetensors"] = path


def seeded():
    torch.manual_seed(SEED); np.random.seed(SEED)


def randomize(model):
    """Random-init is already done by the constructors; re-draw norm/bias params off their defaults so they matter."""
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.ndim == 1:
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g) if ("norm" in n or "scale" in n) else 0.05 * torch.randn(p.shape, generator=g))


def ids_image(h, w):
    # diffusers Flux2 _prepare_latent_ids: (t, h, w, l) with t=0, l=0
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return torch.stack([torch.zeros(h * w), hh.reshape(-1), ww.reshape(-1), torch.zeros(h * w)], dim=-1).float()


def ids_text(n):
    # diffusers Flux2 _prepare_text_ids: (t, h, w, l) with l = arange(n)
    z = torch.zeros(n)
    return torch.stack([z, z, z, torch.arange(n).float()], dim=-1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs"); ap.add_argument("--tokenizer-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "modelpack", "hf", "mlx-community_FLUX.2-klein-4B-bf16", "tokenizer")); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import diffusers, transformers
    from diffusers import Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
    from transformers import Qwen3Config, Qwen3Model
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    versions = dict(torch=torch.__version__, diffusers=diffusers.__version__, transformers=transformers.__version__,
                    numpy=np.__version__, seed=SEED, tap_layers=list(TAP_LAYERS))

    # ---- 3b tokens: REAL tokenizer files (pinned folder), chat template, enable_thinking=False ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_dir, local_files_only=True)
    prompts = ["a red fox in a snowy forest at sunrise", "A street sign that reads \"OPEN\"", "", "caf\u00e9 au lait, \u732b \U0001F431"]
    tokcases = []
    for pr in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": pr}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        real = int(enc["attention_mask"].sum())
        tokcases.append(dict(prompt=pr, templated=text, input_ids=enc["input_ids"][0][enc["attention_mask"][0] == 1].tolist() if tok.padding_side == "right" else enc["input_ids"][0][-real:].tolist(),
                             real_length=real, padded_length=int(enc["input_ids"].shape[1]), padding_side=tok.padding_side, pad_token_id=tok.pad_token_id,
                             pad_ids_head=enc["input_ids"][0][:3].tolist(), mask_head=enc["attention_mask"][0][:3].tolist()))
    with open(os.path.join(a.out, "g3b_tokens.json"), "w") as f: json.dump(tokcases, f, indent=1, ensure_ascii=False)
    written["g3b_tokens.json"] = os.path.join(a.out, "g3b_tokens.json")
    ids = torch.tensor([tokcases[0]["input_ids"][:8]], dtype=torch.long)   # tiny-encoder input: first 8 real ids, mod vocab
    ids = ids % TINY_QWEN["vocab_size"]
    mask = torch.ones_like(ids)

    # ---- 3c text encoder: Qwen3Model, hidden states at the tap layers, stacked (B, S, 3*H) as the pipeline does ----
    seeded()
    qcfg = Qwen3Config(**TINY_QWEN)
    qwen = Qwen3Model(qcfg).float().eval(); randomize(qwen)
    with torch.no_grad():
        o = qwen(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False)
    hs = o.hidden_states  # len = layers + 1, index 0 = embeddings
    stacked = torch.stack([hs[i] for i in TAP_LAYERS], dim=1)           # (B, 3, S, H)
    prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(1, ids.shape[1], -1)  # (B, S, 3*H)
    t = {"w." + k: v for k, v in qwen.state_dict().items()}
    t.update({"in.input_ids": ids.float(), "out.embed": hs[0], "out.final_norm": o.last_hidden_state, "out.prompt_embeds": prompt_embeds})
    for i in TAP_LAYERS: t[f"out.hidden_{i}"] = hs[i]
    dump(a.out, "g3c_textencoder", t, {"config": json.dumps(TINY_QWEN), "taps": TAP_LAYERS})

    # ---- 3a transformer ----
    seeded()
    tf = Flux2Transformer2DModel(**TINY_TF).float().eval(); randomize(tf)
    H = W = 4                                    # packed latent grid
    img = torch.randn(1, H * W, TINY_TF["in_channels"])
    txt = prompt_embeds.detach()
    img_ids = ids_image(H, W)[None]; txt_ids = ids_text(txt.shape[1])[None]
    ts = torch.tensor([0.7])
    inter = {}
    hooks = []
    for i, blk in enumerate(tf.transformer_blocks):
        hooks.append(blk.register_forward_hook(lambda m, inp, out, i=i: inter.update({f"double_{i}_img": out[1], f"double_{i}_txt": out[0]})))
    for i, blk in enumerate(tf.single_transformer_blocks):
        hooks.append(blk.register_forward_hook(lambda m, inp, out, i=i: inter.update({f"single_{i}": out if torch.is_tensor(out) else out[0]})))
    with torch.no_grad():
        pred = tf(hidden_states=img, encoder_hidden_states=txt, timestep=ts, img_ids=img_ids, txt_ids=txt_ids, guidance=None, return_dict=False)[0]
    for h in hooks: h.remove()
    t = {"w." + k: v for k, v in tf.state_dict().items()}
    t.update({"in.hidden_states": img, "in.encoder_hidden_states": txt, "in.timestep": ts, "in.img_ids": img_ids, "in.txt_ids": txt_ids, "out.pred": pred})
    t.update({"mid." + k: v for k, v in inter.items()})
    dump(a.out, "g3a_transformer", t, {"config": json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in TINY_TF.items()})})

    # ---- 3d vae ----
    seeded()
    vae = AutoencoderKLFlux2(**TINY_VAE).float().eval(); randomize(vae)
    with torch.no_grad():
        for n, b in vae.named_buffers():
            if n.endswith("running_mean"): b.copy_(0.2 * torch.randn(b.shape))
            if n.endswith("running_var"): b.copy_(1.0 + 0.3 * torch.rand(b.shape))
    x = torch.rand(1, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        post = vae.encode(x).latent_dist
        mean = post.mean
        rec = vae.decode(mean, return_dict=False)[0]
    # BN denorm + unpatchify, as the pipeline does before decode: z_packed (B, 4*C, h, w) -> (B, C, 2h, 2w)
    C = TINY_VAE["latent_channels"]
    zp = torch.randn(1, 4 * C, 2, 2)
    bn = vae.bn
    mean_b = bn.running_mean.view(1, -1, 1, 1); std_b = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + TINY_VAE["batch_norm_eps"])
    zd = zp * std_b + mean_b
    zu = zd.reshape(1, C, 2, 2, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape(1, C, 4, 4)
    with torch.no_grad():
        dec = vae.decode(zu, return_dict=False)[0]
    t = {"w." + k: v for k, v in vae.state_dict().items()}
    t.update({"in.image": x, "out.latent_mean": mean, "out.recon": rec, "in.z_packed": zp, "out.z_denorm": zd, "out.z_unpatchified": zu, "out.decoded": dec})
    dump(a.out, "g3d_vae", t, {"config": json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in TINY_VAE.items()})})

        # ---- 3e ids / patchify / pack, straight from the pipeline's own static methods ----
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P, compute_empirical_mu, retrieve_timesteps
    t = {}
    for (gh, gw) in ((2, 3), (4, 4)):
        z = torch.arange(1 * 5 * gh * gw, dtype=torch.float32).reshape(1, 5, gh, gw)
        t[f"latent_ids_{gh}x{gw}"] = P._prepare_latent_ids(z)[0].float()
        t[f"pack_in_{gh}x{gw}"] = z; t[f"pack_out_{gh}x{gw}"] = P._pack_latents(z)
    t["text_ids_6"] = P._prepare_text_ids(torch.zeros(1, 6, 1))[0].float()
    z16 = torch.arange(1 * 3 * 4 * 6, dtype=torch.float32).reshape(1, 3, 4, 6)          # unpatchified (B,C,2h,2w)
    pz = P._patchify_latents(z16)                                                         # (B,4C,h,w)
    t["patchify_in"] = z16; t["patchify_out"] = pz; t["unpatchify_roundtrip"] = P._unpatchify_latents(pz)
    gid = P._prepare_latent_ids(pz)
    pk = P._pack_latents(pz)
    t["unpack_ids"] = gid[0].float(); t["unpack_in"] = pk; t["unpack_out"] = P._unpack_latents_with_ids(pk, gid)
    dump(a.out, "g3e_ids", t)

    # ---- 3f scheduler: pipeline's own mu + retrieve_timesteps, the four cases, + Euler step ----
    scal = []
    t = {}
    for seq, steps in ((256, 4), (1024, 4), (4096, 4), (4096, 8), (4096, 1), (16384, 4)):
        mu = compute_empirical_mu(image_seq_len=seq, num_steps=steps)
        sch = FlowMatchEulerDiscreteScheduler(**SCHED)
        sig = np.linspace(1.0, 1 / steps, steps)
        ts, _ = retrieve_timesteps(sch, steps, "cpu", sigmas=sig, mu=mu)
        scal.append(dict(image_seq_len=seq, steps=steps, mu=mu, input_sigmas=sig.tolist(), sigmas=[float(x) for x in sch.sigmas], timesteps=[float(x) for x in ts]))
        t[f"s{seq}_n{steps}.sigmas"] = sch.sigmas.float(); t[f"s{seq}_n{steps}.timesteps"] = ts.float()
    with open(os.path.join(a.out, "g3f_scalars.json"), "w") as f: json.dump(scal, f, indent=1)
    written["g3f_scalars.json"] = os.path.join(a.out, "g3f_scalars.json")
    sch = FlowMatchEulerDiscreteScheduler(**SCHED)
    retrieve_timesteps(sch, 4, "cpu", sigmas=np.linspace(1.0, 0.25, 4), mu=compute_empirical_mu(1024, 4))
    sample = torch.randn(1, 16, 8); model_out = torch.randn(1, 16, 8)
    stepped = [sch.step(model_out, sch.timesteps[0], sample, return_dict=False)[0]]
    t.update({"step.sample": sample, "step.model_output": model_out, "step.result0": stepped[0], "step.sigmas": sch.sigmas.float()})
    dump(a.out, "g3f_scheduler", t, {"config": json.dumps(SCHED)})

    # ---- 3g glue: 2-step denoise on the tiny transformer, then the pipeline's decode glue on the tiny VAE ----
    seeded()
    Cl = TINY_VAE["latent_channels"]; gh = gw = 2
    lat = torch.randn(1, 4 * Cl, gh, gw)                                  # (B, 4C, h, w), 4C = tiny in_channels
    lid = P._prepare_latent_ids(lat)
    packed = P._pack_latents(lat)
    sch = FlowMatchEulerDiscreteScheduler(**SCHED)
    tss, _ = retrieve_timesteps(sch, 2, "cpu", sigmas=np.linspace(1.0, 0.5, 2), mu=compute_empirical_mu(gh * gw, 2))
    x_t = packed.clone(); traj = {}
    tids = ids_text(txt.shape[1])[None]
    for i, tt in enumerate(tss):
        with torch.no_grad():
            p = tf(hidden_states=x_t, encoder_hidden_states=txt, timestep=(tt.expand(1) / 1000), img_ids=lid, txt_ids=tids, guidance=None, return_dict=False)[0]
        x_t = sch.step(p, tt, x_t, return_dict=False)[0]
        traj[f"step{i}.pred"] = p; traj[f"step{i}.latents"] = x_t
    un = P._unpack_latents_with_ids(x_t, lid)
    bmean = vae.bn.running_mean.view(1, -1, 1, 1); bstd = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    den = un * bstd + bmean
    unp = P._unpatchify_latents(den)
    with torch.no_grad(): img_out = vae.decode(unp, return_dict=False)[0]
    dump(a.out, "g3g_glue", {"in.latents": lat, "in.txt": txt, "out.ids": lid[0].float(), "out.packed": packed, "out.timesteps": tss.float(), "out.sigmas": sch.sigmas.float(),
                             "out.unpacked": un, "out.bn_denorm": den, "out.unpatchified": unp, "out.image": img_out, **traj})

    with open(os.path.join(a.out, "versions.json"), "w") as f: json.dump(versions, f, indent=2, sort_keys=True)
    total = sum(os.path.getsize(p) for p in written.values())
    for n, p in sorted(written.items()):
        print(hashlib.sha256(open(p, "rb").read()).hexdigest(), os.path.getsize(p), n)
    print("total bytes", total)
    assert total <= OUT_BUDGET, f"outputs {total} exceed the 2 MB budget"


if __name__ == "__main__":
    main()
