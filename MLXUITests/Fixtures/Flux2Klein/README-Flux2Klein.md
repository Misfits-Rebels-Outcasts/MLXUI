# FLUX.2 Klein parity fixtures (MP7-3a)

Generated **once** on 2026-10-31 by `tiny_goldens.py` (this folder) from the Hugging Face PyTorch classes, with `requirements.txt`
pinned (torch 2.9.1, diffusers 0.37.1, transformers 4.57.3; `versions.json`). Tiny random-init modules, fixed seed 20261031, fp32,
one thread, no model weights, no network (the tokenizer comes from the pinned folder `mlx-community/FLUX.2-klein-4B-bf16 @ 2f7dd4ac`).
Re-running gives byte-identical files. **Never edit these to make a change pass**: a mismatch is a bug in `Modules/Flux2Klein` or a
question for the owner.

| File | For slice | Holds |
|---|---|---|
| `g3a_transformer.safetensors` | 3a/3e..3h | tiny `Flux2Transformer2DModel` weights (`w.`), inputs (`in.`), per-block intermediates (`mid.`), output (`out.pred`) |
| `g3b_tokens.json` | 3b | real Qwen3 tokenizer, chat template with `enable_thinking=False`: templated text, real token ids, lengths, padding side |
| `g3c_textencoder.safetensors` | 3c | tiny Qwen3 weights, input ids, hidden states at taps 1/2/3, stacked `prompt_embeds` |
| `g3d_vae.safetensors` | 3d | tiny `AutoencoderKLFlux2` weights incl. `bn`, image, latent mean, reconstruction, BN denormalise + unpatchify |
| `g3e_ids.safetensors` | 3a | the pipeline's own ids / pack / unpack / patchify outputs |
| `g3f_scalars.json`, `g3f_scheduler.safetensors` | 3a | `compute_empirical_mu`, `retrieve_timesteps` sigmas and timesteps for six (seq, steps) cases; one Euler step |
| `g3g_glue.safetensors` | 3a/3g | a 2-step denoise on the tiny transformer, then unpack, BN denormalise, unpatchify, decode |
| `g3d_embedders.safetensors` | 3d | MP7-3d, made later by `embedders_goldens.py` with g3a's own weights (asserted, none written): `time_proj`, `temb`, x/context embedder outputs, the three modulations and their chunks, `norm_out`/`proj_out`, RoPE cos/sin for three id layouts (a 4x4 and a 2x8 image grid of equal length, and text), rotated q/k |

sha256 of each file is pinned in `Flux2KleinGlueTests.swift` (`Flux2KleinFixtureIntegrityTests`). Total size is checked there (<= 2 MB).

MP7-3d additions (run twice, byte-identical): `g3d_embedders.safetensors` 47,392 bytes, sha256 `1b7695bf722fc266c5a14d87c92e79f91b168fe331097766e30d5da893d8fc33`;
`embedders_goldens.py` sha256 `e47115ad06ab8cdb9b9aa6fa6a413ce369a110895016eb7c633904929a12d784` (pinned with the file in `Flux2KleinEmbeddersFixtureTests`, not in the
list above). Fixture folder total after this slice: 1745069 bytes (budget 2 MB for the goldens).

MP7-3f addition (run twice, byte-identical): `g3f_dit_stack.safetensors` 160,168 bytes, sha256 `d85451a5f2b5c53f3c27932719c4b7446f5710c4a5f0dbebed8bca2b6e68cc1a`; made by
`dit_stack_goldens.py` (sha256 `561066efb4001a36120abe9b1208dd4c24fc29bb8ae9f532051082ff99952e6a`, 7,415 bytes; both pinned in `Flux2KleinDiTStackFixtureTests`, not in the list above).
A separate tiny `Flux2Transformer2DModel` (3 double + 3 single blocks, heads 2 x head_dim 8 = inner 16, in_channels 8, joint 12, axes (2,2,2,2), fp32, seed 20261031, weights drawn at
N(0, 1/fan_in) so every residual branch has real size): `w.*` weights, `in.*` inputs, `mid.double_N_{txt,img}` and `mid.single_N` after EVERY block, `out.pred`; the same set at a
second sigma under `alt.*` (primary sigma 0.005, alt sigma 0.7). g3a is untouched. Fixture folder total after this slice: 1,913,542 bytes (owner ruling: under 2,000,000).

Regenerate (only with the owner's say-so): in a venv with `requirements.txt` installed, `python3 tiny_goldens.py --out <dir>`.
