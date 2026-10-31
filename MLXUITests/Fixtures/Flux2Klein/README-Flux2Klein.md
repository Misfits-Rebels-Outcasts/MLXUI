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

sha256 of each file is pinned in `Flux2KleinGlueTests.swift` (`Flux2KleinFixtureIntegrityTests`). Total size is checked there (<= 2 MB).

Regenerate (only with the owner's say-so): in a venv with `requirements.txt` installed, `python3 tiny_goldens.py --out <dir>`.
