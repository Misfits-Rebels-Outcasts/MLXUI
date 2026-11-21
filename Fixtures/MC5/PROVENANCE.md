# mc5_convert_tiny.safetensors

Reference data for the ACE-Step weight conversions (`MLXUI/Modules/AceStep/AceStepWeights.swift`, tests in `MLXUITests/AceStepWeightsConversionTests.swift`). Do not edit; regenerate with the script.

- Inputs (`in.<original name>`): REAL tensors from the pinned checkpoint `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22` (DiT: `acestep-v15-turbo/model.safetensors`; VAE: `vae/diffusion_pytorch_model.safetensors`), bf16 upcast exactly to f32. Big tensors are sliced on dim 0 (`proj_in.1.weight` and `proj_out.1.weight` first 16 rows, VAE `decoder.block.0.conv_t1` weight_g/v first 8, `decoder.conv1` weight_g/v first 16). Slicing dim 0 commutes with the weight-norm fuse, which takes its norm per dim-0 row. The slices are recorded in the file's metadata.
- Outputs (`out.dit.<name>`, `out.vae.<name>`): the output of UPSTREAM's own converters, unchanged, `ace-step/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866` (`acestep/models/mlx/dit_convert.convert_decoder_weights` and `vae_convert.convert_vae_weights`), run on those inputs through minimal state_dict holders. torch 2.10.0 (from the file's metadata). f32 throughout.
- Made by `dump_convert_tiny.py` (this folder) in the pinned upstream venv: `python -I dump_convert_tiny.py <Ace-Step1.5 dir> mc5_convert_tiny.safetensors`. 48 tensors, 7.3 MB.
- sha256 of the file: `2094b0789552bc7645b03a2006f42d39df2a90ba3f3dd429ecee0ad51556595a`. The tests do not pin it: they compare every `out.*` against our conversion of the matching `in.*` (max abs diff <= 1e-6).

# mc5_tokens.json

Golden for the ACE-Step text side (`MLXUI/Modules/AceStep/AceStepTextSide.swift`, tests in `MLXUITests/AceStepTextSideTests.swift`). Do not edit.

- Produced by the lead from the real upstream pipeline, tokenizer only: `mc5/scripts/oracle_stages.py a --tokens-only` (`ace-step/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866`, weights `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22`; the pins are inside the file). Upstream's `generate_music` drives the prompt formatting; the tokenizer is `Qwen3-Embedding-0.6B/tokenizer.json` through HF `AutoTokenizer` with truncation to 256 (text) / 2048 (lyric) ids.
- `prompts.<id>.text_str|lyric_str|text_ids|lyric_ids` for the three prompts (inputs: `mc5/scripts/prompts.json`, duration 10 s); `truncation.*` holds a long caption (cut to exactly 256 ids) and a long lyric (cut to exactly 2048).
- Readme T25: the HF tokenizer appends one `<|endoftext|>` (151643) after the template's own literal one, on both sequences; truncation reserves room for it: a prompt over the limit is cut to its first N-1 content ids and that appended EOS is the last id (a long caption is exactly 256 ids ending `.., 448, 9450, 151643`).

# mc5_cond_tiny.safetensors (sha256 9098c72e715d1493db0891e546d1cd7551f1ecdd911d7df01634dd5998ad2d75)

Reference data for the condition encoder (`MLXUI/Modules/AceStep/AceStepConditionEncoder.swift`, tests in `MLXUITests/AceStepConditionEncoderTests.swift`). Do not edit; regenerate with `dump_cond_tiny.py`.

- Made by `dump_cond_tiny.py` (this folder): upstream's REAL remote-code class `AceStepConditionEncoder` from the pinned checkpoint folder (`ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22`, code of `ace-step/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866`) built with a TINY config (hidden 32, 4 heads / 2 kv, head_dim 8, text_hidden 16, timbre_hidden 8, 2 lyric + 2 timbre layers, sliding window 4, layer types [sliding, full]; the config JSON is in the file's metadata), seeded random weights (seed 1234), fp32 CPU, eager attention. torch 2.10.0.
- `w.<name>`: weights under the real file's names minus `encoder.`. Two cases, `valid` and `padded` (padding masks that exercise `pack_sequences`' stable sort), each with inputs `<case>.in.*` and outputs `<case>.out.*` (encoder_hidden_states, encoder_mask, lyric_encoder, text_projector, timbre.0, timbre.1). The tiny timbre input is `[1, 6, 8]` (6 frames), not 750. Masks are stored as uint8.
