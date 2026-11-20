# mc5_convert_tiny.safetensors

Reference data for the ACE-Step weight conversions (`MLXUI/Modules/AceStep/AceStepWeights.swift`, tests in `MLXUITests/AceStepWeightsConversionTests.swift`). Do not edit; regenerate with the script.

- Inputs (`in.<original name>`): REAL tensors from the pinned checkpoint `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22` (DiT: `acestep-v15-turbo/model.safetensors`; VAE: `vae/diffusion_pytorch_model.safetensors`), bf16 upcast exactly to f32. Big tensors are sliced on dim 0 (`proj_in.1.weight` and `proj_out.1.weight` first 16 rows, VAE `decoder.block.0.conv_t1` weight_g/v first 8, `decoder.conv1` weight_g/v first 16). Slicing dim 0 commutes with the weight-norm fuse, which takes its norm per dim-0 row. The slices are recorded in the file's metadata.
- Outputs (`out.dit.<name>`, `out.vae.<name>`): the output of UPSTREAM's own converters, unchanged, `ace-step/ACE-Step-1.5` @ `ca1e85fe9430179831e6bc6be790c332190a3866` (`acestep/models/mlx/dit_convert.convert_decoder_weights` and `vae_convert.convert_vae_weights`), run on those inputs through minimal state_dict holders. torch 2.10.0 (from the file's metadata). f32 throughout.
- Made by `dump_convert_tiny.py` (this folder) in the pinned upstream venv: `python -I dump_convert_tiny.py <Ace-Step1.5 dir> mc5_convert_tiny.safetensors`. 48 tensors, 7.3 MB.
- sha256 of the file: `2094b0789552bc7645b03a2006f42d39df2a90ba3f3dd429ecee0ad51556595a`. The tests do not pin it: they compare every `out.*` against our conversion of the matching `in.*` (max abs diff <= 1e-6).
