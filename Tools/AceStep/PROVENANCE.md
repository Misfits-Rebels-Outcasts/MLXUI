# silence_latent_f32.bin (MC5-3b, R6)

- Resource: `MLXUI/Resources/AceStep/silence_latent_f32.bin`, bundled in both editions. Raw little-endian float32, row-major `[15000, 64]`, 3,840,000 bytes.
- Source: `acestep-v15-turbo/silence_latent.pt` from `ACE-Step/Ace-Step1.5` @ `19671f406d603126926c1b7e2adc169acbcade22` (MIT). 3,841,215 bytes, sha256 `a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b`. `torch.load(weights_only=True)` gives a float32 tensor `[1, 64, 15000]`.
- Layout: the output is the source transposed to `[1, 15000, 64]` (as upstream does at load) with the batch dimension dropped, so element `[frame, channel]` is the source's `[0, channel, frame]`.
- Output sha256: `a7ee13d8902f0c02def49249f05a3e5dd99550ae8aed263299be43329b330e23` (pinned in `MLXUITests/AceStepInstallTests.swift`).
- Re-run (torch from the pinned oracle venv; the script refuses an input with a different sha256):

      python3 -I Tools/AceStep/convert_silence_latent.py <silence_latent.pt> MLXUI/Resources/AceStep/silence_latent_f32.bin

- The app has no `.pt` or pickle reader and `ModelFileSelector` still skips `.pt`; this file is why.
