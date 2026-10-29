# RIFE parity fixtures (RIFE-3b)

Generated **once** on 2026-10-28 by `make_rife_golden.py` from the Python reference `xocialize/rife-mlx` (MIT; `mlx` 0.31.2) on
`mlx-community/RIFE-4.25` (`model.safetensors` sha256 in `golden.json`). **Never edit these to make a change pass**: a
mismatch is a `SPEC_QUESTIONS.md` entry, or a bug in `Modules/RIFE`.

- `frame0.u8`, `frame1.u8`: a synthetic pair, 96 x 130 RGB uint8 (H x W x 3, row-major), a textured scene whose disc moves (+6, +3) px.
  The width needs padding (130 -> 192), so the pad-and-crop path is covered.
- `t050_s100.f32`, `t025_s100.f32`, `t050_s050.f32`: the reference's output for timestep 0.5 / 0.25 and pyramid scale 1.0 / 1.0 / 0.5,
  float32 little-endian, H x W x 3, **before** 8-bit quantisation.
- `golden.json`: sizes, seed, per-case statistics, weights hash.

Regenerate (only with the owner's say-so): `python3 -I make_rife_golden.py <weights dir> <this dir> <rife-mlx clone>`.
