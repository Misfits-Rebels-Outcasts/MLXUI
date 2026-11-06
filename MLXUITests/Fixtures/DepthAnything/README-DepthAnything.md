# Depth Anything V2 Small parity fixtures (MC3-3)

Generated **once** (2026-11-06) by `make_depth_golden.py` from the Python reference
`transformers`' `DepthAnythingForDepthEstimation` + `DPTImageProcessor` on the official
`depth-anything/Depth-Anything-V2-Small-hf` checkpoint at revision
`5426e4f0f36572d16453bbda7a8389317b1bef99` (`weights_sha256` in `depth-golden.json`).

**Never edit these to make a change pass**: a mismatch is a `SPEC_QUESTIONS.md` entry, or a bug in
`Modules/DepthAnything`.

```
python3 -m venv --system-site-packages /private/tmp/da-venv
/private/tmp/da-venv/bin/pip install --ignore-installed pillow numpy
/private/tmp/da-venv/bin/python -I make_depth_golden.py \
    /private/tmp/depth-anything MLXUITests/Fixtures/DepthAnything <repo-root>
```

Recorded versions: **transformers 5.12.0, torch 2.7.0, numpy 2.2.6, pillow 13.0.0**.
(`--system-site-packages` reuses the installed torch/transformers; pillow/numpy are installed into
the throwaway venv because the system copies live in the user site, which `python3 -I` hides.)

## Files

All activations are raw little-endian **float32**, layout **NHWC `[1, H, W, C]`** (PyTorch NCHW
transposed so a Swift/MLX reader reshapes directly). `predicted_depth` is `H x W` float32.

| file | what |
|---|---|
| `a_input.rgb` | the synthetic scene, 140 x 196 RGB uint8 row-major (the deterministic input) |
| `a_preprocessed.f32` | `[1, 364, 518, 3]` — the reference's processed tensor for (a) |
| `a_predicted_depth.f32` | `[364, 518]` — the reference's `predicted_depth` for (a) |
| `a_final.png` | 196 x 140 8-bit grayscale, the decision-4 output for (a) |
| `probe_preprocessed.f32` | `[1, 112, 154, 3]` — the same scene through the same reference processor at `size = 112` |
| `probe_feature_{0..3}.f32` | `[1, 89, 384]` — the backbone feature maps after its layer norm (CLS + patch tokens) |
| `probe_reassemble_{0..3}.f32` | the four reassemble outputs (projection + up/down resize) |
| `probe_fusion.f32` | `[1, 64, 88, 64]` — the neck's final fused map (the head's input) |
| `probe_predicted_depth.f32` | `[112, 154]` |
| `b_preprocessed.f32` | `[1, 518, 784, 3]` — the reference's processed tensor for `MLXUI/Resources/Samples/sample-image.png` |
| `b_predicted_depth.f32` | `[518, 784]` |
| `b_final.png` | 1200 x 800 8-bit grayscale, the decision-4 output for (b) |
| `repo-files.json` | the HF file list at the pinned revision (MC3-2) |
| `depth-golden.json` | versions, revision, weights hash, per-file shapes, the preprocessing size table, tolerances |

The `golden.json` name the issue used is `depth-golden.json` here: the test target bundles fixtures
by file name and RIFE already ships `Fixtures/RIFE/golden.json` (two resources with that name fail the
build). The `README.md` name is `README-DepthAnything.md` for the same reason.

## Implementer's calls, pending owner confirmation

- **Layer parity runs on the 112 x 154 probe, not on (a) at full size.** At the model's real
  preprocessed size one fusion map is `208 x 296 x 64` (15.8 MB) and the four feature maps are 5.9 MB;
  storing every stage f32 would put this folder at ~33 MB, far past the issue's ~4 MB guidance. The probe
  is non-square (patch grid 8 x 11, interpolated from the pretrained 37 x 37 grid), so it exercises the
  same position-embedding interpolation and every layer. (a) and (b) keep full-size end-to-end evidence
  (`preprocessed`, `predicted_depth`, final PNG). The folder is ~15 MB.
- **How (b) was fed the same pixels:** `PIL.Image.open`, then `ImageOps.exif_transpose`, then
  `.convert("RGB")`; no other colour transform. `b_preprocessed.f32` is the tensor the reference model
  saw, so the model-parity test is exact-input.
- **Preprocessing tolerance** (normalised units): the Swift resize is Core Graphics `.high`, the reference
  is PIL bicubic, so the tensors are not bit-identical by construction — the differences are localised at
  sharp edges. Measured on (a): median `0`, p99 `0.16`, max `0.47`; the test asserts median ≤ 0.01,
  p99 ≤ 0.25 and max ≤ 0.6.
- **Final PNG rule:** bicubic `F.interpolate(..., align_corners=False)` to the input's pixel size, then
  per-image min–max to 0..255 (a constant prediction is all black). The reference's own
  `post_process_depth_estimation` only does the resize; the min–max quantisation is the app's contract
  from owner decision 4, applied here so the comparison is exact.
