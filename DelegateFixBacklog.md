# DelegateFixBacklog — WAN 2.1 Video Noise Debug

**Branch:** `add-wan21`  
**Fix branch to work on:** `fix-wan21-noise` (create from add-wan21)  
**Goal:** Get WAN 2.1 T2V-1.3B to produce a recognizable video (not noise) from a text prompt on macOS.

---

## What this branch builds

WAN 2.1 T2V-1.3B is a text-to-video model from Wan-AI. The branch adds:

- Full Swift/MLX pipeline: T5-XXL text encoder → 30-layer DiT backbone → 3D causal VAE decoder
- Flow-matching Euler sampler (diffusers `FlowMatchEulerDiscreteScheduler` compatible)
- Stage-scoped memory management to avoid OOM on 16 GB Macs
- SwiftUI run view with resolution presets, step slider, frame count, AVAssetWriter MP4 export
- Catalog entry, installer support, model-type wiring

### Files in `MLXUI/Modules/WanVideo/`

| File | Role |
|---|---|
| `WanVideoDiT.swift` | 30-layer/12-head/dim=1536 DiT backbone, adaLN-zero, 3D RoPE, cross-attn |
| `WanVideoEngine.swift` | Orchestrator: T5 encode → DiT denoise → VAE decode; stage helpers |
| `WanVideoRoPE.swift` | 3D RoPE — temporal + spatial positional encoding |
| `WanVideoSampler.swift` | Euler ODE step + sigma schedule with flow_shift=3.0 |
| `WanVideoVAE.swift` | 3D causal VAE decoder: convIn, midBlock (attn+resnets), 4 upBlocks, convOut |
| `WanVideoRunView.swift` | SwiftUI UI, AVPlayer video playback, MP4 save |

---

## Bugs found and fixed (already in this branch)

These are confirmed fixed and committed. Do not re-introduce them.

### B1 — Missing per-block `time_embedder` weights *(critical, was causing 54× latent explosion)*

Each of the 30 `WanDiTBlock`s has its own `time_embedder: Linear(1536, 6×1536)` that expands
the shared time vector into 6 adaLN modulation signals. The sanitize function had no mapping
for `blocks.{i}.time_embedder.weight/bias`, so all 30 were silently dropped and stayed
randomly initialized.

**Symptom:** `h after all blocks std=52.8` (was 54×), `scaleMSA mean=+0.58` (adaLN bias huge).  
**Fix:** Added `WanDiTBlock.timeEmbedder: Linear` field + sanitize mapping; removed the old
shared `WanTimeEmbed.proj` (which had no checkpoint key anyway).  
**After fix:** `h after all blocks std=1.594`, `scaleMSA mean=-0.108`. ✓

### B2 — Wrong cross-attention T5 projection *(was using MLP instead of Linear)*

The DiT has two separate T5 projections:
- `condition_embedder.text_proj` — simple `Linear(4096, 1536)` for cross-attn keys/values
- `condition_embedder.text_embedder` — two-layer MLP for pooled T5 → time conditioning

Original code fed `textEmbed(T5)` (the MLP output) into cross-attn instead of `textProj(T5)`.
This is a dimension+semantic mismatch.

**Fix:** Added `textProj: Linear` field + sanitize mapping. Cross-attn now uses `textProj(textCond)`.

### B3 — OOM (T5 not released before DiT loads)

T5-XXL is ~5.5 GB at 4-bit. When all stage locals lived in the same function scope,
`Memory.clearCache()` was a no-op while the T5 object was still referenced. DiT then tried
to allocate on top of live T5 memory → OOM crash.

**Fix:** Extracted T5 encode and DiT denoise into `private static func loadAndEncodeText(…)`
and `private static func loadAndDenoise(…)`. Stage model + raw weight dicts go out of scope
when each helper returns, so `Memory.clearCache()` actually frees GPU memory.

### B4 — AVAssetWriter "Cannot create file" (NSURLErrorDomain -3000)

App sandbox does not allow `FileManager.default.temporaryDirectory` for media writing.

**Fix:** Changed to `FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)`.

---

## Current symptom

After the above fixes, the DiT is working correctly:

```
[WAN-DBG] textCond: mean=-0.000 std=0.091  shape=[1, 13, 4096]
[WAN-DBG] init latent: mean=-0.002 std=1.003
[DIT-DBG] block0 temb: mean=0.0005 std=0.0621          ← correct small signal at σ=1
[DIT-DBG] block0 scaleMSA (full): mean=-0.1076 std=0.0898  ← near-zero, correct adaLN-zero
[DIT-DBG] h after all blocks: std=1.5940               ← no explosion
[DIT-DBG] projOut(h) std=1.6506                        ← good velocity magnitude
[WAN-DBG] step0: σ=1.0000 u_std=1.651 c_std=1.662 diff_std=0.458 guided_std=2.666
[WAN-DBG] final latent: mean=-0.168 std=1.132
[WAN-DBG] decoded: mean=0.132 std=0.376 min=-1.000 max=1.000
```

`u_std` improved 24× (0.076 → 1.651). The sampler and DiT are producing reasonable
velocity magnitudes. CFG is adding meaningful guidance (guided_std > u_std).

**But the video output is still random noise.** The VAE decoder is the primary suspect.

---

## ✅ RESOLVED — noise investigation outcome (2026-08-18)

**Root cause:** the 20-step denoising schedule was amplifying CFG at low sigma. With
`flow_shift=3.0` the sigma schedule spends most steps at high noise and only a handful refining
the low-noise end where structure emerges; `u + 5·(c−u)` guidance applied over those few coarse
steps pushed the latent toward collapse (std divergence, then a std≈0.3 collapsed decode).

**Fix:** denoise with **50 steps** (matching the reference diffusers setup for WAN 2.1) instead
of 20. Final config:

- `WanVideoEngine.generate` and `generateAll` default `numSteps: 50`.
- `WanVideoRunView` default `numSteps = 50`, slider `1...50` step 1 (1-step sanity check).
- `cfgScale = 5.0` (reverted after the temporary 1.0 A/B test).
- `numFrames = 3` default for the diagnostic runs (latT=1); frame picker still offers 17/81.

**Diagnostics cleanup:** the `[VEL-DBG]`, `[STEP-DBG]`, `[DIT-DBG]`, `[WAN-DBG]`, `[FFN-DBG]`,
and `[VAE-DBG]` prints are now gated behind `#if DEBUG` (still available in Debug builds; silent
in Release). The DiT/VAE stage instrumentation stays available for future model bring-ups.

**Remaining (accepted) risk:** if a future run still shows blank content at 50 steps / latT≥5,
re-open P8 (DiT conditioning was corrected, but low-σ velocity was not re-validated end-to-end).

---

## ONGOING — dynamic CFG for guided-velocity divergence (2026-08-19)

**Root cause:** `diff_std` grows throughout denoising and can spike anywhere in σ=0.25–0.71
depending on seed. CFG×5 amplifies the spike into net energy addition per step; with
`flow_shift=3.0` Euler step sizes Δσ grow from 0.007 (step 0) to 0.06 (step 48), so
high guided_std at large-Δσ steps is especially damaging. Final latent std overshoots
the target ~1.0 clean-video distribution (e.g. 1.808 run A, 1.584 run B).

**Iteration 1 (threshold=0.5, 2026-08-19):** ramp from full CFG at σ≥0.5 to floor 1.5.
Run B: "some patterns but still like noise", final std=1.584. The spike in run B peaked at
step 36 (σ=0.5209) with cfg still at 5.0 — threshold was too late.

**Iteration 2 (threshold=0.9, 2026-08-19):** ramp now starts at σ=0.9 (≈ step 13).

```swift
let effectiveCfg = sigmaFrom >= 0.9
    ? cfgScale
    : max(1.5, cfgScale * (sigmaFrom / 0.9))
let noisePred = u + (c - u) * effectiveCfg
```

- σ=0.9 (step 13): cfg≈5.0 (just entering ramp)
- σ=0.7 (step 27): cfg≈3.89 → estimated guided_std ~2.0 (was 2.57)
- σ=0.52 (step 36): cfg≈2.89 → estimated guided_std ~2.1 (was 3.09)
- σ≤0.27: cfg=1.5 floor

Expected: mid-σ energy addition cut ~40%; final latent std should land ~1.0–1.3.
`[VEL-DBG]` print includes `cfg=X.XX` to confirm the ramp.

**Applied 2026-08-19.** Run latT=1, 50 steps, 480×480 and report final latent std
and decoded mean/std to verify improvement.

---

## Priority investigation items

Work through these in order. Stop after each SMOKE TEST and check the video before continuing.

---

### P1 — Verify FFN: GEGLU vs plain GELU *(diagnose first, 15 min)*

In diffusers WAN 2.1, the FFN uses GEGLU (gated activation). The checkpoint key is
`ffn.net.0.proj.weight`. If GEGLU: weight shape `[17920, 1536]` (2×ffnDim). If plain GELU:
`[8960, 1536]`.

Our `WanDiTFFN.fc1 = Linear(1536, 8960)` initializes weight at `[8960, 1536]`. After update
with `verify:.none`, fc1.weight takes whatever shape the checkpoint has. If GEGLU and
`fc1.weight=[17920,1536]`, then `fc1(x)=[B,S,17920]`, then `fc2` crashes (inner dim mismatch).
Since the model runs without crashing, fc1 is probably `[8960, 1536]`. But **verify this.**

**Action:** In `WanVideoEngine.loadAndDenoise`, after loading `ditRaw`, add ONE diagnostic print:

```swift
if let w = ditRaw["blocks.0.ffn.net.0.proj.weight"] {
    print("[FFN-DBG] blocks.0 ffn.net.0.proj.weight shape: \(w.shape)")
}
if let w = ditRaw["blocks.0.ffn.net.2.weight"] {
    print("[FFN-DBG] blocks.0 ffn.net.2.weight shape: \(w.shape)")
}
```

- If `net.0.proj.weight` is `[17920, 1536]`: **FFN must be fixed to GEGLU** (see fix below).
- If `[8960, 1536]`: FFN is correct, skip to P2.

**GEGLU fix** (only apply if shape is [17920, 1536]):

In `WanDiTFFN`:
```swift
// Change init:
self._fc1.wrappedValue = Linear(config.dim, config.ffnDim * 2)   // 1536 → 17920

// Change callAsFunction:
func callAsFunction(_ x: MLXArray) -> MLXArray {
    let proj = fc1(x)                          // [B, S, 17920]
    let half = proj.dim(-1) / 2
    let gate = proj[.ellipsis, 0 ..< half]     // [B, S, 8960]
    let val  = proj[.ellipsis, half...]        // [B, S, 8960]
    return fc2(gelu(gate) * val)
}
```

---

### P2 — Verify VAE channel progression *(diagnose, ~20 min)*

Our `WanVAEDecoder` init declares:

```swift
upBlocks[0]: inCh=384, outCh=384   upsampler: 384→192
upBlocks[1]: inCh=192, outCh=384   upsampler: 384→192   ← increases channels, unusual
upBlocks[2]: inCh=192, outCh=192   upsampler: 192→96
upBlocks[3]: inCh=96,  outCh=96    no upsampler
```

`upBlocks[1]` having `outCh=384` when it receives `inCh=192` is suspicious for a decoder
(channels usually decrease as spatial resolution increases). If the real checkpoint has
`up_blocks.1.resnets.*.conv1.weight: [192, *, *, *, 192]` (not 384), then weight shapes
load into mismatched modules — producing numerical garbage silently.

**Action:** In `WanVideoEngine`, after `let vaeRaw = try await loadShards(...)`, add:

```swift
let keysOfInterest = ["up_blocks.0.resnets.0.conv1.weight",
                      "up_blocks.1.resnets.0.conv1.weight",
                      "up_blocks.2.resnets.0.conv1.weight",
                      "up_blocks.3.resnets.0.conv1.weight"]
for k in keysOfInterest {
    if let w = vaeRaw["decoder." + k] ?? vaeRaw[k] { print("[VAE-DBG] \(k): \(w.shape)") }
}
```

- If `up_blocks.1.resnets.0.conv1.weight` has shape `[192, 3, 3, 3, 192]` (not 384 out-ch):
  **upBlocks[1] outCh must be changed from `inner=384` to `b*2=192`.**

  Change in `WanVAEDecoder.init`:
  ```swift
  // Was:
  WanVAEUpBlock(inCh: b * 2, outCh: inner, resnets: 3, upsample: .timeAndSpace(inCh: inner, outCh: b * 2)),
  // Fix:
  WanVAEUpBlock(inCh: b * 2, outCh: b * 2, resnets: 3, upsample: .timeAndSpace(inCh: b * 2, outCh: b)),
  ```
  And adjust subsequent block inCh accordingly.

**Also check for dropped keys.** After sanitize, add:

```swift
var droppedKeys: [String] = []
for key in vaeRaw.keys {
    guard !key.hasPrefix("encoder.") && !key.hasPrefix("quant_") else { continue }
    let k = key.hasPrefix("decoder.") ? String(key.dropFirst("decoder.".count)) : key
    // Quick check: if not in vaeParams, it was dropped
    let mapped = WanVAEDecoder.sanitize([key: vaeRaw[key]!])
    if mapped.isEmpty { droppedKeys.append(key) }
}
print("[VAE-DBG] Dropped \(droppedKeys.count) decoder keys: \(droppedKeys.prefix(10))")
```

Any dropped keys = modules left at random init = likely noise source.

---

### P3 — Add VAE stage diagnostics *(instrument, run, check, ~30 min)*

If P1 and P2 don't fix the output, instrument `WanVAEDecoder.callAsFunction` to track std
at each stage:

```swift
func callAsFunction(_ z: MLXArray) -> MLXArray {
    func s(_ a: MLXArray, _ tag: String) -> MLXArray {
        let f = a.asType(.float32); eval(f)
        let std = sqrt(pow(f - f.mean(), 2).mean()).item(Float.self)
        print("[VAE-DBG] \(tag): std=\(String(format:"%.4f", std)) shape=\(a.shape)")
        return a
    }
    var h = s(convIn(z), "convIn")
    h = s(midBlock(h), "midBlock")
    for (i, block) in upBlocks.enumerated() { h = s(block(h), "upBlock[\(i)]") }
    h = s(silu(normOut(h)), "normOut")
    let raw = s(convOut(h), "convOut")
    return minimum(maximum(raw, MLXArray(-1.0)), MLXArray(1.0))
}
```

Expected healthy std progression (approximate): `convIn~0.5, midBlock~0.5, upBlocks~0.3–0.5, convOut~0.3`.
If std explodes (>10) or collapses (<0.01) at a stage, that stage has a weight bug.

> **Implemented 2026-08-18:** per-stage `[VAE-DBG]` std prints added to
> `WanVAEDecoder.callAsFunction` (convIn → midBlock → upBlock[0..3] → normOut → convOut),
> exactly as specified above. Also verified `vatMeanF`/`vatStdF` against `vae/config.json`:
> **both match the repo's 16 values exactly** (this machine had no installed WAN model, so the
> authoritative `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/config.json` was used — the installed copy
> is the same file the installer downloads). Denormalization constants are correct.
>
> **Per-step latent tracker (reviewer, 2026-08-18):** a `[STEP-DBG]` line is now printed after
> every euler step in `loadAndDenoise` (step number, σ_from→σ_to, latent mean/std). Healthy
> behavior: latent stays near std≈1.0 and shrinks slowly toward the low-noise end (std→0) as
> denoising converges; a sudden collapse (std→~0) or blow-up (std>5) isolates the faulty step.
> **1-step sanity check:** call `loadAndDenoise` with `numSteps=1` (temporarily, or via a tiny
> harness) — one `[STEP-DBG]` line confirms the single euler step moves the latent in the
> predicted direction and by the right magnitude (`latent = x + v·(σ_to−σ_from)`).

**SMOKE TEST 1:** Run with prompt "A cat walks on the grass, realistic style.", numFrames=3,
numSteps=50 (P4 default), resolution=square 480×480. Check if the video shows any non-noise
content (even vague blobs of the right color are progress). Check the `[VAE-DBG]` std progression.

---

### P4 — Increase denoising steps *(quick test, 5 min)*

Change the default `numSteps` in `WanVideoRunView` from 20 to 50:

```swift
@State private var numSteps = 50
```

20 steps is marginal for flow matching with `flow_shift=3.0`. The sigma schedule spends most
steps at high noise (σ near 1.0) and few steps refining fine detail. 50 steps gives more
resolution at the low-noise end where video structure emerges.

> **Implemented 2026-08-18:** `WanVideoRunView` default is now `@State private var numSteps = 50`,
> and the slider allows `1...50` step 1 (so the 1-step sanity check is selectable in the UI).
> `WanSampler.buildSigmas` handles `numSteps==1` (avoids division by `numSteps-1` → NaN).

**SMOKE TEST 2:** Re-run with 50 steps. Check if image quality improves.

> Note: the single-frame `WanVideoEngine.generate` entry point still hardcodes `numSteps: 20`;
> it is a separate preview/stage path, untouched by P4.

---

### P5 — Test with more temporal frames *(medium effort, 15 min)*

With `numFrames=3 → latT=1`, the DiT processes a single 60×60 latent frame (900 tokens).
WAN 2.1 was trained primarily with latT≥5 (17+ video frames). A single latent temporal
frame may not have enough context for the model to produce coherent content.

Try setting the default frame count to 17 in `WanVideoRunView`:

```swift
@State private var numFrames = 17
```

This gives `latT = max(1, (17+3)/4) = 5`. The DiT sees 5×30×30 = 4500 tokens instead of 900.

Note: 17 frames requires more memory. Only do this after P3 memory check passes.

> **Status 2026-08-18:** P5 (numFrames=17, latT=5) was tried but **reverted for the current
> diagnostic runs** — `WanVideoRunView` default is back to `numFrames = 3` (latT=1) to keep the
> per-step sanity checks fast and memory-light. The frame picker still offers 17/81. Re-apply
> `numFrames = 17` once the pipeline produces recognizable content.

**SMOKE TEST 3:** Run with numFrames=17, numSteps=50. Verify no OOM. Check if video has structure.
If 50 steps / latT=5 still produces blank content with no spatial structure, that points to a
subtler DiT bug (see P8 notes) rather than insufficient denoising.

---

## CFG scale=1.0 test *(test run done, 2026-08-18 — reverted)*

`WanVideoEngine.cfgScale` was temporarily set to 1.0, then **reverted to 5.0**. Run 20 steps at default settings and
read the `[VEL-DBG]`/`[STEP-DBG]` lines:

- **If latent std keeps decreasing (no divergence):** CFG amplification (`u + 5·(c−u)`) is the
  root cause → keep diagnosing guidance, and/or re-tune scale.
- **If latent std still diverges at σ≈0.8:** the DiT velocity field itself is broken at lower σ,
  independent of guidance → investigate the P8.1–P8.4 fixes / velocity at low σ.

**Result:** reverted to `cfgScale = 5.0`; production run config is `numSteps = 50` (P4 default).

---

### P6 — Verify uncond tokenization *(sanity check, 10 min)*

Empty-string negative prompt tokenizes to 2 tokens (`[BOS, EOS]` or similar). In diffusers
WAN 2.1, the unconditional uses an empty string. We're doing the same. But `pad: false` in
our tokenizer means we don't pad to 512.

Check: does T5-XXL handle a 2-token sequence correctly, or does it need at minimum N tokens?

**Diagnostic:** Print `negCond` shape in the run. Already printing: `shape=[1, 2, 4096]`.
This is 2 tokens → should be fine (T5 handles variable length).

If suspicious, try passing a single space `" "` as the negative prompt via the UI toggle
to see if it changes the output.

---

### P7 — Norm1 / cross-attn norm check *(low priority, only if P1–P5 don't help)*

In diffusers WAN 2.1 `WanTransformerBlock`, check if there are separate `norm1` and `norm2`
weight keys (separate pre-norms for self-attn vs FFN). Our sanitize only maps `norm2.*`.

**Diagnostic:** After loading `ditRaw`, check for `norm1` keys:

```swift
let norm1Keys = ditRaw.keys.filter { $0.contains("norm1") }
print("[DIT-DBG] norm1 keys in checkpoint: \(norm1Keys.prefix(5))")
```

If `blocks.0.norm1.weight` exists AND differs from `blocks.0.norm2.weight`, add a second
`norm1: LayerNorm` to `WanDiTBlock` and use it for the self-attn pre-norm path.

> **Resolved by checkpoint inspection:** there is no `norm1`/`norm3` weight in the checkpoint
> (they are affine=False). Only `norm2` has weights and it is the **cross-attn** pre-norm.
> The bug is that our code applies `norm2` to the self-attn and FFN pre-norms and skips the
> cross-attn pre-norm. See **P8.2**.

---

## P1 & P2 — Implemented (2026-08-18)

Verified P1 and P2 directly against the actual checkpoint (safetensors headers fetched from
`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`; tensor names + shapes are in the file headers, no full
download needed). Both diagnostics were added to `WanVideoEngine`; neither structural fix that
P1/P2 hypothesized was triggered, **but the P2 dropped-key check exposed one real bug that was fixed**.

### P1 — FFN is plain GELU, NOT GEGLU → no change needed

`blocks.0.ffn.net.0.proj.weight` = `[8960, 1536]`, `blocks.0.ffn.net.2.weight` = `[1536, 8960]`
(confirmed across all 30 blocks). diffusers `FeedForward(dim, inner_dim=8960, activation_fn="gelu-approximate")`.
Our `WanDiTFFN` (`fc1(1536→8960)`, `gelu`, `fc2(8960→1536)`) matches. Do **not** switch to GEGLU.
Note: diffusers uses tanh-approximate GELU here — `WanDiTFFN` calls `gelu()` (exact). Minor numerical
difference, not a noise source. Diagnostic wired into `loadAndDenoise` → `[FFN-DBG]`.

### P2 — VAE channel progression matches the checkpoint → no change needed

Actual decoder channels (from checkpoint, PyTorch layout `[O,I,kD,kH,kW]`):

| upblock | conv1 weight | meaning | our code |
|---|---|---|---|
| `up_blocks.0.resnets.0` | `[384,384,3,3,3]` | 384→384 | ✓ |
| `up_blocks.1.resnets.0` | `[384,192,3,3,3]` | **192→384** (channels increase — unusual but correct) | ✓ |
| `up_blocks.2.resnets.0` | `[192,192,3,3,3]` | 192→192 | ✓ |
| `up_blocks.3.resnets.0` | `[96,96,3,3,3]` | 96→96 | ✓ |

`up_blocks.1` genuinely goes 192→384 (then 384→192 through the upsampler), so the code's
`WanVAEUpBlock(inCh: b*2, outCh: inner, …)` is correct. Upsamplers also match:
`upsamplers.0.time_conv` `[768,384,3,1,1]` (384→768, temporal ×2) + `resample.1` `[192,384,3,3]`
(spatial 384→192); block 2 is `spaceOnly` (no `time_conv`); block 3 has no upsampler.
Temporal pixel-shuffle ordering, causal padding, and nearest-exact ×2 all match diffusers
`WanResample`/`WanCausalConv3d`/`WanUpBlock` exactly.

**Dropped-key check: 0 decoder keys are dropped.** `WanVAEDecoder.sanitize` covers every
`decoder.*` key. Diagnostics wired into `loadAndDecode` → `[VAE-DBG]`.

### P2 fix that WAS needed — `post_quant_conv` was being dropped

diffusers `AutoencoderKLWan._decode` runs the latent through `post_quant_conv` (a trained
1×1×1 causal conv, `[16,16,1,1,1]` + bias) **before** the decoder. Our sanitize's guard
explicitly skips `post_quant_conv.*` (it lives outside `decoder.`), and the engine never applied
it — so every decoded frame was produced from an unprocessed latent. **Fixed** in
`WanVideoEngine`: `loadPostQuantConv(vaeRaw)` now builds the 1×1×1 conv from the raw top-level
keys and `loadAndDecode` applies it as `latentVAE = postQuant(latentVAE)` before `vae(latentVAE)`
(the P2 `[VAE-DBG] Dropped 2 decoder keys` line now flags `post_quant_conv.weight/bias` on purpose).

---

## P8 — DiT conditioning divergences *(implemented 2026-08-18)*

Verifying P1/P2 against the checkpoint surfaced **unambiguous DiT bugs** of the same class as B1
(silently dropped weights → random init → wrong output). Evidence: checkpoint headers + diffusers
v0.33.0 `transformer_wan.py` + original `Wan-Video/Wan2.1` `wan/modules/model.py`.
All four fixes (P8.1–P8.4) are now implemented in `WanVideoDiT.swift`. They should be validated
with SMOKE TEST 1 before spending time on P3–P7.

**P8.1 — ✅ FIXED: `condition_embedder.time_proj` ([9216, 1536] = 6×1536) is now loaded and used**

Added shared `WanDiT.timeProj: Linear(1536, 6·1536)`, mapped `condition_embedder.time_proj.*`
in sanitize, dropped the (nonexistent) per-block `time_embedder.*` mapping, removed
`WanDiTBlock.timeEmbedder`. Forward: `mods = timeProj(silu(timeVec)).reshaped([B, 6, dim])`
computed once; each block computes `mod = shiftTable + mods` then chunks 6 — matching
`(scale_shift_table + time_proj(silu(temb))).chunk(6)`.

**P8.2 — ✅ FIXED: norm routing is now norm1/norm2/norm3**

`WanDiTBlock` now has `norm1`/`norm3` = `LayerNorm(affine: false)` (self-attn & FFN pre-norms,
no weights in checkpoint) and `norm2` = affine `LayerNorm` loaded from `blocks.{i}.norm2.*`,
applied as the **cross-attn** pre-norm: `h = h + crossAttn(norm2(h), cross)`. (Sanitize mapping
renamed `norm2.*` → `norm2.*`.)

**P8.3 — ✅ FIXED: cross-attn uses `textEmbed` (the MLP) on the full T5 sequence**

`cross = textEmbed(textCond)` replaces `textProj(textCond)`. The pooled-T5→time addition
(`textVec`) is removed; `WanTimeEmbed` no longer takes a `textVec`. `condition_embedder.text_proj`
mapping removed — that `Linear` is an unused leftover key in the checkpoint and is now dropped
by sanitize (correct).

**P8.4 — ✅ FIXED: output modulation now includes `+ temb`**

```swift
let outMod = outShiftTable.reshaped([-1, c.dim]).expandedDimensions(axis: 0)
             + timeVec.expandedDimensions(axis: 1)      // [B, 2, dim]
let shift = outMod[0..., 0, 0...]
let scale = outMod[0..., 1, 0...]
h = outNorm(h) * (1 + scale) + shift
```

`outNorm` is now `LayerNorm(affine: false)` (the checkpoint has no `norm_out` weights).

**P8.5 — not fixed (cosmetic):** the FFN uses exact `gelu` vs diffusers `gelu-approximate` (tanh).

Also fixed: `WanVideoTests.ditOutputShape` asserted the latent-resolution shape `[1,2,4,4,4]`
but the DiT correctly unpatchifies to the input resolution `[1,2,8,8,4]` (stale since the
"fix DiT output unpatch" commit). Test updated; all `WanVideoTests` pass.

---

## Diagnostics — gated behind `#if DEBUG` (2026-08-18)

All `[WAN-DBG]`, `[DIT-DBG]`, `[VAE-DBG]`, `[FFN-DBG]`, `[VEL-DBG]`, and `[STEP-DBG]` prints are
**now wrapped in `#if DEBUG`** — they still print in Debug builds, and are compiled out of
Release. Locations, if you ever need to remove them entirely:
- `WanVideoEngine.swift`: `loadAndEncodeText`, `loadAndDenoise` (incl. the `[VEL-DBG]`
  per-step velocity log and the `[STEP-DBG]` per-step latent tracker), `loadAndDecode`,
  and the helper funcs `printFFNShapeDiagnostics`, `printVAEChannelDiagnostics`,
  `printDroppedVAEKeys` (the `[VAE-DBG] post_quant_conv not found` print in
  `loadPostQuantConv` too). Keep `loadPostQuantConv` itself — it is a required decode step.
- `WanVideoDiT.swift`: the two `[DIT-DBG]` blocks in `callAsFunction` + `static var dbgDone`
  (all gated).
- `WanVideoVAE.swift`: the per-stage `[VAE-DBG]` std tracker in `WanVAEDecoder.callAsFunction` (added in P3).

---

## What is confirmed working / do not change

- **Sampler** (`WanVideoSampler.swift`): sigma schedule and Euler step are correct.
  `eulerStep: x + noisePred * (sigmaTo - sigmaFrom)` is the right sign.
- **RoPE** (`WanVideoRoPE.swift`): 3D RoPE with interleaved convention verified.
- **T5 encoding**: textCond std=0.091, negCond std=0.058 — reasonable T5 output range.
- **DiT velocity**: u_std=1.651 (was 0.076 before fixes), 24× improvement — model is running.
- **Memory management**: T5 + DiT scope-released between stages. Do not merge these back.
- **MP4 export**: AVAssetWriter + cachesDirectory works. Do not change path back to tempDirectory.
- **`pad: false` tokenization**: Padding to 512 collapses textVec to near-zero mean and
  breaks all adaLN gate modulation. Must stay `pad: false`.

---

## Architecture reference (WAN 2.1 T2V-1.3B)

> Corrected 2026-08-18 from the actual checkpoint + diffusers v0.33.0 / original Wan-AI code.
> The previous "per-block time_embedder / textProj cross-attn / pooled-T5 time conditioning" lines
> were wrong — see P8. The Swift DiT does NOT yet match this corrected reference.

- DiT: 30 layers, 12 heads, dim=1536, ffnDim=8960, headDim=128
- adaLN-zero: **shared** `condition_embedder.time_proj: Linear(1536, 6×1536)`; per block
  `mod = scale_shift_table + time_proj(silu(temb))`, chunked into
  [shiftMSA, scaleMSA, gateMSA, shiftMLP, scaleMLP, gateMLP]. No per-block time_embedder exists.
- Norms: self-attn pre-norm = affine=False LayerNorm; cross-attn pre-norm = `norm2` (affine,
  has weights); FFN pre-norm = affine=False LayerNorm.
- Cross-attn: T5 sequence projected by `text_embedder` (2-layer GELU-tanh MLP: 4096→1536→1536).
  `text_proj` (`Linear(4096, 1536)`) is an unused leftover key — do not use it.
- Time conditioning: sinusoidal → `time_embedder` (fc1→silu→fc2) → `temb`; `temb` (base, [B,1536])
  is added to the top-level `scale_shift_table` for the output scale/shift. No text in time path.
- FFN: plain GELU (tanh-approx), `[8960, 1536]`, NOT GEGLU.
- Sampler: flow matching, `flow_shift=3.0`, `num_train_timesteps=1000`
- CFG scale: 5.0 (unconditional = empty string negative prompt)
- VAE: 3D causal conv, latent_channels=16, 4 upblocks
  (384→384 upsample3d, 192→384 upsample3d, 192→192 upsample2d, 96→96 none);
  decode = `post_quant_conv` (1×1×1) → decoder → clamp[-1,1].
- VAE normalization: `latents = latent*std + mean` from `vae/config.json`, applied before
  `post_quant_conv`.

---

*Written by Claude Sonnet 4.6 on 2026-08-18. See conversation transcript for full diagnostic logs.*
