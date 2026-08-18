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

**SMOKE TEST 1:** Run with prompt "A cat walks on the grass, realistic style.", numFrames=3,
numSteps=20, resolution=square 480×480. Check if the video shows any non-noise content
(even vague blobs of the right color are progress). Check the `[VAE-DBG]` std progression.

---

### P4 — Increase denoising steps *(quick test, 5 min)*

Change the default `numSteps` in `WanVideoRunView` from 20 to 50:

```swift
@State private var numSteps = 50
```

20 steps is marginal for flow matching with `flow_shift=3.0`. The sigma schedule spends most
steps at high noise (σ near 1.0) and few steps refining fine detail. 50 steps gives more
resolution at the low-noise end where video structure emerges.

**SMOKE TEST 2:** Re-run with 50 steps. Check if image quality improves.

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

**SMOKE TEST 3:** Run with numFrames=17, numSteps=50. Verify no OOM. Check if video has structure.

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

---

## Removing diagnostics when video works

Once the video produces recognizable content, remove all `[WAN-DBG]`, `[DIT-DBG]`, `[VAE-DBG]`,
and `[FFN-DBG]` print statements. They are in:
- `WanVideoEngine.swift`: `loadAndEncodeText`, `loadAndDenoise`, `generateAll`
- `WanVideoDiT.swift`: `callAsFunction` (wrapped in `if dbgThisCall { … }` guard)
- `WanVideoVAE.swift`: (none yet — implementer adds in P3, then removes)
- Also remove `static var dbgDone = false` from `WanDiT` and the `WanDiT.dbgDone` reset
  in the engine if you add one.

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

- DiT: 30 layers, 12 heads, dim=1536, ffnDim=8960, headDim=128
- adaLN-zero: each block has own `time_embedder: Linear(1536, 6×1536)`; shared `norm2`
- Cross-attn: T5 sequence projected via `textProj: Linear(4096, 1536)` (simple linear)
- Time conditioning: pooled T5 → `textEmbed` MLP (4096→1536) → added to sinusoidal time emb
- Sampler: flow matching, `flow_shift=3.0`, `num_train_timesteps=1000`
- CFG scale: 5.0 (unconditional = empty string negative prompt)
- VAE: 3D causal conv, latent_channels=16, 4 upblocks (2× temporal, 2× temporal+spatial, 1× spatial, 0×)
- VAE normalization: per-channel latents_mean and latents_std from `vae/config.json`

---

*Written by Claude Sonnet 4.6 on 2026-08-18. See conversation transcript for full diagnostic logs.*
