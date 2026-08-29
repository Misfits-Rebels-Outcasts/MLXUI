// MLX forward-pass tests — commented out (slow; run manually with ⌘U)
/*
import Testing
import Foundation
import MLX
import MLXRandom
import MLXNN
@testable import MLXUI

/// SD-ENG2 — VAE decoder shape test + the PyTorch→MLX conv-weight sanitize both SDXL conv
/// modules need. Serialized like `DeepSeekOCRMLXTests` (MLX default stream isn't parallel-safe).
@Suite(.serialized)
struct SDXLTurboVAETests {

    /// `[1, 4, 64, 64]` latent → `[1, 3, 512, 512]` (8× upscale through 3 upsample stages).
    /// Unloaded (random) weights — plumbing check, not numerics.
    @Test func vaeDecoderUpscalesByEight() {
        let vae = SDVAE()
        let latent = MLXRandom.normal([1, 4, 64, 64])
        let out = vae.decode(latent)
        eval(out)
        #expect(out.shape == [1, 3, 512, 512])
    }

    /// PyTorch `(out,in,kH,kW)` conv weights transpose to MLX `(out,kH,kW,in)`. 1×1 `post_quant_conv`
    /// and 3×3 decoder convs both hit the rule.
    @Test func sanitizeTransposesPyTorchConvWeights() {
        let cleaned = SDWeights.sanitize([
            "post_quant_conv.weight": MLXArray.zeros([4, 4, 1, 1]),
            "decoder.conv_in.weight": MLXArray.zeros([512, 4, 3, 3]),
            "decoder.up_blocks.2.resnets.0.conv_shortcut.weight": MLXArray.zeros([256, 512, 1, 1]),
        ])
        #expect(cleaned["post_quant_conv.weight"]?.shape == [4, 1, 1, 4])
        #expect(cleaned["decoder.conv_in.weight"]?.shape == [512, 3, 3, 4])
        #expect(cleaned["decoder.up_blocks.2.resnets.0.conv_shortcut.weight"]?.shape == [256, 1, 1, 512])
    }

    /// Linear/Embedding/LayerNorm weights (≤2-D) and already-MLX 4-D weights pass through untouched.
    @Test func sanitizeLeavesNonConvWeightsAlone() {
        let linear = MLXArray.zeros([512, 512])
        let mlxConv = MLXArray.zeros([512, 3, 3, 512])
        let cleaned = SDWeights.sanitize([
            "decoder.mid_block.attentions.0.to_q.weight": linear,
            "decoder.mid_block.attentions.0.to_q.bias": MLXArray.zeros([512]),
            "decoder.up_blocks.0.upsamplers.0.conv.weight": mlxConv,   // already MLX shape
        ])
        #expect(cleaned["decoder.mid_block.attentions.0.to_q.weight"]?.shape == [512, 512])
        #expect(cleaned["decoder.mid_block.attentions.0.to_q.bias"]?.shape == [512])
        #expect(cleaned["decoder.up_blocks.0.upsamplers.0.conv.weight"]?.shape == [512, 3, 3, 512])
    }

    /// SD-ENG4 load-path regression: `ff.net.{0,2}` (diffusers `FeedForward` ModuleList with the
    /// weightless dropout at `net.1`) must be remapped to non-numeric children, else
    /// `ModuleParameters.unflattened` builds an array-with-a-hole and `update(parameters:)` throws
    /// `incompatibleItems` (reproduced by running the real model in the app).
    @Test func sanitizeRemapsFeedForwardGap() {
        let cleaned = SDWeights.sanitize([
            "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight": MLXArray.zeros([10240, 1280]),
            "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.bias": MLXArray.zeros([10240]),
            "mid_block.attentions.0.transformer_blocks.0.ff.net.2.weight": MLXArray.zeros([1280, 5120]),
        ])
        #expect(cleaned["mid_block.attentions.0.transformer_blocks.0.ff.net.geglu.proj.weight"]?.shape == [10240, 1280])
        #expect(cleaned["mid_block.attentions.0.transformer_blocks.0.ff.net.out.weight"]?.shape == [1280, 5120])
        #expect(cleaned.keys.contains { $0.contains("ff.net.0.proj") } == false)
        #expect(cleaned.keys.contains { $0.contains("ff.net.2") } == false)
    }

    /// Full load-path round trip: build a checkpoint-style raw weight dict from the tiny UNet's
    /// own parameter tree (reversing the sanitize transforms), run `sanitize → unflattened →
    /// update(verify:.none)` exactly like the engine does. Must not throw (the `ff.net` gap bug
    /// surfaced as `incompatibleItems` at run time).
    @Test func unetLoadPathAcceptsCheckpointStyleWeights() throws {
        let config = SDUNetConfig(
            inChannels: 4, outChannels: 4, blockOutChannels: [8, 16],
            layersPerBlock: [2, 2], transformerLayersPerBlock: [1, 2],
            crossAttentionDim: 32, attentionHeadDim: 4,
            timeEmbedDim: 8, addTimeEmbedDim: 8, textEmbedDim: 8,
            normGroups: 2, normEps: 1e-5)
        let unet = SDUNet(config: config)

        var raw: [String: MLXArray] = [:]
        for (key, array) in unet.parameters().flattened() {
            var k = key
            if k.contains("ff.net.geglu.proj") {
                k = k.replacingOccurrences(of: "ff.net.geglu.proj", with: "ff.net.0.proj")
            } else if k.contains("ff.net.out") {
                k = k.replacingOccurrences(of: "ff.net.out", with: "ff.net.2")
            }
            var v = MLXRandom.normal(array.shape, dtype: .float32)
            if v.ndim == 4 { v = v.transposed(0, 3, 1, 2) }   // MLX (out,kH,kW,in) → PyTorch (out,in,kH,kW)
            raw[k] = v
        }

        let sanitized = SDWeights.sanitize(raw)
        let params = ModuleParameters.unflattened(sanitized)
        try unet.update(parameters: params, verify: .none)     // throws on structural mismatch
    }

    /// Same load-path round trip for the VAE decoder (to_out.0 array, optional conv_shortcut,
    /// upsampler arrays) — guards the rest of the engine's `update` path against structural errors.
    @Test func vaeLoadPathAcceptsCheckpointStyleWeights() throws {
        let vae = SDVAE()
        var raw: [String: MLXArray] = [:]
        for (key, array) in vae.parameters().flattened() {
            var v = MLXRandom.normal(array.shape, dtype: .float32)
            if v.ndim == 4 { v = v.transposed(0, 3, 1, 2) }   // MLX → PyTorch conv layout
            raw[key] = v
        }
        let sanitized = SDWeights.sanitize(raw)
        try vae.update(parameters: ModuleParameters.unflattened(sanitized), verify: .none)
    }

    /// SD-ENG3 — a minimal SDXL UNet (small channels, few transformer layers) keeps the full
    /// forward shape contract: `[1, H, W, 4]` latent + scalar timestep + `[1, seq, cross]`
    /// encoder states + pooled text embeds + 6 time ids → `[1, H, W, 4]` noise prediction.
    /// The real model is ~2.6B params — too heavy for a unit gate — so this exercises the
    /// down/up/mid/residual/attention plumbing with tiny channel counts.
    @Test func tinyUNetPreservesLatentShape() {
        let config = SDUNetConfig(
            inChannels: 4, outChannels: 4, blockOutChannels: [8, 16],
            layersPerBlock: [2, 2], transformerLayersPerBlock: [1, 2],
            crossAttentionDim: 32, attentionHeadDim: 4,
            timeEmbedDim: 8, addTimeEmbedDim: 8, textEmbedDim: 8,
            normGroups: 2, normEps: 1e-5)
        let unet = SDUNet(config: config)

        // 32×32 latent → down0 downsamples to 16 → mid at 16 → up0 upsamples to 32.
        let sample = MLXRandom.normal([1, 32, 32, 4])
        let timestep = MLXArray([Float(999)])
        let encoderStates = MLXRandom.normal([1, 16, 32])
        let textEmbeds = MLXRandom.normal([1, 8])
        let timeIDs = MLXArray([Float(1024), 1024, 0, 0, 1024, 1024]).reshaped([1, 6])

        let out = unet(sample: sample, timestep: timestep, encoderHiddenStates: encoderStates,
                       textEmbeds: textEmbeds, timeIDs: timeIDs)
        eval(out)
        #expect(out.shape == [1, 32, 32, 4])
    }

    /// The SDXL `add_embedding` contract: 6 time ids → 6×addTimeEmbedDim, + text embeds = the
    /// checkpoint's 2816 input (pinned by `additionEmbedDim`). Guards the diffusers
    /// `time_ids.flatten()` reshape used in the forward.
    @Test func sdxlConfigMatchesCheckpointAdditionDim() {
        #expect(SDUNetConfig.sdxl.additionEmbedDim == 2816)
        #expect(SDUNetConfig.sdxl.tembChannels == 1280)
        #expect(SDUNetConfig.sdxl.heads(for: 1280) == 20)
        #expect(SDUNetConfig.sdxl.heads(for: 640) == 10)
        #expect(SDUNetConfig.sdxl.heads(for: 320) == 5)
    }
}

*/
