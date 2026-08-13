import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXUI

/// MG-ENG1 — shape/construction tests for the MusicGen T5-base text encoder. Instantiate with
/// unloaded (random-init) weights, feed `[1, 32]` int32 token ids, assert `[1, 32, 768]`.
/// No real weights — a forward-pass plumbing check; key-loading is verified by the engine's
/// `loadWeights` at run time and numerical parity by human run-smoke (row 27).
///
/// Serialized like `DeepSeekOCRMLXTests`/`SDXLTurboEncoderTests`: MLX's default stream isn't
/// safe under Swift Testing's parallelism (SIGTRAP).
@Suite(.serialized)
struct MusicGenT5EncoderTests {

    @Test func t5EncoderOutputsHiddenSequence() {
        let encoder = MusicGenT5Encoder()
        let ids = MLXArray.arange(32).expandedDimensions(axis: 0)   // [1, 32] int32
        let out = encoder(ids)
        eval(out)
        #expect(out.shape == [1, 32, 768])
    }

    @Test func t5SmallTinyConfigKeepsShape() {
        // A tiny config (2 layers) exercises the same plumbing with fewer layers; the real
        // `MusicGenT5Config()` default (12 layers) is covered above.
        var config = MusicGenT5Config()
        config.layers = 2
        let encoder = MusicGenT5Encoder(config: config)
        let ids = MLXArray.arange(8).expandedDimensions(axis: 0)
        let out = encoder(ids)
        eval(out)
        #expect(out.shape == [1, 8, 768])
    }

    @Test func relativePositionBucketsMatchT5Contract() {
        // T5's `_relative_position_bucket` bidirectional mapping: positive positions land in
        // the second half (16..31), small absolutes map to themselves, huge map to 31.
        let rp = MLXArray([Int32(0), 1, -1, 100, -100, 5, 15, 31]).expandedDimensions(axis: 0)
        let buckets = MusicGenT5SelfAttention.relativePositionBuckets(
            rp, numBuckets: 32, maxDistance: 128)
        eval(buckets)
        let b = buckets.asArray(Int32.self)
        // exact small positive → 16 + value
        #expect(b[1] == 17)
        // exact small negative → value (no offset)
        #expect(b[2] == 1)
        // large positive → clamped into the positive half (≥ 16, ≤ 31)
        #expect(b[3] >= 16 && b[3] <= 31)
        #expect(b[4] >= 0 && b[4] <= 15)
    }

    /// MG-FIX1 residual regression: pre-norm + residual must keep the input `x` alive through the
    /// block. Zero the attention output projection (`o`) and the FFN output (`wo`) — then every
    /// sublayer contributes exactly 0, so a correct block is the identity (`out == x`), while a
    /// block that drops its residuals discards `x` and emits 0. This test fails on the old code.
    @Test func t5BlockPreservesInputViaResiduals() throws {
        let config = MusicGenT5Config()
        let block = MusicGenT5Block(config: config)

        // Zero `self_attn.o` and `ff.wo` only; keep everything else as-initialized.
        var flat: [String: MLXArray] = [:]
        for (key, array) in block.parameters().flattened() {
            if key == "self_attn.o.weight" || key == "ff.wo.weight" {
                flat[key] = MLXArray.zeros(array.shape)
            } else {
                flat[key] = array
            }
        }
        try block.update(parameters: ModuleParameters.unflattened(flat), verify: .none)
        eval(block)

        let x = MLXArray.ones([1, 8, config.dModel])
        let out = block(x)
        eval(out)
        let diff = MLX.max(abs(out - x))
        eval(diff)
        #expect(diff.item() < 1e-6)
    }
}
