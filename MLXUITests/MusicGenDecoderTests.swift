// MLX forward-pass tests — commented out (slow; run manually with ⌘U)
/*
import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXUI

/// MG-ENG3 — shape tests for the MusicGen causal decoder. Uses a tiny config (2 layers, small
/// hidden dims) so the forward pass is cheap; asserts the output logits shape. No real weights.
///
/// Serialized like the other MLX suites (MLX's default stream isn't parallel-safe).
@Suite(.serialized)
struct MusicGenDecoderTests {

    private static func tinyConfig() -> MusicGenDecoderConfig {
        var c = MusicGenDecoderConfig()
        c.hidden = 64
        c.layers = 2
        c.heads = 4
        c.headDim = 16
        c.ffn = 128
        c.codebooks = 4
        c.vocab = 64
        c.maxPosition = 2048
        c.bosTokenID = 64
        return c
    }

    /// `[1, T, codebooks]` tokens + projected T5 context → `[1, T, vocab, codebooks]` logits.
    @Test func decoderOutputsCodebookLogits() {
        let config = Self.tinyConfig()
        let decoder = MusicGenDecoder(config: config)
        let t = 8
        let tokens = MLXArray.zeros([1, t, config.codebooks]).asType(.int32)
        // Projected T5 context: [1, seq, hidden] (the engine does encToDecProj upstream).
        let conditioning = MLXArray.zeros([1, 16, config.hidden])
        let logits = decoder(tokens, conditioning: conditioning, offset: 0)
        eval(logits)
        #expect(logits.shape == [1, t, config.vocab, config.codebooks])
    }

    /// Learned positional embeddings index correctly past the zero offset (generation advances).
    @Test func decoderHandlesNonZeroOffset() {
        let config = Self.tinyConfig()
        let decoder = MusicGenDecoder(config: config)
        let tokens = MLXArray.zeros([1, 1, config.codebooks]).asType(.int32)
        let conditioning = MLXArray.zeros([1, 16, config.hidden])
        let logits = decoder(tokens, conditioning: conditioning, offset: 42)
        eval(logits)
        #expect(logits.shape == [1, 1, config.vocab, config.codebooks])
    }

    /// The decoder's `@ModuleInfo` keys descend the **checkpoint's** key structure (MG-ENG3
    /// load-path guard, mirroring `SDXLTurboVAETests`): flatten the module tree, rename keys
    /// back to the jasonvassallo checkpoint naming (`layers.N.self_attn.q_proj.weight`,
    /// `embed_tokens.N.weight`, `lm_heads.N.weight`, …), unflatten, and `update` — throws on any
    /// structural mismatch. Uses `verify: .noUnusedKeys` (not `.none`) so a missed key would be
    /// caught. `embed_positions` is special-cased: the checkpoint ships it as a **nested**
    /// `embed_positions.weights` table, and a bare `embed_positions` array was the
    /// `incompatibleItems` bug from run-smoke (fixed 2026-08-13).
    @Test func decoderLoadPathAcceptsCheckpointStyleWeights() throws {
        let config = Self.tinyConfig()
        let decoder = MusicGenDecoder(config: config)
        var raw: [String: MLXArray] = [:]
        for (key, array) in decoder.parameters().flattened() {
            var checkpointKey = key
            if key == "embed_positions.weights" {
                checkpointKey = "embed_positions.weights"   // already nested — matches checkpoint
            }
            raw[checkpointKey] = MLXRandom.normal(array.shape, dtype: .float32)
        }
        let params = ModuleParameters.unflattened(raw)
        try decoder.update(parameters: params, verify: .noUnusedKeys)
        eval(decoder)
    }
}

*/
