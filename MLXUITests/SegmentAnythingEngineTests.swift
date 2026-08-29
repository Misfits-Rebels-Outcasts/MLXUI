// MLX forward-pass tests — commented out (slow; run manually with ⌘U)
/*
import Testing
import MLX
import MLXNN
@testable import MLXUI

/// Shape-correctness tests for the SAM3 image encoder and mask decoder (SA-AM4).
/// These use toy configurations so they run in milliseconds without real weights.
/// Serialized because MLX's random state (used by Linear.init weight init) is not
/// thread-safe across concurrent test executions.
@Suite(.serialized)
struct SegmentAnythingEngineTests {

    // MARK: - Tiny config helpers

    /// 2-layer, 8-dim, 2-head backbone; 28px image, 14px patch → 2×2 grid, 4 neck channels.
    private func tinyBackbone(windowSize: Int = 0, globalLayers: Set<Int> = [0, 1]) -> SAM3ViTBackbone {
        SAM3ViTBackbone(
            hiddenSize: 8,
            numLayers: 2,
            numHeads: 2,
            intermediateSize: 16,
            imageSize: 28,
            patchSize: 14,
            windowSize: windowSize,
            globalAttnLayers: globalLayers,
            neckChannels: 4
        )
    }

    // MARK: - Image encoder

    /// Global-attention path: [1,28,28,3] → [1,2,2,4]
    @Test func encoderOutputShape() {
        let out = tinyBackbone()(MLXArray.zeros([1, 28, 28, 3]))
        #expect(out.shape == [1, 2, 2, 4])
    }

    /// Windowed-attention path produces the same output shape.
    @Test func encoderWindowedAttentionPath() {
        let backbone = tinyBackbone(windowSize: 1, globalLayers: [])
        let out = backbone(MLXArray.zeros([1, 28, 28, 3]))
        #expect(out.shape == [1, 2, 2, 4])
    }

    // MARK: - Mask decoder

    /// 3 × 2× upsample: grid 2 → 16;  [1,2,2,4] embed → [1,16,16,3] masks.
    @Test func decoderOutputShape() {
        let decoder = SAM3PixelDecoder(inChannels: 4, numMasks: 3)
        let embed  = MLXArray.zeros([1, 2, 2, 4])
        let tokens = MLXArray.zeros([1, 3, 4])
        let masks  = decoder(embed, promptTokens: tokens)
        #expect(masks.shape == [1, 16, 16, 3])
    }

    // MARK: - IoU predictor

    /// Scores should be [1, 3] and all ∈ (0, 1) — sigmoid(0) = 0.5 for zero weights.
    @Test func iouPredictorShape() {
        let predictor = SAM3IoUPredictor(inChannels: 4, numMasks: 3)
        let scores = predictor(MLXArray.zeros([1, 2, 2, 4]))
        #expect(scores.shape == [1, 3])
        #expect(scores.asArray(Float.self).allSatisfy { $0 > 0 && $0 < 1 })
    }

    // MARK: - Prompt encoder

    /// 2 points + 1 no-mask token → [1, 3, 8].
    @Test func promptEncoderOutputShape() {
        let encoder = SAM3PromptEncoder(promptDim: 8)
        let points = MLXArray([Float(0.5), Float(0.5), Float(0.2), Float(0.8)], [2, 2])
        let labels = MLXArray([Int32(1), Int32(0)])
        let tokens = encoder(points: points, labels: labels)
        #expect(tokens.shape == [1, 3, 8])
    }
}

*/
