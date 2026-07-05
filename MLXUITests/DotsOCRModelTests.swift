import Testing
import MLX
import MLXRandom
import MLXLMCommon
@testable import MLXUI

/// SUP-3 slices 3–4 MLX compute smoke, **serialized**: MLX's default stream isn't safe under Swift
/// Testing's parallel execution (concurrent GPU eval traps with SIGTRAP), so all DotsOCR tests that
/// actually run MLX live in this one `.serialized` suite. They can't check numeric parity (that's the
/// human run-smoke) — they assert the tensor plumbing so a shape bug crashes here, not at load time.
@Suite(.serialized)
struct DotsOCRMLXTests {

    // MARK: - Vision tower (slice 3)

    private func tinyVisionConfig() -> DotsOCRConfig.Vision {
        DotsOCRConfig.Vision(
            embedDim: 32, hiddenSize: 24, intermediateSize: 16,
            numHiddenLayers: 2, numAttentionHeads: 4,   // headDim 8 (÷4 → valid vision RoPE)
            numChannels: 3, patchSize: 2, postNorm: true, rmsNormEps: 1e-5,
            spatialMergeSize: 2, temporalPatchSize: 1, useBias: false)
    }

    @Test func visionTowerMergesPatchesToHiddenDim() {
        let cfg = tinyVisionConfig()
        let vision = DotsVisionModel(cfg)
        let grid = (t: 1, h: 4, w: 4)  // 16 patches
        let pixels = MLXRandom.normal([grid.h * grid.w, cfg.numChannels * cfg.temporalPatchSize
                                       * cfg.patchSize * cfg.patchSize])
        let out = vision(pixels, gridTHW: [grid])
        eval(out)
        #expect(out.ndim == 2)
        #expect(out.dim(0) == (grid.h / cfg.spatialMergeSize) * (grid.w / cfg.spatialMergeSize)) // 4
        #expect(out.dim(1) == cfg.hiddenSize)  // 24
    }

    // MARK: - Model merge + prepare (slice 4)

    private func tinyConfig() -> DotsOCRConfig {
        DotsOCRConfig(
            modelType: "dots_ocr",
            hiddenSize: 24, intermediateSize: 16, numHiddenLayers: 2,
            numAttentionHeads: 4, numKeyValueHeads: 2, vocabSize: 32,
            rmsNormEps: 1e-6, ropeTheta: 1_000_000, attentionBias: true,
            tieWordEmbeddings: false, maxPositionEmbeddings: 128,
            imageTokenId: 99, videoTokenId: 98,
            visionConfig: DotsOCRConfig.Vision(
                embedDim: 16, hiddenSize: 24, intermediateSize: 16,
                numHiddenLayers: 1, numAttentionHeads: 2,   // headDim 8 (÷4 → valid vision RoPE)
                numChannels: 3, patchSize: 2, postNorm: true, rmsNormEps: 1e-5,
                spatialMergeSize: 2, temporalPatchSize: 1, useBias: false),
            quantization: nil)
    }

    @Test func mergePlacesImageFeaturesAtImageTokenPositions() {
        let ids = MLXArray([Int32(1), 99, 2, 99, 3])          // image tokens (id 99) at pos 1,3
        let embeds = MLXArray(0 ..< 15).asType(.float32).reshaped(5, 3)
        let feats = MLXArray([Float]([100, 100, 100, 200, 200, 200])).reshaped(2, 3)  // f0=100s, f1=200s

        let merged = DotsOCR.merge(inputIds: ids, inputEmbeds: embeds, imageFeatures: feats, imageTokenId: 99)
        eval(merged)

        #expect(merged.shape == [1, 5, 3])
        let flat = merged.reshaped(5, 3)
        #expect(flat[1].asArray(Float.self) == [100, 100, 100])  // pos 1 → feature 0
        #expect(flat[3].asArray(Float.self) == [200, 200, 200])  // pos 3 → feature 1
        #expect(flat[0].asArray(Float.self) == [0, 1, 2])         // pos 0 → original text embed
        #expect(flat[2].asArray(Float.self) == [6, 7, 8])         // pos 2 → original text embed
    }

    @Test func prepareProducesLogitsOverPromptWithImage() throws {
        let cfg = tinyConfig()
        let model = DotsOCR(cfg)

        let grid = THW(1, 4, 4)  // 16 patches → spatial-merge → 4 image tokens
        let ids = MLXArray([Int32](repeating: Int32(cfg.imageTokenId), count: 4) + [Int32(1), Int32(2)])
        let pixels = MLXRandom.normal([grid.h * grid.w,
                                       cfg.visionConfig.numChannels * cfg.visionConfig.temporalPatchSize
                                       * cfg.visionConfig.patchSize * cfg.visionConfig.patchSize])

        let input = LMInput(text: .init(tokens: ids), image: .init(pixels: pixels, frames: [grid]))
        let cache = model.newCache(parameters: nil)

        let result = try model.prepare(input, cache: cache, windowSize: nil)
        guard case let .logits(out) = result else {
            Issue.record("prepare should return .logits")
            return
        }
        eval(out.logits)
        #expect(out.logits.dim(0) == 1)
        #expect(out.logits.dim(1) == 6)               // one logit row per prompt token
        #expect(out.logits.dim(2) == cfg.vocabSize)   // 32
    }
}
