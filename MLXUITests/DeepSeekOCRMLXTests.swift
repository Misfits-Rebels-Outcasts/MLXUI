import Testing
import Foundation
import CoreGraphics
import MLX
import MLXRandom
import MLXLMCommon
@testable import MLXUI

/// SUP-2 MLX compute smoke, **serialized** (MLX's default stream isn't safe under Swift Testing's
/// parallelism — SIGTRAP). Shape-only assertions; numeric parity is the human run-smoke (slice 8).
/// Later DeepSeek-OCR slices (Qwen2 encoder, LM, model assembly) add their tests to this suite.
@Suite(.serialized)
struct DeepSeekOCRMLXTests {

    /// Tiny but structurally faithful SAM config (real one is 12 layers / 768-wide / 1024²).
    private func tinySAMVision() -> DeepSeekOCRConfig.Vision {
        DeepSeekOCRConfig.Vision(
            samWidth: 64, samLayers: 2, samHeads: 4,
            samGlobalAttnIndexes: [1], samDownsampleChannels: [512, 1024],
            samWindowSize: 4, samImageSize: 64, samFinalOutChannels: 32,
            qwen2Dim: 64)
    }

    @Test func samEncoderDownsamplesToFinalChannels() {
        let sam = SAMEncoder(tinySAMVision())
        // NHWC image; grid = 64/16 = 4 → neck(4) → net_2 stride2 (2) → net_3 stride2 (1).
        let x = MLXRandom.normal([1, 64, 64, 3])
        let out = sam(x)
        eval(out)
        #expect(out.ndim == 4)
        #expect(out.dim(1) == 1)
        #expect(out.dim(2) == 1)
        #expect(out.dim(3) == 32)   // final_out_chans
    }

    @Test func samConvSanitizeTransposesPyTorchWeightsOnly() {
        let pytorch = MLXArray.zeros([256, 768, 1, 1])   // neck.0 (out,in,kH,kW): kH≠? guard catches
        let key = "sam_model.neck.0.weight"
        let cleaned = SAMEncoder.sanitize([
            key: pytorch,
            "sam_model.blocks.0.attn.rel_pos_h": MLXArray.zeros([27, 64]),   // not a conv → untouched
        ])
        #expect(cleaned[key]?.shape == [256, 1, 1, 768])                     // transposed to MLX layout
        #expect(cleaned["sam_model.blocks.0.attn.rel_pos_h"]?.shape == [27, 64])
    }

    // MARK: - Qwen2 vision encoder + projector (slice 4)

    private func tinyQwen2Vision() -> DeepSeekOCRConfig.Vision {
        DeepSeekOCRConfig.Vision(
            samWidth: 64, samLayers: 1, samHeads: 4,
            samGlobalAttnIndexes: [], samDownsampleChannels: [512, 1024],
            qwen2Dim: 16, qwen2Layers: 2, qwen2Heads: 4, qwen2KVHeads: 2, qwen2IntermediateSize: 16)
    }

    @Test func qwen2EncoderReturnsQueryTokensThenProjects() {
        let v = tinyQwen2Vision()
        let vision = DeepSeekOCRVisionModel(v)

        // SAM features (B, H, W, dim); 12×12 = 144 image tokens → query_768 path → 144 query tokens.
        let sam = MLXRandom.normal([1, 12, 12, v.qwen2Dim])
        let encoded = vision(sam)
        eval(encoded)
        #expect(encoded.dim(0) == 1)
        #expect(encoded.dim(1) == 144)          // query tokens returned
        #expect(encoded.dim(2) == v.qwen2Dim)   // 16

        let projector = DeepSeekOCRProjector(
            DeepSeekOCRConfig.Projector(inputDim: v.qwen2Dim, nEmbed: 24, projectorType: "linear"))
        let projected = projector(encoded)
        eval(projected)
        #expect(projected.dim(1) == 144)
        #expect(projected.dim(2) == 24)         // LM embedding width
    }

    @Test func qwen2EncoderUses1024QueryFor256Tokens() {
        let v = tinyQwen2Vision()
        let vision = DeepSeekOCRVisionModel(v)
        // 16×16 = 256 image tokens → query_1024 path → 256 query tokens.
        let sam = MLXRandom.normal([1, 16, 16, v.qwen2Dim])
        let encoded = vision(sam)
        eval(encoded)
        #expect(encoded.dim(1) == 256)
    }

    // MARK: - DeepSeek-V2 MoE LM (slice 5)

    /// Tiny DeepSeek-V2 config via JSON (the config only has a custom `init(from:)`): 2 layers
    /// (layer 0 dense, layer 1 MoE), 4 routed + 1 shared experts, top-2.
    private static let tinyLMJSON = """
    {
        "hidden_size": 16, "intermediate_size": 16, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "vocab_size": 32, "first_k_dense_replace": 1,
        "n_group": 1, "topk_group": 1, "n_routed_experts": 4, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "topk_method": "greedy",
        "projector_config": {"input_dim": 64, "n_embed": 16, "projector_type": "linear"},
        "vision_config": {"width": {"qwen2-0-5b": {"dim": 16},
            "sam_vit_b": {"width": 64, "layers": 1, "heads": 4, "global_attn_indexes": [], "downsample_channels": [512, 1024]}}}
    }
    """

    private func tinyLMConfig() throws -> DeepSeekOCRConfig {
        try JSONDecoder().decode(DeepSeekOCRConfig.self, from: Data(Self.tinyLMJSON.utf8))
    }

    @Test func languageModelProducesLogits() throws {
        let cfg = try tinyLMConfig()
        let lm = DeepSeekOCRLanguage(cfg)
        let ids = MLXArray([Int32(1), 2, 3, 4]).reshaped(1, 4)
        let logits = lm(ids)   // no cache → causal mask; exercises dense (L0) + MoE (L1)
        eval(logits)
        #expect(logits.dim(0) == 1)
        #expect(logits.dim(1) == 4)
        #expect(logits.dim(2) == 32)   // vocab
    }

    @Test func expertJoinSanitizeStacksExpertsForSwitchGLU() {
        // Fake per-expert weights for layer 1 → expect a single stacked switch_mlp weight, experts gone.
        var weights: [String: MLXArray] = [:]
        for e in 0 ..< 4 {
            weights["language_model.model.layers.1.mlp.experts.\(e).gate_proj.weight"] = MLXArray.zeros([8, 16])
        }
        let out = DeepSeekOCRLanguage.sanitizeExperts(weights, numLayers: 2, numExperts: 4)
        #expect(out["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 8, 16])
        #expect(out["language_model.model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)
    }

    // MARK: - Full model assembly (slice 6)

    /// Tiny end-to-end config with a small `image_token_index` so ids stay within the tiny vocab.
    private static let tinyModelJSON = """
    {
        "hidden_size": 16, "intermediate_size": 16, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "vocab_size": 32, "first_k_dense_replace": 1, "image_token_index": 5,
        "n_group": 1, "topk_group": 1, "n_routed_experts": 4, "n_shared_experts": 1,
        "num_experts_per_tok": 2, "topk_method": "greedy",
        "projector_config": {"input_dim": 16, "n_embed": 16, "projector_type": "linear"},
        "vision_config": {"image_size": 64, "width": {
            "qwen2-0-5b": {"dim": 16, "layers": 1, "heads": 4, "kv_heads": 2, "intermediate_size": 16},
            "sam_vit_b": {"width": 64, "layers": 1, "heads": 4, "final_out_chans": 16, "global_attn_indexes": [], "downsample_channels": [512, 1024]}}}
    }
    """

    @Test func prepareRunsFullPipelineToLogits() throws {
        let cfg = try JSONDecoder().decode(DeepSeekOCRConfig.self, from: Data(Self.tinyModelJSON.utf8))
        let model = DeepSeekOCR(cfg)

        // One global tile (64²). SAM→Qwen2 yields `query_1024` = 256 tokens; + view_separator = 257.
        let h = 64, w = 64
        let pixels = MLXRandom.normal([h * w * 3])                 // flat 1-D buffer
        let frames = [THW(1, h, w)]
        let numImageTokens = 256 + 1
        let ids = MLXArray([Int32](repeating: 5, count: numImageTokens) + [Int32(1), Int32(2)])

        let input = LMInput(
            text: .init(tokens: ids),
            image: .init(pixels: pixels, frames: frames))
        let cache = model.newCache(parameters: nil)

        let result = try model.prepare(input, cache: cache, windowSize: nil)
        guard case let .logits(out) = result else {
            Issue.record("prepare should return .logits")
            return
        }
        eval(out.logits)
        #expect(out.logits.dim(0) == 1)
        #expect(out.logits.dim(1) == numImageTokens + 2)   // one logit row per prompt token
        #expect(out.logits.dim(2) == 32)                   // vocab
    }

    // MARK: - Processor tiling (pure `dynamic_preprocess` grid; no MLX)

    /// Grids reproduce the reference `dynamic_preprocess` (min 1 / max 6 / 768²): closest aspect ratio,
    /// tie broken toward the finer grid only when the image is large enough to fill it.
    @Test(arguments: [
        (CGSize(width: 991, height: 916), (cols: 1, rows: 1)),      // near-square, below the finer-grid area
        (CGSize(width: 1280, height: 1280), (cols: 2, rows: 2)),    // square + large → 2×2
        (CGSize(width: 1700, height: 2200), (cols: 2, rows: 3)),    // portrait page → 6
        (CGSize(width: 1024, height: 768), (cols: 3, rows: 2)),     // landscape → 6
        (CGSize(width: 600, height: 1600), (cols: 1, rows: 3)),     // tall receipt → 3
    ])
    func tileGridMatchesReference(size: CGSize, expected: (cols: Int, rows: Int)) {
        let grid = DeepSeekOCRProcessor.tileGrid(for: size)
        #expect(grid.cols == expected.cols)
        #expect(grid.rows == expected.rows)
        // Token budget the processor emits must equal the model's [patches·144, global 256, sep 1].
        let patches = grid.cols * grid.rows
        #expect(patches >= 1 && patches <= 6)
    }

    @Test func tileGridClampsDegenerateSize() {
        #expect(DeepSeekOCRProcessor.tileGrid(for: .zero) == (cols: 1, rows: 1))
    }

    // MARK: - SAM rel-pos dtype (tiling regression)

    /// 768² local tiles (SAM grid 48) interpolate the 1024²-sized rel-pos table (grid 64). The result
    /// MUST keep the table's dtype — a float32 bias makes `scaledDotProductAttention` reject the mask
    /// ("must promote to output type bfloat16") in the bf16 model. Reproduces the tiling run crash.
    @Test func relPosInterpolationPreservesBF16Dtype() {
        let table = MLXArray.zeros([127, 8]).asType(.bfloat16)   // 2·64−1 (global block at grid 64)
        let interpolated = SAMMath.getRelPos(48, 48, table)      // 2·48−1 = 95 ≠ 127 → interpolates
        eval(interpolated)
        #expect(interpolated.dtype == .bfloat16)
        #expect(interpolated.dim(0) == 48 && interpolated.dim(1) == 48 && interpolated.dim(2) == 8)
    }

    /// The no-interpolation branch (sizes already match) also returns the table dtype unchanged.
    @Test func relPosPassthroughKeepsDtype() {
        let table = MLXArray.zeros([127, 8]).asType(.bfloat16)
        let out = SAMMath.getRelPos(64, 64, table)               // 2·64−1 = 127 → no interpolation
        eval(out)
        #expect(out.dtype == .bfloat16)
    }
}
