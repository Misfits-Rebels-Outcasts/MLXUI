import MLX
import Testing
@testable import MLXUI

struct LayaModelTests {
    // MARK: layaSanitizeWeights (model.py::sanitize_weights)

    @Test func sanitizeWeightsRenamesInProjAndSequentialIndices() {
        let raw: [String: MLXArray] = [
            "head.layers.0.self_attn.in_proj_weight": MLXArray.zeros([3]),
            "head.layers.0.self_attn.in_proj_bias": MLXArray.zeros([3]),
            "scorer.0.weight": MLXArray.zeros([1]),
            "scorer.1.weight": MLXArray.zeros([1]),
            "scorer.layers.3.weight": MLXArray.zeros([1]),  // already-renamed key: must not double-rename
            "act_head.0.weight": MLXArray.zeros([1]),
            "encoder.embeddings.tok_embeddings.weight": MLXArray.zeros([1]),
        ]
        let out = layaSanitizeWeights(raw)
        #expect(out["head.layers.0.self_attn.in_proj.weight"] != nil)
        #expect(out["head.layers.0.self_attn.in_proj.bias"] != nil)
        #expect(out["scorer.layers.0.weight"] != nil)
        #expect(out["scorer.layers.1.weight"] != nil)
        #expect(out["scorer.layers.3.weight"] != nil)
        #expect(out["act_head.layers.0.weight"] != nil)
        #expect(out["encoder.embeddings.tok_embeddings.weight"] != nil)
        #expect(out.count == raw.count)
    }

    // MARK: modernBERTAttentionMasks (model.py::attention_masks)

    @Test func slidingWindowMaskRespectsLocalDistanceAndPadding() {
        let attentionMask = MLXArray([Int32(1), 1, 1, 0], [1, 4]).asType(.bool)
        let (full, sliding) = modernBERTAttentionMasks(attentionMask: attentionMask, window: 2)
        let fullArr = full.asArray(Bool.self)
        #expect(fullArr == [true, true, true, false])

        let slidingArr = sliding.asArray(Bool.self)
        func at(_ q: Int, _ k: Int) -> Bool { slidingArr[q * 4 + k] }
        #expect(at(0, 0) == true)    // distance 0
        #expect(at(0, 1) == true)    // distance 1 <= window/2 (1)
        #expect(at(0, 2) == false)   // distance 2 > 1, key valid → masked
        #expect(at(0, 3) == false)   // key 3 is padding → masked regardless of distance
        // Padded queries (row 3) may see every valid key, so a fully-masked softmax row
        // never happens (`model.py::attention_masks`'s own justification).
        #expect(at(3, 0) == true)
        #expect(at(3, 3) == false)
    }

    // MARK: end-to-end forward shape/finiteness smoke (tiny random-init weights, no checkpoint)

    @Test func modernBERTEncoderForwardProducesFiniteOutput() {
        let config = ModernBERTEncoderConfig(
            vocabularySize: 50, hiddenSize: 32, intermediateSize: 16, numHiddenLayers: 3,
            numAttentionHeads: 4, localAttention: 4, globalAttnEveryNLayers: 2)
        let encoder = ModernBERTEncoder(config, slidingWindowEnabled: true)
        let ids = MLXArray([Int32](repeating: 1, count: 12), [2, 6])
        let mask = MLXArray([Int32](repeating: 1, count: 12), [2, 6]).asType(.bool)
        let out = encoder(inputIds: ids, attentionMask: mask)
        out.eval()
        #expect(out.shape == [2, 6, 32])
        #expect(out.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    @Test func layaDecisionModelForwardProducesFiniteLogits() {
        let encoderConfig = ModernBERTEncoderConfig(
            vocabularySize: 50, hiddenSize: 32, intermediateSize: 16, numHiddenLayers: 2,
            numAttentionHeads: 4, localAttention: 4, globalAttnEveryNLayers: 2)
        let model = LayaDecisionModel(encoderConfig: encoderConfig, headLayers: 1, actCostCount: 1)
        let b = 2, length = 6, count = 3
        let ids = MLXArray([Int32](repeating: 1, count: b * length), [b, length])
        let attentionMask = MLXArray([Int32](repeating: 1, count: b * length), [b, length]).asType(.bool)
        let markerPos = MLXArray([Int32](repeating: 1, count: b * count), [b, count])
        let markerMask = MLXArray([Int32](repeating: 1, count: b * count), [b, count]).asType(.bool)
        let qtype = MLXArray([Int32](repeating: 0, count: b))

        let (logits, action) = model(
            inputIds: ids, attentionMask: attentionMask, markerPos: markerPos, markerMask: markerMask,
            qtype: qtype)
        logits.eval()
        action.eval()
        #expect(logits.shape == [b, count])
        #expect(action.shape[0] == b)
        #expect(logits.asArray(Float.self).allSatisfy { $0.isFinite })
    }
}
