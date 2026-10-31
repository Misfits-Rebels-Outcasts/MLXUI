import Foundation
import MLX
import Testing
@testable import MLXUI

// MP7-3c: the Klein Qwen3 tap encoder against diffusers 0.37.1 on the tiny fp32 golden (g3c_textencoder.safetensors, never edited), plus the loader's
// streaming rules. The tiny config has hidden 16, so it cannot be quantized (MLX affine groups start at 32); the quantized path is exercised by the
// opt-in real-weight suite (Flux2KleinTextEncoderRealLoadTests).

enum KleinCosine {
    /// Cosine similarity of two equally-sized arrays, read through the error guard (rule 17a), in Double.
    static func of(_ a: MLXArray, _ b: MLXArray) throws -> Double {
        let x = try MLXErrorGuard.read(a.asType(.float32).reshaped(-1), as: Float.self)
        let y = try MLXErrorGuard.read(b.asType(.float32).reshaped(-1), as: Float.self)
        return cosine(x, y)
    }

    static func cosine(_ x: [Float], _ y: [Float]) -> Double {
        var dot = 0.0, nx = 0.0, ny = 0.0
        for i in 0 ..< x.count { dot += Double(x[i]) * Double(y[i]); nx += Double(x[i]) * Double(x[i]); ny += Double(y[i]) * Double(y[i]) }
        return dot / (nx.squareRoot() * ny.squareRoot())
    }
}

@Suite(.serialized) struct Flux2KleinTextEncoderTests {
    static let golden: [String: MLXArray] = {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein/g3c_textencoder.safetensors")
        return (try? loadArrays(url: url)) ?? [:]
    }()

    /// TINY_QWEN of tiny_goldens.py; taps (1, 2, 3) of 4 layers stand in for (9, 18, 27) of 36.
    static let tiny = Flux2KleinQwen3Shape(hidden: 16, intermediate: 32, heads: 4, kvHeads: 2, headDim: 8, eps: 1e-6, ropeTheta: 1_000_000,
                                           vocab: 128, taps: [1, 2, 3])

    /// The golden's `w.*` tensors as one shard.
    static func weights(dropping: (String) -> Bool = { _ in false }) -> [String: MLXArray] {
        golden.filter { $0.key.hasPrefix("w.") && !dropping($0.key) }
    }

    static func encoder(shards: [Flux2KleinTextEncoder.Shard]? = nil) throws -> Flux2KleinTextEncoder {
        try Flux2KleinTextEncoder.load(shape: tiny, prefix: "w.", shards: shards ?? [{ weights() }], bits: nil)
    }

    static func ids(_ array: MLXArray) -> MLXArray { array.asType(.int32) }

    // Measured 2026-10-31 on this port against the golden: 1 - cosine <= 7.1e-15 on every tap (max-abs <= 7.5e-9). The floor sits far below that
    // (set after measuring, not tuned to pass); the owner's target was >= 0.99999.
    static let cosineFloor = 0.999999

    @Test func goldenIsPresent() {
        #expect(Self.golden["out.prompt_embeds"] != nil && Self.golden["w.embed_tokens.weight"] != nil)
    }

    @Test func everyTapMatchesDiffusersOnTheTinyModel() throws {
        let g = Self.golden
        let input = Self.ids(try #require(g["in.input_ids"]))
        let mask = MLXArray.ones(like: input)
        let encoder = try Self.encoder()
        let out = try encoder(ids: input, attentionMask: mask)
        eval(out)
        let expected = try #require(g["out.prompt_embeds"])
        #expect(out.shape == expected.shape && out.shape == [1, 8, 48])
        for (n, tap) in Self.tiny.taps.enumerated() {
            let mine = out[0..., 0..., (n * 16) ..< ((n + 1) * 16)]
            let want = try #require(g["out.hidden_\(tap)"])
            let c = try KleinCosine.of(mine, want)
            let worst = try MLXErrorGuard.read(abs(mine - want).max().reshaped(1), as: Float.self)[0]
            print("TINYPARITY tap after \(tap) layers: cosine \(c) (1-cos = \(1 - c)) max-abs \(worst)")
            #expect(c >= Self.cosineFloor, "tap after \(tap) layers")
        }
        let all = try KleinCosine.of(out, expected)
        print("TINYPARITY concatenated: cosine \(all) (1-cos = \(1 - all))")
        #expect(all >= Self.cosineFloor)
    }

    @Test func theLastTapIsBeforeAnyFinalNorm() throws {
        // hidden_states[3] of the 4-layer golden is NOT the normed last_hidden_state; the port has no final norm, so it must differ from it.
        let g = Self.golden
        let input = Self.ids(try #require(g["in.input_ids"]))
        let out = try Self.encoder()(ids: input, attentionMask: MLXArray.ones(like: input))
        let lastTap = out[0..., 0..., 32 ..< 48]
        let afterThree = try #require(g["out.hidden_3"]), final = try #require(g["out.final_norm"])
        let toTap = try MLXErrorGuard.read(abs(lastTap - afterThree).max().reshaped(1), as: Float.self)[0]
        let toFinal = try MLXErrorGuard.read(abs(lastTap - final).max().reshaped(1), as: Float.self)[0]
        #expect(toTap < 1e-4 && toFinal > 1e-2, "tap-to-hidden_3 \(toTap), tap-to-final-norm \(toFinal)")
    }

    @Test func aPaddedKeyIsInvisibleToEveryRowAfterIt() throws {
        // Right-padded to 8: 5 real + 3 pad. Changing the id sitting at the FIRST pad position must change that row only. A plain causal
        // mask would let rows 6 and 7 attend to it; the padding mask must not.
        let real: [Int32] = [5, 9, 17, 40, 77]
        func run(_ firstPad: Int32) throws -> MLXArray {
            let ids = MLXArray(real + [firstPad, 3, 3], [1, 8])
            let mask = MLXArray([1, 1, 1, 1, 1, 0, 0, 0] as [Int32], [1, 8])
            let out = try Self.encoder()(ids: ids, attentionMask: mask)
            eval(out)
            return out
        }
        let a = try run(11), b = try run(99)
        func rows(_ x: MLXArray, _ r: Range<Int>) -> MLXArray { x[0..., r, 0...] }
        #expect(try KleinCosine.of(rows(a, 0 ..< 5), rows(b, 0 ..< 5)) > 0.999999, "real rows do not see padding at all")
        let drift = try MLXErrorGuard.read(abs(rows(a, 6 ..< 8) - rows(b, 6 ..< 8)).max().reshaped(1), as: Float.self)[0]
        #expect(drift < 1e-5, "rows after a padded key attend to real tokens only; drift \(drift)")
        let own = try MLXErrorGuard.read(abs(rows(a, 5 ..< 6) - rows(b, 5 ..< 6)).max().reshaped(1), as: Float.self)[0]
        #expect(own > 1e-3, "the padded row itself still depends on its own embedding; diff \(own)")
    }

    @Test func realRowsDoNotChangeWhenPaddingIsAppended() throws {
        let input = Self.ids(try #require(Self.golden["in.input_ids"]))
        let short = try Self.encoder()(ids: input, attentionMask: MLXArray.ones(like: input))
        let padded = MLX.concatenated([input, MLXArray([3, 3, 3, 3] as [Int32], [1, 4])], axis: 1)
        let mask = MLX.concatenated([MLXArray.ones(like: input), MLXArray.zeros([1, 4], dtype: .int32)], axis: 1)
        let long = try Self.encoder()(ids: padded, attentionMask: mask)
        eval(short, long)
        #expect(try KleinCosine.of(short, long[0..., 0 ..< 8, 0...]) > 0.999999)
    }

    @Test func aLayerSplitAcrossTwoShardsLoadsTheSame() throws {
        let whole = Self.golden["in.input_ids"].map(Self.ids)!
        let mask = MLXArray.ones(like: whole)
        let one = try Self.encoder()(ids: whole, attentionMask: mask)
        // Shard 1: embeddings + layers 0-1 + half of layer 2; shard 2: the rest of layer 2, layer 3 (past the deepest tap) and the final norm.
        func first(_ k: String) -> Bool { k.contains("embed_tokens") || k.contains("layers.0.") || k.contains("layers.1.") || (k.contains("layers.2.") && k.contains("self_attn")) }
        let two = try Self.encoder(shards: [{ Self.weights().filter { first($0.key) } }, { Self.weights().filter { !first($0.key) } }])(ids: whole, attentionMask: mask)
        eval(one, two)
        #expect(try KleinCosine.of(one, two) > 0.9999999)
    }

    @Test func layersPastTheDeepestTapAreNeitherNeededNorBuilt() throws {
        let encoder = try Self.encoder(shards: [{ Self.weights { $0.contains("layers.3.") || $0.contains(".norm.") && !$0.contains("layernorm") && !$0.contains("q_norm") && !$0.contains("k_norm") } }])
        #expect(encoder.layers.count == 3)
    }

    @Test func aMissingWeightIsNamed() {
        #expect(throws: Flux2KleinTextEncoderError.self) {
            _ = try Self.encoder(shards: [{ Self.weights { $0.contains("layers.1.mlp.down_proj") } }])
        }
        #expect(throws: Flux2KleinTextEncoderError.missingWeight("w.embed_tokens.weight")) {
            _ = try Self.encoder(shards: [{ Self.weights { $0.contains("embed_tokens") } }])
        }
    }

    @Test(arguments: [1, 2, 7, 9, 16]) func unsupportedBitsAreRefusedWithASentence(bits: Int) {
        #expect(throws: Flux2KleinTextEncoderError.unsupportedBits(bits)) {
            _ = try Flux2KleinTextEncoder.load(shape: Self.tiny, prefix: "w.", shards: [{ Self.weights() }], bits: bits)
        }
        #expect(Flux2KleinTextEncoderError.unsupportedBits(7).errorDescription?.contains("3, 4, 5, 6 or 8") == true)
    }

    @Test func aCancelledLoadThrowsCancellationNotAnMLXError() async {
        let task = Task { () throws -> Bool in
            withUnsafeCurrentTask { $0?.cancel() }
            _ = try Self.encoder()
            return true
        }
        let result = await task.result
        if case let .failure(error) = result { #expect(error is CancellationError) } else { Issue.record("a cancelled load must stop") }
    }

    @Test func realShapeMatchesTheDecodedConfig() throws {
        let config = try Flux2KleinTextEncoderConfig.decode(from: Data(Klein3c.textEncoderJSON.utf8))
        let s = Flux2KleinQwen3Shape(config)
        #expect(s.taps == [9, 18, 27] && s.layerCount == 27)
        #expect(s.hidden == 2560 && s.heads == 32 && s.kvHeads == 8 && s.headDim == 128 && s.vocab == 151_936)
    }
}

enum Klein3c {
    static let textEncoderJSON = """
{"hidden_size": 2560, "intermediate_size": 9728, "num_hidden_layers": 36, "num_attention_heads": 32, "num_key_value_heads": 8,
 "head_dim": 128, "vocab_size": 151936, "rms_norm_eps": 1e-06, "rope_theta": 1000000, "tie_word_embeddings": true}
"""
}
