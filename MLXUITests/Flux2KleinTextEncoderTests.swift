import CryptoKit
import Foundation
import MLX
import MLXNN
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

    // MARK: MP7-3c-FIX-1

    @Test func aLayerMissingSeveralTensorsNamesThemAll() {
        #expect(throws: Flux2KleinTextEncoderError.missingWeight("w.layers.1.mlp.down_proj.weight, w.layers.1.self_attn.q_norm.weight")) {
            _ = try Self.encoder(shards: [{ Self.weights { $0.contains("layers.1.mlp.down_proj") || $0.contains("layers.1.self_attn.q_norm") } }])
        }
        // A layer that never arrives lists all eleven.
        do {
            _ = try Self.encoder(shards: [{ Self.weights { $0.contains("layers.2.") } }])
            Issue.record("a layer with no tensors must be refused")
        } catch let Flux2KleinTextEncoderError.missingWeight(names) {
            #expect(names.components(separatedBy: ", ").count == 11 && names.hasPrefix("w.layers.2.") )
        } catch { Issue.record("\(error)") }
    }

    @Test func anExtraOrRenamedPerLayerTensorIsRefusedByName() throws {
        var extra = Self.weights()
        extra["w.layers.0.self_attn.q_proj.bias"] = MLXArray.zeros([16])
        #expect(throws: Flux2KleinTextEncoderError.unexpectedWeight("w.layers.0.self_attn.q_proj.bias")) {
            _ = try Self.encoder(shards: [{ extra }])
        }
        var renamed = Self.weights()
        renamed["w.layers.2.mlp.up_projection.weight"] = renamed.removeValue(forKey: "w.layers.2.mlp.up_proj.weight")
        #expect(throws: Flux2KleinTextEncoderError.unexpectedWeight("w.layers.2.mlp.up_projection.weight")) {
            _ = try Self.encoder(shards: [{ renamed }])
        }
        // An extra tensor on a layer that is still incomplete is refused too, not silently parked.
        var partial = Self.weights { $0.contains("layers.1.mlp.down_proj") }
        partial["w.layers.1.mlp.surprise.weight"] = MLXArray.zeros([4])
        #expect(throws: Flux2KleinTextEncoderError.unexpectedWeight("w.layers.1.mlp.surprise.weight")) {
            _ = try Self.encoder(shards: [{ partial }])
        }
        #expect(Flux2KleinQwen3Layer.expectedTensorNames.count == 11)
    }

    // A shape whose widths are multiples of 64, so MLX can really quantize it; random weights (no golden: this tests the policy, not the numbers).
    static let quantizable = Flux2KleinQwen3Shape(hidden: 64, intermediate: 128, heads: 2, kvHeads: 1, headDim: 32, eps: 1e-6, ropeTheta: 1_000_000,
                                                  vocab: 128, taps: [1, 2])

    static func randomWeights(_ s: Flux2KleinQwen3Shape) -> [String: MLXArray] {
        var w: [String: MLXArray] = ["w.embed_tokens.weight": MLXRandom.normal([s.vocab, s.hidden]) * 0.05]
        let shapes: [String: [Int]] = [
            "input_layernorm.weight": [s.hidden], "post_attention_layernorm.weight": [s.hidden],
            "self_attn.q_proj.weight": [s.heads * s.headDim, s.hidden], "self_attn.k_proj.weight": [s.kvHeads * s.headDim, s.hidden],
            "self_attn.v_proj.weight": [s.kvHeads * s.headDim, s.hidden], "self_attn.o_proj.weight": [s.hidden, s.heads * s.headDim],
            "self_attn.q_norm.weight": [s.headDim], "self_attn.k_norm.weight": [s.headDim],
            "mlp.gate_proj.weight": [s.intermediate, s.hidden], "mlp.up_proj.weight": [s.intermediate, s.hidden], "mlp.down_proj.weight": [s.hidden, s.intermediate],
        ]
        for layer in 0 ..< s.layerCount {
            for (name, shape) in shapes { w["w.layers.\(layer).\(name)"] = name.contains("norm") ? MLXArray.ones(shape) : MLXRandom.normal(shape) * 0.05 }
        }
        return w
    }

    @Test(arguments: [3, 4, 5, 6, 8]) func theTextEncoderIsNeverQuantizedBelowEightBits(requested: Int) throws {
        let weights = Self.randomWeights(Self.quantizable)
        let encoder = try Flux2KleinTextEncoder.load(shape: Self.quantizable, prefix: "w.", shards: [{ weights }], bits: requested, groupSize: 64)
        #expect(encoder.bits == 8, "asked for \(requested), loaded at \(String(describing: encoder.bits))")
        #expect((encoder.embedTokens as? QuantizedEmbedding)?.bits == 8, "the embedding table too")
        for layer in encoder.layers {
            #expect((layer.attention.qProj as? QuantizedLinear)?.bits == 8)
            #expect((layer.mlp.down as? QuantizedLinear)?.bits == 8)
        }
        let ids = MLXArray([3, 9, 20, 5] as [Int32], [1, 4])
        let out = try encoder(ids: ids, attentionMask: MLXArray.ones(like: ids))
        eval(out)
        let reference = try Flux2KleinTextEncoder.load(shape: Self.quantizable, prefix: "w.", shards: [{ weights }], bits: nil)
        let ref = try reference(ids: ids, attentionMask: MLXArray.ones(like: ids))
        eval(ref)
        #expect(try KleinCosine.of(out, ref) > 0.99, "an 8-bit encoder tracks the unquantized one")
        #expect(reference.bits == nil && !(reference.embedTokens is QuantizedEmbedding))
    }

    @Test func theBitFloorLivesInOnePlaceWithTheRowWording() {
        #expect(Flux2KleinQuantization.textEncoderMinimumBits == 8)
        #expect(Flux2KleinQuantization.textEncoderBits(for: 4) == 8 && Flux2KleinQuantization.textEncoderBits(for: 8) == 8)
        #expect(Flux2KleinQuantization.textEncoderBits(for: nil) == nil)
        #expect(Flux2KleinQuantization.textEncoderNote.contains("text encoder stays 8-bit"))
    }

    // MARK: Padded input against diffusers (g3c_padded.safetensors, made by padded_goldens.py with g3c's own weights, never edited)

    static let padded: [String: MLXArray] = {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein/g3c_padded.safetensors")
        return (try? loadArrays(url: url)) ?? [:]
    }()

    @Test(arguments: [("pad5_11", 5), ("pad7_9", 7)]) func paddedRowsMatchDiffusersToo(name: String, real: Int) throws {
        let ids = Self.ids(try #require(Self.padded["\(name).in.input_ids"])), mask = Self.ids(try #require(Self.padded["\(name).in.mask"]))
        let expected = try #require(Self.padded["\(name).out.prompt_embeds"])
        #expect(ids.shape == [1, 16] && expected.shape == [1, 16, 48])
        let out = try Self.encoder()(ids: ids, attentionMask: mask)
        eval(out)
        let all = try KleinCosine.of(out, expected)
        let pad = try KleinCosine.of(out[0..., real..., 0...], expected[0..., real..., 0...])
        let front = try KleinCosine.of(out[0..., 0 ..< real, 0...], expected[0..., 0 ..< real, 0...])
        let worst = try MLXErrorGuard.read(abs(out - expected).max().reshaped(1), as: Float.self)[0]
        var tapLine = ""
        for n in 0 ..< 3 { tapLine += " tap\(n): \(try KleinCosine.of(out[0..., 0..., (n * 16) ..< ((n + 1) * 16)], expected[0..., 0..., (n * 16) ..< ((n + 1) * 16)]))" }
        print("PADPARITY \(name): all \(all) real rows \(front) pad rows \(pad) max-abs \(worst)\(tapLine)")
        #expect(all >= Self.paddedCosineFloor && pad >= Self.paddedCosineFloor && front >= Self.paddedCosineFloor)
        #expect(Float(worst) < Self.paddedMaxAbsCeiling)
    }

    // Measured 2026-10-31 (fp32, diffusers 0.37.1 golden): 1 - cosine <= 7e-15 on every tap and block, max-abs <= 1.1e-8, for both cases. Floors set after.
    static let paddedCosineFloor = 0.999999
    static let paddedMaxAbsCeiling: Float = 1e-5

    @Test func thePaddedGoldenIsTheOneThatWasGenerated() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein/g3c_padded.safetensors")
        let digest = SHA256.hash(data: try Data(contentsOf: url)).map { String(format: "%02x", $0) }.joined()
        #expect(digest == "3a70abe2b522e44975a4f7cc3119324be92105cd9342177506be2883feeca339")
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
