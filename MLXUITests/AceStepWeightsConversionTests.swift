import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3c: the ACE-Step weight conversions against upstream's own output (`Fixtures/MC5/mc5_convert_tiny.safetensors`,
/// provenance in `Fixtures/MC5/PROVENANCE.md`: real checkpoint tensors sliced on dim 0, and `ace-step/ACE-Step-1.5` @ ca1e85f's
/// converters run unchanged), the strict-names helper, and the selective reader on small synthetic files. No real weights.
struct AceStepWeightsConversionTests {

    // MARK: Fixture

    static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()   // MLXUITests
        .deletingLastPathComponent()   // repo root
        .appendingPathComponent("Fixtures/MC5/mc5_convert_tiny.safetensors")

    static let ditInputs = [
        "in.decoder.proj_in.1.weight", "in.decoder.proj_in.1.bias",
        "in.decoder.proj_out.1.weight", "in.decoder.proj_out.1.bias",
        "in.decoder.norm_out.weight", "in.decoder.time_embed.linear_1.weight",
    ]

    /// (dit inputs by their original `decoder.` name, vae inputs by their original name, upstream dit outputs, upstream vae outputs)
    static func load() throws -> (dit: [String: MLXArray], vae: [String: MLXArray],
                                  ditOut: [String: MLXArray], vaeOut: [String: MLXArray]) {
        let index = try SafetensorsIndex(url: fixtureURL)
        let all = try index.read(index.names)
        var dit: [String: MLXArray] = [:], vae: [String: MLXArray] = [:]
        var ditOut: [String: MLXArray] = [:], vaeOut: [String: MLXArray] = [:]
        for (key, value) in all {
            if key.hasPrefix("in.") {
                let name = String(key.dropFirst(3))
                if ditInputs.contains(key) { dit[name] = value } else { vae[name] = value }
            } else if key.hasPrefix("out.dit.") {
                ditOut[String(key.dropFirst("out.dit.".count))] = value
            } else if key.hasPrefix("out.vae.") {
                vaeOut[String(key.dropFirst("out.vae.".count))] = value
            }
        }
        return (dit, vae, ditOut, vaeOut)
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    // MARK: Tiny-tier parity

    @Test func fixtureHasTheExpectedShape() throws {
        let f = try Self.load()
        #expect(f.dit.count == 6 && f.ditOut.count == 6)
        #expect(f.vae.count == 21)
        #expect(f.vaeOut.count == 15)
        #expect(f.dit.count + f.vae.count + f.ditOut.count + f.vaeOut.count == 48)
    }

    @Test func ditConversionMatchesUpstream() throws {
        let f = try Self.load()
        let got = try AceStepWeights.convertDecoder(f.dit)
        #expect(Set(got.keys) == Set(f.ditOut.keys))
        for (name, want) in f.ditOut {
            let g = try #require(got[name], "\(name)")
            #expect(g.shape == want.shape, "\(name): \(g.shape) vs \(want.shape)")
            #expect(Self.maxAbsDiff(g, want) <= 1e-6, "\(name)")
        }
        // The layouts, spelled out.
        #expect(got["proj_in.weight"]?.shape == [16, 2, 192])
        #expect(got["proj_out.weight"]?.shape == [64, 2, 16])
        #expect(got["time_embed.linear_1.weight"]?.shape == [2048, 256])
        #expect(got["norm_out.weight"]?.shape == [2048])
    }

    @Test func vaeConversionMatchesUpstream() throws {
        let f = try Self.load()
        let got = try AceStepWeights.convertVAE(f.vae)
        #expect(Set(got.keys) == Set(f.vaeOut.keys))
        for (name, want) in f.vaeOut {
            let g = try #require(got[name], "\(name)")
            #expect(g.shape == want.shape, "\(name): \(g.shape) vs \(want.shape)")
            #expect(g.dtype == .float32, "\(name) is float32")
            #expect(Self.maxAbsDiff(g, want) <= 1e-6, "\(name)")
        }
        // Three Conv1d pairs, two conv_t1 pairs, snake, and conv2 without a bias.
        #expect(got["decoder.conv1.weight"]?.shape == [16, 7, 64])
        #expect(got["decoder.block.4.res_unit1.conv1.weight"]?.shape == [128, 7, 128])
        #expect(got["decoder.block.4.res_unit1.conv2.weight"]?.shape == [128, 1, 128])
        #expect(got["decoder.block.0.conv_t1.weight"]?.shape == [1024, 20, 8])
        #expect(got["decoder.block.4.conv_t1.weight"]?.shape == [128, 4, 128])
        #expect(got["decoder.snake1.alpha"]?.shape == [128])
        #expect(got["decoder.block.4.res_unit1.snake1.beta"]?.shape == [128])
        #expect(got["decoder.conv2.bias"] == nil)
        #expect(got["decoder.conv2.weight"] != nil)
        #expect(!got.keys.contains { $0.hasSuffix(".weight_g") || $0.hasSuffix(".weight_v") })
    }

    /// Fuse-then-transpose is the only correct order. For the ConvTranspose1d `block.4.conv_t1` (`[in=128, out=128, K=4]`)
    /// transposing first moves `in` off dim 0, so the per-row norm and the `g` scale land on the wrong axis: that result
    /// must differ from upstream's by far more than the 1e-3 floor, so a converter that swapped the order fails
    /// `vaeConversionMatchesUpstream` above.
    @Test func fuseThenTransposeOrderMatters() throws {
        let f = try Self.load()
        let base = "decoder.block.4.conv_t1"
        let g = try #require(f.vae[base + ".weight_g"]), v = try #require(f.vae[base + ".weight_v"])
        let want = try #require(f.vaeOut[base + ".weight"])
        let right = AceStepWeights.fuseWeightNorm(g: g, v: v).transposed(1, 2, 0)
        #expect(Self.maxAbsDiff(right, want) <= 1e-6)
        let wrong = AceStepWeights.fuseWeightNorm(g: g, v: v.transposed(1, 2, 0))
        #expect(wrong.shape == want.shape)
        #expect(Self.maxAbsDiff(wrong, want) > 1e-3, "transpose-then-fuse must not reproduce upstream")
    }

    /// The epsilon is added to the norm (`norm + 1e-9`), not used as a clamp floor: a zero row stays zero, not NaN.
    @Test func epsilonIsAddedToTheNorm() {
        let g = MLXArray([Float(2), 3], [2, 1, 1])
        let v = MLXArray([Float(0), 0, 0, 0, 3, 4, 0, 0], [2, 2, 2])   // row 0 is all zero, row 1 has norm 5
        let w = AceStepWeights.fuseWeightNorm(g: g, v: v)
        let flat = w.asArray(Float.self)
        #expect(flat.allSatisfy { $0.isFinite })
        #expect(flat[0..<4].allSatisfy { $0 == 0 })
        #expect(abs(flat[4] - 3 * 3 / 5) < 1e-6)
        #expect(abs(flat[5] - 3 * 4 / 5) < 1e-6)
    }

    @Test func decoderConversionDropsRotaryAndRejectsForeignNames() throws {
        let t = MLXArray.zeros([2, 2])
        let got = try AceStepWeights.convertDecoder(["decoder.rotary_emb.inv_freq": t, "decoder.layers.0.x.weight": t])
        #expect(Set(got.keys) == ["layers.0.x.weight"])
        #expect(throws: AceStepWeightsError.unexpectedTensor("encoder.a")) {
            _ = try AceStepWeights.convertDecoder(["encoder.a": t])
        }
    }

    @Test func decoderConversionKeepsBF16() throws {
        let w = MLXArray.zeros([4, 6, 2]).asType(.bfloat16)
        let got = try AceStepWeights.convertDecoder(["decoder.proj_in.1.weight": w, "decoder.layers.0.q.weight": w])
        #expect(got["proj_in.weight"]?.dtype == .bfloat16)
        #expect(got["layers.0.q.weight"]?.dtype == .bfloat16)
        #expect(got["proj_in.weight"]?.shape == [4, 2, 6])
    }

    @Test func aLoneWeightNormHalfIsNamed() {
        let v = MLXArray.zeros([2, 2, 2])
        #expect(throws: AceStepWeightsError.missingTensor("decoder.conv1.weight_g")) {
            _ = try AceStepWeights.convertVAE(["decoder.conv1.weight_v": v])
        }
        #expect(throws: AceStepWeightsError.missingTensor("decoder.conv1.weight_v")) {
            _ = try AceStepWeights.convertVAE(["decoder.conv1.weight_g": MLXArray.zeros([2, 1, 1])])
        }
    }

    // MARK: Strict names

    @Test func strictNamesNameTheFirstMissingThenTheFirstUnused() {
        let expected = ["a.weight", "b.weight", "c.weight"]
        #expect(throws: Never.self) { try AceStepWeights.checkStrict(actual: expected, expected: expected) }
        // First missing, in sorted order, even when others are unused too.
        #expect(throws: AceStepWeightsError.missingTensor("b.weight")) {
            try AceStepWeights.checkStrict(actual: ["a.weight", "z.weight", "y.weight"], expected: expected)
        }
        #expect(throws: AceStepWeightsError.unusedTensor("y.weight")) {
            try AceStepWeights.checkStrict(actual: expected + ["z.weight", "y.weight"], expected: expected)
        }
        let message = AceStepWeightsError.missingTensor("b.weight").description
        #expect(message.contains("'b.weight'"))
    }

    // MARK: Selective reader (synthetic files, no real weights)

    struct Spec { let name: String; let dtype: String; let shape: [Int]; let bytes: Data }

    static func f32(_ values: [Float], _ shape: [Int], _ name: String) -> Spec {
        Spec(name: name, dtype: "F32", shape: shape, bytes: values.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    static func bf16(_ values: [Float], _ shape: [Int], _ name: String) -> Spec {
        var d = Data()
        for v in values { var h = UInt16(v.bitPattern >> 16); withUnsafeBytes(of: &h) { d.append(contentsOf: $0) } }
        return Spec(name: name, dtype: "BF16", shape: shape, bytes: d)
    }

    static func write(_ specs: [Spec], to url: URL) throws {
        var header: [String: Any] = [:]
        var body = Data()
        for s in specs {
            header[s.name] = ["dtype": s.dtype, "shape": s.shape, "data_offsets": [body.count, body.count + s.bytes.count]]
            body.append(s.bytes)
        }
        let json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var out = Data()
        var len = UInt64(json.count).littleEndian
        withUnsafeBytes(of: &len) { out.append(contentsOf: $0) }
        out.append(json); out.append(body)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try out.write(to: url)
    }

    /// A tiny installed folder: the three files at the real relative paths with real-shaped (small) tensors.
    static func makeModelDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-weights-\(UUID().uuidString)")
        try write([
            bf16(Array(repeating: 1, count: 4 * 6 * 2), [4, 6, 2], "decoder.proj_in.1.weight"),
            bf16([1, 2, 3, 4], [4], "decoder.proj_in.1.bias"),
            bf16(Array(repeating: 2, count: 4 * 3 * 2), [4, 3, 2], "decoder.proj_out.1.weight"),
            bf16([5, 6], [2], "decoder.layers.0.q.weight"),
            bf16([7, 8], [2], "decoder.rotary_emb.inv_freq"),
            bf16([9, 10], [2], "encoder.text_projector.weight"),
            bf16([11], [1, 1, 1], "null_condition_emb"),
            bf16([12, 13], [2], "tokenizer.quantizer.weight"),
            bf16([14, 15], [2], "detokenizer.proj.weight"),
        ], to: dir.appendingPathComponent(AceStepWeights.turboPath))
        try write([
            bf16([1, 2, 3, 4, 5, 6], [3, 2], "embed_tokens.weight"),
            bf16([1], [1], "norm.weight"),
            bf16([1], [1], "layers.0.self_attn.q_proj.weight"),
        ], to: dir.appendingPathComponent(AceStepWeights.qwenPath))
        try write([
            f32([3, 4, 0, 0], [1, 2, 2], "decoder.conv1.weight_v"),
            f32([2], [1, 1, 1], "decoder.conv1.weight_g"),
            f32([0.5], [1], "decoder.conv1.bias"),
            f32([1, 2], [1, 2, 1], "decoder.snake1.alpha"),
            f32([9, 9], [2], "encoder.block.0.conv1.bias"),
        ], to: dir.appendingPathComponent(AceStepWeights.vaePath))
        return dir
    }

    @Test func turboReaderTakesOnlyTheAskedGroupsAndKeepsBF16() throws {
        let dir = try Self.makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let got = try AceStepWeights.readTurbo(directory: dir, prefixes: AceStepWeights.turboReadPrefixes,
                                               names: AceStepWeights.turboReadNames)
        #expect(Set(got.keys) == ["decoder.proj_in.1.weight", "decoder.proj_in.1.bias", "decoder.proj_out.1.weight",
                                  "decoder.layers.0.q.weight", "decoder.rotary_emb.inv_freq",
                                  "encoder.text_projector.weight", "null_condition_emb"])
        #expect(got.values.allSatisfy { $0.dtype == .bfloat16 }, "bf16 stays bf16 (R3)")
        #expect(got["decoder.proj_in.1.bias"]?.asType(.float32).asArray(Float.self) == [1, 2, 3, 4])
    }

    @Test func theTokenizerAndDetokenizerAreNeverRead() throws {
        #expect(!AceStepWeights.turboReadPrefixes.contains { $0.hasPrefix("tokenizer") || $0.hasPrefix("detokenizer") })
        #expect(AceStepWeights.turboNeverReadPrefixes == ["tokenizer.", "detokenizer."])
        let dir = try Self.makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        for forbidden in ["tokenizer.", "detokenizer.", "tokenizer.quantizer.weight", ""] {
            #expect(throws: AceStepWeightsError.forbiddenGroup(forbidden)) {
                _ = try AceStepWeights.readTurbo(directory: dir, prefixes: [forbidden])
            }
        }
        #expect(throws: AceStepWeightsError.forbiddenGroup("detokenizer.proj.weight")) {
            _ = try AceStepWeights.readTurbo(directory: dir, prefixes: ["decoder."], names: ["detokenizer.proj.weight"])
        }
        let all = try AceStepWeights.readTurbo(directory: dir, prefixes: ["decoder.", "encoder."], names: ["null_condition_emb"])
        #expect(!all.keys.contains { $0.hasPrefix("tokenizer.") || $0.hasPrefix("detokenizer.") })
    }

    @Test func groupLoadersOnASyntheticFolder() throws {
        let dir = try Self.makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }

        let dit = try AceStepWeights.loadDecoder(directory: dir)
        #expect(Set(dit.keys) == ["proj_in.weight", "proj_in.bias", "proj_out.weight", "layers.0.q.weight"])
        #expect(dit["proj_in.weight"]?.shape == [4, 2, 6])
        #expect(dit["proj_out.weight"]?.shape == [3, 2, 4])
        #expect(dit.values.allSatisfy { $0.dtype == .bfloat16 })

        let enc = try AceStepWeights.loadConditionEncoder(directory: dir)
        #expect(Set(enc.keys) == ["encoder.text_projector.weight"])
        #expect(try AceStepWeights.loadNullConditionEmbedding(directory: dir).shape == [1, 1, 1])

        let table = try AceStepWeights.loadQwenEmbedTokens(directory: dir)
        #expect(table.shape == [3, 2] && table.dtype == .bfloat16)
        let qwen = try AceStepWeights.readQwen(directory: dir, prefixes: ["layers."])
        #expect(Set(qwen.keys) == ["layers.0.self_attn.q_proj.weight"])

        // The VAE's own `encoder.*` is never read; the decoder pair is fused: 2 * [3, 4, 0, 0] / 5 = [1.2, 1.6, 0, 0].
        let vae = try AceStepWeights.loadVAEDecoder(directory: dir)
        #expect(Set(vae.keys) == ["decoder.conv1.weight", "decoder.conv1.bias", "decoder.snake1.alpha"])
        #expect(vae["decoder.conv1.weight"]?.shape == [1, 2, 2])
        #expect(vae["decoder.snake1.alpha"]?.shape == [2])
        let w = try #require(vae["decoder.conv1.weight"]).asArray(Float.self)
        // v [1,2,2] = [[3,4],[0,0]] (in=2 rows of K=2) fused then swapped on axes 1,2 -> [[1.2,0],[1.6,0]].
        #expect(w.enumerated().allSatisfy { abs($0.element - [1.2, 0, 1.6, 0][$0.offset]) < 1e-6 })
    }

    @Test func missingFilesAndTensorsAreNamed() throws {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-empty-\(UUID().uuidString)")
        #expect(throws: AceStepWeightsError.missingFile("model.safetensors")) {
            _ = try AceStepWeights.loadDecoder(directory: empty)
        }
        let dir = try Self.makeModelDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: AceStepWeightsError.missingTensor("nope")) {
            _ = try AceStepWeights.readQwen(directory: dir, prefixes: [], names: ["nope"])
        }
    }
}
