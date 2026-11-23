import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXUI

/// MC5-3f: the DiT against upstream's own tiny-config output (`Fixtures/MC5/mc5_dit_tiny.safetensors`: torch weights, the torch forward
/// `fwd.out.vt_torch` and the native-MLX forward `fwd.out.vt_mlx`). Weights go through the real 3c `convertDecoder`. No real weights.
struct AceStepDiTTests {

    static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/MC5/mc5_dit_tiny.safetensors")

    struct Fixture {
        let config: AceStepDiTConfig
        let weights: [String: MLXArray]      // converted (MLX layout, no `decoder.` prefix)
        let tensors: [String: MLXArray]
    }

    static func load() throws -> Fixture {
        let index = try SafetensorsIndex(url: fixtureURL)
        let all = try index.read(index.names)
        let json = try #require(index.metadata["config"]).data(using: .utf8) ?? Data()
        var raw: [String: MLXArray] = [:]
        for (k, v) in all where k.hasPrefix("w.") { raw["decoder." + k.dropFirst(2)] = v }
        return Fixture(config: try AceStepDiTConfig(json: json), weights: try AceStepWeights.convertDecoder(raw), tensors: all)
    }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private func run(_ net: AceStepDiT, _ f: Fixture, hidden: MLXArray? = nil, context: MLXArray? = nil) throws -> MLXArray {
        func t(_ k: String) throws -> MLXArray { try #require(f.tensors[k], "\(k)") }
        return net(hidden: try hidden ?? t("fwd.in.xt"), context: try context ?? t("fwd.in.ctx"), encoder: try t("fwd.in.enc"),
                   timestep: try t("fwd.in.t"), timestepR: try t("fwd.in.tr"))
    }

    @Test func tinyConfigParses() throws {
        let c = try Self.load().config
        #expect(c.hiddenSize == 32 && c.attentionHeads == 4 && c.kvHeads == 2 && c.headDim == 8 && c.layers == 4)
        #expect(c.inChannels == 192 && c.outChannels == 64 && c.patchSize == 2 && c.slidingWindow == 4)
        #expect(c.slidingLayers == [true, false, true, false])
    }

    @Test func forwardMatchesUpstreamTorchAndMLX() throws {
        let f = try Self.load()
        let net = try AceStepDiT.load(config: f.config, weights: f.weights)
        let out = try run(net, f)
        let torch = try #require(f.tensors["fwd.out.vt_torch"]), mlx = try #require(f.tensors["fwd.out.vt_mlx"])
        #expect(out.shape == torch.shape)
        let dTorch = Self.maxAbs(out, torch), dMLX = Self.maxAbs(out, mlx)
        print("ACESTEP-DIT-TINY maxAbs vs torch=\(dTorch) vs upstream-mlx=\(dMLX)")
        // Upstream's own two references differ by 3.7e-5 here (its dump script asserts < 1e-4; outputs peak at 5.3), so torch gets that bound.
        #expect(dTorch <= 1e-4, "vs torch")
        #expect(dMLX <= 1e-5, "vs upstream MLX")
    }

    @Test func strictNamesRefuseMissingAndExtra() throws {
        let f = try Self.load()
        #expect(Set(AceStepDiT.expectedParameterNames(f.config)) == Set(f.weights.keys))
        var missing = f.weights
        missing["layers.2.mlp.up_proj.weight"] = nil
        #expect(throws: AceStepWeightsError.self) { _ = try AceStepDiT.load(config: f.config, weights: missing) }
        var extra = f.weights
        extra["layers.9.mlp.up_proj.weight"] = MLXArray.zeros([1])
        #expect(throws: AceStepWeightsError.self) { _ = try AceStepDiT.load(config: f.config, weights: extra) }
    }

    @Test func realConfigNamesAre476() {
        #expect(AceStepDiT.expectedParameterNames(AceStepDiTConfig()).count == 476)
    }

    /// One layer, window 4 tokens: a perturbation at token 0 must not reach token 5 (distance 5) when the layer is sliding, and must when full.
    @Test func slidingLayersDoNotAttendBeyondTheWindow() throws {
        let f = try Self.load()
        let oneLayer = f.weights.filter { !$0.key.hasPrefix("layers.") || $0.key.hasPrefix("layers.0.") }
        var results: [Bool: Float] = [:]
        for sliding in [true, false] {
            var c = f.config
            c.layers = 1; c.slidingLayers = [sliding]
            let net = try AceStepDiT.load(config: c, weights: oneLayer)
            let base = try #require(f.tensors["fwd.in.xt"])
            let bumped = concatenated([base[0..., 0 ..< 2, 0...] + 5, base[0..., 2..., 0...]], axis: 1)
            let a = try run(net, f), b = try run(net, f, hidden: bumped)
            results[sliding] = Self.maxAbs(a[0..., 10 ..< 12, 0...], b[0..., 10 ..< 12, 0...])      // token 5
            #expect(Self.maxAbs(a[0..., 0 ..< 2, 0...], b[0..., 0 ..< 2, 0...]) > 1e-3, "the perturbed token itself changes")
        }
        #expect(results[true] == 0, "sliding: token 5 is outside token 0's window (got \(results[true] ?? -1))")
        #expect((results[false] ?? 0) > 1e-6, "full attention sees token 0")
    }

    /// Odd T is zero-padded to a patch multiple and cropped back (upstream).
    @Test func oddLengthPadsAndCrops() throws {
        let f = try Self.load()
        let net = try AceStepDiT.load(config: f.config, weights: f.weights)
        let xt = try #require(f.tensors["fwd.in.xt"]), ctx = try #require(f.tensors["fwd.in.ctx"])
        let odd = try run(net, f, hidden: xt[0..., 0 ..< 11, 0...], context: ctx[0..., 0 ..< 11, 0...])
        #expect(odd.shape == [1, 11, 64])
        let zeros = { (c: Int) in MLXArray.zeros([1, 1, c]) }
        let even = try run(net, f, hidden: concatenated([xt[0..., 0 ..< 11, 0...], zeros(64)], axis: 1),
                           context: concatenated([ctx[0..., 0 ..< 11, 0...], zeros(128)], axis: 1))
        #expect(Self.maxAbs(odd, even[0..., 0 ..< 11, 0...]) == 0)
    }

    /// bf16 weights with f32 inputs (the real run's dtypes): f32 output, close to the all-f32 result.
    @Test func bf16WeightsWithF32InputsGiveF32Output() throws {
        let f = try Self.load()
        let full = try run(try AceStepDiT.load(config: f.config, weights: f.weights), f)
        let low = try run(try AceStepDiT.load(config: f.config, weights: f.weights.mapValues { $0.asType(.bfloat16) }), f)
        #expect(low.dtype == .float32)
        let cos = ((full * low).sum() / (sqrt(full.square().sum()) * sqrt(low.square().sum()))).item(Float.self)
        #expect(cos > 0.99, "cosine \(cos)")
    }

    @Test func timestepEmbeddingIsCosThenSinOfScaledTime() {
        let e = AceStepTimestepEmbedding.sinusoid(MLXArray([Float(0.5)]))
        #expect(e.shape == [1, 256])
        let v = e.asArray(Float.self)
        #expect(abs(v[0] - cos(Float(500))) < 1e-3 && abs(v[128] - sin(Float(500))) < 1e-3)
    }
}
