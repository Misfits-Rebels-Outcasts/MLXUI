import Testing
import Foundation
import MLX
import MLXNN
@testable import MLXUI

/// MC5-3e: the condition encoder against upstream's own tiny-config output (`Fixtures/MC5/mc5_cond_tiny.safetensors`), `pack_sequences`,
/// the strict key set, the cached silence timbre and `context_latents`. No real weights.
struct AceStepConditionEncoderTests {

    static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/MC5/mc5_cond_tiny.safetensors")

    struct Fixture {
        let config: AceStepConditionConfig
        let weights: [String: MLXArray]      // names without `encoder.`
        let tensors: [String: MLXArray]      // every tensor, as stored
    }

    static func load() throws -> Fixture {
        let index = try SafetensorsIndex(url: fixtureURL)
        let all = try index.read(index.names)
        let json = try #require(index.metadata["config"]).data(using: .utf8) ?? Data()
        var w: [String: MLXArray] = [:]
        for (k, v) in all where k.hasPrefix("w.") { w[String(k.dropFirst(2))] = v }
        return Fixture(config: try AceStepConditionConfig(json: json), weights: w, tensors: all)
    }

    static func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    static func ints(_ a: MLXArray) -> [Int32] { a.asType(.int32).asArray(Int32.self) }

    // MARK: Parity

    @Test func tinyConfigParsesFromTheFixture() throws {
        let f = try Self.load()
        #expect(f.config.hiddenSize == 32 && f.config.attentionHeads == 4 && f.config.kvHeads == 2 && f.config.headDim == 8)
        #expect(f.config.textHiddenDim == 16 && f.config.timbreHiddenDim == 8)
        #expect(f.config.lyricLayers == 2 && f.config.timbreLayers == 2 && f.config.slidingWindow == 4)
        #expect(Array(f.config.slidingLayers.prefix(2)) == [true, false])
    }

    @Test(arguments: ["valid", "padded"])
    func everyOutputMatchesUpstream(_ name: String) throws {
        let f = try Self.load()
        let enc = try AceStepConditionEncoder(config: f.config, weights: f.weights)
        func t(_ k: String) throws -> MLXArray { try #require(f.tensors["\(name).\(k)"], "\(name).\(k)") }
        let textHidden = try t("in.text_hidden"), textMask = try t("in.text_mask").asType(.int32)
        let lyricEmbeds = try t("in.lyric_embeds"), lyricMask = try t("in.lyric_mask").asType(.int32)
        let refer = try t("in.refer")
        let order = Self.ints(try t("in.refer_order")).map(Int.init)

        // text_projector
        let text = enc.projectText(textHidden)
        #expect(text.shape == (try t("out.text_projector")).shape)
        #expect(Self.maxAbs(text, try t("out.text_projector")) <= 1e-5, "\(name) text_projector")

        // lyric encoder
        let lyric = enc.encodeLyrics(lyricEmbeds, mask: lyricMask)
        #expect(lyric.shape == (try t("out.lyric_encoder")).shape)
        #expect(Self.maxAbs(lyric, try t("out.lyric_encoder")) <= 1e-5, "\(name) lyric_encoder")

        // timbre encoder
        let (timbre, timbreMask) = enc.encodeTimbre(referPacked: refer, order: order)
        #expect(timbre.shape == (try t("out.timbre.0")).shape)
        #expect(Self.maxAbs(timbre, try t("out.timbre.0")) <= 1e-5, "\(name) timbre.0")
        #expect(Self.ints(timbreMask) == Self.ints(try t("out.timbre.1")), "\(name) timbre mask")

        // final packed
        let (hidden, mask) = try AceStepConditionEncoder.packBoth(
            lyric: lyric, lyricMask: lyricMask, timbre: timbre, timbreMask: timbreMask, text: text, textMask: textMask)
        #expect(hidden.shape == (try t("out.encoder_hidden_states")).shape)
        #expect(Self.maxAbs(hidden, try t("out.encoder_hidden_states")) <= 1e-5, "\(name) encoder_hidden_states")
        #expect(Self.ints(mask) == Self.ints(try t("out.encoder_mask")), "\(name) encoder mask")
        print("ACESTEP-COND-TINY case=\(name) maxAbs text=\(Self.maxAbs(text, try t("out.text_projector"))) lyric=\(Self.maxAbs(lyric, try t("out.lyric_encoder"))) timbre=\(Self.maxAbs(timbre, try t("out.timbre.0"))) hidden=\(Self.maxAbs(hidden, try t("out.encoder_hidden_states")))")
    }

    /// The same, end to end through `encode(_:)` with the cached silence timbre swapped for the fixture's reference clip: proves the whole
    /// path wires the pieces in upstream's order.
    @Test func encodeUsesTheCachedSilenceTimbreAndComputesItOnce() throws {
        let f = try Self.load()
        // A silence latent with the tiny timbre width; the encoder takes the first 750 frames of it.
        let silence = MLXRandom.normal([800, f.config.timbreHiddenDim], key: MLXRandom.key(3))
        let enc = try AceStepConditionEncoder(config: f.config, weights: f.weights, silenceLatent: silence)
        #expect(enc.timbreEvaluations == 0)
        let a = try enc.silenceTimbre()
        let b = try enc.silenceTimbre()
        #expect(enc.timbreEvaluations == 1, "the second call must not recompute")
        #expect(Self.maxAbs(a.embedding, b.embedding) == 0)
        #expect(a.embedding.shape == [1, 1, f.config.hiddenSize] && Self.ints(a.mask) == [1])

        // The cached value is what the timbre encoder gives for silence[:750] as one clip.
        let refer = silence[0 ..< 750].expandedDimensions(axis: 0)
        let (want, _) = enc.encodeTimbre(referPacked: refer, order: [0])
        #expect(Self.maxAbs(a.embedding, want) <= 1e-6)

        let inputs = AceStepTextInputs(
            textIds: [1, 2, 3], textMask: MLXArray.ones([1, 3], dtype: .int32), lyricIds: [4, 5],
            lyricMask: MLXArray.ones([1, 2], dtype: .int32),
            textHidden: MLXRandom.normal([1, 3, f.config.textHiddenDim], key: MLXRandom.key(4)),
            lyricEmbeds: MLXRandom.normal([1, 2, f.config.textHiddenDim], key: MLXRandom.key(5)))
        let out1 = try enc.encode(inputs)
        let out2 = try enc.encode(inputs)
        #expect(enc.timbreEvaluations == 1, "encode never re-runs the cached timbre")
        #expect(out1.hidden.shape == [1, 2 + 1 + 3, f.config.hiddenSize])
        #expect(Self.ints(out1.mask) == [1, 1, 1, 1, 1, 1])
        #expect(Self.maxAbs(out1.hidden, out2.hidden) == 0)
    }

    // MARK: pack_sequences

    @Test func packSequencesIsAStableValidFirstSort() throws {
        let h1 = MLXArray([Float(10), 20, 30], [1, 3, 1]), h2 = MLXArray([Float(40), 50], [1, 2, 1])
        let m1 = MLXArray([Int32(1), 0, 1], [1, 3]), m2 = MLXArray([Int32(0), 1], [1, 2])
        let (h, m) = try AceStepConditionEncoder.packSequences(h1, h2, mask1: m1, mask2: m2)
        // valid tokens first in their original order (10, 30, 50), then the padding in its original order (20, 40).
        #expect(h.asArray(Float.self) == [10, 30, 50, 20, 40])
        #expect(Self.ints(m) == [1, 1, 1, 0, 0])
        #expect(h.shape == [1, 5, 1])
    }

    @Test func packSequencesKeepsEachBatchRowSeparate() throws {
        let h1 = MLXArray([Float(1), 2, 3, 4], [2, 2, 1]), h2 = MLXArray([Float(5), 6, 7, 8], [2, 2, 1])
        let m1 = MLXArray([Int32(0), 1, 1, 1], [2, 2]), m2 = MLXArray([Int32(1), 0, 0, 0], [2, 2])
        let (h, m) = try AceStepConditionEncoder.packSequences(h1, h2, mask1: m1, mask2: m2)
        // row 0: tokens 1(0) 2(1) 5(1) 6(0) -> 2, 5, 1, 6 ; row 1: 3(1) 4(1) 7(0) 8(0) -> 3, 4, 7, 8
        #expect(h.asArray(Float.self) == [2, 5, 1, 6, 3, 4, 7, 8])
        #expect(Self.ints(m) == [1, 1, 0, 0, 1, 1, 0, 0])
    }

    @Test func allValidKeepsTheConcatenationOrder() throws {
        let h1 = MLXArray([Float(1), 2], [1, 2, 1]), h2 = MLXArray([Float(3)], [1, 1, 1])
        let ones2 = MLXArray.ones([1, 2], dtype: .int32), ones1 = MLXArray.ones([1, 1], dtype: .int32)
        let (h, m) = try AceStepConditionEncoder.packSequences(h1, h2, mask1: ones2, mask2: ones1)
        #expect(h.asArray(Float.self) == [1, 2, 3] && Self.ints(m) == [1, 1, 1])
    }

    @Test func unpackPlacesEachClipAtItsBatchItemAndPosition() {
        let packed = MLXArray([Float(1), 2, 3, 4, 5], [5, 1])
        let (out, mask) = AceStepTimbreEncoder.unpack(packed, order: [0, 1, 0, 1, 1])
        // item 0 holds clips 0 and 2; item 1 holds clips 1, 3, 4.
        #expect(out.shape == [2, 3, 1])
        #expect(out.asArray(Float.self) == [1, 3, 0, 2, 4, 5])
        #expect(Self.ints(mask) == [1, 1, 0, 1, 1, 1])
    }

    // MARK: Names

    /// The analytic key set equals the module's own parameter names, so it cannot drift from the code.
    @Test func expectedNamesEqualTheModulesParameters() throws {
        let f = try Self.load()
        let net = AceStepConditionNetwork(f.config)
        let actual = Set(net.parameters().flattened().map(\.0))
        #expect(actual == Set(AceStepConditionEncoder.expectedParameterNames(f.config)))
        #expect(Set(f.weights.keys) == actual, "the fixture carries exactly the real file's names")
    }

    /// Synthetic key list of the real shape: 140 keys for the real config (1 projector + lyric 2+1+1+8*11 + timbre 2+1+1+4*11 + special token).
    @Test func theRealConfigHas140KeysAndTheFileHeaderAgrees() throws {
        let real = AceStepConditionConfig()
        let names = AceStepConditionEncoder.expectedParameterNames(real)
        #expect(names.count == 140)
        #expect(Set(names).count == 140)
        #expect(names.contains("timbre_encoder.special_token"))
        // Header-level check against the real turbo file when the oracle folder is present; otherwise the count/shape above is what is checked.
        guard let dir = AceStepTextFixtures.modelDirectory,
              FileManager.default.fileExists(atPath: dir.appendingPathComponent(AceStepWeights.turboPath).path) else { return }
        let index = try SafetensorsIndex(url: dir.appendingPathComponent(AceStepWeights.turboPath))
        let actual = index.names(withPrefixes: ["encoder."]).map { String($0.dropFirst("encoder.".count)) }
        #expect(actual.count == 140)
        try AceStepWeights.checkStrict(actual: actual, expected: names)
        // The config the module reads from the real folder is the default one.
        let cfg = try AceStepConditionConfig(json: Data(contentsOf: dir.appendingPathComponent("acestep-v15-turbo/config.json")))
        #expect(cfg.hiddenSize == 2048 && cfg.lyricLayers == 8 && cfg.timbreLayers == 4 && cfg.slidingWindow == 128)
        #expect(cfg.slidingLayers.prefix(8).enumerated().allSatisfy { $0.element == ($0.offset % 2 == 0) })
    }

    @Test func strictLoadNamesTheFirstMissingAndTheFirstUnused() throws {
        let f = try Self.load()
        var missing = f.weights
        missing.removeValue(forKey: "lyric_encoder.layers.1.mlp.up_proj.weight")
        missing.removeValue(forKey: "timbre_encoder.norm.weight")
        #expect(throws: AceStepWeightsError.missingTensor("lyric_encoder.layers.1.mlp.up_proj.weight")) {
            _ = try AceStepConditionEncoder(config: f.config, weights: missing)
        }
        var extra = f.weights
        extra["zzz.extra"] = MLXArray.zeros([1])
        extra["aaa.extra"] = MLXArray.zeros([1])
        #expect(throws: AceStepWeightsError.unusedTensor("aaa.extra")) {
            _ = try AceStepConditionEncoder(config: f.config, weights: extra)
        }
        // The checkpoint's `encoder.` prefix is accepted.
        var prefixed: [String: MLXArray] = [:]
        for (k, v) in f.weights { prefixed["encoder." + k] = v }
        _ = try AceStepConditionEncoder(config: f.config, weights: prefixed)
    }

    // MARK: Masks, silence, context latents

    @Test func slidingMaskIsBidirectionalAndPaddingKeysAreMasked() {
        let m = AceStepMask.bidirectional(length: 5, keyMask: MLXArray([Int32(1), 1, 1, 1, 0], [1, 5]), window: 1, dtype: .float32)
        #expect(m.shape == [1, 1, 5, 5])
        let rows = m.reshaped(5, 5).asArray(Float.self)
        func visible(_ i: Int, _ j: Int) -> Bool { rows[i * 5 + j] == 0 }
        #expect(visible(2, 1) && visible(2, 3) && !visible(2, 0) && !visible(2, 4), "|i-j| <= 1 both ways")
        #expect(!visible(3, 4) && !visible(0, 4), "the padded key 4 is never visible")
        #expect(rows[0 * 5 + 4] == -Float.greatestFiniteMagnitude, "finite minimum, not -inf")
        #expect(AceStepMask.minValue(.float16) == -65504)
        #expect(AceStepMask.minValue(.bfloat16) < -3.38e38)
    }

    @Test func latentFramesFollowUpstream() {
        #expect(AceStepConditionEncoder.latentFrames(duration: 10) == 250)
        #expect(AceStepConditionEncoder.latentFrames(duration: 240) == 6000)
        #expect(AceStepConditionEncoder.latentFrames(duration: 59.9) == 1497, "int(59.9 * 48000) // 1920")
        #expect(AceStepConditionEncoder.latentFrames(duration: 3) == 128, "at least 128 frames")
    }

    @Test func contextLatentsAreSilenceThenAChunkMaskOfOnes() throws {
        let f = try Self.load()
        let silence = MLXRandom.normal([300, 8], key: MLXRandom.key(9))
        let enc = try AceStepConditionEncoder(config: f.config, weights: f.weights, silenceLatent: silence)
        let ctx = try enc.contextLatents(duration: 10)
        #expect(ctx.shape == [1, 250, 16])
        #expect(Self.maxAbs(ctx[0, 0..., 0 ..< 8], silence[0 ..< 250]) == 0)
        #expect(Self.maxAbs(ctx[0, 0..., 8 ..< 16], MLXArray.ones([250, 8])) == 0)
        // Tiling past the stored length, as `_get_silence_latent_slice` does.
        let long = AceStepSilenceLatent.slice(silence, length: 700)
        #expect(long.shape == [700, 8] && Self.maxAbs(long[300 ..< 600], silence) == 0 && Self.maxAbs(long[600 ..< 700], silence[0 ..< 100]) == 0)
    }

    @Test func theBundledSilenceLatentLoads() throws {
        let s = try AceStepSilenceLatent.load()
        #expect(s.shape == [15_000, 64] && s.dtype == .float32)
        #expect(abs(s[0, 0].item(Float.self) - 0.5706896781921387) < 1e-7, "frame 0 channel 0, read from the .pt (3b)")
        #expect(abs(s[14_999, 63].item(Float.self) - 1.2298461198806763) < 1e-7)
    }

    @Test func withoutASilenceLatentTheDerivedValuesThrowByName() throws {
        let f = try Self.load()
        let enc = try AceStepConditionEncoder(config: f.config, weights: f.weights)
        #expect(throws: AceStepWeightsError.missingFile("silence_latent_f32.bin")) { _ = try enc.silenceTimbre() }
        #expect(throws: AceStepWeightsError.missingFile("silence_latent_f32.bin")) { _ = try enc.contextLatents(duration: 10) }
    }

    /// MC5-FIX-finish-gather: an empty sequence is refused by name, not handed to MLX as a zero-size gather.
    @Test func packSequencesRefusesEmptyInputByName() {
        let empty = MLXArray.zeros([1, 0, 4]), mask = MLXArray.zeros([1, 0], dtype: .int32)
        #expect(throws: AceStepEngineError.emptyTokens("tokens to pack")) {
            try AceStepConditionEncoder.packSequences(empty, empty, mask1: mask, mask2: mask)
        }
    }
}
