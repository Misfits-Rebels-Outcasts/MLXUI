import CryptoKit
import Foundation
import MLX
import Testing
@testable import MLXUI

// MP7-3a: the Klein glue math, checked against diffusers 0.37.1 on tiny random-init modules (fixtures in
// MLXUITests/Fixtures/Flux2Klein, made by tiny_goldens.py, never edited to pass) plus values from the real pipeline functions.
// Config literals are the pinned checkpoint's config.json files (mlx-community/FLUX.2-klein-4B-bf16 @ 2f7dd4ac); the sandboxed
// test host cannot read RSI/, so they are embedded.

private enum Klein {
    static let transformerJSON = """
{
  "_class_name": "Flux2Transformer2DModel",
  "_diffusers_version": "0.37.0.dev0",
  "_name_or_path": "/raid/yiyi/klein-4b-distilled-diffusers/transformer",
  "attention_head_dim": 128,
  "axes_dims_rope": [
    32,
    32,
    32,
    32
  ],
  "eps": 1e-06,
  "guidance_embeds": false,
  "in_channels": 128,
  "joint_attention_dim": 7680,
  "mlp_ratio": 3.0,
  "num_attention_heads": 24,
  "num_layers": 5,
  "num_single_layers": 20,
  "out_channels": null,
  "patch_size": 1,
  "rope_theta": 2000,
  "timestep_guidance_channels": 256
}
"""
    static let textEncoderJSON = """
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "dtype": "bfloat16",
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "transformers_version": "4.56.1",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
"""
    static let vaeJSON = """
{
  "_class_name": "AutoencoderKLFlux2",
  "_diffusers_version": "0.37.0.dev0",
  "_name_or_path": "black-forest-labs/FLUX.2-dev",
  "act_fn": "silu",
  "batch_norm_eps": 0.0001,
  "batch_norm_momentum": 0.1,
  "block_out_channels": [
    128,
    256,
    512,
    512
  ],
  "down_block_types": [
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D",
    "DownEncoderBlock2D"
  ],
  "force_upcast": true,
  "in_channels": 3,
  "latent_channels": 32,
  "layers_per_block": 2,
  "mid_block_add_attention": true,
  "norm_num_groups": 32,
  "out_channels": 3,
  "patch_size": [
    2,
    2
  ],
  "sample_size": 1024,
  "up_block_types": [
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D",
    "UpDecoderBlock2D"
  ],
  "use_post_quant_conv": true,
  "use_quant_conv": true
}
"""
    static let schedulerJSON = """
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "_diffusers_version": "0.37.0.dev0",
  "base_image_seq_len": 256,
  "base_shift": 0.5,
  "invert_sigmas": false,
  "max_image_seq_len": 4096,
  "max_shift": 1.15,
  "num_train_timesteps": 1000,
  "shift": 3.0,
  "shift_terminal": null,
  "stochastic_sampling": false,
  "time_shift_type": "exponential",
  "use_beta_sigmas": false,
  "use_dynamic_shifting": true,
  "use_exponential_sigmas": false,
  "use_karras_sigmas": false
}
"""

    static var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein")
    }

    static func arrays(_ name: String) throws -> [String: MLXArray] {
        try MLX.loadArrays(url: fixtures.appendingPathComponent(name + ".safetensors"))
    }

    static func floats(_ a: MLXArray) -> [Float] { a.asType(.float32).asArray(Float.self) }

    static func int32(_ a: MLXArray) -> [Int32] { a.asType(.int32).asArray(Int32.self) }

    /// Largest absolute difference; shapes must match.
    static func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        guard a.shape == b.shape else { return .infinity }
        return zip(floats(a), floats(b)).map { abs($0 - $1) }.max() ?? 0
    }

    struct ScalarCase: Decodable {
        let image_seq_len: Int
        let steps: Int
        let mu: Double
        let sigmas: [Float]
        let timesteps: [Float]
    }
}

@Suite struct Flux2KleinConfigTests {
    @Test func theFourPinnedConfigsDecodeAndMatchTheDimensionsThePortIsWrittenFor() throws {
        let tf = try Flux2KleinTransformerConfig.decode(from: Data(Klein.transformerJSON.utf8))
        #expect(tf.innerDim == 3072)
        #expect(tf.axesDimsRope == [32, 32, 32, 32])
        let te = try Flux2KleinTextEncoderConfig.decode(from: Data(Klein.textEncoderJSON.utf8))
        #expect(te.hiddenSize * Flux2KleinDims.tapLayers.count == tf.jointAttentionDim)
        let vae = try Flux2KleinVAEConfig.decode(from: Data(Klein.vaeJSON.utf8))
        #expect(vae.latentChannels * 4 == tf.inChannels)
        let sch = try Flux2KleinSchedulerConfig.decode(from: Data(Klein.schedulerJSON.utf8))
        #expect(sch.baseShift == 0.5 && sch.maxShift == 1.15 && sch.baseImageSeqLen == 256 && sch.maxImageSeqLen == 4096)
    }

    @Test func aDifferentDimensionIsRefusedWithTheFieldNamed() throws {
        let wrong = Klein.transformerJSON.replacingOccurrences(of: "\"num_single_layers\": 20", with: "\"num_single_layers\": 24")
        #expect(wrong != Klein.transformerJSON)
        do {
            _ = try Flux2KleinTransformerConfig.decode(from: Data(wrong.utf8))
            Issue.record("a 24-single-block config decoded")
        } catch let e as Flux2KleinConfigError {
            #expect(e == .unexpected(file: "transformer/config.json", field: "num_single_layers", expected: "20", actual: "24"))
            #expect(e.localizedDescription.contains("num_single_layers"))
        }
        let badVAE = Klein.vaeJSON.replacingOccurrences(of: "\"batch_norm_eps\": 0.0001", with: "\"batch_norm_eps\": 1e-05")
        #expect(badVAE != Klein.vaeJSON)
        #expect(throws: Flux2KleinConfigError.self) { try Flux2KleinVAEConfig.decode(from: Data(badVAE.utf8)) }
        let bad9B = Klein.textEncoderJSON.replacingOccurrences(of: "\"hidden_size\": 2560", with: "\"hidden_size\": 4096")
        #expect(throws: Flux2KleinConfigError.self) { try Flux2KleinTextEncoderConfig.decode(from: Data(bad9B.utf8)) }
    }
}

@Suite struct Flux2KleinFixtureIntegrityTests {
    /// sha256 of each golden as generated (tiny_goldens.py, torch 2.9.1, diffusers 0.37.1, transformers 4.57.3). A hand edit fails here.
    static let hashes: [String: String] = [
        "g3a_transformer.safetensors": "f372424d3334c0b94a7bdc4554dfdd5acd3ed77e5556660c540dc8cd10973b8e",
        "g3b_tokens.json": "dc90f38ba8a0cff8d5cf7038cdf9bb56a53985004aa725b9bba853fec89111f0",
        "g3b_tokens_edge.json": "ab0ceb216256bf8c368e51dcc6a8f51b55cb562f55b90fee3c56cc7e20bab3e4",
        "g3c_textencoder.safetensors": "cd767b07a1917ea1cc4581d4089a04a1a93f7761791afd32f07ac655b85b5050",
        "g3d_vae.safetensors": "e2830030dbafcb3be688ff0337af6cd8b402cba81a92eee43048a6192a9e351b",
        "g3e_ids.safetensors": "f8eb78eda0fdaa55999bb3af7acd4d551ec2e3644240d76e2950eaf9eef9d67b",
        "g3f_scalars.json": "e4b74bd9779fd55ed57c9ad1013346dbb4ab45ba9370bb6223f366fa35bdce90",
        "g3f_scheduler.safetensors": "7fe30e628538349fa3ab4877c37270ecf8920af91868e661641cf1d52159f395",
        "g3g_glue.safetensors": "64d95a6a39b7f5cf641bd394db228de91e0c6420dbceea89a70532f5754216f8",
    ]

    @Test func everyGoldenIsByteIdenticalToWhatTheScriptWrote() throws {
        var total = 0
        for (name, expected) in Self.hashes {
            let data = try Data(contentsOf: Klein.fixtures.appendingPathComponent(name))
            total += data.count
            let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            #expect(digest == expected, Comment(rawValue: name))
        }
        #expect(total <= 2 * 1024 * 1024, "goldens stay within the 2 MB budget")
    }
}

@Suite struct Flux2KleinSamplerTests {
    /// diffusers 0.37.1 `compute_empirical_mu` + `retrieve_timesteps` (g3f_scalars.json, sha256 e4b74bd9...). Literals for the four
    /// cases the lead named; the whole file is checked below.
    static let literals: [(seq: Int, steps: Int, mu: Double, sigmas: [Float])] = [
        (256, 4, 1.965567161408539, [1.0, 0.9553907513618469, 0.8771342635154724, 0.7041116952896118, 0.0]),
        (1024, 4, 2.0306897079499455, [1.0, 0.9580853581428528, 0.8839818835258484, 0.7174965739250183, 0.0]),
        (4096, 4, 2.291179894115571, [1.0, 0.9673840403556824, 0.908143937587738, 0.7671999335289001, 0.0]),
        (4096, 8, 2.2678904387254573,
         [1.0, 0.9854254722595215, 0.9666411280632019, 0.9415147304534912, 0.906182587146759, 0.8528416752815247,
          0.7630144953727722, 0.5798067450523376, 0.0]),
    ]

    @Test func muAndSigmasMatchDiffusersAtTheFourNamedCases() {
        for c in Self.literals {
            #expect(abs(Flux2KleinSampler.empiricalMu(imageSeqLen: c.seq, steps: c.steps) - c.mu) < 1e-12, "mu (\(c.seq), \(c.steps))")
            let s = Flux2KleinSampler.sigmas(steps: c.steps, imageSeqLen: c.seq)
            #expect(s.count == c.steps + 1)
            for (a, b) in zip(s, c.sigmas) { #expect(abs(a - b) <= 2e-7, "sigma (\(c.seq), \(c.steps)): \(a) vs \(b)") }
            let t = Flux2KleinSampler.timesteps(steps: c.steps, imageSeqLen: c.seq)
            #expect(t.count == c.steps)
            for (a, b) in zip(t, c.sigmas) { #expect(abs(a - b * 1000) <= 2e-4) }
        }
    }

    @Test func everyCaseInTheGoldenFileMatchesIncludingTheLargeAndTheOneStepCase() throws {
        let data = try Data(contentsOf: Klein.fixtures.appendingPathComponent("g3f_scalars.json"))
        let cases = try JSONDecoder().decode([Klein.ScalarCase].self, from: data)
        #expect(cases.count == 6)
        for c in cases {
            #expect(abs(Flux2KleinSampler.empiricalMu(imageSeqLen: c.image_seq_len, steps: c.steps) - c.mu) < 1e-12)
            let s = Flux2KleinSampler.sigmas(steps: c.steps, imageSeqLen: c.image_seq_len)
            #expect(s.count == c.sigmas.count)
            for (a, b) in zip(s, c.sigmas) { #expect(abs(a - b) <= 2e-7, "(\(c.image_seq_len), \(c.steps)): \(a) vs \(b)") }
            let t = Flux2KleinSampler.timesteps(steps: c.steps, imageSeqLen: c.image_seq_len)
            for (a, b) in zip(t, c.timesteps) { #expect(abs(a - b) <= 2e-4) }
        }
    }

    @Test func theSameSigmasComeOutOfTheSafetensorsGolden() throws {
        let g = try Klein.arrays("g3f_scheduler")
        for (seq, steps) in [(256, 4), (1024, 4), (4096, 4), (4096, 8), (4096, 1), (16384, 4)] {
            let want = Klein.floats(try #require(g["s\(seq)_n\(steps).sigmas"]))
            let got = Flux2KleinSampler.sigmas(steps: steps, imageSeqLen: seq)
            #expect(want.count == got.count)
            for (a, b) in zip(got, want) { #expect(abs(a - b) <= 2e-7) }
        }
    }

    @Test func theEulerStepMatchesDiffusersAndIsTheFluxOneFormula() throws {
        let g = try Klein.arrays("g3f_scheduler")
        let sigmas = Klein.floats(try #require(g["step.sigmas"]))
        let sample = try #require(g["step.sample"]), out = try #require(g["step.model_output"]), want = try #require(g["step.result0"])
        let got = Flux2KleinSampler.step(sample: sample, velocity: out, sigma: sigmas[0], sigmaNext: sigmas[1])
        #expect(Klein.maxDiff(got, want) <= 1e-6)
        // Same arithmetic as the FLUX.1 sampler's step, so the formula is shared; the types are not.
        let flux1 = FluxSampler.step(pred: out, xT: sample, t: Double(sigmas[0]), tPrev: Double(sigmas[1]))
        #expect(Klein.maxDiff(got, flux1) <= 1e-6)
    }

    @Test func theLoopEndsAtZeroAndTheTerminalStepLandsOnTheDenoisedSample() {
        let s = Flux2KleinSampler.sigmas(steps: 4, imageSeqLen: 1024)
        #expect(s.first == 1 && s.last == 0)
        #expect(zip(s, s.dropFirst()).allSatisfy { $0 > $1 }, "strictly decreasing")
    }
}

@Suite struct Flux2KleinLatentTests {
    @Test func positionIdsMatchTheDiffusersPipeline() throws {
        let g = try Klein.arrays("g3e_ids")
        for (h, w) in [(2, 3), (4, 4)] {
            let want = Klein.int32(try #require(g["latent_ids_\(h)x\(w)"]))
            let got = Flux2KleinLatents.imageIDs(gridHeight: h, gridWidth: w)
            #expect(got.shape == [h * w, 4])
            #expect(Klein.int32(got) == want)
        }
        let text = Flux2KleinLatents.textIDs(count: 6)
        #expect(text.shape == [6, 4])
        #expect(Klein.int32(text) == Klein.int32(try #require(g["text_ids_6"])))
    }

    @Test func packMatchesDiffusersAndUnpackInvertsIt() throws {
        let g = try Klein.arrays("g3e_ids")
        for (h, w) in [(2, 3), (4, 4)] {
            let input = try #require(g["pack_in_\(h)x\(w)"])
            let packed = Flux2KleinLatents.packToSequence(input)
            #expect(Klein.maxDiff(packed, try #require(g["pack_out_\(h)x\(w)"])) == 0)
            let back = Flux2KleinLatents.unpackFromSequence(packed, gridHeight: h, gridWidth: w)
            #expect(Klein.maxDiff(back, input) == 0)
        }
        // diffusers unpacks by scattering on the ids; the grid reshape must agree on a non-square grid.
        let unpacked = Flux2KleinLatents.unpackFromSequence(try #require(g["unpack_in"]), gridHeight: 2, gridWidth: 3)
        #expect(Klein.maxDiff(unpacked, try #require(g["unpack_out"])) == 0)
    }

    @Test func patchifyUsesChannelTimesFourPlusPatchRowTimesTwoPlusPatchColumn() throws {
        let g = try Klein.arrays("g3e_ids")
        let input = try #require(g["patchify_in"])
        let patched = Flux2KleinLatents.patchify(input)
        #expect(patched.shape == [1, 12, 2, 3])
        #expect(Klein.maxDiff(patched, try #require(g["patchify_out"])) == 0)
        #expect(Klein.maxDiff(Flux2KleinLatents.unpatchify(patched), input) == 0)
        #expect(Klein.maxDiff(Flux2KleinLatents.unpatchify(patched), try #require(g["unpatchify_roundtrip"])) == 0)
        // Spelled out: output channel c*4 + p1*2 + p2 at (i, j) is input channel c at (2i + p1, 2j + p2).
        let want = Klein.floats(input), got = Klein.floats(patched)
        let (H, W) = (4, 6), (h, w) = (2, 3)
        for c in 0..<3 { for p1 in 0..<2 { for p2 in 0..<2 { for i in 0..<h { for j in 0..<w {
            #expect(got[((c * 4 + p1 * 2 + p2) * h + i) * w + j] == want[(c * H + 2 * i + p1) * W + 2 * j + p2])
        } } } } }
    }

    @Test func theBatchNormDenormaliseAndTheWholeDecodeGlueMatchDiffusers() throws {
        let vae = try Klein.arrays("g3d_vae")
        let mean = try #require(vae["w.bn.running_mean"]), variance = try #require(vae["w.bn.running_var"])
        let denorm = Flux2KleinLatents.denormalize(try #require(vae["in.z_packed"]), mean: mean, variance: variance)
        #expect(Klein.maxDiff(denorm, try #require(vae["out.z_denorm"])) <= 1e-6)
        #expect(Klein.maxDiff(Flux2KleinLatents.unpatchify(denorm), try #require(vae["out.z_unpatchified"])) <= 1e-6)

        // Whole chain on the last denoise step of the tiny run: unpack -> denormalise (eps 1e-4) -> unpatchify.
        let glue = try Klein.arrays("g3g_glue")
        let unpacked = Flux2KleinLatents.unpackFromSequence(try #require(glue["step1.latents"]), gridHeight: 2, gridWidth: 2)
        #expect(Klein.maxDiff(unpacked, try #require(glue["out.unpacked"])) <= 1e-6)
        let den = Flux2KleinLatents.denormalize(unpacked, mean: mean, variance: variance)
        #expect(Klein.maxDiff(den, try #require(glue["out.bn_denorm"])) <= 1e-5)
        #expect(Klein.maxDiff(Flux2KleinLatents.unpatchify(den), try #require(glue["out.unpatchified"])) <= 1e-5)
        // The tiny run's ids and packed layout are the same functions.
        #expect(Klein.int32(Flux2KleinLatents.imageIDs(gridHeight: 2, gridWidth: 2)) == Klein.int32(try #require(glue["out.ids"])))
        // The tiny run's sigmas and timesteps are the sampler's, for a 2 x 2 grid at 2 steps.
        let sigmas = Flux2KleinSampler.sigmas(steps: 2, imageSeqLen: 4)
        for (a, b) in zip(sigmas, Klein.floats(try #require(glue["out.sigmas"]))) { #expect(abs(a - b) <= 2e-7) }
    }

    @Test func noiseHasTheLatentLayoutAndIsReproduciblePerSeed() throws {
        let a = try Flux2KleinLatents.makeNoise(width: 512, height: 768, seed: 7)
        #expect(a.shape == [1, 128, 48, 32], "[1, 128, H/16, W/16]")
        let b = try Flux2KleinLatents.makeNoise(width: 512, height: 768, seed: 7)
        let c = try Flux2KleinLatents.makeNoise(width: 512, height: 768, seed: 8)
        #expect(Klein.maxDiff(a, b) == 0)
        #expect(Klein.maxDiff(a, c) > 0)
        let v = Klein.floats(a)
        let mean = v.reduce(0, +) / Float(v.count)
        let sd = (v.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(v.count)).squareRoot()
        #expect(abs(mean) < 0.02 && abs(sd - 1) < 0.02, "standard normal: mean \(mean), sd \(sd)")
    }

    @Test func sizesThatAreNotMultiplesOfSixteenAreRefusedNotRounded() throws {
        let g = try Flux2KleinLatents.validateSize(width: 1024, height: 512)
        #expect(g.gridWidth == 64 && g.gridHeight == 32)
        #expect(try Flux2KleinLatents.imageSequenceLength(width: 1024, height: 1024) == 4096)
        for (w, h) in [(1000, 1024), (1024, 1020), (1023, 1023), (8, 16), (24, 24)] {
            #expect(throws: Flux2KleinSizeError.notMultipleOf16(width: w, height: h)) { try Flux2KleinLatents.validateSize(width: w, height: h) }
        }
        for (w, h) in [(0, 512), (512, -16), (-1, -1)] {
            #expect(throws: Flux2KleinSizeError.notPositive(width: w, height: h)) { try Flux2KleinLatents.validateSize(width: w, height: h) }
        }
        #expect(throws: Flux2KleinSizeError.self) { try Flux2KleinLatents.makeNoise(width: 1000, height: 1000, seed: 1) }
        // SPEC-Q242 wording: width first, names the first offending setting only.
        #expect(Flux2KleinSizeError.notMultipleOf16(width: 1000, height: 600).localizedDescription
            == "Width 1000 isn't a multiple of 16. The nearest sizes are 992 and 1008.")
        #expect(Flux2KleinSizeError.notMultipleOf16(width: 1024, height: 1020).localizedDescription
            == "Height 1020 isn't a multiple of 16. The nearest sizes are 1008 and 1024.")
        #expect(Flux2KleinSizeError.notMultipleOf16(width: 8, height: 512).localizedDescription
            == "Width 8 isn't a multiple of 16. The nearest size is 16.")
        do { try Flux2KleinLatents.validateSize(width: 1000, height: 1020) } catch {
            #expect(error.localizedDescription == "Width 1000 isn't a multiple of 16. The nearest sizes are 992 and 1008.")
        }
    }
}

@Suite struct Flux2KleinGapRemovalTests {
    /// The MP7-2 gap kept Klein off the workflow rows while it had no runner. MP7-4 registered the module and removed the gap from `ModelSupport`, so the
    /// catalog's own display name resolves it as runnable. MP7-W added the `CatalogBridge` row "FLUX.2 Klein 4B" (the catalog's displayName is the same string), so the
    /// resolution is no longer the display-name fallback (`.same`) but the bridge's `.requantized` onto the catflow pin (MP7-W: owner-ordered; also `Flux2KleinWorkflowTests`).
    @Test func theDisplayNameFallbackIsRunnableNowThatTheKleinGapIsRemoved() throws {
        let data = try Data(contentsOf: URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/browser.json"))
        let browser = try JSONDecoder().decode(BrowserData.self, from: data)
        let klein = try #require(browser.domains.flatMap { $0.allModels }.first { $0.hfModelId == "mlx-community/FLUX.2-klein-4B-bf16" })

        #expect(ModelSupport.unsupportedReason(for: klein) == nil, "the gap is gone")
        #expect(ModelSupport.unsupportedReason(for: klein, ignoringGaps: ["flux.2-klein"]) == nil)
        guard case let .runnable(target, equivalence, _) = CatalogBridge.resolve(klein.displayName, catalog: [klein]) else {
            Issue.record("the display-name fallback still refuses Klein with the gap removed"); return
        }
        #expect(equivalence == .requantized)
        if case let .cataloged(model) = target { #expect(model.hfModelId == klein.hfModelId) } else { Issue.record("not a catalog target") }
    }
}
