import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXUI

// MP7-3g: the Klein VAE decoder against diffusers 0.37.1 on the tiny random-init AutoencoderKLFlux2 of g3d_vae.safetensors (made by tiny_goldens.py in 3d, never
// edited), the key set, and the loader on tiny random-weight directories written here. No real weights in this file (see Flux2KleinVAERealLoadTests for those).
// Floors were set AFTER measuring (MP7-3g-report.md); they pin measured behaviour of this port, not image quality.

enum Klein3gFix {
    static var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein")
    }

    /// The tiny golden: weights (prefix `w.` stripped), inputs/outputs (`in.`/`out.`), and the shape from the file's own metadata.
    struct Tiny {
        var weights: [String: MLXArray] = [:]
        var io: [String: MLXArray] = [:]
        var shape: Flux2KleinVAEShape
        var configJSON: Data
    }

    static func tiny() throws -> Tiny {
        let (arrays, meta) = try loadArraysAndMetadata(url: fixtures.appendingPathComponent("g3d_vae.safetensors"))
        let metaJSON = try #require(meta["meta"]?.data(using: .utf8))
        let outer = try #require(JSONSerialization.jsonObject(with: metaJSON) as? [String: Any])
        let configString = try #require(outer["config"] as? String)
        let configJSON = Data(configString.utf8)
        let shape = Flux2KleinVAEShape(try JSONDecoder().decode(Flux2KleinVAEConfig.self, from: configJSON))
        var t = Tiny(shape: shape, configJSON: configJSON)
        for (k, v) in arrays {
            if k.hasPrefix("w.") { t.weights[String(k.dropFirst(2))] = v } else { t.io[k] = v }
        }
        return t
    }

    static func floats(_ a: MLXArray) throws -> [Float] { try MLXErrorGuard.read(a.asType(.float32).reshaped(-1), as: Float.self) }

    struct Errors { var maxAbs: Double, cosine: Double, psnr: Double }

    /// Max-abs, cosine, and PSNR with the usual image convention: values in [-1, 1] are mapped to [0, 1] (x/2 + 1/2), so the peak is 1 and MSE is the
    /// mean squared difference / 4. Identical arrays give +infinity.
    static func compare(_ a: MLXArray, _ b: MLXArray) throws -> Errors {
        let x = try floats(a), y = try floats(b)
        precondition(x.count == y.count)
        var maxAbs = 0.0, se = 0.0
        for i in 0 ..< x.count { let d = Double(x[i]) - Double(y[i]); maxAbs = max(maxAbs, abs(d)); se += d * d }
        let mse = se / Double(x.count) / 4
        return Errors(maxAbs: maxAbs, cosine: KleinCosine.cosine(x, y), psnr: mse == 0 ? .infinity : 10 * log10(1 / mse))
    }

    /// The model's parameters back in diffusers layout (convs [O, I, kH, kW]), as a checkpoint would carry them.
    static func diffusersTensors(of model: Module) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in model.parameters().flattened() {
            out[k] = k.hasSuffix(".weight") && v.ndim == 4 ? v.transposed(0, 3, 1, 2) : v
        }
        return out
    }

    /// A tiny random-weight checkpoint: values from MLX's default init for the shape, plus the names a real checkpoint carries that the decoder ignores.
    static func randomCheckpoint(shape: Flux2KleinVAEShape, seed: UInt64 = 7) -> (source: Flux2KleinVAE, tensors: [String: MLXArray]) {
        MLXRandom.seed(seed)
        let source = Flux2KleinVAE(shape)
        // random values for the norms and statistics too (the default init is all ones / zeros, which would hide a mix-up)
        var randomized: [String: MLXArray] = [:]
        for (k, v) in source.parameters().flattened() {
            randomized[k] = k == "bn.running_var" ? abs(MLXRandom.normal(v.shape)) + 0.5 : (v.ndim == 1 ? MLXRandom.normal(v.shape) * 0.3 + 1 : v)
        }
        source.update(parameters: ModuleParameters.unflattened(randomized))
        eval(source)
        var tensors = diffusersTensors(of: source)
        tensors["bn.num_batches_tracked"] = MLXArray(Int64(3))
        tensors["quant_conv.weight"] = MLXRandom.normal([2 * shape.latentChannels, 2 * shape.latentChannels, 1, 1])
        tensors["quant_conv.bias"] = MLXRandom.normal([2 * shape.latentChannels])
        tensors["encoder.conv_in.weight"] = MLXRandom.normal([shape.blockOutChannels[0], 3, 3, 3])
        tensors["encoder.conv_in.bias"] = MLXRandom.normal([shape.blockOutChannels[0]])
        return (source, tensors)
    }
}

/// Floors for the tiny fp32 golden. Measured first (MP7-3g-report.md): decode max-abs 2.53e-6, 1-cosine 4.4e-13, PSNR 130.6 dB; BN denormalise and unpatchify max-abs 0.0.
/// Cosine and PSNR keep the spike's owner-approved targets (>= 0.99999 and >= 80 dB), far under the measured values; the max-abs floors pin measured behaviour with a margin
/// (about 4x) and are not quality claims. The margins are the implementer's call, pending owner confirmation.
enum Klein3gFloors {
    static let decodeCosine = 0.99999
    static let decodePSNR = 80.0
    static let decodeMaxAbs = 1e-5
    static let glueMaxAbs = 1e-6
    static let chunkedAttentionMaxAbs = 1e-5   // measured 2.2e-6 (same fp32 rounding scale as the parity error)
}

// MARK: - Parity against diffusers on the tiny golden

@Suite(.serialized) struct Flux2KleinVAEParityTests {
    @Test func theTinyGoldenDecodesLikeDiffusers() throws {
        let g = try Klein3gFix.tiny()
        #expect(g.shape == Flux2KleinVAEShape(blockOutChannels: [8, 8, 16, 16], layersPerBlock: 1, groups: 4, latentChannels: 4))
        let vae = try Flux2KleinVAE.load(shape: g.shape, tensors: g.weights, computeDType: nil)
        #expect(vae.computeDType == .float32)

        // pipeline glue through the loaded BatchNorm vectors: denormalise, unpatchify
        let zPacked = try #require(g.io["in.z_packed"])
        let denorm = Flux2KleinLatents.denormalize(zPacked, mean: vae.bn.runningMean, variance: vae.bn.runningVar, eps: g.shape.batchNormEps)
        let glue = try Klein3gFix.compare(denorm, try #require(g.io["out.z_denorm"]))
        let unpatch = try Klein3gFix.compare(Flux2KleinLatents.unpatchify(denorm), try #require(g.io["out.z_unpatchified"]))
        print("VAE3G tiny z_denorm maxabs \(glue.maxAbs) cos \(glue.cosine); z_unpatchified maxabs \(unpatch.maxAbs)")
        #expect(glue.maxAbs <= Klein3gFloors.glueMaxAbs && unpatch.maxAbs <= Klein3gFloors.glueMaxAbs)

        // decode from the unpatchified latent, and from the packed latent (whole tail)
        let want = try #require(g.io["out.decoded"])
        let fromZ = try vae.decode(latents: try #require(g.io["out.z_unpatchified"]))
        let fromPacked = try vae.decode(packedLatents: zPacked)
        #expect(fromZ.shape == [1, 3, 32, 32] && fromPacked.shape == [1, 3, 32, 32])
        for (name, got) in [("latents", fromZ), ("packed", fromPacked)] {
            let e = try Klein3gFix.compare(got, want)
            print("VAE3G tiny decode(\(name)) maxabs \(e.maxAbs) cos \(e.cosine) 1-cos \(1 - e.cosine) psnr \(e.psnr) dB")
            #expect(e.cosine >= Klein3gFloors.decodeCosine, Comment(rawValue: "\(name) cosine \(e.cosine)"))
            #expect(e.psnr >= Klein3gFloors.decodePSNR, Comment(rawValue: "\(name) psnr \(e.psnr)"))
            #expect(e.maxAbs <= Klein3gFloors.decodeMaxAbs, Comment(rawValue: "\(name) maxabs \(e.maxAbs)"))
        }
        // the decode is not trivially the input: the output differs from a zero image and from a decode without the BatchNorm denormalise
        let noDenorm = try vae.decode(latents: Flux2KleinLatents.unpatchify(zPacked))
        #expect(try Klein3gFix.compare(noDenorm, want).maxAbs > 1e-2, "skipping the denormalise must change the picture")
    }

    @Test func aBatchedDecodeEqualsTwoSingleDecodes() throws {
        let g = try Klein3gFix.tiny()
        let vae = try Flux2KleinVAE.load(shape: g.shape, tensors: g.weights, computeDType: nil)
        let z = try #require(g.io["in.z_packed"])
        MLXRandom.seed(5)
        let z2 = MLXRandom.normal(z.shape)
        let both = try vae.decode(packedLatents: concatenated([z, z2], axis: 0))
        let a = try vae.decode(packedLatents: z), b = try vae.decode(packedLatents: z2)
        #expect(try Klein3gFix.compare(both[0 ..< 1], a).maxAbs <= 1e-5)
        #expect(try Klein3gFix.compare(both[1 ..< 2], b).maxAbs <= 1e-5)
    }

    @Test func chunkedMidBlockAttentionIsExact() throws {
        let g = try Klein3gFix.tiny()
        let vae = try Flux2KleinVAE.load(shape: g.shape, tensors: g.weights, computeDType: nil)
        let z = try #require(g.io["in.z_packed"])
        let whole = try vae.decode(packedLatents: z)
        vae.setAttentionQueryChunk(3)         // 4 x 4 = 16 pixels at the mid block -> slices of 3, 3, 3, 3, 3, 1
        let chunked = try vae.decode(packedLatents: z)
        vae.setAttentionQueryChunk(nil)
        let e = try Klein3gFix.compare(whole, chunked)
        print("VAE3G tiny chunked-vs-whole attention maxabs \(e.maxAbs) cos \(e.cosine)")
        #expect(e.maxAbs <= Klein3gFloors.chunkedAttentionMaxAbs)
    }

    @Test func badInputsAreNamedNotCrashed() throws {
        let g = try Klein3gFix.tiny()
        let vae = try Flux2KleinVAE.load(shape: g.shape, tensors: g.weights, computeDType: nil)
        #expect(throws: Flux2KleinVAEError.self) { _ = try vae.decode(packedLatents: MLXArray.zeros([1, 15, 2, 2])) }
        #expect(throws: Flux2KleinVAEError.self) { _ = try vae.decode(packedLatents: MLXArray.zeros([16, 2, 2])) }
        #expect(throws: Flux2KleinVAEError.self) { _ = try vae.decode(latents: MLXArray.zeros([1, 16, 2, 2])) }
    }
}

// MARK: - Key sets

@Suite(.serialized) struct Flux2KleinVAEKeyTests {
    @Test func theTinyModelKeySetIsTheGoldensKeySetMinusWhatTheDecoderIgnores() throws {
        let g = try Klein3gFix.tiny()
        let mine = Set(Flux2KleinVAE(g.shape).parameters().flattened().map { $0.0 })
        let theirs = Set(g.weights.keys.filter { !Flux2KleinVAE.isIgnoredName($0) })
        #expect(mine == theirs, "model minus golden \(mine.subtracting(theirs).sorted()), golden minus model \(theirs.subtracting(mine).sorted())")
        let ignored = g.weights.keys.filter(Flux2KleinVAE.isIgnoredName)
        #expect(ignored.allSatisfy { $0.hasPrefix("encoder.") || $0.hasPrefix("quant_conv.") || $0 == "bn.num_batches_tracked" })
        #expect(ignored.contains("quant_conv.weight") && ignored.contains("bn.num_batches_tracked"))
    }

    @Test func theKlein4BModelKeySetIsPinned() throws {
        let params = Flux2KleinVAE(.klein4B).parameters().flattened()
        let keys = params.map { $0.0 }
        // mid 26 (2 resnets x 8 + attention 10) + up blocks 106 (12 resnets x 8 + 2 shortcuts x 2 + 3 upsamplers x 2) + conv_in/norm_out/conv_out 6 + post_quant_conv 2 + bn 2
        #expect(keys.count == 142, "keys: \(keys.count)")
        #expect(keys.filter { $0.hasPrefix("decoder.mid_block.") }.count == 26)
        #expect(keys.filter { $0.hasPrefix("decoder.up_blocks.") }.count == 106)
        for k in ["post_quant_conv.weight", "post_quant_conv.bias", "bn.running_mean", "bn.running_var", "decoder.conv_in.weight", "decoder.conv_in.bias",
                  "decoder.mid_block.resnets.0.norm1.weight", "decoder.mid_block.attentions.0.group_norm.weight", "decoder.mid_block.attentions.0.to_q.weight",
                  "decoder.mid_block.attentions.0.to_out.0.bias", "decoder.mid_block.resnets.1.conv2.weight", "decoder.up_blocks.0.upsamplers.0.conv.weight",
                  "decoder.up_blocks.2.resnets.0.conv_shortcut.weight", "decoder.up_blocks.3.resnets.2.conv2.bias", "decoder.conv_norm_out.weight", "decoder.conv_out.weight"] {
            #expect(keys.contains(k), Comment(rawValue: k))
        }
        #expect(!keys.contains { $0.hasPrefix("encoder.") || $0.hasPrefix("quant_conv.") || $0.contains("num_batches") })
        #expect(!keys.contains("decoder.up_blocks.3.upsamplers.0.conv.weight"), "the last up block has no upsampler")
        #expect(!keys.contains("decoder.up_blocks.0.resnets.0.conv_shortcut.weight"), "512 -> 512 needs no shortcut")
        let shapes = Dictionary(uniqueKeysWithValues: params.map { ($0.0, $0.1.shape) })
        #expect(shapes["decoder.conv_in.weight"] == [512, 3, 3, 32], "MLX layout [O, kH, kW, I]")
        #expect(shapes["post_quant_conv.weight"] == [32, 1, 1, 32])
        #expect(shapes["decoder.mid_block.attentions.0.to_q.weight"] == [512, 512])
        #expect(shapes["bn.running_mean"] == [128] && shapes["bn.running_var"] == [128])
        #expect(shapes["decoder.up_blocks.3.resnets.0.conv_shortcut.weight"] == [128, 1, 1, 256])
        let decoderElements = params.reduce(0) { $0 + $1.1.size }
        print("VAE3G klein4B decoder parameters \(decoderElements) = \(Double(decoderElements) * 2 / 1e6) MB bf16")
    }
}

// MARK: - Loader

@Suite(.serialized) struct Flux2KleinVAELoaderTests {
    static let shape = Flux2KleinVAEShape(blockOutChannels: [8, 8, 16, 16], layersPerBlock: 1, groups: 4, latentChannels: 4)

    @Test func aRandomCheckpointLoadsAndDecodesExactlyLikeItsSource() throws {
        let (source, tensors) = Klein3gFix.randomCheckpoint(shape: Self.shape)
        let loaded = try Flux2KleinVAE.load(shape: Self.shape, tensors: tensors, computeDType: nil)
        MLXRandom.seed(11)
        let z = MLXRandom.normal([1, 16, 3, 2])
        let a = try source.decode(packedLatents: z), b = try loaded.decode(packedLatents: z)
        #expect(a.shape == [1, 3, 48, 32])
        #expect(try Klein3gFix.compare(a, b).maxAbs == 0)
    }

    @Test func convolutionKernelsAreTransposedFromDiffusersToMLXLayout() throws {
        let (_, tensors) = Klein3gFix.randomCheckpoint(shape: Self.shape)
        // distinct values: element (o, i, y, x) = 1000 o + 100 i + 10 y + x
        func ramp(_ o: Int, _ i: Int, _ kh: Int, _ kw: Int) -> MLXArray {
            var v: [Float] = []
            for oo in 0 ..< o { for ii in 0 ..< i { for y in 0 ..< kh { for x in 0 ..< kw { v.append(Float(1000 * oo + 100 * ii + 10 * y + x)) } } } }
            return MLXArray(v, [o, i, kh, kw])
        }
        var t = tensors
        t["decoder.conv_in.weight"] = ramp(16, 4, 3, 3)
        t["post_quant_conv.weight"] = ramp(4, 4, 1, 1)
        let loaded = try Flux2KleinVAE.load(shape: Self.shape, tensors: t, computeDType: nil)
        let w = loaded.decoder.convIn.weight
        #expect(w.shape == [16, 3, 3, 4])
        let got = try Klein3gFix.floats(w)
        for o in [0, 5, 15] { for i in 0 ..< 4 { for y in 0 ..< 3 { for x in 0 ..< 3 {
            #expect(got[((o * 3 + y) * 3 + x) * 4 + i] == Float(1000 * o + 100 * i + 10 * y + x), Comment(rawValue: "conv_in [\(o),\(i),\(y),\(x)]"))
        } } } }
        let p = try Klein3gFix.floats(loaded.postQuantConv.weight)
        #expect(loaded.postQuantConv.weight.shape == [4, 1, 1, 4])
        for o in 0 ..< 4 { for i in 0 ..< 4 { #expect(p[o * 4 + i] == Float(1000 * o + 100 * i)) } }
        // Linear weights are NOT transposed (2-D)
        let q = try Klein3gFix.floats(loaded.decoder.midBlock.attentions[0].toQ.weight)
        let src = try Klein3gFix.floats(try #require(tensors["decoder.mid_block.attentions.0.to_q.weight"]))
        #expect(q == src)
    }

    @Test func theEncoderQuantConvAndStepCounterAreIgnoredByName() throws {
        let (_, tensors) = Klein3gFix.randomCheckpoint(shape: Self.shape)
        #expect(tensors.keys.contains { $0.hasPrefix("encoder.") } && tensors.keys.contains("quant_conv.bias") && tensors.keys.contains("bn.num_batches_tracked"))
        _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: tensors, computeDType: nil)
    }

    @Test func missingTensorsAreNamedAndAnythingElseUnexpectedThrows() throws {
        let (_, tensors) = Klein3gFix.randomCheckpoint(shape: Self.shape)
        var short = tensors
        short.removeValue(forKey: "decoder.mid_block.attentions.0.to_k.bias")
        short.removeValue(forKey: "bn.running_var")
        do { _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: short, computeDType: nil); Issue.record("expected a throw") } catch let Flux2KleinVAEError.missingWeight(names) {
            #expect(names == "bn.running_var, decoder.mid_block.attentions.0.to_k.bias")
        }

        var extra = tensors
        extra["decoder.up_blocks.3.upsamplers.0.conv.weight"] = MLXArray.zeros([8, 3, 3, 8])
        #expect(throws: Flux2KleinVAEError.unexpectedWeight("decoder.up_blocks.3.upsamplers.0.conv.weight")) { _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: extra, computeDType: nil) }

        var renamed = tensors
        renamed["decoder.mid_block.attentions.0.to_out.weight"] = renamed.removeValue(forKey: "decoder.mid_block.attentions.0.to_out.0.weight")
        #expect(throws: Flux2KleinVAEError.unexpectedWeight("decoder.mid_block.attentions.0.to_out.weight")) { _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: renamed, computeDType: nil) }

        var scaled = tensors                                  // an FLUX.1 style key the decoder must not accept silently
        scaled["decoder.conv_in.conv2d.weight"] = MLXArray.zeros([16, 4, 3, 3])
        #expect(throws: Flux2KleinVAEError.unexpectedWeight("decoder.conv_in.conv2d.weight")) { _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: scaled, computeDType: nil) }
    }

    @Test func aWrongShapeThrowsWithTheNameInIt() throws {
        let (_, tensors) = Klein3gFix.randomCheckpoint(shape: Self.shape)
        var bad = tensors
        bad["post_quant_conv.weight"] = MLXArray.zeros([4, 5, 1, 1])
        do { _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: bad, computeDType: nil); Issue.record("expected a throw") } catch let Flux2KleinVAEError.unexpectedWeight(message) {
            #expect(message.hasPrefix("post_quant_conv.weight has shape"), Comment(rawValue: message))
        }
        var already = tensors            // a checkpoint already in MLX layout [O, kH, kW, I] must not slip through for non-square cases
        already["decoder.conv_in.weight"] = tensors["decoder.conv_in.weight"]?.transposed(0, 2, 3, 1)
        #expect(throws: Flux2KleinVAEError.self) { _ = try Flux2KleinVAE.load(shape: Self.shape, tensors: already, computeDType: nil) }
    }

    @Test func dtypeIsKeptByDefaultAndUpcastOnRequest() throws {
        let (_, tensors) = Klein3gFix.randomCheckpoint(shape: Self.shape)
        let bf16 = tensors.mapValues { $0.dtype == .float32 ? $0.asType(.bfloat16) : $0 }
        let kept = try Flux2KleinVAE.load(shape: Self.shape, tensors: bf16, computeDType: nil)
        #expect(kept.computeDType == .bfloat16 && kept.bn.runningMean.dtype == .bfloat16 && kept.decoder.midBlock.attentions[0].toQ.weight.dtype == .bfloat16)
        let up = try Flux2KleinVAE.load(shape: Self.shape, tensors: bf16, computeDType: .float32)
        #expect(up.computeDType == .float32 && up.bn.runningVar.dtype == .float32)
        #expect(up.parameterBytes() == 2 * kept.parameterBytes())
        // both decode (the bf16 one in bf16, returning float32) and agree loosely
        MLXRandom.seed(3)
        let z = MLXRandom.normal([1, 16, 2, 2])
        let a = try kept.decode(packedLatents: z), b = try up.decode(packedLatents: z)
        #expect(a.dtype == .float32 && b.dtype == .float32)
        let e = try Klein3gFix.compare(a, b)
        print("VAE3G tiny bf16-vs-fp32 decode (same bf16 weights): maxabs \(e.maxAbs) cos \(e.cosine) psnr \(e.psnr)")
        #expect(e.cosine > 0.99)
    }

    @Test func aDirectoryWithConfigAndSafetensorsLoadsAndNoWeightFilesIsANamedError() throws {
        let g = try Klein3gFix.tiny()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("klein3g-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try g.configJSON.write(to: dir.appendingPathComponent("config.json"))
        // two files, to prove shards are merged and a duplicate name is refused
        let keys = g.weights.keys.sorted()
        let half = Set(keys.prefix(keys.count / 2))
        try MLX.save(arrays: g.weights.filter { half.contains($0.key) }, url: dir.appendingPathComponent("a.safetensors"))
        try MLX.save(arrays: g.weights.filter { !half.contains($0.key) }, url: dir.appendingPathComponent("b.safetensors"))
        let vae = try Flux2KleinVAE.loadForTesting(vaeDirectory: dir, computeDType: nil)
        let want = try #require(g.io["out.decoded"])
        #expect(try Klein3gFix.compare(try vae.decode(packedLatents: try #require(g.io["in.z_packed"])), want).psnr >= Klein3gFloors.decodePSNR)

        try MLX.save(arrays: ["decoder.conv_out.bias": g.weights["decoder.conv_out.bias"] ?? MLXArray(0)], url: dir.appendingPathComponent("c.safetensors"))
        #expect(throws: Flux2KleinVAEError.unexpectedWeight("decoder.conv_out.bias (duplicate)")) { _ = try Flux2KleinVAE.loadForTesting(vaeDirectory: dir, computeDType: nil) }

        let empty = dir.appendingPathComponent("empty")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        try g.configJSON.write(to: empty.appendingPathComponent("config.json"))
        #expect(throws: Flux2KleinVAEError.noWeightFiles(empty.path)) { _ = try Flux2KleinVAE.loadForTesting(vaeDirectory: empty, computeDType: nil) }
    }
}
