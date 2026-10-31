import Darwin
import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import MLXUI

// MP7-3g, rule 13b, opt-in: the Klein VAE on the REAL installed vae/ folder. Unsandboxed host (reads ~/Library/Application Support/AI Browser). Nothing is downloaded and
// the installed folder is only read.
//  * Header check: names + shapes only, no tensor data. Runs whenever the installed model is present.
//  * Peak measurement: needs TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_SIZES=512,1024 (comma list of sizes, multiples of 16) and TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_DTYPE=bf16|fp32;
//    optional TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_CACHE_MB (Memory.cacheLimit) and TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_CHUNK (attention query rows per slice).
//    ONE dtype per process so peaks are not polluted. Prints VAE3G-PEAK lines. Random latent, no accuracy claim.

enum Klein3gReal {
    nonisolated static let directory: URL? = {
        guard let t = Klein3eHeader.directory else { return nil }
        let v = t.deletingLastPathComponent().appendingPathComponent("vae")
        let fm = FileManager.default
        return fm.fileExists(atPath: v.appendingPathComponent("diffusion_pytorch_model.safetensors").path) && fm.fileExists(atPath: v.appendingPathComponent("config.json").path) ? v : nil
    }()
    nonisolated static var present: Bool { directory != nil }
    nonisolated static let skipReason: Comment = "needs the installed FLUX.2 Klein (mlx-community--FLUX.2-klein-4B-bf16) vae/ or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR, and an unsandboxed test host (reads the safetensors header only)"

    nonisolated static let env = ProcessInfo.processInfo.environment
    nonisolated static let sizes: [Int] = (env["MLXUI_TEST_KLEIN_VAE_SIZES"] ?? "").split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    nonisolated static let dtype: DType? = ["bf16": DType.bfloat16, "fp32": DType.float32][env["MLXUI_TEST_KLEIN_VAE_DTYPE"] ?? ""]
    nonisolated static let cacheMB: Int? = env["MLXUI_TEST_KLEIN_VAE_CACHE_MB"].flatMap { Int($0) }
    nonisolated static let chunk: Int? = env["MLXUI_TEST_KLEIN_VAE_CHUNK"].flatMap { Int($0) }
    nonisolated static let evalEach: Bool? = env["MLXUI_TEST_KLEIN_VAE_EVAL_EACH"].map { $0 == "1" }       // nil = the decoder's default (on)
    nonisolated static let peakEnabled = present && !sizes.isEmpty && dtype != nil
    nonisolated static let peakSkipReason: Comment = "needs the installed Klein vae/, TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_SIZES=512,1024 and TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_DTYPE=bf16|fp32, and an unsandboxed test host"
}

@Suite(.serialized) struct Flux2KleinVAERealHeaderTests {
    @Test(.enabled(if: Klein3gReal.present, Klein3gReal.skipReason))
    func everyTensorTheDecoderNeedsIsInTheRealHeaderWithTheRightShape() throws {
        let dir = try #require(Klein3gReal.directory)
        let header = try Klein3eHeader.read(dir.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        let config = try Flux2KleinVAEConfig.decode(from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let params = Flux2KleinVAE(config).parameters().flattened()
        // module shapes are MLX layout; the header is diffusers layout [O, I, kH, kW] for convolutions
        var mine: [String: [Int]] = [:]
        for (k, v) in params { mine[k] = k.hasSuffix(".weight") && v.ndim == 4 ? [v.dim(0), v.dim(3), v.dim(1), v.dim(2)] : v.shape }
        let used = header.filter { !Flux2KleinVAE.isIgnoredName($0.key) }
        print("REALKEYS vae header \(header.count) tensors; decoder uses \(used.count), model has \(mine.count) parameters")
        #expect(mine.count == 142 && used.count == 142)
        #expect(Set(used.keys) == Set(mine.keys), "header minus model: \(Set(used.keys).subtracting(mine.keys).sorted().prefix(5)), model minus header: \(Set(mine.keys).subtracting(used.keys).sorted().prefix(5))")
        var mismatches: [String] = []
        for (k, t) in used {
            if mine[k] != t.shape { mismatches.append("\(k): model \(String(describing: mine[k])) header \(t.shape)") }
            #expect(t.dtype == "BF16", Comment(rawValue: "\(k) dtype \(t.dtype)"))
        }
        #expect(mismatches.isEmpty, Comment(rawValue: mismatches.prefix(5).joined(separator: "; ")))
        // what the header carries that the decoder does not use
        let unused = header.filter { Flux2KleinVAE.isIgnoredName($0.key) }
        let enc = unused.filter { $0.key.hasPrefix("encoder.") }.count
        let quant = unused.filter { $0.key.hasPrefix("quant_conv.") }.keys.sorted()
        let other = unused.keys.filter { !$0.hasPrefix("encoder.") && !$0.hasPrefix("quant_conv.") }.sorted()
        let elements = { (m: [String: Flux2KleinHeaderTensor]) in m.values.reduce(0) { $0 + $1.shape.reduce(1, *) } }
        print("REALKEYS vae unused: encoder \(enc) tensors, quant_conv \(quant), other \(other); decoder elements \(elements(used)) = \(Double(elements(used)) * 2 / 1e6) MB bf16, whole file \(elements(header)) elements")
        #expect(enc + quant.count + other.count == header.count - used.count)
        #expect(other == ["bn.num_batches_tracked"] && quant == ["quant_conv.bias", "quant_conv.weight"])
        #expect(header["bn.running_mean"]?.shape == [128] && header["bn.running_var"]?.shape == [128])
    }
}

typealias Flux2KleinHeaderTensor = Klein3eHeader.Tensor

@Suite(.enabled(if: Klein3gReal.peakEnabled, Klein3gReal.peakSkipReason), .serialized)
struct Flux2KleinVAERealPeakTests {
    @Test func decodeAtEachSizeAndRecordThePeaks() throws {
        let dir = try #require(Klein3gReal.directory)
        let dtype = try #require(Klein3gReal.dtype)
        let label = dtype == .bfloat16 ? "bf16" : "fp32"
        if let mb = Klein3gReal.cacheMB { Memory.cacheLimit = mb * 1024 * 1024 }
        let cacheNote = Klein3gReal.cacheMB.map { "cacheLimit \($0) MB" } ?? "default cache"
        let chunkNote0 = Klein3gReal.chunk.map { "attention chunk \($0)" } ?? "attention unchunked"

        Memory.clearCache(); Memory.peakMemory = 0
        let fpLoad = KleinFootprint()
        let before = KleinFootprint.gb(KleinFootprint.now())
        fpLoad.start()
        let t0 = Date()
        let vae = try Flux2KleinVAE.load(vaeDirectory: dir, computeDType: dtype)
        if let each = Klein3gReal.evalEach { vae.setEvaluatesBetweenLayers(each) }
        let loadSeconds = Date().timeIntervalSince(t0)
        let loadFootprint = KleinFootprint.gb(fpLoad.stop())
        print(String(format: "VAE3G-PEAK %@ LOAD: %.2f s, weights %.3f GB, mlx peak %.3f GB, footprint peak %.2f GB (before %.2f), %@, %@", label, loadSeconds,
                     KleinFootprint.gb(vae.parameterBytes()), KleinFootprint.gb(Memory.peakMemory), loadFootprint, before, cacheNote, chunkNote0))
        vae.setAttentionQueryChunk(Klein3gReal.chunk)
        let chunkNote = chunkNote0 + (vae.evaluatesBetweenLayers ? ", eval per layer" : ", single eval")

        for size in Klein3gReal.sizes {
            let grid = size / 16
            Memory.clearCache()
            MLXRandom.seed(20261031)
            // a normalised packed latent, bf16 like the DiT's output; evaluated before the timed region
            let packed = MLXRandom.normal([1, 128, grid, grid]).asType(.bfloat16)
            try MLXErrorGuard.evaluate(packed)
            Memory.peakMemory = 0
            let fp = KleinFootprint()
            let footBefore = KleinFootprint.gb(KleinFootprint.now())
            fp.start()
            let t = Date()
            let image = try vae.decode(packedLatents: packed)
            let seconds = Date().timeIntervalSince(t)
            let mlxPeak = KleinFootprint.gb(Memory.peakMemory)
            let footPeak = KleinFootprint.gb(fp.stop())
            let stats = try MLXErrorGuard.read(stacked([isFinite(image).all().asType(.float32), image.min(), image.max(), sqrt((image * image).mean())]), as: Float.self)
            print(String(format: "VAE3G-PEAK %@ %dx%d: decode %.2f s, mlx peak %.3f GB, footprint peak %.2f GB (before %.2f), finite %@, range [%.2f, %.2f], rms %.3f, shape %@, %@, %@",
                         label, size, size, seconds, mlxPeak, footPeak, footBefore, stats[0] == 1 ? "yes" : "NO", stats[1], stats[2], stats[3], "\(image.shape)", cacheNote, chunkNote))
            #expect(image.shape == [1, 3, size, size] && stats[0] == 1)
        }
        Memory.clearCache()
    }
}

/// Opt-in (TEST_RUNNER_MLXUI_TEST_KLEIN_VAE_PRECISION=1): the same random normalised latent decoded by the real VAE with the shipped bf16 weights and with the weights
/// upcast to float32, so the cost of bf16 compute alone is on record before the owner's diffusers run exists. A random latent is not a natural image's latent: this is
/// an indication, not a quality claim (the diffusers comparison is Flux2KleinRealVAEGoldenTests).
@Suite(.enabled(if: Klein3gReal.present && ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_VAE_PRECISION"] == "1", Klein3gReal.peakSkipReason), .serialized)
struct Flux2KleinVAERealPrecisionTests {
    @Test func bf16VersusFloat32OnTheRealWeights() throws {
        let dir = try #require(Klein3gReal.directory)
        Memory.cacheLimit = 256 * 1024 * 1024
        let fp32 = try Flux2KleinVAE.load(vaeDirectory: dir, computeDType: .float32)
        let bf16 = try Flux2KleinVAE.load(vaeDirectory: dir, computeDType: nil)
        #expect(bf16.computeDType == .bfloat16 && fp32.computeDType == .float32)
        for seed in [UInt64(1), 2, 3] {
            MLXRandom.seed(seed)
            let packed = MLXRandom.normal([1, 128, 16, 16])           // 256 x 256
            let a = try fp32.decode(packedLatents: packed), b = try bf16.decode(packedLatents: packed.asType(.bfloat16))
            let e = try Klein3gFix.compare(b, a)
            print(String(format: "VAE3G-PREC seed %llu 256x256 bf16 vs fp32 (same weights, random latent): PSNR %.2f dB, cosine %.8f, max-abs %.4f", seed, e.psnr, e.cosine, e.maxAbs))
            #expect(e.psnr > 30)
        }
    }
}
