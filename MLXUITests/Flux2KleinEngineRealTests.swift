import CoreGraphics
import Darwin
import Foundation
import ImageIO
import MLX
import MLXRandom
import Testing
import UniformTypeIdentifiers
@testable import MLXUI

// MP7-3h, rule 13b, OPT-IN: the staged Klein engine on the REAL installed model. Unsandboxed host (reads ~/Library/Application Support/AI Browser, or
// TEST_RUNNER_MLXUI_TEST_KLEIN_DIR). Nothing is downloaded; the installed folder is only read. ONE of these per process, with the Klein app CLOSED (`pgrep -fl MLXUI` empty).
//
//  1. End to end (done-when h): TEST_RUNNER_MLXUI_TEST_KLEIN_E2E_OUT=<dir> (where the PNG is written). One 1024 x 1024 image, quantize 8, 4 steps, encoder 8-bit, DiT 8-bit, VAE bf16.
//     Run on the main actor inside MLXErrorGuard.guardedRun, as the Run UI calls it, so it is also the KICKOFF 17c main-actor probe on real weights.
//     Optional: ..._E2E_SIZE (default 1024), ..._E2E_QUANTIZE (default 8), ..._E2E_EACH_BLOCK=0 (per-block evaluation is ON by default).
//  2. DiT working memory (done-when c, d): TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_SIZES=512,768,1024 and TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_EACH=0|1 (ONE value per process):
//     the streamed DiT at TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_QUANTIZE (default 8), one forward per size on random inputs, MLX peak and footprint peak per size.
//  3. Real golden comparison (owner-run klein-ref/real_e2e_goldens.py): TEST_RUNNER_MLXUI_TEST_KLEIN_E2E_GOLDEN_DIR=<dir with real_e2e.safetensors>. Skips without it.

enum Klein3hReal {
    nonisolated static let env = ProcessInfo.processInfo.environment
    nonisolated static var modelDirectory: URL? { Klein3eHeader.directory?.deletingLastPathComponent() }
    nonisolated static var installedAllComponents: Bool {
        guard let dir = modelDirectory else { return false }
        let fm = FileManager.default
        return ["text_encoder/config.json", "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
            .allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) }
    }

    nonisolated static let e2eOut: URL? = env["MLXUI_TEST_KLEIN_E2E_OUT"].map { URL(fileURLWithPath: $0, isDirectory: true) }
    nonisolated static let e2eEnabled = installedAllComponents && e2eOut != nil
    nonisolated static let e2eSkip: Comment = "needs the installed Klein model, TEST_RUNNER_MLXUI_TEST_KLEIN_E2E_OUT=<dir> and an unsandboxed host; loads about 16 GB of weights in stages, app closed"

    nonisolated static let ditSizes: [Int] = (env["MLXUI_TEST_KLEIN_DIT_SIZES"] ?? "").split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    nonisolated static let ditEach: Bool? = env["MLXUI_TEST_KLEIN_DIT_EACH"].map { $0 == "1" }
    nonisolated static let ditEnabled = installedAllComponents && !ditSizes.isEmpty && ditEach != nil
    nonisolated static let ditSkip: Comment = "needs the installed Klein model, TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_SIZES=512,1024 and TEST_RUNNER_MLXUI_TEST_KLEIN_DIT_EACH=0|1, unsandboxed host"

    nonisolated static let goldenFile: URL? = {
        guard let dir = env["MLXUI_TEST_KLEIN_E2E_GOLDEN_DIR"] else { return nil }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("real_e2e.safetensors")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }()
    nonisolated static let goldenEnabled = installedAllComponents && goldenFile != nil
    nonisolated static let goldenSkip: Comment = "needs TEST_RUNNER_MLXUI_TEST_KLEIN_E2E_GOLDEN_DIR (folder with real_e2e.safetensors from real_e2e_goldens.py), the installed Klein model and an unsandboxed host"

    nonisolated static let prompt = "a red fox in a snowy forest at sunrise"

    static func writePNG(_ image: CGImage, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}

/// Footprint peak per stage, driven by the engine's own events.
final class Klein3hFootprints: @unchecked Sendable {
    private let lock = NSLock()
    private var current: (String, KleinFootprint)?
    private var done: [(String, Double)] = []
    private var lines: [String] = []
    let t0 = Date()

    func observe(_ e: Flux2KleinLifecycleEvent) {
        lock.lock(); defer { lock.unlock() }
        let t = String(format: "%.1f", Date().timeIntervalSince(t0))
        let mlx = String(format: "mlx active %.2f GB, peak %.2f GB, cache %.2f GB, footprint %.2f GB", KleinFootprint.gb(Memory.activeMemory), KleinFootprint.gb(Memory.peakMemory),
                         KleinFootprint.gb(Memory.cacheMemory), KleinFootprint.gb(KleinFootprint.now()))
        lines.append("KLEIN3H-E2E t=\(t)s \(e): \(mlx)")
        print(lines.last ?? "")
        if case let .cacheLimitApplied(c, _) = e {
            if let (label, sampler) = current { done.append((label, KleinFootprint.gb(sampler.stop()))) }
            let sampler = KleinFootprint(); sampler.start()
            current = (c.rawValue, sampler)
        }
    }

    func finish() -> [(String, Double)] {
        lock.lock(); defer { lock.unlock() }
        if let (label, sampler) = current { done.append((label, KleinFootprint.gb(sampler.stop()))); current = nil }
        return done
    }
}

@Suite(.enabled(if: Klein3hReal.e2eEnabled, Klein3hReal.e2eSkip), .serialized)
struct Flux2KleinEngineRealE2ETests {
    @Test @MainActor func oneImageEndToEndFromTheMainActorInsideGuardedRun() async throws {
        let dir = try #require(Klein3hReal.modelDirectory)
        let out = try #require(Klein3hReal.e2eOut)
        let size = Klein3hReal.env["MLXUI_TEST_KLEIN_E2E_SIZE"].flatMap { Int($0) } ?? 1024
        let quantize = Klein3hReal.env["MLXUI_TEST_KLEIN_E2E_QUANTIZE"].flatMap { Int($0) } ?? 8
        let each = Klein3hReal.env["MLXUI_TEST_KLEIN_E2E_EACH_BLOCK"] != "0"      // default on, like the request
        let limitBefore = Memory.cacheLimit
        let footprints = Klein3hFootprints()
        let whole = KleinFootprint()
        Memory.clearCache()
        let before = KleinFootprint.gb(KleinFootprint.now())
        whole.start()
        let engine = Flux2KleinEngine(stages: Flux2KleinInstalledStages(modelDirectory: dir), observer: { footprints.observe($0) })
        var request = Flux2KleinRequest(prompt: Klein3hReal.prompt, width: size, height: size, steps: 4, seed: 7, quantize: quantize)
        request.evaluatesEachBlock = each
        let started = Date()
        let result = try await MLXErrorGuard.guardedRun { try await engine.generate(request) }
        let total = Date().timeIntervalSince(started)
        let perStage = footprints.finish()
        let wholePeak = KleinFootprint.gb(whole.stop())

        for s in result.stages {
            print(String(format: "KLEIN3H-E2E STAGE %@: %.1f s, mlx peak %.2f GB", s.component.rawValue, s.seconds, KleinFootprint.gb(s.mlxPeakBytes)))
        }
        for (label, peak) in perStage { print(String(format: "KLEIN3H-E2E FOOTPRINT PEAK %@: %.2f GB", label, peak)) }
        print(String(format: "KLEIN3H-E2E TOTAL: %.1f s, footprint peak %.2f GB (before %.2f GB), %dx%d quantize %d, 4 steps, eval each block %@, cacheLimit restored %@",
                     total, wholePeak, before, size, size, quantize, each ? "yes" : "no", Memory.cacheLimit == limitBefore ? "yes" : "NO"))
        let name = "klein-\(size)-q\(quantize)-seed7\(each ? "" : "-noeval").png"
        try Klein3hReal.writePNG(result.image, to: out.appendingPathComponent(name))
        print("KLEIN3H-E2E PNG: \(out.appendingPathComponent(name).path)")

        #expect(result.image.width == size && result.image.height == size)
        #expect(Memory.cacheLimit == limitBefore, "the process-global cache limit must be put back")
        #expect(result.stages.map(\.component) == [.textEncoder, .transformer, .vae])
        // STOP LINE (lead, 2026-11-01): 14 GB footprint for this one run.
        #expect(wholePeak < 14.0, "STOP LINE: footprint peaked at \(wholePeak) GB")
    }
}

@Suite(.enabled(if: Klein3hReal.ditEnabled, Klein3hReal.ditSkip), .serialized)
struct Flux2KleinDiTWorkingMemoryTests {
    @Test func oneForwardPerSizeAtTheStreamedQuantize() throws {
        let dir = try #require(Klein3hReal.modelDirectory)
        let each = try #require(Klein3hReal.ditEach)
        let quantize = Klein3hReal.env["MLXUI_TEST_KLEIN_DIT_QUANTIZE"].flatMap { Int($0) } ?? 8
        let capMB = Klein3hReal.env["MLXUI_TEST_KLEIN_DIT_CACHE_MB"].flatMap { Int($0) } ?? 256
        let limitBefore = Memory.cacheLimit
        defer { Memory.cacheLimit = limitBefore }
        Memory.cacheLimit = min(limitBefore, capMB * 1_048_576)

        Memory.clearCache(); Memory.peakMemory = 0
        let load = KleinFootprint()
        let before = KleinFootprint.gb(KleinFootprint.now())
        load.start()
        let t0 = Date()
        let dit = try Flux2KleinInstalledStages(modelDirectory: dir).makeTransformer(quantize: quantize)
        let loadSeconds = Date().timeIntervalSince(t0)
        print(String(format: "DIT3H LOAD q%d: %.1f s, weights %.3f GB, mlx peak %.2f GB, footprint peak %.2f GB (before %.2f), cacheLimit %d MB", quantize, loadSeconds,
                     KleinFootprint.gb(dit.parameterBytes()), KleinFootprint.gb(Memory.peakMemory), KleinFootprint.gb(load.stop()), before, capMB))
        let weights = KleinFootprint.gb(Memory.activeMemory)

        for size in Klein3hReal.ditSizes {
            let g = size / 16
            let n = g * g
            MLXRandom.seed(99)
            let hidden = MLXRandom.normal([1, n, 128]).asType(.bfloat16)
            let encoder = MLXRandom.normal([1, 512, 7680]).asType(.bfloat16)
            let imgIDs = Flux2KleinLatents.imageIDs(gridHeight: g, gridWidth: g), txtIDs = Flux2KleinLatents.textIDs(count: 512)
            try MLXErrorGuard.evaluate(hidden, encoder)
            Memory.clearCache(); Memory.peakMemory = 0
            let sampler = KleinFootprint()
            sampler.start()
            let t = Date()
            let out = try dit(hidden: hidden, encoder: encoder, sigma: MLXArray([0.9] as [Float]), imgIDs: imgIDs, txtIDs: txtIDs, evaluatesEachBlock: each)
            let seconds = Date().timeIntervalSince(t)
            let mlxPeak = KleinFootprint.gb(Memory.peakMemory)
            let footprint = KleinFootprint.gb(sampler.stop())
            let finite = try MLXErrorGuard.read(isFinite(out.asType(.float32)).all().reshaped(1), as: Bool.self)[0]
            print(String(format: "DIT3H FORWARD q%d %dx%d (%d image tokens) eval-each-block %@: %.2f s, mlx peak %.2f GB (weights %.2f, working %.2f), footprint peak %.2f GB, cacheLimit %d MB, finite %@",
                         quantize, size, size, n, each ? "yes" : "no", seconds, mlxPeak, weights, mlxPeak - weights, footprint, capMB, "\(finite)"))
            #expect(finite)
            #expect(footprint < 10.0, "STOP LINE (10 GB): DiT forward at \(size) peaked at footprint \(footprint) GB")
        }
    }
}

// MARK: - Real golden comparison (owner-run real_e2e_goldens.py)

@Suite(.enabled(if: Klein3hReal.goldenEnabled, Klein3hReal.goldenSkip), .serialized)
struct Flux2KleinEngineRealGoldenTests {
    /// `in.noise` = the script's initial seeded noise latents, float32 [1,128,size/16,size/16] (patchified, before packing); both tests inject exactly it and never draw their own.
    /// Names the script writes (and `requiredKeys` checks first, so a mismatch is a clear failure, not a nil).
    static let requiredKeys = ["in.noise", "in.prompt_embeds", "out.final_latents_packed", "out.image", "out.step0", "out.step1", "out.step2", "out.step3"]

    /// PROVISIONAL floors, labelled as such (implementer's call, pending owner confirmation): there is no measurement yet. They only say "the picture is the same picture"
    /// (not noise, not a different seed's image); the real numbers go to MP7-3h-report.md after the owner's first run and these are then pinned just under them.
    static let provisionalImageCosine = 0.90
    static let provisionalLatentCosine = 0.90

    struct Golden {
        let arrays: [String: MLXArray]
        let prompt: String
        let size: Int
        let steps: Int
    }

    static func load() throws -> Golden {
        let url = try #require(Klein3hReal.goldenFile)
        let (arrays, meta) = try loadArraysAndMetadata(url: url)
        let missing = requiredKeys.filter { arrays[$0] == nil }
        if !missing.isEmpty {
            throw Klein3hGoldenError.missingKeys("the golden file \(url.lastPathComponent) is missing \(missing). `in.noise` is the seeded float32 noise real_e2e_goldens.py injected into diffusers; this test must feed THOSE latents (torch and MLX RNGs differ), so regenerate the golden with the current script")
        }
        return Golden(arrays: arrays, prompt: meta["prompt"] ?? Klein3hReal.prompt, size: Int(meta["size"] ?? "512") ?? 512, steps: Int(meta["steps"] ?? "4") ?? 4)
    }

    static func report(_ label: String, result: Flux2KleinResult, golden: Golden, steps: [(Int, MLXArray)]) throws {
        for (i, x) in steps.sorted(by: { $0.0 < $1.0 }) {
            guard let want = golden.arrays["out.step\(i)"] else { continue }
            let c = try KleinCosine.of(x, want)
            let rel = try Klein3hRelL2(x, want)
            print(String(format: "E2EGOLDEN %@ step %d latents vs diffusers bf16: cosine %.6f, relative L2 %.4f", label, i, c, rel))
            #expect(c >= provisionalLatentCosine, "\(label) step \(i)")
        }
        let want = try #require(golden.arrays["out.image"])
        let e = try Klein3gFix.compare(result.pixels, want)
        print(String(format: "E2EGOLDEN %@ image vs diffusers bf16: PSNR %.2f dB, cosine %.6f, max-abs %.4f", label, e.psnr, e.cosine, e.maxAbs))
        #expect(e.cosine >= provisionalImageCosine, "\(label): the picture is not the same picture")
    }

    /// Full pipeline of this port (its own 8-bit encoder, 8-bit DiT, bf16 VAE) from the golden's noise, against diffusers bf16 from the same noise.
    @Test @MainActor func fullPipelineFromInjectedNoise() async throws {
        let golden = try Self.load()
        let dir = try #require(Klein3hReal.modelDirectory)
        let collector = Flux2KleinEngineGoldenTests.StepCollector()
        var injection = Flux2KleinInjection(noise: try #require(golden.arrays["in.noise"]))
        injection.stepProbe = { i, x in collector.add(i, x) }
        let engine = Flux2KleinEngine(stages: Flux2KleinInstalledStages(modelDirectory: dir))
        let request = Flux2KleinRequest(prompt: golden.prompt, width: golden.size, height: golden.size, steps: golden.steps, seed: 7, quantize: 8)
        let result = try await MLXErrorGuard.guardedRun { try await engine.generate(request, injection: injection) }
        try Self.report("own-encoder", result: result, golden: golden, steps: collector.all)
    }

    /// Same, but diffusers' own prompt embeddings are injected, so the encoder's 8-bit error is out of the picture and only the DiT (8-bit) and VAE (bf16) differ.
    @Test @MainActor func ditAndVAEOnlyFromInjectedNoiseAndEmbeddings() async throws {
        let golden = try Self.load()
        let dir = try #require(Klein3hReal.modelDirectory)
        let collector = Flux2KleinEngineGoldenTests.StepCollector()
        var injection = Flux2KleinInjection(noise: try #require(golden.arrays["in.noise"]), embeddings: try #require(golden.arrays["in.prompt_embeds"]))
        injection.stepProbe = { i, x in collector.add(i, x) }
        let engine = Flux2KleinEngine(stages: Flux2KleinInstalledStages(modelDirectory: dir))
        let request = Flux2KleinRequest(prompt: golden.prompt, width: golden.size, height: golden.size, steps: golden.steps, seed: 7, quantize: 8)
        let result = try await MLXErrorGuard.guardedRun { try await engine.generate(request, injection: injection) }
        try Self.report("golden-embeddings", result: result, golden: golden, steps: collector.all)
    }
}

nonisolated enum Klein3hGoldenError: Error, CustomStringConvertible {
    case missingKeys(String)
    var description: String { switch self { case let .missingKeys(m): return m } }
}

/// ||a - b|| / ||b||
func Klein3hRelL2(_ a: MLXArray, _ b: MLXArray) throws -> Double {
    let x = try MLXErrorGuard.read(a.asType(.float32).reshaped(-1), as: Float.self)
    let y = try MLXErrorGuard.read(b.asType(.float32).reshaped(-1), as: Float.self)
    var num = 0.0, den = 0.0
    for i in 0 ..< x.count { let d = Double(x[i]) - Double(y[i]); num += d * d; den += Double(y[i]) * Double(y[i]) }
    return (num / den).squareRoot()
}
