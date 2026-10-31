import Darwin
import Foundation
import MLX
import Testing
@testable import MLXUI

// MP7-3c, rule 13b: the Klein text encoder on the REAL installed weights, opt-in. Gated on `text_encoder/model-00001-of-00002.safetensors` being present:
// TEST_RUNNER_MLXUI_TEST_KLEIN_DIR (folder with text_encoder/ and tokenizer/), else the Direct store's mlx-community--FLUX.2-klein-4B-bf16.
// The sandboxed MLXUI test host cannot read ~/Library/Application Support, so the real run needs the unsandboxed host `model-smoke.sh` uses:
//   TEST_RUNNER_MLXUI_TEST_KLEIN_DIR=<dir> xcodebuild test -scheme MLXUI -destination 'platform=macOS' -parallel-testing-enabled NO \
//     CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO -only-testing:MLXUITests/Flux2KleinEncoderMemoryTests
// (then ...Flux2KleinEncoderQuantizationTests, ...Flux2KleinEncoderMainActorProbeTests). ONE process loads these weights at a time. No goldens from
// diffusers exist for the real encoder yet; the quantization suite compares this port against ITSELF unquantized and claims nothing about diffusers.

enum KleinRealInstall {
    nonisolated static let directory: URL? = {
        let url = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_DIR"].map { URL(fileURLWithPath: $0) }
            ?? ModelStore.shared.directory(forHFModelID: "mlx-community/FLUX.2-klein-4B-bf16")
        let shard = url.appendingPathComponent("text_encoder/model-00001-of-00002.safetensors")
        return FileManager.default.fileExists(atPath: shard.path) ? url : nil
    }()
    nonisolated static let enabled = directory != nil
    nonisolated static let skipReason: Comment = "needs the installed FLUX.2 Klein (mlx-community--FLUX.2-klein-4B-bf16) or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR, and an unsandboxed test host"
}

/// Process `phys_footprint` (what Activity Monitor shows), decimal GB, with a background sampler for the peak (RSI/evals/app-footprint.swift measures it the same way).
final class KleinFootprint: @unchecked Sendable {
    private let lock = NSLock()
    private var peakBytes: UInt64 = 0
    private var running = false

    static func now() -> UInt64 {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { ptr in
            ptr.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
        }
        return rc == 0 ? info.ri_phys_footprint : 0
    }

    static func gb(_ bytes: UInt64) -> Double { Double(bytes) / 1e9 }
    static func gb(_ bytes: Int) -> Double { Double(bytes) / 1e9 }

    func start() {
        lock.lock(); peakBytes = Self.now(); running = true; lock.unlock()
        Thread.detachNewThread { [self] in
            while true {
                lock.lock(); let go = running; if go { peakBytes = max(peakBytes, Self.now()) }; lock.unlock()
                if !go { return }
                usleep(20_000)
            }
        }
    }

    func stop() -> UInt64 {
        lock.lock(); defer { lock.unlock() }
        running = false
        return max(peakBytes, Self.now())
    }
}

struct KleinPhase {
    var bits: Int?
    var loadSeconds = 0.0, encodeSeconds = 0.0
    var mlxPeakGB = 0.0, footprintPeakGB = 0.0, footprintBeforeGB = 0.0, footprintAfterFreeGB = 0.0, mlxActiveAfterFreeGB = 0.0
    var label: String { bits.map { "int\($0)" } ?? "bf16" }
}

/// Loads, encodes the given prompts, frees, and measures. Returns the embeddings (evaluated) of each prompt.
@discardableResult
func kleinMeasure(bits: Int?, prompts: [Flux2KleinEncodedPrompt], dir: URL, into phase: inout KleinPhase,
                  keep: ((Int, MLXArray) throws -> Void)? = nil, narrate: Bool = false) throws -> Int {
    let sampler = KleinFootprint()
    Memory.clearCache()
    Memory.peakMemory = 0
    phase.footprintBeforeGB = KleinFootprint.gb(KleinFootprint.now())
    sampler.start()
    let t0 = Date()
    var encoder: Flux2KleinTextEncoder? = try Flux2KleinTextEncoder.load(
        textEncoderDirectory: dir.appendingPathComponent("text_encoder"), bits: bits) { stage in
        if narrate || stage.hasPrefix("shard") || stage == "embedding ready" || stage == "layer 0 ready" || stage == "layer 26 ready" {
            print(String(format: "STAGE %@ %@: mlx active %.2f GB, mlx peak %.2f GB, footprint %.2f GB", phase.label, stage,
                         KleinFootprint.gb(Memory.activeMemory), KleinFootprint.gb(Memory.peakMemory), KleinFootprint.gb(KleinFootprint.now())))
        }
    }
    phase.loadSeconds = Date().timeIntervalSince(t0)
    let t1 = Date()
    for (i, p) in prompts.enumerated() {
        let out = try #require(encoder).embeddings(for: p)
        try keep?(i, out)
    }
    phase.encodeSeconds = Date().timeIntervalSince(t1) / Double(max(prompts.count, 1))
    phase.mlxPeakGB = KleinFootprint.gb(Memory.peakMemory)
    phase.footprintPeakGB = KleinFootprint.gb(sampler.stop())
    encoder = nil
    Memory.clearCache()
    phase.mlxActiveAfterFreeGB = KleinFootprint.gb(Memory.activeMemory)
    phase.footprintAfterFreeGB = KleinFootprint.gb(KleinFootprint.now())
    Thread.sleep(forTimeInterval: 2)
    print(String(format: "SMOKE-NUMBERS %@: footprint 2 s after free %.2f GB", phase.label, KleinFootprint.gb(KleinFootprint.now())))
    print(String(format: "SMOKE-NUMBERS %@: load %.1f s, encode %.2f s/prompt, mlx peak %.2f GB, footprint peak %.2f GB (before %.2f), after free: footprint %.2f GB, mlx active %.3f GB",
                 phase.label, phase.loadSeconds, phase.encodeSeconds, phase.mlxPeakGB, phase.footprintPeakGB, phase.footprintBeforeGB,
                 phase.footprintAfterFreeGB, phase.mlxActiveAfterFreeGB))
    return prompts.count
}

func kleinPrompts(_ dir: URL, names: [String]? = nil) async throws -> (names: [String], prompts: [Flux2KleinEncodedPrompt]) {
    struct G: Decodable { let prompt: String; let name: String? }
    let base = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein")
    var all: [G] = []
    for f in ["g3b_tokens.json", "g3b_tokens_edge.json"] { all += try JSONDecoder().decode([G].self, from: Data(contentsOf: base.appendingPathComponent(f))) }
    let tokenizer = try await HFTokenizerLoader().load(from: dir.appendingPathComponent("tokenizer"))
    var labels: [String] = [], encoded: [Flux2KleinEncodedPrompt] = []
    for (i, g) in all.enumerated() {
        let label = g.name ?? "p\(i)"
        if let names, !names.contains(label) { continue }
        labels.append(label)
        encoded.append(try Flux2KleinPrompt.encode(g.prompt, tokenizer: tokenizer))
    }
    return (labels, encoded)
}

@Suite(.enabled(if: KleinRealInstall.enabled, KleinRealInstall.skipReason), .serialized)
struct Flux2KleinEncoderMemoryTests {
    /// Streamed quantize-at-load at int4 and int8: peak MLX memory and process footprint, one prompt each. The owner's stop line is ~8 GB.
    @Test func streamedLoadStaysUnderTheStopLine() async throws {
        let dir = try #require(KleinRealInstall.directory)
        let (_, prompts) = try await kleinPrompts(dir, names: ["p0"])
        for bits in [4, 8] {
            var phase = KleinPhase(bits: bits)
            try kleinMeasure(bits: bits, prompts: prompts, dir: dir, into: &phase)
            #expect(phase.mlxPeakGB < 8.0 && phase.footprintPeakGB < 8.0, "int\(bits): STOP LINE (mlx \(phase.mlxPeakGB) GB, footprint \(phase.footprintPeakGB) GB)")
        }
    }
}

@Suite(.enabled(if: KleinRealInstall.enabled, KleinRealInstall.skipReason), .serialized)
struct Flux2KleinEncoderQuantizationTests {
    struct Stats { var all = 0.0, taps: [Double] = [], front = 0.0, pad = 0.0, worstRow = 1.0 }

    static func stats(_ a: MLXArray, _ ref: MLXArray) throws -> Stats {
        var s = Stats()
        s.all = try KleinCosine.of(a, ref)
        for t in 0 ..< 3 { s.taps.append(try KleinCosine.of(a[0..., 0..., (t * 2560) ..< ((t + 1) * 2560)], ref[0..., 0..., (t * 2560) ..< ((t + 1) * 2560)])) }
        s.front = try KleinCosine.of(a[0..., 0 ..< 48, 0...], ref[0..., 0 ..< 48, 0...])
        s.pad = try KleinCosine.of(a[0..., 500 ..< 512, 0...], ref[0..., 500 ..< 512, 0...])
        let x = try MLXErrorGuard.read(a.asType(.float32).reshaped(512, 7680), as: Float.self)
        let y = try MLXErrorGuard.read(ref.asType(.float32).reshaped(512, 7680), as: Float.self)
        for r in 0 ..< 512 { s.worstRow = min(s.worstRow, KleinCosine.cosine(Array(x[(r * 7680) ..< ((r + 1) * 7680)]), Array(y[(r * 7680) ..< ((r + 1) * 7680)]))) }
        return s
    }

    // Floors set AFTER the first measurement (2026-10-31, 13 golden prompts, own-port bf16 as the reference), each a margin below the measured minimum:
    //   int8 minima: all 0.999305, tap 0.998764, rows 0-47 0.999975, rows 500-511 0.964933
    //   int4 minima: all 0.936666, tap 0.907350, rows 0-47 0.999490, rows 500-511 0.344044 (pad rows are the weak spot; the worst single row is -0.86)
    // These pin the measured behaviour so a regression shows; they are not a claim that int4 pad rows are good enough for the DiT (3h's end-to-end
    // image check decides that).
    static let floors: [Int: (all: Double, tap: Double, front: Double, pad: Double)] = [
        8: (all: 0.998, tap: 0.997, front: 0.9999, pad: 0.95),
        4: (all: 0.92, tap: 0.89, front: 0.998, pad: 0.30),
    ]

    /// This port unquantized (bf16, the reference for the quantized runs) vs int8 and int4 of the same port, on all 13 golden prompts.
    @Test func int8AndInt4StayCloseToTheUnquantizedPort() async throws {
        let dir = try #require(KleinRealInstall.directory)
        let (names, prompts) = try await kleinPrompts(dir)
        #expect(prompts.count == 13)
        var reference: [MLXArray] = []
        var bf16 = KleinPhase(bits: nil)
        try kleinMeasure(bits: nil, prompts: prompts, dir: dir, into: &bf16) { _, out in
            // finite everywhere, including the 500...511 pad rows
            let probe = try MLXErrorGuard.read(out[0..., 500 ..< 512, 0 ..< 64].asType(.float32).reshaped(-1), as: Float.self)
            #expect(probe.allSatisfy { $0.isFinite })
            reference.append(out)
        }
        #expect(bf16.mlxPeakGB < 8.0 && bf16.footprintPeakGB < 8.0, "bf16 reference: STOP LINE (mlx \(bf16.mlxPeakGB) GB, footprint \(bf16.footprintPeakGB) GB)")
        for bits in [8, 4] {
            var phase = KleinPhase(bits: bits)
            var worst: [Stats] = []
            try kleinMeasure(bits: bits, prompts: prompts, dir: dir, into: &phase) { i, out in
                let s = try Self.stats(out, reference[i])
                worst.append(s)
                print(String(format: "QUANTPARITY int%d %@: all %.6f taps %.6f %.6f %.6f rows0-47 %.6f rows500-511 %.6f worst-row %.6f", bits, names[i], s.all,
                             s.taps[0], s.taps[1], s.taps[2], s.front, s.pad, s.worstRow))
            }
            let minAll = worst.map(\.all).min() ?? 0, minFront = worst.map(\.front).min() ?? 0, minPad = worst.map(\.pad).min() ?? 0
            let minTap = worst.flatMap(\.taps).min() ?? 0, minRow = worst.map(\.worstRow).min() ?? 0
            print(String(format: "QUANTPARITY int%d MIN over 13: all %.6f tap %.6f rows0-47 %.6f rows500-511 %.6f worst-row %.6f", bits, minAll, minTap, minFront, minPad, minRow))
            if let f = Self.floors[bits] {
                #expect(minAll >= f.all && minTap >= f.tap && minFront >= f.front && minPad >= f.pad, "int\(bits)")
            }
        }
    }
}

@Suite(.enabled(if: KleinRealInstall.enabled, KleinRealInstall.skipReason), .serialized)
@MainActor
struct Flux2KleinEncoderMainActorProbeTests {
    /// The Run UI's call shape: `@MainActor`, inside `guardedRun`, the work hops off the main actor (`@concurrent`), the result comes back evaluated and is read on the main thread.
    @Test func encodeFromTheMainActorReturnsEmbeddingsAndRecordsNoMLXError() async throws {
        let dir = try #require(KleinRealInstall.directory)
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let result = try await MLXErrorGuard.guardedRun {
            try await Flux2KleinTextEncoding.embed(prompt: "a red fox in a snowy forest at sunrise", modelDirectory: dir, bits: 4)
        }
        #expect(result.array.shape == [1, 512, 7680])
        #expect(result.realLength > 12 && result.cutTokenCount == 0)
        // Read on the MAIN thread an array built on a pool thread (rule 17: thread-bound streams).
        let front = try MLXErrorGuard.read(result.array[0..., 0 ..< 4, 0 ..< 32].asType(.float32).reshaped(-1), as: Float.self)
        let pad = try MLXErrorGuard.read(result.array[0..., 508 ..< 512, 0 ..< 32].asType(.float32).reshaped(-1), as: Float.self)
        #expect(front.allSatisfy { $0.isFinite } && front.contains { $0 != 0 })
        #expect(pad.allSatisfy { $0.isFinite } && pad.contains { $0 != 0 })
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        print("PROBE main-actor encode: ok, real tokens \(result.realLength)")
    }

    /// Cancelling mid-run is the user pressing Stop: a CancellationError (or an already-finished result), never an MLX error.
    @Test func aCancelledRunIsNotAnError() async throws {
        let dir = try #require(KleinRealInstall.directory)
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let task = Task { @MainActor in
            try await MLXErrorGuard.guardedRun {
                try await Flux2KleinTextEncoding.embed(prompt: "a lighthouse at dusk", modelDirectory: dir, bits: 4)
            }
        }
        try await Task.sleep(for: .milliseconds(1500))
        task.cancel()
        switch await task.result {
        case .failure(let error): #expect(error is CancellationError, "unexpected error: \(error)")
        case .success: print("PROBE cancel: the run finished before the cancel landed")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        Memory.clearCache()
    }
}
