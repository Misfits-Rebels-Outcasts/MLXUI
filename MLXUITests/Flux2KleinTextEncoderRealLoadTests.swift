import Darwin
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

// MP7-3c, rule 13b: the Klein text encoder on the REAL installed weights, opt-in. Gated on `text_encoder/model-00001-of-00002.safetensors` being present:
// TEST_RUNNER_MLXUI_TEST_KLEIN_DIR (folder with text_encoder/ and tokenizer/), else the Direct store's mlx-community--FLUX.2-klein-4B-bf16.
// The sandboxed MLXUI test host cannot read ~/Library/Application Support, so the real run needs the unsandboxed host `model-smoke.sh` uses:
//   TEST_RUNNER_MLXUI_TEST_KLEIN_DIR=<dir> xcodebuild test -scheme MLXUI -destination 'platform=macOS' -parallel-testing-enabled NO \
//     CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO -only-testing:MLXUITests/Flux2KleinEncoderMemoryTests
// (then ...Flux2KleinEncoderQuantizationTests, ...Flux2KleinEncoderMainActorProbeTests, ...Flux2KleinRealShapeGoldenTests). ONE process loads these weights at a time. No goldens from
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
    /// Streamed quantize-at-load at the encoder's 8-bit floor: peak MLX memory and process footprint, one prompt. The owner's stop line is ~8 GB.
    @Test func streamedLoadStaysUnderTheStopLine() async throws {
        let dir = try #require(KleinRealInstall.directory)
        let (_, prompts) = try await kleinPrompts(dir, names: ["p0"])
        for bits in [8] {   // below 8 is not shipped for the encoder (MP7-3c-FIX-1); int4 numbers are in the 3c report
            var phase = KleinPhase(bits: bits)
            try kleinMeasure(bits: bits, prompts: prompts, dir: dir, into: &phase)
            #expect(phase.mlxPeakGB < 8.0 && phase.footprintPeakGB < 8.0, "int\(bits): STOP LINE (mlx \(phase.mlxPeakGB) GB, footprint \(phase.footprintPeakGB) GB)")
        }
    }
}

@Suite(.enabled(if: KleinRealInstall.enabled, KleinRealInstall.skipReason), .serialized)
struct Flux2KleinEncoderQuantizationTests {
    struct Stats {
        var all = 0.0, taps: [Double] = [], front = 0.0, pad = 0.0, worstRow = 1.0
        /// Relative L2 error ||a - ref|| / ||ref||: whole, rows 0-47, pad rows 500-511, and the worst single row.
        var relAll = 0.0, relFront = 0.0, relPad = 0.0, relWorstRow = 0.0
    }

    static func relL2(_ x: [Float], _ y: [Float]) -> Double {
        var d = 0.0, n = 0.0
        for i in 0 ..< x.count { let e = Double(x[i]) - Double(y[i]); d += e * e; n += Double(y[i]) * Double(y[i]) }
        return (d / max(n, 1e-30)).squareRoot()
    }

    static func stats(_ a: MLXArray, _ ref: MLXArray) throws -> Stats {
        var s = Stats()
        s.all = try KleinCosine.of(a, ref)
        for t in 0 ..< 3 { s.taps.append(try KleinCosine.of(a[0..., 0..., (t * 2560) ..< ((t + 1) * 2560)], ref[0..., 0..., (t * 2560) ..< ((t + 1) * 2560)])) }
        s.front = try KleinCosine.of(a[0..., 0 ..< 48, 0...], ref[0..., 0 ..< 48, 0...])
        s.pad = try KleinCosine.of(a[0..., 500 ..< 512, 0...], ref[0..., 500 ..< 512, 0...])
        let x = try MLXErrorGuard.read(a.asType(.float32).reshaped(512, 7680), as: Float.self)
        let y = try MLXErrorGuard.read(ref.asType(.float32).reshaped(512, 7680), as: Float.self)
        func row(_ v: [Float], _ r: Int) -> [Float] { Array(v[(r * 7680) ..< ((r + 1) * 7680)]) }
        for r in 0 ..< 512 {
            s.worstRow = min(s.worstRow, KleinCosine.cosine(row(x, r), row(y, r)))
            s.relWorstRow = max(s.relWorstRow, relL2(row(x, r), row(y, r)))
        }
        s.relAll = relL2(x, y)
        s.relFront = relL2(Array(x[0 ..< (48 * 7680)]), Array(y[0 ..< (48 * 7680)]))
        s.relPad = relL2(Array(x[(500 * 7680)...]), Array(y[(500 * 7680)...]))
        return s
    }

    // Floors set AFTER measuring (2026-10-31, 13 golden prompts, this port's bf16 as the reference), each a margin beyond the worst measured value.
    // They pin measured behaviour; they are not a claim about image quality. Worst measured (13 prompts): cosine all 0.999305, tap 0.998764, rows 0-47 0.999975,
    // pad 0.964933, worst row 0.646; relative L2 all 0.03803, rows 0-47 0.01052, pad 0.26277, worst single row 1.465 (the whitespace prompt).
    static let floors = (all: 0.998, tap: 0.997, front: 0.9999, pad: 0.95, relAll: 0.06, relFront: 0.015, relPad: 0.35, relWorstRow: 2.0)

    /// int8 (the encoder's floor; a request for int4 loads int8 too) vs this port unquantized, on all 13 golden prompts.
    @Test func int8StaysCloseToTheUnquantizedPort() async throws {
        let dir = try #require(KleinRealInstall.directory)
        let (names, prompts) = try await kleinPrompts(dir)
        #expect(prompts.count == 13)
        var reference: [MLXArray] = []
        var bf16 = KleinPhase(bits: nil)
        try kleinMeasure(bits: nil, prompts: prompts, dir: dir, into: &bf16) { _, out in
            let probe = try MLXErrorGuard.read(out[0..., 500 ..< 512, 0 ..< 64].asType(.float32).reshaped(-1), as: Float.self)
            #expect(probe.allSatisfy { $0.isFinite })
            reference.append(out)
        }
        #expect(bf16.mlxPeakGB < 8.0 && bf16.footprintPeakGB < 8.0, "bf16 reference: STOP LINE (mlx \(bf16.mlxPeakGB) GB, footprint \(bf16.footprintPeakGB) GB)")
        // The floor, on the real checkpoint: a request for 4 bits is an 8-bit encoder.
        do {   // scoped, so this encoder is gone before the measured int8 phase below
            let asked4 = try Flux2KleinTextEncoder.load(textEncoderDirectory: dir.appendingPathComponent("text_encoder"), bits: 4)
            #expect(asked4.bits == 8 && (asked4.embedTokens as? QuantizedEmbedding)?.bits == 8)
            #expect((asked4.layers[0].attention.qProj as? QuantizedLinear)?.bits == 8)
        }
        Memory.clearCache()

        var phase = KleinPhase(bits: 8)
        var all: [Stats] = []
        try kleinMeasure(bits: 8, prompts: prompts, dir: dir, into: &phase) { i, out in
            let s = try Self.stats(out, reference[i])
            all.append(s)
            print(String(format: "QUANTPARITY int8 %@: cos all %.6f taps %.6f %.6f %.6f rows0-47 %.6f pad %.6f worst-row %.6f | relL2 all %.5f rows0-47 %.5f pad %.5f worst-row %.5f",
                         names[i], s.all, s.taps[0], s.taps[1], s.taps[2], s.front, s.pad, s.worstRow, s.relAll, s.relFront, s.relPad, s.relWorstRow))
        }
        let minAll = all.map(\.all).min() ?? 0, minFront = all.map(\.front).min() ?? 0, minPad = all.map(\.pad).min() ?? 0
        let minTap = all.flatMap(\.taps).min() ?? 0, minRow = all.map(\.worstRow).min() ?? 0
        let maxAll = all.map(\.relAll).max() ?? 9, maxFront = all.map(\.relFront).max() ?? 9, maxPad = all.map(\.relPad).max() ?? 9
        let maxRow = all.map(\.relWorstRow).max() ?? 9
        print(String(format: "QUANTPARITY int8 WORST over 13: cos all %.6f tap %.6f rows0-47 %.6f pad %.6f worst-row %.6f | relL2 all %.5f rows0-47 %.5f pad %.5f worst-row %.5f",
                     minAll, minTap, minFront, minPad, minRow, maxAll, maxFront, maxPad, maxRow))
        let f = Self.floors
        #expect(minAll >= f.all && minTap >= f.tap && minFront >= f.front && minPad >= f.pad)
        #expect(maxAll <= f.relAll && maxFront <= f.relFront && maxPad <= f.relPad && maxRow <= f.relWorstRow)
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
            try await Flux2KleinTextEncoding.embed(prompt: "a red fox in a snowy forest at sunrise", modelDirectory: dir, bits: 8)
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
        Memory.clearCache()
        let cacheBefore = Memory.cacheMemory
        let task = Task { @MainActor in
            try await MLXErrorGuard.guardedRun {
                try await Flux2KleinTextEncoding.embed(prompt: "a lighthouse at dusk", modelDirectory: dir, bits: 8)
            }
        }
        try await Task.sleep(for: .milliseconds(1500))
        task.cancel()
        switch await task.result {
        case .failure(let error): #expect(error is CancellationError, "unexpected error: \(error)")
        case .success: print("PROBE cancel: the run finished before the cancel landed")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        // embed's `defer` empties MLX's buffer cache even when the run is cancelled part-way through the load.
        let cacheAfter = Memory.cacheMemory
        print("PROBE cancel: MLX cache \(KleinFootprint.gb(cacheBefore)) GB before, \(KleinFootprint.gb(cacheAfter)) GB after")
        #expect(cacheAfter <= cacheBefore + 64_000_000, "cache after a cancelled run: \(cacheAfter) bytes (before \(cacheBefore))")
    }
}

/// Owner-run, opt-in: this port's bf16 encoder vs diffusers' bf16 text encoder on the REAL checkpoint, from files made by
/// RSI/notes/claude-code/modelpack/klein-ref/real_encoder_goldens.py (not committed). TEST_RUNNER_MLXUI_TEST_KLEIN_REAL_GOLDEN_DIR names the folder
/// holding real_encoder.safetensors. Without those files the suite skips with the reason; it never passes without them.
enum KleinRealGolden {
    nonisolated static let file: URL? = {
        guard let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_REAL_GOLDEN_DIR"] else { return nil }
        let url = URL(fileURLWithPath: dir).appendingPathComponent("real_encoder.safetensors")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }()
    nonisolated static let enabled = file != nil && KleinRealInstall.enabled
}

@Suite(.enabled(if: KleinRealGolden.enabled, "needs TEST_RUNNER_MLXUI_TEST_KLEIN_REAL_GOLDEN_DIR (folder with real_encoder.safetensors from real_encoder_goldens.py) and the installed Klein"),
       .serialized)
struct Flux2KleinRealShapeGoldenTests {
    /// Per-tap cosine floor for the port's bf16 vs diffusers, over both row blocks. Set after the owner's golden was measured (MP7-3d-FIX-1): the worst tap measured
    /// 0.995853 (cjk pad rows, tap 27); the front rows are pinned much tighter by the variants test below. A pin of measured behaviour, not a quality claim.
    static let provisionalTapCosine = 0.99

    @Test func thePortMatchesDiffusersBf16OnTheRealCheckpoint() throws {
        let dir = try #require(KleinRealInstall.directory)
        let golden = try loadArrays(url: try #require(KleinRealGolden.file))
        let encoder = try Flux2KleinTextEncoder.load(textEncoderDirectory: dir.appendingPathComponent("text_encoder"), bits: nil)
        for name in ["short", "long", "cjk"] {
            let ids = try #require(golden["\(name).ids"]).asType(.int32), mask = try #require(golden["\(name).mask"]).asType(.int32)
            let out = try encoder(ids: ids, attentionMask: mask)
            eval(out)
            for (block, rows) in [("rows0-47", 0 ..< 48), ("pad500-511", 500 ..< 512)] {
                let want = try #require(golden["\(name).\(block == "rows0-47" ? "front" : "pad")"]).reshaped(1, rows.count, 7680)
                let mine = out[0..., rows, 0...]
                let whole = try KleinCosine.of(mine, want)
                var taps: [Double] = []
                for t in 0 ..< 3 { taps.append(try KleinCosine.of(mine[0..., 0..., (t * 2560) ..< ((t + 1) * 2560)], want[0..., 0..., (t * 2560) ..< ((t + 1) * 2560)])) }
                let x = try MLXErrorGuard.read(mine.asType(.float32).reshaped(-1), as: Float.self)
                let y = try MLXErrorGuard.read(want.asType(.float32).reshaped(-1), as: Float.self)
                let rel = Flux2KleinEncoderQuantizationTests.relL2(x, y)
                print(String(format: "REALGOLDEN %@ %@: cos all %.6f taps %.6f %.6f %.6f relL2 %.5f", name, block, whole, taps[0], taps[1], taps[2], rel))
                #expect(taps.allSatisfy { $0 >= Self.provisionalTapCosine }, "\(name) \(block)")
            }
        }
    }

    // MARK: MP7-3d-FIX-1 -- per tap, per row block, four numbers per variant, against the diffusers golden.

    struct Variant { let name: String; let bits: Int?; let keepEmbeddingBF16: Bool }
    /// The golden holds only rows 0-47 (front) and 500-511 (pad), so every number below is over those 60 rows, never all 512.
    struct Measure { var cos = 0.0, rel = 0.0, tapCos: [Double] = [], tapRel: [Double] = [], worstRow = -1, worstRel = 0.0, goldNorm = 0.0, portNorm = 0.0 }

    static let variants = [Variant(name: "1 bf16 (port, bits nil)", bits: nil, keepEmbeddingBF16: false),
                           Variant(name: "2 int8 layers + bf16 embedding", bits: 8, keepEmbeddingBF16: true),
                           Variant(name: "3 int8 everything (shipping)", bits: 8, keepEmbeddingBF16: false)]

    /// rows x 7680 row-major arrays; rowOffset is the first row's real index in the 512-row output (for the worst-row report).
    static func measure(_ x: [Float], _ y: [Float], rows: Int, rowOffset: Int) -> Measure {
        var m = Measure()
        m.cos = KleinCosine.cosine(x, y); m.rel = Flux2KleinEncoderQuantizationTests.relL2(x, y)
        for t in 0 ..< 3 {
            var xt: [Float] = [], yt: [Float] = []
            for r in 0 ..< rows { xt += x[(r * 7680 + t * 2560) ..< (r * 7680 + (t + 1) * 2560)]; yt += y[(r * 7680 + t * 2560) ..< (r * 7680 + (t + 1) * 2560)] }
            m.tapCos.append(KleinCosine.cosine(xt, yt)); m.tapRel.append(Flux2KleinEncoderQuantizationTests.relL2(xt, yt))
        }
        for r in 0 ..< rows {
            let a = Array(x[(r * 7680) ..< ((r + 1) * 7680)]), b = Array(y[(r * 7680) ..< ((r + 1) * 7680)])
            let rel = Flux2KleinEncoderQuantizationTests.relL2(a, b)
            if rel > m.worstRel {
                m.worstRel = rel; m.worstRow = rowOffset + r
                m.goldNorm = b.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
                m.portNorm = a.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
            }
        }
        return m
    }

    /// results[variant][prompt][block]
    nonisolated(unsafe) static var results: [String: [String: [String: Measure]]] = [:]

    @Test func measuresEveryVariantAgainstDiffusersPerTapAndRowBlock() throws {
        let dir = try #require(KleinRealInstall.directory)
        let golden = try loadArrays(url: try #require(KleinRealGolden.file))
        let blocks: [(String, String, Int, Int)] = [("front rows0-47", "front", 48, 0), ("pad rows500-511", "pad", 12, 500)]
        for v in Self.variants {
            let sampler = KleinFootprint()
            Memory.clearCache(); Memory.peakMemory = 0
            sampler.start()
            let t0 = Date()
            var loadSeconds = 0.0
            var perPrompt: [String: [String: Measure]] = [:]
            var encodeSeconds = 0.0
            do {
            let encoder = try Flux2KleinTextEncoder.load(textEncoderDirectory: dir.appendingPathComponent("text_encoder"), bits: v.bits,
                                                                              testOnlyKeepEmbeddingBF16: v.keepEmbeddingBF16)
            loadSeconds = Date().timeIntervalSince(t0)
            for name in ["short", "long", "cjk"] {
                let ids = try #require(golden["\(name).ids"]).asType(.int32), mask = try #require(golden["\(name).mask"]).asType(.int32)
                let t1 = Date()
                let out = try encoder(ids: ids, attentionMask: mask)
                eval(out)
                encodeSeconds += Date().timeIntervalSince(t1)
                var byBlock: [String: Measure] = [:]
                for (label, key, count, offset) in blocks {
                    let want = try #require(golden["\(name).\(key)"]).reshaped(1, count, 7680)
                    let mine = out[0..., offset ..< (offset + count), 0...]
                    let x = try MLXErrorGuard.read(mine.asType(.float32).reshaped(-1), as: Float.self)
                    let y = try MLXErrorGuard.read(want.asType(.float32).reshaped(-1), as: Float.self)
                    byBlock[label] = Self.measure(x, y, rows: count, rowOffset: offset)
                }
                perPrompt[name] = byBlock
            }
            }
            Memory.clearCache()
            let peak = sampler.stop()
            Self.results[v.name] = perPrompt
            print(String(format: "REALGOLDEN-RUN %@: load %.1f s, encode 3 prompts %.1f s, MLX peak %.2f GB, process footprint peak %.2f GB", v.name, loadSeconds, encodeSeconds,
                         KleinFootprint.gb(Memory.peakMemory), KleinFootprint.gb(peak)))
            for name in ["short", "long", "cjk"] {
                for (label, _, _, _) in blocks {
                    guard let m = perPrompt[name]?[label] else { continue }
                    print(String(format: "REALGOLDEN-TABLE | %@ | %@ | %@ | cos all %.6f taps %.6f %.6f %.6f | relL2 all %.5f taps %.5f %.5f %.5f | worst row %d relL2 %.5f goldNorm %.4f portNorm %.4f",
                                 v.name, name, label, m.cos, m.tapCos[0], m.tapCos[1], m.tapCos[2], m.rel, m.tapRel[0], m.tapRel[1], m.tapRel[2], m.worstRow, m.worstRel, m.goldNorm, m.portNorm))
                }
            }
        }
                Self.assertFloors()
    }

    /// FLOORS, set AFTER measuring (MP7-3d-FIX-1; measured values in MP7-3d-FIX-1-report.md): a fixed small margin under the worst value over the three prompts.
    /// These are pins of measured behaviour, NOT quality claims. Implementer's call, pending owner confirmation. Each tuple is
    /// (front cos all/tap, front relL2 all max, front worst-row relL2 max, pad cos all, pad cos tap, pad relL2 all max, pad worst-row relL2 max).
    static let floors: [String: (frontCos: Double, frontRel: Double, frontRow: Double, padCos: Double, padTapCos: Double, padRel: Double, padRow: Double)] = [
        "1 bf16 (port, bits nil)": (0.9999, 0.005, 0.25, 0.99, 0.99, 0.12, 0.25),
        "2 int8 layers + bf16 embedding": (0.9999, 0.015, 0.45, 0.97, 0.97, 0.25, 0.40),
        "3 int8 everything (shipping)": (0.9999, 0.015, 0.60, 0.95, 0.93, 0.33, 0.65),
    ]

    static func assertFloors() {
        for (name, f) in floors {
            for prompt in ["short", "long", "cjk"] {
                if let m = results[name]?[prompt]?["front rows0-47"] {
                    #expect(m.cos >= f.frontCos && m.tapCos.allSatisfy { $0 >= f.frontCos }, "\(name) \(prompt) front cosine")
                    #expect(m.rel <= f.frontRel, "\(name) \(prompt) front relL2")
                    #expect(m.worstRel <= f.frontRow, "\(name) \(prompt) front worst row")
                } else { Issue.record("missing front measure \(name) \(prompt)") }
                if let m = results[name]?[prompt]?["pad rows500-511"] {
                    #expect(m.cos >= f.padCos && m.tapCos.allSatisfy { $0 >= f.padTapCos }, "\(name) \(prompt) pad cosine")
                    #expect(m.rel <= f.padRel, "\(name) \(prompt) pad relL2")
                    #expect(m.worstRel <= f.padRow, "\(name) \(prompt) pad worst row")
                } else { Issue.record("missing pad measure \(name) \(prompt)") }
            }
        }
    }
}
