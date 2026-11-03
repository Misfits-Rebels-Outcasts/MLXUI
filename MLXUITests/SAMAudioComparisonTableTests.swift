import Darwin
import Foundation
import Testing
import MLX
import MLXAudioSTS
@testable import MLXUI

/// MC1-9 (E) — the window-length comparison table, real weights. Skipped unless the R5 opt-in is set
/// (`TEST_RUNNER_MLXUI_TEST_SAMAUDIO=1`, see `SAMAudioRealLoad.optedIn`) **and** `TEST_RUNNER_MLXUI_TEST_SAMAUDIO_TABLE=1`
/// (about 25 minutes, ~12.5 GB: one real-model process at a time, under the smoke lock).
///
/// The clip is the 65 s deterministic voice-over-music clip the windowed test uses. Rows: our window loop at 10 s, 20 s
/// and 30 s windows (3 s overlap), and the library's own `separateLong` at 10 s. Seeds 1, 2, 3. One parseable line per
/// run: `SAMTABLE row=… seed=… iso=… other=… wall=… x=… peakGB=… baselineGB=… seeded=…`.
///
/// **The library row is seeded differently, and that is a limit of the library.** `separateLong` takes no `noise:`;
/// each chunk draws from the global `MLXRandom` state. The closest fair reproducibility is `MLXRandom.seed(seed)` once
/// before the call, in this test only (the app never touches global state). Its noise therefore differs from ours for
/// the same seed number: the rows compare the *method*, not identical noise. Those lines say `seeded=global`.
@Suite(.serialized)
struct SAMAudioComparisonTableTests {
    static let env = ProcessInfo.processInfo.environment
    static var enabled: Bool {
        SAMAudioRealLoad.canRun && (env["MLXUI_TEST_SAMAUDIO_TABLE"] ?? env["TEST_RUNNER_MLXUI_TEST_SAMAUDIO_TABLE"]) == "1"
    }

    /// Rows to run: `MLXUI_TEST_SAMAUDIO_TABLE_ROWS=ours30,ours10` (default: all four). Seeds: `MLXUI_TEST_SAMAUDIO_TABLE_SEEDS=1-10`
    /// or `1,2,3` (default 1,2,3). Both also read with the `TEST_RUNNER_` prefix.
    static func setting(_ name: String) -> String? { env[name] ?? env["TEST_RUNNER_" + name] }
    static var rows: [Row] {
        guard let raw = setting("MLXUI_TEST_SAMAUDIO_TABLE_ROWS") else { return Row.allCases }
        let parsed = raw.split(separator: ",").compactMap { Row(rawValue: $0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? Row.allCases : parsed
    }
    static var seeds: [UInt64] {
        guard let raw = setting("MLXUI_TEST_SAMAUDIO_TABLE_SEEDS") else { return [1, 2, 3] }
        if let dash = raw.firstIndex(of: "-"), let a = UInt64(raw[..<dash]), let b = UInt64(raw[raw.index(after: dash)...]), a <= b {
            return Array(a...b)
        }
        let parsed = raw.split(separator: ",").compactMap { UInt64($0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? [1, 2, 3] : parsed
    }

    /// A window is "bad" when the speech leaks into the other track or the isolated track lost the speech.
    static func isBad(iso: Double, other: Double) -> Bool { other >= 0.3 || iso < 0.5 }

    enum Row: String, CaseIterable, Sendable {
        case ours10, ours20, ours30, library10
        var chunk: Float { switch self { case .ours10, .library10: return 10; case .ours20: return 20; case .ours30: return 30 } }
    }

    /// The library's `separateLong` on the installed folder, with the text encoder and weights loaded the same way the
    /// engine loads them. `@concurrent`, like the engine, so the MLX loop never runs on the main actor.
    @concurrent nonisolated static func librarySeparateLong(_ samples: [Float], seed: UInt64, modelDirectory: URL,
                                                            chunkSeconds: Float, overlapSeconds: Float) async throws -> (isolated: [Float], other: [Float]) {
        let loaded = try SAMAudioEngine.load(modelDirectory: modelDirectory)
        defer { Memory.clearCache() }
        MLXRandom.seed(seed)
        let audios = MLXArray(samples).reshaped([1, 1, samples.count])
        try MLXErrorGuard.evaluate(audios)
        let result = try await loaded.model.separateLong(
            audios: audios, descriptions: ["speech"], chunkSeconds: chunkSeconds, overlapSeconds: overlapSeconds,
            odeDecodeChunkSize: SAMAudioEngine.odeDecodeChunkSize)
        guard let target = result.target.first, let residual = result.residual.first else {
            throw MLXError.caught("the library returned no tracks")
        }
        var isolated = try MLXErrorGuard.read(target, as: Float.self)
        var other = try MLXErrorGuard.read(residual, as: Float.self)
        isolated = Array(isolated.prefix(samples.count)) + [Float](repeating: 0, count: max(samples.count - isolated.count, 0))
        other = Array(other.prefix(samples.count)) + [Float](repeating: 0, count: max(samples.count - other.count, 0))
        return (isolated, other)
    }

    @Test(.enabled(if: SAMAudioComparisonTableTests.enabled))
    func theWindowLengthTable() async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        try await SAMAudioRealLoad.withClip(seconds: 65) { clip in
            let overlap: Float = 3
            for row in Self.rows {
                for seed in Self.seeds {
                    Memory.clearCache()
                    Memory.peakMemory = 0
                    let baseline = FootprintTracker.phys()
                    let tracker = FootprintTracker()
                    tracker.note(baseline)
                    let sampler = Task.detached {
                        while !Task.isCancelled {
                            tracker.note(FootprintTracker.phys())
                            try? await Task.sleep(for: .milliseconds(20))
                        }
                    }
                    let started = Date()
                    let tracks: (isolated: [Float], other: [Float])
                    do {
                        if row == .library10 {
                            tracks = try await Self.librarySeparateLong(clip.mix, seed: seed, modelDirectory: dir,
                                                                       chunkSeconds: row.chunk, overlapSeconds: overlap)
                        } else {
                            let r = try await SAMAudioEngine.separate(clip.mix, description: "speech", modelDirectory: dir,
                                                                      seed: seed, chunkSeconds: row.chunk, overlapSeconds: overlap)
                            tracks = (r.isolated, r.other)
                        }
                    } catch {
                        sampler.cancel()
                        Issue.record("row \(row.rawValue) seed \(seed): \(error)")
                        return
                    }
                    let wall = Date().timeIntervalSince(started)
                    tracker.note(FootprintTracker.phys())
                    sampler.cancel()
                    let iso = SAMAudioRealLoad.correlation(tracks.isolated, clip.speech)
                    let other = SAMAudioRealLoad.correlation(tracks.other, clip.speech)
                    // Per-window scoring (MC1-9b): the FINAL stitched tracks (after the crossfade), scored against the speech
                    // reference over each window's own sample range, with the engine's own window plan and the same
                    // `correlation` as the clip-level numbers.
                    let plan = SAMAudioEngine.chunkPlan(sampleCount: clip.mix.count,
                                                        chunkSamples: Int(Float(48_000) * row.chunk),
                                                        overlapSamples: Int(Float(48_000) * overlap))
                    var bad = 0
                    for (index, range) in plan.windows.enumerated() {
                        let wIso = SAMAudioRealLoad.correlation(Array(tracks.isolated[range]), Array(clip.speech[range]))
                        let wOther = SAMAudioRealLoad.correlation(Array(tracks.other[range]), Array(clip.speech[range]))
                        let isBad = Self.isBad(iso: wIso, other: wOther)
                        if isBad { bad += 1 }
                        print(String(format: "SAMWIN row=%@ seed=%llu win=%d range=%d..%d iso=%.3f other=%.3f bad=%d",
                                     row.rawValue, seed, index, range.lowerBound, range.upperBound, wIso, wOther, isBad ? 1 : 0))
                    }
                    print(String(format: "SAMTABLE row=%@ seed=%llu iso=%.3f other=%.3f wall=%.1f x=%.2f peakGB=%.3f baselineGB=%.3f seeded=%@ windows=%d bad=%d",
                                 row.rawValue, seed, iso, other, wall, 65 / max(wall, 0.001),
                                 Double(tracker.runPeak) / 1e9, Double(baseline) / 1e9,
                                 row == .library10 ? "global" : "key", plan.windows.count, bad))
                    #expect(SAMAudioRealLoad.finite(tracks.isolated) && SAMAudioRealLoad.finite(tracks.other))
                }
            }
        }
    }
}
