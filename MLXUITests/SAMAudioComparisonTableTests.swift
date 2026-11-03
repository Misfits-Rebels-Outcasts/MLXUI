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
            for row in Row.allCases {
                for seed in [UInt64(1), 2, 3] {
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
                    print(String(format: "SAMTABLE row=%@ seed=%llu iso=%.3f other=%.3f wall=%.1f x=%.2f peakGB=%.3f baselineGB=%.3f seeded=%@",
                                 row.rawValue, seed, iso, other, wall, 65 / max(wall, 0.001),
                                 Double(tracker.runPeak) / 1e9, Double(baseline) / 1e9,
                                 row == .library10 ? "global" : "key"))
                    #expect(SAMAudioRealLoad.finite(tracks.isolated) && SAMAudioRealLoad.finite(tracks.other))
                }
            }
        }
    }
}
