import Darwin
import Foundation
import MLX
import Testing
@testable import MLXUI

/// MC1-6 — the footprint measurement for SAM-Audio Base. It **measures**; it changes no `ramGB`, limit or budget (the
/// maintainers set `ramGB` from their own smoke run). Shape: `LFMAudioFootprintTests` / the DeepFilterNet harness —
/// `phys_footprint` (`task_info(TASK_VM_INFO)`, what Activity Monitor and jetsam count) through the shared
/// `FootprintTracker`, sampled every 20 ms, plus `Memory.snapshot()`.
///
/// Skipped unless the weights are installed (see `SAMAudioRealLoad.directory`) **and**
/// `MLXUI_TEST_SAMAUDIO_FOOTPRINT=1` (`TEST_RUNNER_MLXUI_TEST_SAMAUDIO_FOOTPRINT=1` on the xcodebuild line).
/// `MLXUI_TEST_SAMAUDIO_FOOTPRINT_PHASES=10,60` limits the phases; the default is all three (10 s, 60 s, 30 minutes).
///
/// One line per phase: `SAMFOOTPRINT seconds=… peakGB=… baselineMiB=… mlxPeakGB=… wall=… s speed=…x tracksMiB=…`.
@Suite(.serialized)
struct SAMAudioFootprintTests {
    static let env = ProcessInfo.processInfo.environment

    static var enabled: Bool {
        SAMAudioRealLoad.canRun && (env["MLXUI_TEST_SAMAUDIO_FOOTPRINT"] ?? env["TEST_RUNNER_MLXUI_TEST_SAMAUDIO_FOOTPRINT"]) == "1"
    }

    static var phases: [Double] {
        guard let raw = env["MLXUI_TEST_SAMAUDIO_FOOTPRINT_PHASES"] ?? env["TEST_RUNNER_MLXUI_TEST_SAMAUDIO_FOOTPRINT_PHASES"] else { return [10, 60, 1_800] }
        let parsed = raw.split(separator: ",").compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? [10, 60, 1_800] : parsed
    }

    /// One phase: load the model, separate a clip of `seconds`, report the whole-process peak.
    @Test(.enabled(if: SAMAudioFootprintTests.enabled), arguments: SAMAudioFootprintTests.phases)
    func footprint(seconds: Double) async throws {
        let dir = try #require(SAMAudioRealLoad.directory)
        let mix = try SAMAudioRealLoad.voiceOverMusicMix(seconds: seconds)

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
        let result = try await SAMAudioEngine.separate(mix, description: "speech", modelDirectory: dir)
        let wall = Date().timeIntervalSince(started)
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let mib = 1_048_576.0
        let tracksMiB = Double(result.isolated.count + result.other.count) * 4 / mib
        print(String(format: "SAMFOOTPRINT seconds=%.0f peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f mlxPeakGB=%.3f wall=%.1f s speed=%.2fx tracksMiB=%.0f samples=%d",
                     seconds, Double(tracker.runPeak) / 1e9, Double(tracker.runPeak) / mib, Double(baseline) / mib,
                     Double(Memory.snapshot().peakMemory) / 1e9, wall, seconds / max(wall, 0.001), tracksMiB,
                     result.isolated.count))
        #expect(result.sampleRate == 48_000)
        #expect(abs(Double(result.isolated.count) - seconds * 48_000) < 48_000,
                "the track keeps the recording's length (\(result.isolated.count))")
        #expect(result.isolated.contains { $0 != 0 })
    }
}
