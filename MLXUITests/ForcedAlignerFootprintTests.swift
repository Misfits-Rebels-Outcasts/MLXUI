import Testing
import Foundation
import MLX
@testable import MLXUI

/// FA-7 for issue #13: the `.align` footprint, measured through the Run path's own shape — `phys_footprint`
/// (`task_info(TASK_VM_INFO)`, what Activity Monitor and jetsam count) sampled every 20 ms, plus MLX's own counters.
/// This test **measures**; the `ModelLoadProfile.entry(for: .align)` row it justifies is booked from the worst case.
///
/// Opt-in twice over: gated like `ForcedAlignerRealLoad` (`MLXUI_TEST_ALIGN=1` + `MLXUI_TEST_ALIGN_DIR`) **and** on
/// `MLXUI_TEST_ALIGN_FOOTPRINT=1` (`TEST_RUNNER_MLXUI_TEST_ALIGN_FOOTPRINT=1` through xcodebuild). One line per phase:
///
/// `FAFOOTPRINT phase=… peakGB=… peakMiB=… baselineMiB=… mlxPeakGB=… wall=…s seconds=… units=…`
///
/// The phases are the three decision 5 allows: load + a 5 s clip, a 60 s clip, and the worst case — a ~300 s clip with
/// its full transcript. Speech is generated with `say` (never a personal recording).
@Suite(.serialized)
struct ForcedAlignerFootprintTests {
    static let env = ProcessInfo.processInfo.environment

    static var enabled: Bool {
        ForcedAlignerRealLoad.enabled
            && (env["MLXUI_TEST_ALIGN_FOOTPRINT"] ?? env["TEST_RUNNER_MLXUI_TEST_ALIGN_FOOTPRINT"]) == "1"
    }

    static var phases: [Int] {
        guard let raw = env["MLXUI_TEST_ALIGN_FOOTPRINT_PHASES"]
                ?? env["TEST_RUNNER_MLXUI_TEST_ALIGN_FOOTPRINT_PHASES"] else { return [1, 2, 3] }
        let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? [1, 2, 3] : parsed
    }

    static let sentence = "The quick brown fox jumps over the lazy dog."

    /// One `say` clip whose transcript is exactly the text it speaks: `repeats` copies of `sentence` at 180 words/min
    /// (about 2.7 s a copy, so 2, 22 and 110 copies are about 5, 60 and 295 seconds).
    nonisolated static func sayClip(repeats: Int) throws -> (url: URL, text: String) {
        let text = Array(repeating: sentence, count: repeats).joined(separator: " ")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("fa-footprint-\(repeats)-\(UUID().uuidString).wav")
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        process.arguments = ["-r", "180", "-o", url.path, "--data-format=LEI16@16000", text]
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0, FileManager.default.fileExists(atPath: url.path) else {
            throw CocoaError(.fileReadUnknown)
        }
        return (url, text)
    }

    @Test(.enabled(if: ForcedAlignerFootprintTests.enabled), arguments: ForcedAlignerFootprintTests.phases)
    func footprint(phase: Int) async throws {
        // The sandboxed test host cannot write the aligner's generated tokenizer.json into /private/tmp, so measure on a
        // writable clone of the pointed-at folder (the same working copy the real-load tests use).
        let copy = try ForcedAlignerRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let (repeats, label): (Int, String)
        switch phase {
        case 1: (repeats, label) = (2, "load+5s")
        case 2: (repeats, label) = (22, "60s")
        default: (repeats, label) = (110, "300s-worst-case")
        }
        let clip = try Self.sayClip(repeats: repeats)
        defer { try? FileManager.default.removeItem(at: clip.url) }
        let buffer = try AudioFileReader.read(clip.url)
        let samples = try AudioResampler.resample(buffer, to: ForcedAlignerEngine.sampleRate).samples
        let seconds = Double(samples.count) / Double(ForcedAlignerEngine.sampleRate)
        try ForcedAlignerEngine.requireWithinLimit(seconds: seconds)

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
        let units = try await ForcedAlignerEngine.align(samples, transcript: clip.text, language: "en",
                                                        modelDirectory: dir)
        let wall = Date().timeIntervalSince(started)
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let mib = 1_048_576.0
        let line = String(format: "FAFOOTPRINT phase=%@ peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f mlxPeakGB=%.3f wall=%.1f s seconds=%.1f units=%d",
                          label, Double(tracker.runPeak) / 1e9, Double(tracker.runPeak) / mib,
                          Double(baseline) / mib, Double(Memory.snapshot().peakMemory) / 1e9,
                          wall, seconds, units.count)
        print(line)
        // What the `.align` booking says, for the one line the PR needs: does it cover the measured peak?
        let files = ModelWeightCharge.safetensorsBytes(in: dir)
        // The engine cache books the weights' charge plus the kind's per-call working set (EngineCache.footprint).
        let booked = ModelLoadProfile.charge(kind: .align, directory: dir, catalogFallback: nil)
            + ModelLoadProfile.workingSetBytes(for: .align)
        let booking = String(format: "FAFOOTPRINT booking rows=align filesGB=%.3f bookedGB=%.3f peakGB=%.3f covered=%@",
                             Double(files) / 1e9, Double(booked) / 1e9, Double(tracker.runPeak) / 1e9,
                             tracker.runPeak <= booked ? "yes" : "NO")
        print(booking)
        // The harness cannot read a test host's stdout: keep the PR's evidence in a file too, one phase per append.
        let path = NSTemporaryDirectory() + "forcedaligner-footprint.txt"
        let previous = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        try? (previous + line + "\n" + booking + "\n").write(toFile: path, atomically: true, encoding: .utf8)

        #expect(!units.isEmpty)
        #expect(units.map(\.text).joined(separator: " ").contains("quick"), "the transcript is timed")
        #expect(tracker.runPeak > baseline)
    }
}
