import Darwin
import Foundation
import MLX
import Testing
@testable import MLXUI

/// PF-8 — the footprint measurement for OpenAI Privacy Filter. It **measures** the whole Run path; it changes no
/// `ramGB` in `browser.json` and no limit (the measured booking is the `.redact` row in `ModelLoadProfile`).
///
/// Shape: `TranslateGemmaFootprintTests` / `SAMAudioFootprintTests` — `phys_footprint` (`task_info(TASK_VM_INFO)`, what
/// Activity Monitor and jetsam count) sampled every 20 ms through the whole run, plus `Memory.snapshot()`.
///
/// Skipped unless the weights are pointed at (`PrivacyFilterRealLoad.directory`) **and**
/// `MLXUI_TEST_PRIVACY_FOOTPRINT=1` (`TEST_RUNNER_MLXUI_TEST_PRIVACY_FOOTPRINT=1` through xcodebuild).
/// `MLXUI_TEST_PRIVACY_FOOTPRINT_PHASES=1,2` limits the phases; the default is all three.
///
/// One line per phase:
/// `PFFOOTPRINT phase=… peakGB=… peakMiB=… baselineMiB=… mlxPeakGB=… wall=…s tokens=… spans=…`
@Suite(.serialized)
struct PrivacyFilterFootprintTests {
    static let env = ProcessInfo.processInfo.environment

    static var enabled: Bool {
        PrivacyFilterRealLoad.canRun
            && (env["MLXUI_TEST_PRIVACY_FOOTPRINT"] ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY_FOOTPRINT"]) == "1"
    }

    static var phases: [Int] {
        guard let raw = env["MLXUI_TEST_PRIVACY_FOOTPRINT_PHASES"]
                ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY_FOOTPRINT_PHASES"] else { return [1, 2, 3] }
        let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? [1, 2, 3] : parsed
    }

    private func directory() throws -> URL {
        try #require(PrivacyFilterRealLoad.directory)
    }

    /// Synthetic text with about `target` tokens, built from numbered paragraphs (never one passage repeated: the issue
    /// asks for a realistic worst case, and numbering keeps it from collapsing to a trivial pattern). The count is the
    /// model's own tokenizer's, not a word estimate.
    private func textForTokens(_ target: Int, directory: URL) async throws -> (text: String, tokens: Int) {
        let tokenizer = try await PrivacyFilterTokenizer(modelDirectory: directory)
        let passage = "The quarterly report records ordinary business details for the regional office and the forms "
            + "are filed with the usual references."
        func build(_ paragraphs: Int) -> String {
            (0 ..< paragraphs).map { "Memo \($0 + 1). " + passage }.joined(separator: " ")
        }
        // Grow until the target is passed, then trim a prefix to land at or below it (the count is monotonic in length
        // for this text). Two or three tokenizations, not one per paragraph.
        var paragraphs = 1
        var full = build(paragraphs)
        while try tokenizer.tokenize(full).tokenIDs.count < target, paragraphs < 16_384 {
            paragraphs *= 2
            full = build(paragraphs)
        }
        var low = 0, high = full.count, best = 0, bestTokens = 0
        while low <= high {
            let mid = (low + high) / 2
            let count = try tokenizer.tokenize(String(full.prefix(mid))).tokenIDs.count
            if count <= target {
                best = mid; bestTokens = count; low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return (String(full.prefix(best)), bestTokens)
    }

    @Test(.enabled(if: PrivacyFilterFootprintTests.enabled), arguments: PrivacyFilterFootprintTests.phases)
    func footprint(phase: Int) async throws {
        let dir = try directory()
        let text: String
        var label: String
        switch phase {
        case 1:
            label = "load+one-sentence"
            text = PrivacyFilterRealLoad.letterSentence
        case 2:
            let built = try await textForTokens(6_000, directory: dir)
            text = built.text
            label = "6000-token-document(\(built.tokens))"
        default:
            // Decision 5's worst case: the most tokens the model accepts.
            let built = try await textForTokens(PrivacyFilterInference.maxTokens, directory: dir)
            text = built.text
            label = "128000-token-worst-case(\(built.tokens))"
        }

        try PrivacyFilterEngine.validate(text)
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
        let outcome = try await PrivacyFilterEngine.redact(text, modelDirectory: dir, holder: .runWindow)
        let wall = Date().timeIntervalSince(started)
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let mib = 1_048_576.0
        let line = String(format:
            "PFFOOTPRINT phase=%@ peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f mlxPeakGB=%.3f wall=%.1f s tokens=%d spans=%d",
            label, Double(tracker.runPeak) / 1e9, Double(tracker.runPeak) / mib,
            Double(baseline) / mib, Double(Memory.snapshot().peakMemory) / 1e9, wall,
            outcome.tokenCount, outcome.spans.count)
        // What the `.redact` booking says, for the one line the PR needs: does it cover the measured peak?
        let files = ModelLoadProfile.safetensorsBytes(underneath: dir)
        let booked = ModelLoadProfile.charge(kind: .redact, directory: dir, catalogFallback: nil)
            + ModelLoadProfile.workingSetBytes(for: .redact)
        let booking = String(format:
            "PFFOOTPRINT booking row=redact filesGB=%.3f bookedGB=%.3f peakGB=%.3f covered=%@",
            Double(files) / 1e9, Double(booked) / 1e9, Double(tracker.runPeak) / 1e9,
            tracker.runPeak <= booked ? "yes" : "NO")
        print(line)
        print(booking)
        // The harness cannot read a test host's stdout: keep the PR's evidence in a file too, one phase per append.
        let path = NSTemporaryDirectory() + "privacy-filter-footprint.txt"
        let previous = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        try? (previous + line + "\n" + booking + "\n").write(toFile: path, atomically: true, encoding: .utf8)
        #expect(!outcome.cancelled)
        #expect(outcome.tokenCount <= PrivacyFilterInference.maxTokens)
        #expect(tracker.runPeak > baseline)
    }
}
