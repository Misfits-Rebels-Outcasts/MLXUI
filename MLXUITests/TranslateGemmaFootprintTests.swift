import Darwin
import Foundation
import MLX
import Testing
@testable import MLXUI

/// MC2-7 — the footprint measurement for TranslateGemma 4B. It **measures**; it changes no `ramGB`, limit or booking
/// (`ramGB` 4.1 and the `.translate` booking come from `TranslateGemmaSmokeTests`' worst-input phase, MC2-11, R2). Shape: `SAMAudioFootprintTests` /
/// `LFMAudioFootprintTests` — `phys_footprint` (`task_info(TASK_VM_INFO)`, what Activity Monitor and jetsam count)
/// sampled every 20 ms through the whole run, plus `Memory.snapshot()`.
///
/// Skipped unless the weights are pointed at (see `TranslateGemmaRealLoad.directory`) **and**
/// `MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT=1` (`TEST_RUNNER_MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT=1` through xcodebuild).
/// `MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT_PHASES=1,2` limits the phases; the default is all three.
///
/// One line per phase:
/// `TGEMFOOTPRINT phase=… peakGB=… baselineMiB=… mlxPeakGB=… wall=…s promptTokens=… generatedTokens=…`
@Suite(.serialized)
struct TranslateGemmaFootprintTests {
    static let env = ProcessInfo.processInfo.environment

    static var enabled: Bool {
        TranslateGemmaRealLoad.canRun
            && (env["MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT"] ?? env["TEST_RUNNER_MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT"]) == "1"
    }

    static var phases: [Int] {
        guard let raw = env["MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT_PHASES"]
                ?? env["TEST_RUNNER_MLXUI_TEST_TRANSLATEGEMMA_FOOTPRINT_PHASES"] else { return [1, 2, 3] }
        let parsed = raw.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        return parsed.isEmpty ? [1, 2, 3] : parsed
    }

    private func directory() throws -> URL {
        try #require(TranslateGemmaRealLoad.directory)
    }

    /// English text whose **rendered** request is as close as possible to `target` tokens, using the model's own
    /// tokenizer. The rendered count (the 2K limit counts the template's own instruction text) is what is hit, not a
    /// word estimate. Paragraphs are numbered so the text is never the same passage repeated — a repeated passage can
    /// make the model stop early, and this phase measures the *worst* case (prompt at the limit, output to the limit).
    private func textForRenderedTokens(_ target: Int, targetCode: String, directory: URL) async throws -> (text: String, tokens: Int) {
        let passage = """
        The river runs past the old mill and turns north towards the sea. Every morning the fishermen mend their nets         on the stones and talk about the weather, the tides and the long winter ahead. You can judge a good landing.
        """
        func build(_ paragraphs: Int) -> String {
            (0 ..< paragraphs).map { "Paragraph \($0 + 1). " + passage }.joined(separator: " ")
        }
        func count(_ text: String) async throws -> Int {
            try await TranslateGemmaTokenCounter.shared.tokenCount(
                for: .init(text: text, sourceCode: "en", targetCode: targetCode), modelDirectory: directory)
        }
        var paragraphs = 1
        var full = build(paragraphs)
        while try await count(full) < target, paragraphs < 400 {
            paragraphs += 1
            full = build(paragraphs)
        }
        // A character-prefix binary search lands within a few tokens of the limit (the count is monotonic in length
        // for this text), so the prompt is a true worst case rather than "one paragraph short".
        var low = 0, high = full.count, best = 0, bestTokens = 0
        while low <= high {
            let mid = (low + high) / 2
            let tokens = try await count(String(full.prefix(mid)))
            if tokens <= target {
                best = mid; bestTokens = tokens; low = mid + 1
            } else {
                high = mid - 1
            }
        }
        return (String(full.prefix(best)), bestTokens)
    }

    @Test(.enabled(if: TranslateGemmaFootprintTests.enabled), arguments: TranslateGemmaFootprintTests.phases)
    func footprint(phase: Int) async throws {
        let dir = try directory()
        let request: TranslateGemmaEngine.Request
        var label: String
        switch phase {
        case 1:
            label = "load+one-sentence"
            request = .init(text: TranslateGemmaRealLoad.englishSentence, sourceCode: "en", targetCode: "ms")
        case 2:
            label = "1000-token-en-zh-Hans"
            let (text, tokens) = try await textForRenderedTokens(1_000, targetCode: "zh-Hans", directory: dir)
            label = "1000-token-en-zh-Hans(\(tokens))"
            request = .init(text: text, sourceCode: "en", targetCode: "zh-Hans")
        default:
            label = "max-prompt-and-output"
            let (text, tokens) = try await textForRenderedTokens(2_048, targetCode: "zh-Hans", directory: dir)
            label = "max-prompt(\(tokens))-and-output"
            request = .init(text: text, sourceCode: "en", targetCode: "zh-Hans")
        }
        try TranslateGemmaEngine.validate(request)

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
        let outcome = try await TranslateGemmaEngine.translate(request, modelDirectory: dir,
                                                              maxOutputTokens: TranslateGemmaLanguages.maxOutputTokens)
        let wall = Date().timeIntervalSince(started)
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let mib = 1_048_576.0
        let preview = outcome.text.replacingOccurrences(of: "\n", with: " ").prefix(300)
        let line = String(format: "TGEMFOOTPRINT phase=%@ preview=%@ peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f mlxPeakGB=%.3f wall=%.1f s promptTokens=%d generatedTokens=%d cancelled=%@",
                          label, String(preview), Double(tracker.runPeak) / 1e9, Double(tracker.runPeak) / mib,
                          Double(baseline) / mib, Double(Memory.snapshot().peakMemory) / 1e9, wall,
                          outcome.promptTokens, outcome.generatedTokens, outcome.cancelled ? "yes" : "no")
        // What the `.llm` booking (files x 1.10) says, for the one line the PR needs: does it cover the measured peak?
        let files = ModelWeightCharge.safetensorsBytes(in: dir)
        let booked = ModelLoadProfile.charge(kind: .translate, directory: dir, catalogFallback: nil)
        let booking = String(format: "TGEMFOOTPRINT booking rows=llm filesGB=%.3f bookedGB=%.3f peakGB=%.3f covered=%@",
                             Double(files) / 1e9, Double(booked) / 1e9, Double(tracker.runPeak) / 1e9,
                             tracker.runPeak <= booked ? "yes" : "NO")
        print(line)
        print(booking)
        // The harness cannot read a test host's stdout: keep the PR's evidence in a file too, one phase per append.
        let path = NSTemporaryDirectory() + "translategemma-footprint.txt"
        let previous = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        try? (previous + line + "\n" + booking + "\n").write(toFile: path, atomically: true, encoding: .utf8)
        #expect(!outcome.cancelled)
        #expect(outcome.promptTokens <= TranslateGemmaLanguages.maxPromptTokens)
        #expect(tracker.runPeak > baseline)
    }
}
