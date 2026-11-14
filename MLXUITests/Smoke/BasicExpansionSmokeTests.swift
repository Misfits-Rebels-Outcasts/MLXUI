import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import MLX
import Testing
@testable import MLXUI

/// BX-4 (`RSI/DelegateBasicExpansionBacklog.md`): Basic flows 14-18 with the **real** models, end to end through the real
/// interpreter and `RealExecutor`, with peak footprint recorded. The BS-4 pattern (`BasicShowcaseSmokeTests`).
///
/// A flow's test runs only when BOTH hold:
///   1. `MLXUI_BASIC_EXPANSION_SMOKE=1` is in the test process's environment (set it in the Xcode scheme's Test action;
///      from the CLI the variable does not reach the test host, so the CLI always skips), and
///   2. every model the flow's rows resolve to is installed in `ModelStore.shared`, i.e. the Direct edition's store
///      `~/Library/Application Support/AI Browser/` when the host is unsandboxed (run on the `MLXUI-Direct` host or an
///      unsandboxed test host; the sandboxed `MLXUI` host looks in its own empty container and skips). Read-only: only
///      `.installed` markers are looked at here, and the engines read the weights.
/// Models: TranslateGemma 4B (14), Depth Anything V2 Small (15), Qwen3.5 9B (16), Gemma 4 E4B + Kokoro 82M (17),
/// Ministral 3B (18).
///
/// Output lines follow `ModelSmoke` (`SMOKE|...|PASS/FAIL/INFO`, `FOOTPRINT|...`). A skipped test never counts as a pass.
/// Flow 17 also prints `RAW|BX4-17|row2|...` and `RAW|BX4-17|row3|...` with the verbatim model outputs (the owner's hard-stop
/// rule: if the story check fails, the raw text is what gets reported; nothing is stripped or worked around here).
nonisolated enum BasicExpansionRealLoad {
    static var optedIn: Bool { ProcessInfo.processInfo.environment["MLXUI_BASIC_EXPANSION_SMOKE"] == "1" }

    static func ready(_ flowID: String) -> Bool {
        guard optedIn, let doc = try? GalleryLoader.loadDocument(flowID: flowID) else { return false }
        let catalog = BasicShowcaseRealLoad.catalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: BasicShowcaseRealLoad.installedIDs(catalog),
                                       totalRAMGB: 19, claimableModelIDs: Set(catalog.map(\.id)))
        return !result.needs.isEmpty && !result.isBlocked && result.toDownload.isEmpty
    }
}

/// Flow 17's story check, as a pure function so a non-model test can prove it (owner ruling, Boo, 2026-11-14: "Narrow flow 17's
/// thinking check. Reject <|channel, <channel|>, <think, and a line consisting only of thought (case-insensitive) at the start of
/// the output. Don't reject the word thought inside prose. Keep the check on # and *."). Returns the names of every rule broken.
nonisolated enum BasicExpansionStoryCheck {
    static func violations(in story: String) -> [String] {
        var broken: [String] = []
        for marker in ["#", "*", "<|channel", "<channel|>", "<think"] where story.lowercased().contains(marker) {
            broken.append(marker)
        }
        // The first non-empty line being only "thought" is the start of a Gemma 4 thought channel with its tag stripped.
        let firstLine = story.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }.first { !$0.isEmpty }
        if firstLine?.lowercased() == "thought" { broken.append("leading line 'thought'") }
        return broken
    }
}

@Suite(.serialized)
@MainActor
struct BasicExpansionSmokeTests {
    static let row = "BX4"

    // MARK: Harness

    private final class RegistryBox: @unchecked Sendable {
        let registry: ModelRegistry
        @MainActor init() {
            registry = ModelRegistry()
            for module in installedModules { module.register(into: registry) }
        }
    }

    /// Wraps a real stage and records every text it returns, verbatim, in call order.
    private nonisolated struct Recording: PipelineStage {
        let inner: any PipelineStage
        let log: TextLog
        var id: String { inner.id }
        var name: String { inner.name }
        var accepts: MediaKind { inner.accepts }
        var produces: MediaKind { inner.produces }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            let out = try await inner.run(input, progress: progress)
            if case .text(let t) = out { log.add(t) }
            return out
        }
    }

    private final class TextLog: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [String] = []
        func add(_ s: String) { lock.withLock { items.append(s) } }
        var values: [String] { lock.withLock { items } }
    }

    private struct Run {
        let dir: URL
        let events: [FlowInterpreter.PathEvent]
        let texts: [String]
        let peakMiB: Double
        let cleanup: URL
    }

    private func runFlow(_ flowID: String, phase: String) async throws -> Run {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("bx4-\(UUID().uuidString)")
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let catalog = BasicShowcaseRealLoad.catalog()
        let box = RegistryBox()
        let texts = TextLog()
        let makeStage: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            let stage: any PipelineStage = try await MainActor.run {
                guard let resolved = box.registry.bestModule(for: model) else {
                    throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
                }
                return try resolved.sdk.makeStage(for: model, config: config)
            }
            return Recording(inner: stage, log: texts)
        }
        let executor = RealExecutor(workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                    makeModelStage: makeStage,
                                    installedModelIDs: BasicShowcaseRealLoad.installedIDs(catalog), catalog: catalog)
        let doc = try GalleryLoader.loadDocument(flowID: flowID)
        let sampler = ModelSmoke.PeakSampler()
        sampler.note(ModelSmoke.physMiB())
        let watcher = Task.detached {
            while !Task.isCancelled {
                sampler.note(ModelSmoke.physMiB())
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        let events: [FlowInterpreter.PathEvent]
        do {
            events = try await ModelSmoke.measure("\(flowID) \(phase)") {
                try await FlowInterpreter.run(doc, executor: executor)
            }
        } catch {
            watcher.cancel()
            throw error
        }
        watcher.cancel()
        sampler.note(ModelSmoke.physMiB())
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let number = flowID.split(separator: "-").first.map(String.init) ?? flowID
        ModelSmoke.info(Self.row, "\(number).peak-rss",
                        String(format: "%@: peak phys_footprint %.0f MiB; process ru_maxrss %.0f MiB", phase, sampler.peak, Double(usage.ru_maxrss) / 1_048_576))
        let failed = events.filter { $0.kind == .rowFailed }
        ModelSmoke.check(Self.row, "\(flowID).no-row-failed", failed.isEmpty, failed.map { "\($0.path): \($0.message ?? "")" }.joined(separator: "; "))
        return Run(dir: ws.directory(for: flowID), events: events, texts: texts.values, peakMiB: sampler.peak, cleanup: base)
    }

    private func decoded(_ url: URL) throws -> CGImage {
        let src = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        return try #require(CGImageSourceCreateImageAtIndex(src, 0, nil))
    }

    /// RGBA bytes of an image, drawn into a known layout.
    private func rgba(_ image: CGImage) -> [UInt8] {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        bytes.withUnsafeMutableBytes { buf in
            if let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                   space: CGColorSpaceCreateDeviceRGB(),
                                   bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) {
                ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            }
        }
        return bytes
    }

    // MARK: 14 - Translate a Letter

    @Test(.enabled(if: BasicExpansionRealLoad.ready("14-TranslateLetter"), "needs MLXUI_BASIC_EXPANSION_SMOKE=1 and TranslateGemma 4B installed"))
    func flow14_malayLetterIsNonEmptyAndKeepsTheDate() async throws {
        let r = try await runFlow("14-TranslateLetter", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        let text = try String(contentsOf: r.dir.appendingPathComponent("letter-malay.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        ModelSmoke.check(Self.row, "14.non-empty", !text.isEmpty, text)
        ModelSmoke.check(Self.row, "14.not-the-input", text != BasicExpansionInputsTests.letterText.trimmingCharacters(in: .whitespacesAndNewlines), "output equals the English letter")
        ModelSmoke.check(Self.row, "14.has-14-November-or-Sabtu", text.contains("14 November") || text.contains("Sabtu"), text)
        print("RAW|BX4-14|letter-malay|\(text)")
        _ = try? ModelSmoke.artifact("BX4-14", "letter-malay.txt", text: text, judge: "Fluent Malay; keeps 14 November, 9 to 1, Block C, the phone number.")
    }

    // MARK: 15 - See the Depth

    @Test(.enabled(if: BasicExpansionRealLoad.ready("15-DepthMap"), "needs MLXUI_BASIC_EXPANSION_SMOKE=1 and Depth Anything V2 Small installed"))
    func flow15_depthPngMatchesTheInputAspectAndIsNotUniform() async throws {
        let r = try await runFlow("15-DepthMap", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        let out = try decoded(r.dir.appendingPathComponent("photo-depth.png"))
        let input = try decoded(r.dir.appendingPathComponent("photo.jpg"))
        let aspectOut = Double(out.width) / Double(out.height), aspectIn = Double(input.width) / Double(input.height)
        ModelSmoke.check(Self.row, "15.same-aspect", abs(aspectOut - aspectIn) < 0.01, "output \(out.width)x\(out.height), input \(input.width)x\(input.height)")
        let px = rgba(out)
        let n = out.width * out.height
        var sum = 0.0, sumSq = 0.0, colourful = 0
        for i in 0..<n {
            let rr = Double(px[i * 4]), gg = Double(px[i * 4 + 1]), bb = Double(px[i * 4 + 2])
            let lum = (rr + gg + bb) / 3
            sum += lum; sumSq += lum * lum
            if abs(rr - gg) > 8 || abs(gg - bb) > 8 { colourful += 1 }
        }
        let mean = sum / Double(n), std = (sumSq / Double(n) - mean * mean).squareRoot()
        ModelSmoke.check(Self.row, "15.pixel-stddev-above-zero", std > 0, String(format: "stddev %.2f, mean %.1f", std, mean))
        let kind = Double(colourful) / Double(n) > 0.05 ? "colour-mapped" : "grey"
        ModelSmoke.info(Self.row, "15.output-kind", "\(kind) (\(colourful) of \(n) pixels have channels differing by more than 8)")
        _ = try? ModelSmoke.artifact("BX4-15", "photo-depth.png", copying: r.dir.appendingPathComponent("photo-depth.png"), judge: "Near things light, far things dark; the kitten reads as nearer than the background.")
    }

    // MARK: 16 - Ask a Document

    @Test(.enabled(if: BasicExpansionRealLoad.ready("16-AskADocument"), "needs MLXUI_BASIC_EXPANSION_SMOKE=1 and Qwen3.5 9B installed"))
    func flow16_answerSaysFourteenAndProbation() async throws {
        let r = try await runFlow("16-AskADocument", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        let text = try String(contentsOf: r.dir.appendingPathComponent("answer.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = text.lowercased()
        ModelSmoke.check(Self.row, "16.non-empty", !text.isEmpty, text)
        ModelSmoke.check(Self.row, "16.mentions-14", text.contains("14"), text)
        ModelSmoke.check(Self.row, "16.mentions-probation", lower.contains("probation"), text)
        // 18 days is the 5-years-of-service figure; it may appear only when tied to that.
        let eighteenForNewStaff = lower.contains("18") && !(lower.contains("5 year") || lower.contains("five year") || lower.contains("5-year"))
        ModelSmoke.check(Self.row, "16.does-not-give-18-days-to-new-staff", !eighteenForNewStaff, text)
        print("RAW|BX4-16|answer|\(text)")
        _ = try? ModelSmoke.artifact("BX4-16", "answer.txt", text: text, judge: "14 days, usable after the 3-month probation; no sick-leave mix-up.")
    }

    // MARK: 17 - Story From a Photo

    @Test(.enabled(if: BasicExpansionRealLoad.ready("17-StoryFromPhoto"), "needs MLXUI_BASIC_EXPANSION_SMOKE=1 and Gemma 4 E4B + Kokoro 82M installed (about 8 GB at the peak)"))
    func flow17_storyIsCleanSpeakableTextAndTheWavIsTwentyToNinetySeconds() async throws {
        let r = try await runFlow("17-StoryFromPhoto", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        // The Recording wrapper sees every text a model stage returned: row 2 (Describe Image) first, row 3 (Draft) second.
        ModelSmoke.check(Self.row, "17.captured-row-2-and-row-3-text", r.texts.count == 2, "captured \(r.texts.count) text outputs")
        let description = r.texts.first ?? ""
        let story = r.texts.count > 1 ? r.texts[1] : ""
        // Verbatim, for the hard-stop report: nothing here strips or edits the model's output.
        print("RAW|BX4-17|row2|\(description)")
        print("RAW|BX4-17|row3|\(story)")
        _ = try? ModelSmoke.artifact("BX4-17", "row2-description.raw.txt", text: description)
        _ = try? ModelSmoke.artifact("BX4-17", "row3-story.raw.txt", text: story, judge: "A gentle bedtime story, about 120 words, plain text.")
        ModelSmoke.check(Self.row, "17.story-non-empty", !story.isEmpty, story)
        let broken = BasicExpansionStoryCheck.violations(in: story)
        ModelSmoke.check(Self.row, "17.story-is-clean-speakable-text", broken.isEmpty, "broken rules: \(broken); story: \(story)")
        let wav = r.dir.appendingPathComponent("story.wav")
        let file = try AVAudioFile(forReading: wav)
        let seconds = Double(file.length) / file.processingFormat.sampleRate
        ModelSmoke.check(Self.row, "17.wav-20-to-90-s", (20...90).contains(seconds), String(format: "%.1f s", seconds))
        ModelSmoke.check(Self.row, "17.peak-rss-under-12-GB", r.peakMiB < 12 * 1024, String(format: "%.0f MiB", r.peakMiB))
        _ = try? ModelSmoke.artifact("BX4-17", "story.wav", copying: wav, judge: "The story read aloud in a natural voice, no markup spoken.")
    }

    // MARK: 18 - Chart a Spreadsheet

    @Test(.enabled(if: BasicExpansionRealLoad.ready("18-ChartSpreadsheet"), "needs MLXUI_BASIC_EXPANSION_SMOKE=1 and Ministral 3B installed"))
    func flow18_chartIs800By400AndTheSummaryNamesTheBestAndWorstMonth() async throws {
        let r = try await runFlow("18-ChartSpreadsheet", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        let png = try decoded(r.dir.appendingPathComponent("sales-chart.png"))
        ModelSmoke.check(Self.row, "18.png-800x400", png.width == 800 && png.height == 400, "\(png.width)x\(png.height)")
        let text = try String(contentsOf: r.dir.appendingPathComponent("sales-summary.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        ModelSmoke.check(Self.row, "18.mentions-Dec", text.contains("Dec"), text)
        ModelSmoke.check(Self.row, "18.mentions-Feb", text.contains("Feb"), text)
        print("RAW|BX4-18|summary|\(text)")
        _ = try? ModelSmoke.artifact("BX4-18", "sales-chart.png", copying: r.dir.appendingPathComponent("sales-chart.png"), judge: "Twelve bars, rising through the year.")
        _ = try? ModelSmoke.artifact("BX4-18", "sales-summary.txt", text: text, judge: "Two sentences: upward trend, December best, February worst.")
    }
}
