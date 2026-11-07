import Foundation
import AVFoundation
import ImageIO
import MLX
import Testing
@testable import MLXUI

/// BS-4 (`RSI/DelegateBasicShowcaseBacklog.md`): Basic flows 9-13 with the **real** models, end to end through the real
/// interpreter and `RealExecutor`, with peak footprint recorded.
///
/// **These tests have NOT been run with real weights by their author** (the implementer had no models and was told to
/// download nothing). They compile and they skip cleanly.
///
/// A flow's test runs only when BOTH hold:
///   1. `MLXUI_BASIC_SHOWCASE_SMOKE=1` is in the test process's environment (set it in the Xcode scheme's Test action;
///      from the CLI the variable does not reach the test host, so the CLI always skips), and
///   2. every model the flow's rows resolve to is installed in `ModelStore.shared`, i.e. the Direct edition's store
///      `~/Library/Application Support/AI Browser/` when the host is unsandboxed (run on the `MLXUI-Direct` host or an
///      unsandboxed test host; the sandboxed `MLXUI` host looks in its own empty container and skips).
/// Models: Kokoro 82M (flow 9), LFM2-VL 1.6B (10), FLUX.2 Klein 4B (11), Qwen3 8B (12), Laya 0.4B (13).
///
/// Output lines follow `ModelSmoke` (`SMOKE|…|PASS/FAIL/INFO`, `FOOTPRINT|…`). A skipped test never counts as a pass.
nonisolated enum BasicShowcaseRealLoad {
    static var optedIn: Bool { ProcessInfo.processInfo.environment["MLXUI_BASIC_SHOWCASE_SMOKE"] == "1" }

    static func catalog() -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard let data = try? Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json")),
              let decoded = try? JSONDecoder().decode(BrowserData.self, from: data) else { return [] }
        return decoded.domains.flatMap { $0.allModels }
    }

    /// Catalog ids with the `.installed` marker in `ModelStore.shared`.
    static func installedIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let store = ModelStore.shared
        return Set(catalog.filter { FileManager.default.fileExists(atPath: store.installedMarker(forModelID: $0.id).path) }.map(\.id))
    }

    /// Opted in, and the flow has model rows, all resolved and installed (never reads anything but marker files).
    static func ready(_ flowID: String) -> Bool {
        guard optedIn, let doc = try? GalleryLoader.loadDocument(flowID: flowID) else { return false }
        let catalog = catalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: installedIDs(catalog),
                                       totalRAMGB: 19, claimableModelIDs: Set(catalog.map(\.id)))
        return !result.needs.isEmpty && !result.isBlocked && result.toDownload.isEmpty
    }
}

@Suite(.serialized)
@MainActor
struct BasicShowcaseSmokeTests {
    static let row = "BS4"

    // MARK: Harness

    private final class RegistryBox: @unchecked Sendable {
        let registry: ModelRegistry
        @MainActor init() {
            registry = ModelRegistry()
            for module in installedModules { module.register(into: registry) }
        }
    }

    private struct Run {
        let dir: URL
        let events: [FlowInterpreter.PathEvent]
        let cleanup: URL
    }

    /// One real run in a fresh flow folder (no cache, so a repeat run is a genuine second generation).
    private func runFlow(_ flowID: String, phase: String) async throws -> Run {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("bs4-\(UUID().uuidString)")
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let catalog = BasicShowcaseRealLoad.catalog()
        let box = RegistryBox()
        let makeStage: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            try await MainActor.run {
                guard let resolved = box.registry.bestModule(for: model) else {
                    throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind)
                }
                return try resolved.sdk.makeStage(for: model, config: config)
            }
        }
        let executor = RealExecutor(workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                    makeModelStage: makeStage,
                                    installedModelIDs: BasicShowcaseRealLoad.installedIDs(catalog), catalog: catalog)
        let doc = try GalleryLoader.loadDocument(flowID: flowID)
        // Peak physical footprint of this process over the run (what Activity Monitor shows), plus the process lifetime ru_maxrss;
        // emitted per run as `SMOKE|BS4|<n>.peak-rss|INFO|…` (the FOOTPRINT line from `measure` is kept too).
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
        return Run(dir: ws.directory(for: flowID), events: events, cleanup: base)
    }

    // MARK: 9 - Read It Aloud

    @Test(.enabled(if: BasicShowcaseRealLoad.ready("9-ReadItAloud"), "needs MLXUI_BASIC_SHOWCASE_SMOKE=1 and Kokoro 82M installed"))
    func flow9_wavLengthAndSeedRepeatability() async throws {
        let a = try await runFlow("9-ReadItAloud", phase: "run 1"); defer { try? FileManager.default.removeItem(at: a.cleanup) }
        let b = try await runFlow("9-ReadItAloud", phase: "run 2"); defer { try? FileManager.default.removeItem(at: b.cleanup) }
        let urlA = a.dir.appendingPathComponent("note.wav")
        let file = try AVAudioFile(forReading: urlA)
        let seconds = Double(file.length) / file.processingFormat.sampleRate
        ModelSmoke.check(Self.row, "9.wav-20-to-60-s", (20...60).contains(seconds), String(format: "%.1f s", seconds))
        // BS-6 (R6): `Speak` now honours `seed=7`, so two runs must be byte-identical (Kokoro draws no random numbers today,
        // so this also fails if anything non-deterministic creeps into the engine).
        let same = try Data(contentsOf: urlA) == Data(contentsOf: b.dir.appendingPathComponent("note.wav"))
        ModelSmoke.check(Self.row, "9.two-seed7-runs-byte-identical", same, same ? "identical" : "the two note.wav files differ")
        try ModelSmoke.artifact("BS4-9", "note.wav", copying: urlA, judge: "Natural voice reading the reminder; no clipping or gaps.")
    }

    // MARK: 10 - Describe a Photo

    @Test(.enabled(if: BasicShowcaseRealLoad.ready("10-DescribePhoto"), "needs MLXUI_BASIC_SHOWCASE_SMOKE=1 and LFM2-VL 1.6B installed"))
    func flow10_descriptionIsShortAndMentionsTheKitten() async throws {
        let r = try await runFlow("10-DescribePhoto", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        let text = try String(contentsOf: r.dir.appendingPathComponent("photo-description.txt"), encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        ModelSmoke.check(Self.row, "10.non-empty", !text.isEmpty, text)
        // R13 (2026-11-04): the row no longer promises two sentences; it must stay short (at most 120 words).
        let wordCount = text.split(whereSeparator: { $0.isWhitespace }).count
        ModelSmoke.check(Self.row, "10.at-most-120-words", wordCount <= 120, "\(wordCount) words: \(text)")
        let lower = text.lowercased()
        ModelSmoke.check(Self.row, "10.mentions-cat-or-kitten", lower.contains("cat") || lower.contains("kitten"), text)
        try ModelSmoke.artifact("BS4-10", "photo-description.txt", text: text, judge: "Describes the kitten photo accurately in plain words.")
    }

    // MARK: 11 - Picture From Words

    @Test(.enabled(if: BasicShowcaseRealLoad.ready("11-PictureFromWords"), "needs MLXUI_BASIC_SHOWCASE_SMOKE=1 and FLUX.2 Klein 4B installed (about 6 GB at its peak)"))
    func flow11_pngIs1024SquareAndTwoSeedSevenRunsMatch() async throws {
        let a = try await runFlow("11-PictureFromWords", phase: "run 1"); defer { try? FileManager.default.removeItem(at: a.cleanup) }
        let b = try await runFlow("11-PictureFromWords", phase: "run 2"); defer { try? FileManager.default.removeItem(at: b.cleanup) }
        let urlA = a.dir.appendingPathComponent("picture.png")
        let src = try #require(CGImageSourceCreateWithURL(urlA as CFURL, nil))
        let img = try #require(CGImageSourceCreateImageAtIndex(src, 0, nil))
        ModelSmoke.check(Self.row, "11.png-1024x1024", img.width == 1024 && img.height == 1024, "\(img.width)x\(img.height)")
        let dataA = try Data(contentsOf: urlA), dataB = try Data(contentsOf: b.dir.appendingPathComponent("picture.png"))
        // Backlog: byte-identical, "or record why not". A difference is reported, not failed.
        ModelSmoke.info(Self.row, "11.two-seed7-runs-byte-identical", dataA == dataB ? "identical" : "different: \(dataA.count) vs \(dataB.count) bytes (record why before filing)")
        try ModelSmoke.artifact("BS4-11", "picture.png", copying: urlA, judge: "A red bicycle leaning on a blue door in morning light, photographic.")
    }

    // MARK: 12 - Invoice to Spreadsheet

    @Test(.enabled(if: BasicShowcaseRealLoad.ready("12-InvoiceToSpreadsheet"), "needs MLXUI_BASIC_SHOWCASE_SMOKE=1 and Qwen3 8B installed"))
    func flow12_csvCarriesTheInvoiceValues() async throws {
        let r = try await runFlow("12-InvoiceToSpreadsheet", phase: "run"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
        let csv = try String(contentsOf: r.dir.appendingPathComponent("invoice.csv"), encoding: .utf8)
        // CRLF: a Swift "\r\n" is one Character, so split on .newlines (BS-3 finding).
        let lines = csv.components(separatedBy: .newlines).filter { !$0.isEmpty }
        ModelSmoke.check(Self.row, "12.header", lines.first == "vendor,invoice_number,date,due_date,total,currency", lines.first ?? "")
        ModelSmoke.check(Self.row, "12.one-data-row", lines.count == 2, "\(lines.count) lines")
        let row = lines.dropFirst().first ?? ""
        for needle in ["BPS-2026-0417", "2026-11-05", "370.60", "SGD"] {
            ModelSmoke.check(Self.row, "12.row-has-\(needle)", row.contains(needle), row)
        }
        try ModelSmoke.artifact("BS4-12", "invoice.csv", text: csv, judge: "Opens in Numbers/Excel as one row under the six headings.")
    }

    // MARK: 13 - Sort an Email

    @Test(.enabled(if: BasicShowcaseRealLoad.ready("13-SortAnEmail"), "needs MLXUI_BASIC_SHOWCASE_SMOKE=1 and Laya 0.4B installed"))
    func flow13_routesToTodayThreeTimes() async throws {
        var routes: [String] = []
        for i in 1...3 {
            let r = try await runFlow("13-SortAnEmail", phase: "run \(i)"); defer { try? FileManager.default.removeItem(at: r.cleanup) }
            let today = FileManager.default.fileExists(atPath: r.dir.appendingPathComponent("reply-today.txt").path)
            let later = FileManager.default.fileExists(atPath: r.dir.appendingPathComponent("reply-later.txt").path)
            ModelSmoke.check(Self.row, "13.run\(i).exactly-one-file", today != later, "today=\(today) later=\(later)")
            routes.append(today ? "today" : later ? "later" : "none")
        }
        let flips = routes.filter { $0 != "today" }.count
        ModelSmoke.info(Self.row, "13.routes", routes.joined(separator: ","))
        ModelSmoke.check(Self.row, "13.routes-to-today-every-time", flips == 0, "flips: \(flips) of 3 (\(routes.joined(separator: ",")))")
    }
}
