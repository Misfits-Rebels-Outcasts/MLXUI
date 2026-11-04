import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// BS-3 (`RSI/DelegateBasicShowcaseBacklog.md`): each of Basic flows 9-13 runs end to end through the
/// real interpreter, the real `RealExecutor` and the real file tools (Read Text / Read Image / Read PDF /
/// Table to Text / Save *), with the *model* stages replaced by deterministic stubs (no weights load).
/// So the saved files are real files in a real flow folder.
struct BasicShowcaseRunTests {

    // MARK: Stubs

    private nonisolated struct Stub: PipelineStage {
        let id = "bs3.stub"
        let name = "BS3 stub"
        let accepts: MediaKind
        let produces: MediaKind
        let body: @Sendable (Media) async throws -> Media
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            try require(input, accepts)
            progress(1.0)
            return try await body(input)
        }
    }

    private final class Log<T>: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [T] = []
        func add(_ x: T) { lock.withLock { items.append(x) } }
        var values: [T] { lock.withLock { items } }
    }

    private static func solidImage(_ side: Int) -> CGImage {
        let ctx = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.8, green: 0.2, blue: 0.2, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: side, height: side))
        return ctx.makeImage()!
    }

    private static func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private struct Harness {
        let ws: FlowWorkspace
        let flowID: String
        let dir: URL
        let doc: FlowDocument
        let executor: RealExecutor
        let cleanup: URL
    }

    /// Prepare the flow folder from the built bundle, as a Run does, then build the executor.
    @MainActor private func harness(_ flowID: String, stages: @escaping @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage,
                                    layaTag: String? = nil) throws -> Harness {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("bs3-\(UUID().uuidString)")
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let catalog = try Self.catalog()
        var real = RealExecutor(workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                makeModelStage: stages, installedModelIDs: Set(catalog.map(\.id)), catalog: catalog)
        if let layaTag {
            real.askLaya = { _, _, _ in
                var answer = LayaAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1],
                                        optionLabels: [], stateTruncated: false)
                answer.choiceLabel = layaTag
                return [answer]
            }
        }
        return Harness(ws: ws, flowID: flowID, dir: ws.directory(for: flowID),
                       doc: try GalleryLoader.loadDocument(flowID: flowID), executor: real, cleanup: base)
    }

    private func run(_ h: Harness) async throws -> [FlowInterpreter.PathEvent] {
        let events = try await FlowInterpreter.run(h.doc, executor: h.executor)
        let failed = events.filter { $0.kind == .rowFailed }
        #expect(failed.isEmpty, "rows failed: \(failed.map { "\($0.path): \($0.message ?? "")" })")
        return events
    }

    // MARK: 9

    @Test @MainActor func flow9SavesNoteWavFromTheNoteText() async throws {
        let prompts = Log<String>()
        let h = try harness("9-ReadItAloud") { _, config in
            Stub(accepts: .text, produces: .audio) { input in
                if case .text(let t) = input { prompts.add(t) }
                return .audio(AudioBuffer(samples: [Float](repeating: 0.1, count: 2400), sampleRate: 24000))
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        _ = try await run(h)
        #expect(prompts.values == [BasicShowcaseInputsTests.noteText.trimmingCharacters(in: .newlines)]
                || prompts.values == [BasicShowcaseInputsTests.noteText])
        let wav = try Data(contentsOf: h.dir.appendingPathComponent("note.wav"))
        #expect(wav.prefix(4) == Data("RIFF".utf8))
    }

    // MARK: 10

    @Test @MainActor func flow10SavesPhotoDescriptionTxt() async throws {
        let h = try harness("10-DescribePhoto") { _, config in
            Stub(accepts: .image, produces: .text) { _ in .text("Two kittens sit side by side. They are indoors.") }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        _ = try await run(h)
        let text = try String(contentsOf: h.dir.appendingPathComponent("photo-description.txt"), encoding: .utf8)
        #expect(text.contains("Two kittens"))
    }

    // MARK: 11

    @Test @MainActor func flow11SavesPictureAndHumanInputDefaultPassesTheTemplateThrough() async throws {
        let prompts = Log<String>()
        let seeds = Log<UInt64?>()
        let h = try harness("11-PictureFromWords") { _, config in
            seeds.add(config.seed)
            return Stub(accepts: .text, produces: .image) { input in
                if case .text(let t) = input { prompts.add(t) }
                return .image(ImageMedia(cgImage: Self.solidImage(16)))
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        let events = try await run(h)
        let png = try Data(contentsOf: h.dir.appendingPathComponent("picture.png"))
        #expect(png.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
        #expect(prompts.values == ["a red bicycle leaning on a blue door, morning light, photograph"])
        #expect(seeds.values == [7])
        #expect(events.contains { $0.kind == .flagRaised && $0.code == "F002" }, "the timeout default is disclosed")
    }

    // MARK: 12

    /// The stubbed model answers the yes/no gate once, then returns each field's value.
    private static func invoiceStages() -> @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage {
        let gates = Log<Void>()
        let values: [String: String] = ["vendor": "Bluebird Print Studio", "invoice_number": "BPS-2026-0417",
                                        "date": "2026-10-06", "due_date": "2026-11-05",
                                        "total": "370.60", "currency": "SGD"]
        return { _, _ in
            Stub(accepts: .text, produces: .text) { input in
                guard case .text(let prompt) = input else { return .text("") }
                if prompt.contains("Answer yes or no") {
                    gates.add(())
                    return .text(gates.values.count == 1 ? "yes" : "no")
                }
                for (field, value) in values where prompt.hasSuffix("\n\(field):") { return .text(value) }
                return .text("")
            }
        }
    }

    @Test @MainActor func flow12SavesInvoiceCsvWithTheExtractedRow() async throws {
        let h = try harness("12-InvoiceToSpreadsheet", stages: Self.invoiceStages())
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        _ = try await run(h)
        let csv = try String(contentsOf: h.dir.appendingPathComponent("invoice.csv"), encoding: .utf8)
        // RECORDED (BS-3): the real Table to Text path writes RFC 4180 CRLF line endings. A Swift
        // `"\r\n"` is ONE Character, so splitting on "\n" finds no break (parseCSV CRLF note); split on
        // `.newlines` instead. The header it produces is exactly the schema, comma-joined, unquoted.
        #expect(csv.contains("\r\n"))
        let lines = csv.components(separatedBy: .newlines).filter { !$0.isEmpty }
        #expect(lines.first == "vendor,invoice_number,date,due_date,total,currency", "header was: \(lines.first ?? "nil")")
        #expect(lines.count == 2, "csv was: \(csv)")
        #expect(lines.last == "Bluebird Print Studio,BPS-2026-0417,2026-10-06,2026-11-05,370.60,SGD", "row was: \(lines.last ?? "nil")")
    }

    // MARK: 13

    @Test @MainActor func flow13RoutesToTodayOnly() async throws {
        try await routed(tag: "today", present: "reply-today.txt", absent: "reply-later.txt")
    }

    @Test @MainActor func flow13RoutesToLaterOnly() async throws {
        try await routed(tag: "later", present: "reply-later.txt", absent: "reply-today.txt")
    }

    private func routed(tag: String, present: String, absent: String) async throws {
        let h = try await MainActor.run {
            try harness("13-SortAnEmail", stages: { _, _ in throw StageError.unsupportedModel(id: "none", kind: .llm) }, layaTag: tag)
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        _ = try await run(h)
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: h.dir.appendingPathComponent(present).path))
        #expect(!fm.fileExists(atPath: h.dir.appendingPathComponent(absent).path), "\(absent) must not be written")
        let saved = try String(contentsOf: h.dir.appendingPathComponent(present), encoding: .utf8)
        #expect(saved == BasicShowcaseInputsTests.emailText, "the email lands unchanged in \(present) (Spec 7.4 R1)")
    }
}
