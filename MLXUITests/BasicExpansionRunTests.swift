import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// BX-3 (`RSI/DelegateBasicExpansionBacklog.md`): each of Basic flows 14-18 runs end to end through the
/// real interpreter, the real `RealExecutor` and the real file tools (Read Text / Read Image / Read PDF /
/// Template / Read CSV / Chart / Table to Text / Save *), with the *model* stages replaced by deterministic
/// stubs (no weights load), the BS-3 pattern. So the saved files are real files in a real flow folder.
struct BasicExpansionRunTests {

    // MARK: Stubs

    private nonisolated struct Stub: PipelineStage {
        let id = "bx3.stub"
        let name = "BX3 stub"
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
        ctx.setFillColor(CGColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1))
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
        let dir: URL
        let doc: FlowDocument
        let executor: RealExecutor
        let cleanup: URL
    }

    @MainActor private func harness(_ flowID: String,
                                    stages: @escaping @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage) throws -> Harness {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("bx3-\(UUID().uuidString)")
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let catalog = try Self.catalog()
        let real = RealExecutor(workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                makeModelStage: stages, installedModelIDs: Set(catalog.map(\.id)), catalog: catalog)
        return Harness(ws: ws, dir: ws.directory(for: flowID),
                       doc: try GalleryLoader.loadDocument(flowID: flowID), executor: real, cleanup: base)
    }

    private func run(_ h: Harness) async throws {
        let events = try await FlowInterpreter.run(h.doc, executor: h.executor)
        let failed = events.filter { $0.kind == .rowFailed }
        #expect(failed.isEmpty, "rows failed: \(failed.map { "\($0.path): \($0.message ?? "")" })")
    }

    private func exists(_ h: Harness, _ name: String) -> Bool {
        FileManager.default.fileExists(atPath: h.dir.appendingPathComponent(name).path)
    }

    // MARK: 14

    @Test @MainActor func flow14SavesLetterGermanTxtFromTheLetterWithTargetDe() async throws {
        let seen = Log<String>()
        let configs = Log<String>()
        let h = try harness("14-TranslateLetter") { entry, config in
            #expect(entry.runnerKind == .translate)
            configs.add("\(config.sourceLanguage ?? "nil")>\(config.targetLanguage ?? "nil")")
            return Stub(accepts: .text, produces: .text) { input in
                if case .text(let t) = input { seen.add(t) }
                return .text("Liebe Bewohner, am Samstag, dem 14. November, gibt es in Block C kein Wasser.")
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        try await run(h)
        #expect(seen.values.count == 1 && seen.values[0].contains("Dear residents"), "the Translate row gets the raw letter")
        #expect(configs.values == ["en>de"])
        let saved = try String(contentsOf: h.dir.appendingPathComponent("letter-german.txt"), encoding: .utf8)
        #expect(saved.contains("14. November"))
    }

    // MARK: 15

    @Test @MainActor func flow15SavesPhotoDepthPng() async throws {
        let sizes = Log<String>()
        let h = try harness("15-DepthMap") { entry, _ in
            #expect(entry.runnerKind == .depth)
            return Stub(accepts: .image, produces: .image) { input in
                if case .image(let img) = input { sizes.add("\(img.cgImage.width)x\(img.cgImage.height)") }
                return .image(ImageMedia(cgImage: Self.solidImage(16)))
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        try await run(h)
        #expect(sizes.values == ["740x503"], "the stage receives the kitten photo: \(sizes.values)")
        let png = try Data(contentsOf: h.dir.appendingPathComponent("photo-depth.png"))
        #expect(png.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
    }

    // MARK: 16

    @Test @MainActor func flow16SavesAnswerTxtAndThePromptCarriesTheDocumentAndTheQuestion() async throws {
        let prompts = Log<String>()
        let h = try harness("16-AskADocument") { entry, _ in
            #expect(entry.runnerKind == .llm)
            return Stub(accepts: .text, produces: .text) { input in
                if case .text(let t) = input { prompts.add(t) }
                return .text("New employees get 14 days, usable after the 3-month probation.")
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        try await run(h)
        #expect(prompts.values.count == 1)
        let prompt = try #require(prompts.values.first)
        #expect(prompt.contains("Riverstone Trading Sdn Bhd"), "the PDF text is in the Answer prompt")
        #expect(prompt.contains("3-month probation") && prompt.contains("31 March"))
        #expect(prompt.contains("Question: How many days of"), "the question is in the Answer prompt")
        #expect(prompt.contains("start using it?"))
        let saved = try String(contentsOf: h.dir.appendingPathComponent("answer.txt"), encoding: .utf8)
        #expect(saved.contains("14 days"))
    }

    // MARK: 17

    @Test @MainActor func flow17SavesStoryWavThroughTheThreeAIChain() async throws {
        let description = "A grey kitten sits on a sofa in a bright living room."
        let story = "Once upon a time, a little grey kitten curled up and fell asleep."
        let draftPrompts = Log<String>()
        let speakInputs = Log<String>()
        let h = try harness("17-StoryFromPhoto") { entry, _ in
            switch entry.runnerKind {
            case .vision:
                return Stub(accepts: .image, produces: .text) { _ in .text(description) }
            case .llm:
                return Stub(accepts: .text, produces: .text) { input in
                    if case .text(let t) = input { draftPrompts.add(t) }
                    return .text(story)
                }
            case .tts:
                return Stub(accepts: .text, produces: .audio) { input in
                    if case .text(let t) = input { speakInputs.add(t) }
                    return .audio(AudioBuffer(samples: [Float](repeating: 0.1, count: 2400), sampleRate: 24000))
                }
            default:
                throw StageError.unsupportedModel(id: entry.id, kind: entry.runnerKind)
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        try await run(h)
        #expect(draftPrompts.values.count == 1)
        #expect(draftPrompts.values.first?.contains(description) == true, "row 3's input is row 2's description")
        #expect(speakInputs.values == [story], "row 4's input is exactly row 3's story")
        let wav = try Data(contentsOf: h.dir.appendingPathComponent("story.wav"))
        #expect(wav.prefix(4) == Data("RIFF".utf8))
    }

    // MARK: 18

    @Test @MainActor func flow18SavesBothTheChartAndTheSummaryAndRowFourReadsRowOnesTable() async throws {
        let prompts = Log<String>()
        let h = try harness("18-ChartSpreadsheet") { entry, _ in
            #expect(entry.runnerKind == .llm)
            return Stub(accepts: .text, produces: .text) { input in
                if case .text(let t) = input { prompts.add(t) }
                return .text("Sales trend up through the year. December was best, February worst.")
            }
        }
        defer { try? FileManager.default.removeItem(at: h.cleanup) }
        try await run(h)
        // Both outputs from the one input.
        let png = try Data(contentsOf: h.dir.appendingPathComponent("sales-chart.png"))
        #expect(png.prefix(4) == Data([0x89, 0x50, 0x4E, 0x47]))
        let cg = try #require(ImageLoader.decodedCGImage(from: png))
        #expect(cg.width == 800 && cg.height == 400)
        let summary = try String(contentsOf: h.dir.appendingPathComponent("sales-summary.txt"), encoding: .utf8)
        #expect(summary.contains("December") && summary.contains("February"))
        // Row 4 reads row 1's table (`(1)`), not row 3's Save Image status: the Summarize prompt holds the markdown table.
        #expect(prompts.values.count == 1)
        let prompt = try #require(prompts.values.first)
        #expect(prompt.contains("| month | sales |") && prompt.contains("| Dec | 7200 |") && prompt.contains("| Feb | 3900 |"))
        #expect(!prompt.contains("sales-chart.png"), "row 3's status must not reach the table row")
        #expect(exists(h, "sales-chart.png") && exists(h, "sales-summary.txt"))
    }
}
