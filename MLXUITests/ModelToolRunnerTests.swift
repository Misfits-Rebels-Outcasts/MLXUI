import Testing
import Foundation
import MLXLMCommon
import ImageIO
import UniformTypeIdentifiers
@testable import MLXUI

/// Records every row a model tool runs (the seam MT1-3 tests read) and answers from `respond`.
nonisolated final class MT13Executor: FlowExecutor, @unchecked Sendable {
    private let lock = NSLock()
    private var _rows: [Row] = []
    private var _inputs: [[Asset]] = []
    private var _made = 0
    private var _blob: URL?
    let respond: @Sendable (Row, [Asset], URL?) throws -> Asset

    init(respond: @escaping @Sendable (Row, [Asset], URL?) throws -> Asset) { self.respond = respond }

    var rows: [Row] { lock.withLock { _rows } }
    var inputs: [[Asset]] { lock.withLock { _inputs } }
    var executorsMade: Int { lock.withLock { _made } }
    func made(blob: URL) { lock.withLock { _made += 1; _blob = blob } }

    func execute(path: String, row: Row, inputs: [Asset], transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?, usedFlowContent: String?) async throws -> Asset {
        let blob = lock.withLock { () -> URL? in _rows.append(row); _inputs.append(inputs); return _blob }
        return try respond(row, inputs, blob)
    }
    var lastCacheHit: Bool { false }
    var lastTag: String? { nil }
    var lastTimeoutFlag: (code: String, message: String)? { nil }
    var lastProviderDeciderFlag: (code: String, message: String)? { nil }
    var lastStaged: (id: String, kind: String, summary: String)? { nil }
    var lastDeciderDetail: (tag: String, confidence: Double, probabilities: [(label: String, probability: Double)],
                            expectedLevel: Double?, stateTruncated: Bool)? { nil }

    static func text(_ value: String) -> MT13Executor {
        MT13Executor { _, _, _ in Asset(items: [Item(kind: .text, value: value)]) }
    }
    /// An executor that writes a small file of `kind` into the blob directory it was given.
    static func file(kind: Kind, ext: String) -> MT13Executor {
        MT13Executor { _, _, blob in
            let dir = blob ?? FileManager.default.temporaryDirectory
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let file = dir.appendingPathComponent("out.\(UUID().uuidString).\(ext)")
            try Data([1, 2, 3]).write(to: file)
            return Asset(items: [Item(kind: kind, path: file)])
        }
    }
}

/// MT1-3 checkpoint (b): validation, the row, the resolver wiring, plain vs JSON results.
@MainActor @Suite struct ModelToolRunnerTests {

    // MARK: fixtures

    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private func claimable(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    private func runtime(installed hf: [String], executor: MT13Executor, ram: Double = 19,
                         roots: [URL]? = nil) throws -> ModelToolRuntime {
        let cat = try catalog()
        let ids = Set(try hf.map { name in try #require(cat.first { $0.hfModelId == name }, "\(name)").id })
        return ModelToolRuntime(catalog: cat, installedModelIDs: ids, claimableModelIDs: claimable(cat), totalRAMGB: ram,
                                makeExecutor: { blob, _ in executor.made(blob: blob); return executor }, pathRoots: roots)
    }

    private func tempRoot() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mt13-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func spec(_ name: String) throws -> ModelToolSpec { try #require(ModelToolSpec.spec(named: name)) }

    private func json(_ text: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any], "not JSON: \(text)")
    }

    private func run(_ tool: String, _ args: [String: JSONValue], runtime rt: ModelToolRuntime,
                     host: ModelToolHost?) async throws -> String {
        let runner = ModelToolRunner(spec: try spec(tool), runtime: rt)
        return await ModelToolHost.$current.withValue(host) { await runner.run(arguments: args) }
    }

    private static let parakeet = "mlx-community/parakeet-tdt-0.6b-v3"
    private static let kokoro = "mlx-community/Kokoro-82M-bf16"
    private static let glmOCR = "mlx-community/GLM-OCR-4bit"

    // MARK: validation (pure)

    @Test func unknownKeysAreRefusedWithTheValidList() throws {
        let result = ModelToolRunner.validate(["wdith": .int(5), "seed": .int(1)], for: try spec("generate_image"))
        guard case .failure(let failure) = result else { Issue.record("accepted"); return }
        #expect(failure.text == "Error: unknown argument \"wdith\" for generate_image. Valid arguments: text, asset, width, height, steps, seed.")
    }

    @Test func aMissingRequiredArgumentIsRefused() throws {
        guard case .failure(let failure) = ModelToolRunner.validate([:], for: try spec("translate")) else { Issue.record("accepted"); return }
        #expect(failure.text.hasPrefix("Error: translate needs the \"to\" argument"))
    }

    @Test func stepsOverFiftyNamesTheLimit() throws {
        guard case .failure(let failure) = ModelToolRunner.validate(["steps": .int(51)], for: try spec("generate_image")) else {
            Issue.record("accepted"); return
        }
        #expect(failure.text == "Error: steps must be at most 50 (got 51).")
        guard case .success = ModelToolRunner.validate(["steps": .int(50)], for: try spec("generate_image")) else { Issue.record("50 refused"); return }
    }

    @Test func integersAreTypeChecked() throws {
        let s = try spec("generate_image")
        guard case .failure(let f1) = ModelToolRunner.validate(["width": .string("wide")], for: s) else { Issue.record("a"); return }
        #expect(f1.text == "Error: width must be a whole number.")
        guard case .failure(let f2) = ModelToolRunner.validate(["width": .double(512.5)], for: s) else { Issue.record("b"); return }
        #expect(f2.text == "Error: width must be a whole number.")
        guard case .failure(let f3) = ModelToolRunner.validate(["width": .int(0)], for: s) else { Issue.record("c"); return }
        #expect(f3.text == "Error: width must be at least 1 (got 0).")
        guard case .success(let ok) = ModelToolRunner.validate(["width": .double(512), "height": .string("256")], for: s) else {
            Issue.record("d"); return
        }
        #expect(ok["width"] == .int(512) && ok["height"] == .int(256))
    }

    @Test func nullArgumentsCountAsAbsent() throws {
        guard case .success(let v) = ModelToolRunner.validate(["language": .null], for: try spec("transcribe_audio")) else {
            Issue.record("refused"); return
        }
        #expect(v.isEmpty)
    }

    // MARK: the row text

    @Test func languageBecomesLangAndAbsentSendsNothing() throws {
        let s = try spec("transcribe_audio")
        #expect(ModelToolRunner.rowText(spec: s, modelDisplay: "Whisper Tiny", values: ["language": .string("ms")])
                == "1. Transcribe   Whisper Tiny; lang=\"ms\"")
        #expect(ModelToolRunner.rowText(spec: s, modelDisplay: "Whisper Tiny", values: [:]) == "1. Transcribe   Whisper Tiny")
    }

    @Test func questionIsTheRowsFirstBareQuotedToken() throws {
        let text = ModelToolRunner.rowText(spec: try spec("describe_image"), modelDisplay: "LFM2-VL 1.6B",
                                           values: ["question": .string("What is the total?")])
        #expect(text == "1. Describe Image   LFM2-VL 1.6B; \"What is the total?\"")
        let row = try #require(try ModelToolRunner.document(rowText: text).rows.first)
        #expect(FlowSettings(row.settings).firstBare() == "What is the total?")
    }

    @Test func generateImageSettingsFollowSpecOrder() throws {
        let text = ModelToolRunner.rowText(
            spec: try spec("generate_image"), modelDisplay: "FLUX.2 Klein 4B",
            values: ["seed": .int(7), "steps": .int(4), "width": .int(512), "height": .int(256)])
        #expect(text == "1. Generate Image   FLUX.2 Klein 4B; width=512; height=256; steps=4; seed=7")
    }

    // MARK: the shared escaper

    // Known limit, not ours to fix here: `FlowSettings.unquote` replaces `\n` and `\t` before `\\`, so a backslash directly
    // followed by n or t (a Windows path `C:\new`, TeX `\newcommand`) does not survive a round trip. Recorded in the journal
    // (MT1-3, Questions for owner); `FlowSettings` is shared with every flow and is left alone.
    @Test func escaperRoundTripsThroughParserAndSerializer() throws {
        for value in ["plain", "with \"quotes\"", "back\\slash", "two\nlines", "a; b = c", "tab\there", "x \"y\"; seed=999", "C:\\Users\\boo\\a.txt"] {
            let text = "mlxflow 0.8\n\n1. Translate   TranslateGemma 4B; \(FlowSettingsEscaper.pair("to", value))\n"
            let doc = try CatParser.parse(text)
            let again = try CatParser.parse(CatSerializer.serialize(doc))
            #expect(CatSerializer.serialize(doc) == CatSerializer.serialize(again))
            let settings = FlowSettings(try #require(again.rows.first).settings)
            #expect(settings.value(for: "to") == value.replacingOccurrences(of: "\t", with: "\t"), "value: \(value)")
        }
    }

    @Test func aQuotedSettingCannotSmuggleAnotherOne() throws {
        let hostile = "a \"x\"; seed=999"
        let text = ModelToolRunner.rowText(spec: try spec("describe_image"), modelDisplay: "LFM2-VL 1.6B",
                                           values: ["question": .string(hostile)])
        let row = try #require(try ModelToolRunner.document(rowText: text).rows.first)
        let settings = FlowSettings(row.settings)
        #expect(settings.value(for: "seed") == nil)
        #expect(settings.firstBare() == hostile)
        let viaPair = FlowSettings(FlowSettingsEscaper.pair("to", hostile))
        #expect(viaPair.value(for: "to") == hostile)
        #expect(viaPair.value(for: "seed") == nil)
    }

    @Test func carriageReturnsBecomeOneLineBreak() {
        #expect(FlowSettingsEscaper.quoted("a\r\nb\rc") == "\"a\\nb\\nc\"")
    }

    // MARK: no host — plain text, exactly as before

    @Test func withoutAHostTranscribeReturnsPlainText() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        let exec = MT13Executor.text("  hello world \n")
        let rt = try runtime(installed: [Self.parakeet], executor: exec, roots: [root])
        let out = try await run("transcribe_audio", ["path": .string(audio.path)], runtime: rt, host: nil)
        #expect(out == "hello world")
        #expect(exec.rows.count == 1)
    }

    @Test func withoutAHostAnEmptyTranscriptIsTheOldSentence() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        let rt = try runtime(installed: [Self.parakeet], executor: .text("  "), roots: [root])
        #expect(try await run("transcribe_audio", ["path": .string(audio.path)], runtime: rt, host: nil)
                == "(no speech detected in the audio)")
    }

    @Test func withoutAHostOCRReturnsPlainText() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("r.png"); try Data([0, 1]).write(to: image)
        let rt = try runtime(installed: [Self.glmOCR], executor: .text("TOTAL 84.20"), roots: [root])
        #expect(try await run("ocr_image", ["path": .string(image.path)], runtime: rt, host: nil) == "TOTAL 84.20")
    }

    @Test func withoutAHostNothingInstalledKeepsTheOldSentences() async throws {
        let rt = try runtime(installed: [], executor: .text("x"))
        #expect(try await run("transcribe_audio", ["url": .string("https://example.com/a.mp3")], runtime: rt, host: nil)
                == "Error: no speech-to-text model is installed. Install one from the Browse tab first.")
        #expect(try await run("ocr_image", ["url": .string("https://example.com/a.png")], runtime: rt, host: nil)
                == "Error: no OCR or vision model is installed. Install one from the Browse tab first.")
    }

    @Test func withoutAHostAMediaResultNeedsAChat() async throws {
        let rt = try runtime(installed: [Self.kokoro], executor: .file(kind: .audio, ext: "wav"))
        let out = try await run("speak", ["text": .string("hello")], runtime: rt, host: nil)
        #expect(out == "Error: speak makes a file, and needs a chat to keep it.")
    }

    // MARK: with a host — JSON

    private func host(_ root: URL, budget: Int = 1000,
                      install: @escaping @Sendable (ModelInstallNeed) async -> Bool = ModelToolHost.declineAllInstalls) -> ModelToolHost {
        ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: root), installRequest: install, inlineBudgetTokens: budget)
    }

    @Test func withAHostATextWithinBudgetComesBackInFullAsJSON() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        let input = try h.assets.importFile(audio, kind: "audio")
        let exec = MT13Executor.text("hello brave new world")
        let rt = try runtime(installed: [Self.parakeet], executor: exec)
        let out = try json(try await run("transcribe_audio", ["asset": .string(input.handle)], runtime: rt, host: h))
        #expect(out["asset"] as? String == "a2")
        #expect(out["kind"] as? String == "text")
        #expect(out["model"] as? String == "Parakeet TDT 0.6B v3")
        #expect(out["words"] as? Int == 4)
        #expect(out["text"] as? String == "hello brave new world")
        #expect(out["preview"] == nil && out["note"] == nil)
        let stored = try #require(h.assets.record("a2"))
        #expect(stored.provenance.inputs == ["a1"])
        #expect(stored.provenance.model == "Parakeet TDT 0.6B v3")
        #expect(stored.provenance.rowText == "1. Transcribe   Parakeet TDT 0.6B v3")
        #expect(h.assets.text(for: "a2") == "hello brave new world")
    }

    @Test func withAHostALongTextComesBackAsAReference() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root, budget: 50)                                   // ~200 characters
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        _ = try h.assets.importFile(audio, kind: "audio")
        let long = Array(repeating: "word", count: 300).joined(separator: " ")
        let rt = try runtime(installed: [Self.parakeet], executor: .text(long))
        let out = try json(try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: h))
        #expect(out["asset"] as? String == "a2")
        #expect(out["words"] as? Int == 300)
        #expect(out["text"] == nil)
        let preview = try #require(out["preview"] as? String)
        #expect(preview.count == ModelToolRunner.previewCharacters + 1 && preview.hasSuffix("…"))
        #expect(out["note"] as? String == "Long text: pass \"a2\" to another tool, or call read_asset.")
        #expect(h.assets.text(for: "a2") == long)
    }

    @Test func theInlineBudgetIsAParameter() {
        let record = ChatAssetRecord(handle: "a1", kind: "text", name: "n", fileName: "a1.txt", bytes: 400, words: 100, duration: nil,
                                     provenance: .init(), createdAt: Date())
        let text = String(repeating: "x", count: 400)                      // 100 estimated tokens
        #expect(ModelToolRunner.textResultJSON(record: record, model: "M", text: text, budgetTokens: 100).contains("\"text\":"))
        #expect(ModelToolRunner.textResultJSON(record: record, model: "M", text: text, budgetTokens: 99).contains("\"preview\":"))
    }

    @Test func anImageResultReportsTheProducedSizeFromTheOutputFile() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let exec = MT13Executor { _, _, blob in
            let dir = try #require(blob)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let file = dir.appendingPathComponent("out.png")
            let ctx = try #require(CGContext(data: nil, width: 6, height: 4, bitsPerComponent: 8, bytesPerRow: 0,
                                             space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
            let image = try #require(ctx.makeImage())
            let dest = try #require(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
            CGImageDestinationAddImage(dest, image, nil)
            #expect(CGImageDestinationFinalize(dest))
            return Asset(items: [Item(kind: .image, path: file)])
        }
        let rt = try runtime(installed: ["mlx-community/FLUX.2-klein-4B-bf16"], executor: exec)
        let out = try json(try await run("generate_image", ["text": .string("a bicycle"), "width": .int(512), "height": .int(512)],
                                         runtime: rt, host: h))
        #expect(out["kind"] as? String == "image")
        #expect(out["width"] as? Int == 6 && out["height"] as? Int == 4)   // the produced file, not the request
        #expect(exec.rows.first?.settings == "width=512; height=512")
    }

    @Test func aCallWithNoInputIsRefusedNamingAllThree() async throws {
        let rt = try runtime(installed: [Self.parakeet, Self.glmOCR], executor: .text("x"))
        for tool in ["transcribe_audio", "ocr_image"] {
            let out = try await run(tool, [:], runtime: rt, host: nil)
            #expect(out == "Error: give the input for \(tool) as exactly one of: asset, path, url.")
        }
    }

    @Test func aMediaResultIsOnlyAHandle() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let exec = MT13Executor.file(kind: .audio, ext: "wav")
        let rt = try runtime(installed: [Self.kokoro], executor: exec)
        let out = try json(try await run("speak", ["text": .string("hello there"), "voice": .string("af_heart")], runtime: rt, host: h))
        #expect(out["asset"] as? String == "a1")
        #expect(out["kind"] as? String == "audio")
        #expect(out["text"] == nil)
        let record = try #require(h.assets.record("a1"))
        #expect(record.fileName == "a1.wav")
        #expect(FileManager.default.fileExists(atPath: h.assets.url(for: record).path))
        #expect(exec.rows.first?.settings == "voice=\"af_heart\"")
        #expect(exec.rows.first?.task == "Speak")
        // the text to speak is the INPUT asset
        #expect(exec.inputs.first?.first?.items.first?.value == "hello there")
    }

    // MARK: F6, the resolver, the decline path

    @Test func f6OnlyParakeetInstalledRunsParakeetNotWhisper() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        _ = try h.assets.importFile(audio, kind: "audio")
        let exec = MT13Executor.text("ok")
        let rt = try runtime(installed: [Self.parakeet], executor: exec)
        _ = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: h)
        let row = try #require(exec.rows.first)
        #expect(row.model == "Parakeet TDT 0.6B v3")
        #expect(row.model != "Whisper Tiny")
    }

    @Test func ocrPicksTheInstalledModelByClaim() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("r.png"); try Data([0, 1]).write(to: image)
        let exec = MT13Executor.text("t")
        let rt = try runtime(installed: [Self.glmOCR], executor: exec, roots: [root])
        _ = try await run("ocr_image", ["path": .string(image.path)], runtime: rt, host: nil)
        #expect(exec.rows.first?.task == "OCR")
        #expect(exec.rows.first?.model == "GLM-OCR")
    }

    /// Replaces the retired id-substring routing tests (paddle / deepseek / dots / olmOCR-as-VLM / case-insensitive):
    /// each OCR model, installed alone, is the one the row names — chosen by the catalog and the registry's claim.
    @Test(arguments: ["mlx-community/PaddleOCR-VL-1.5-4bit", "mlx-community/DeepSeek-OCR-2-bf16", "mlx-community/dots.ocr-4bit",
                      "mlx-community/olmOCR-2-7B-1025-mlx-4bit"])
    func eachInstalledOCRModelIsTheOneTheRowNames(hf: String) async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("r.png"); try Data([0, 1]).write(to: image)
        let exec = MT13Executor.text("t")
        let rt = try runtime(installed: [hf], executor: exec, roots: [root])
        let entry = try #require(rt.catalog.first { $0.hfModelId == hf })
        let out = try await run("ocr_image", ["path": .string(image.path)], runtime: rt, host: nil)
        let status = CapabilityResolver.status(for: try spec("ocr_image"), catalog: rt.catalog, installedModelIDs: rt.installedModelIDs,
                                               claimableModelIDs: rt.claimableModelIDs, totalRAMGB: rt.totalRAMGB)
        #expect(exec.rows.first?.task == "OCR", "\(hf): \(out) | \(status) | claimable=\(rt.claimableModelIDs.contains(entry.id)) ram=\(entry.ramGB)")
        #expect(exec.rows.first?.model == TaskModels.displayName(for: entry), "\(hf)")
    }

    @Test func aNotInstalledModelReturnsTheDeclineSentenceAndRunsNothing() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let asked = MT13Box<[ModelInstallNeed]>([])
        let h = host(root, install: { need in asked.mutate { $0.append(need) }; return false })
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        _ = try h.assets.importFile(audio, kind: "audio")
        let exec = MT13Executor.text("never")
        let rt = try runtime(installed: [], executor: exec)
        let out = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: h)
        let need = try #require(asked.value.first)
        #expect(out == "The user chose not to download \(need.modelDisplayName). Answer without it, or say what it would need.")
        #expect(need.toolName == "transcribe_audio")
        #expect(need.downloadGB > 0)
        #expect(exec.rows.isEmpty)
        #expect(exec.executorsMade == 0)
    }

    @Test func wrongKindAndUnknownHandlesAreCorrective() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        _ = try h.assets.addText("some text", name: "t")
        let rt = try runtime(installed: [Self.glmOCR], executor: .text("x"))
        #expect(try await run("ocr_image", ["asset": .string("a1")], runtime: rt, host: h)
                == "Error: asset a1 is text, but ocr_image needs image.")
        #expect(try await run("ocr_image", ["asset": .string("a7")], runtime: rt, host: h)
                == "Error: no asset \"a7\". Known assets: a1 (text).")
        #expect(try await run("ocr_image", [:], runtime: rt, host: h)
                == "Error: give the input for ocr_image as exactly one of: asset, path, url.")
    }

    @Test func aFreshExecutorIsMadePerCall() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        let exec = MT13Executor.text("words here")
        let rt = try runtime(installed: [Self.parakeet], executor: exec)
        let audio = root.appendingPathComponent("clip.wav"); try Data([0, 1]).write(to: audio)
        _ = try h.assets.importFile(audio, kind: "audio")
        _ = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: h)
        _ = try await run("transcribe_audio", ["asset": .string("a1")], runtime: rt, host: h)
        #expect(exec.executorsMade == 2)
    }

    // MARK: the CW-3 attachment OCR

    @Test func attachmentOCRWithNoModelInstalledIsNilSoTheChipNoteStillShows() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("r.png"); try Data([0, 1]).write(to: image)
        let exec = MT13Executor.text("should not run")
        let rt = try runtime(installed: [], executor: exec)
        #expect(await OCRImageTool.recognizeText(in: image, runtime: rt) == nil)
        #expect(await OCRImageTool.recognizeText(in: image, runtime: nil) == nil)
        #expect(exec.rows.isEmpty)
    }

    @Test func attachmentOCRWithAnInstalledModelReturnsItsText() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let image = root.appendingPathComponent("r.png"); try Data([0, 1]).write(to: image)
        let exec = MT13Executor.text("RECEIPT TOTAL 84.20")
        let rt = try runtime(installed: [Self.glmOCR], executor: exec)
        #expect(await OCRImageTool.recognizeText(in: image, runtime: rt) == "RECEIPT TOTAL 84.20")
        #expect(exec.rows.first?.model == "GLM-OCR")
        #expect(exec.inputs.first?.first?.items.first?.path == image)
    }

    // MARK: the generated tools

    @Test func everySpecRowHasATool() {
        #expect(ModelTool.all.map(\.name) == ModelToolSpec.all.map(\.name))
        #expect(ModelTool.all.count == 12)
    }

    @Test func toolSchemasCarryTheInputsAndRequiredFlags() throws {
        func required(_ name: String) throws -> [String] {
            let tool = try #require(ModelTool.all.first { $0.name == name })
            let fn = tool.toolSpec["function"] as? [String: any Sendable]
            let params = fn?["parameters"] as? [String: any Sendable]
            return params?["required"] as? [String] ?? []
        }
        #expect(try required("transcribe_audio") == [])
        #expect(try required("ocr_image") == [])
        #expect(try required("translate") == ["to"])
        #expect(try required("read_asset") == ["asset"])
        #expect(Set(try #require(ModelTool.all.first { $0.name == "transcribe_audio" }).parameters.map(\.name))
                == ["asset", "path", "url", "language"])
        #expect(Set(try #require(ModelTool.all.first { $0.name == "speak" }).parameters.map(\.name)) == ["text", "asset", "voice"])
    }

    @Test func backCompatToolsKeepTheirNamesAndAcceptUrl() {
        #expect(TranscribeAudioTool().name == "transcribe_audio")
        #expect(OCRImageTool().name == "ocr_image")
        for tool: any AgentTool in [TranscribeAudioTool(), OCRImageTool()] {
            let names = tool.parameters.map(\.name)
            #expect(names.contains("url") && names.contains("asset") && names.contains("path"))
        }
    }

    @Test func noModelIdAppearsInWhatTheRouterReads() throws {
        let cat = try catalog()
        let ids = Set(cat.flatMap { [$0.id, $0.hfModelId] })
        for tool in ModelTool.all {
            var text = tool.toolDescription + tool.parameters.map(\.name).joined()
            let fn = tool.toolSpec["function"] as? [String: any Sendable]
            text += String(describing: fn?["parameters"] ?? "")
            for id in ids { #expect(!text.contains(id), "\(tool.name) names \(id)") }
        }
    }

    @Test func listCapabilitiesReportsStatusWithoutNamingModels() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        _ = try h.assets.addText("hi", name: "note")
        let rt = try runtime(installed: [Self.parakeet], executor: .text("x"))
        let out = await ModelToolRuntime.$current.withValue(rt) {
            await ModelToolHost.$current.withValue(h) { ModelTool.listCapabilities(arguments: [:]) }
        }
        #expect(out.contains("- transcribe_audio: Speech to text. Set language to the spoken language unless it is English. [ready]"))
        #expect(out.contains("- ocr_image: Read the text in an image. [needs a one-time "))
        #expect(out.contains("Files in this chat:\n- a1 note (text)"))
        #expect(!out.contains("Parakeet"))
        #expect(!out.contains("list_capabilities:") && !out.contains("read_asset:"))
    }

    @Test func readAssetPagesALongText() async throws {
        let root = tempRoot(); defer { try? FileManager.default.removeItem(at: root) }
        let h = host(root)
        _ = try h.assets.addText("0123456789", name: "digits")
        func read(_ args: [String: JSONValue]) async -> String {
            await ModelToolHost.$current.withValue(h) { ModelTool.readAsset(arguments: args) }
        }
        let first = try json(await read(["asset": .string("a1"), "length": .int(4)]))
        #expect(first["text"] as? String == "0123" && first["next"] as? Int == 4 && first["total"] as? Int == 10)
        let last = try json(await read(["asset": .string("a1"), "start": .int(8), "length": .int(50)]))
        #expect(last["text"] as? String == "89" && last["next"] == nil)
        #expect(await read(["asset": .string("a1"), "start": .int(11)]).hasPrefix("Error: start 11 is past the end"))
        #expect(await read(["asset": .string("a9")]) == "Error: no asset \"a9\". Known assets: a1 (text).")
        #expect(ModelTool.readAsset(arguments: ["asset": .string("a1")]) == "Error: read_asset is only available in a chat.")
    }
}

/// A tiny lock-protected box for values a `@Sendable` closure records.
nonisolated final class MT13Box<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: T
    init(_ value: T) { stored = value }
    var value: T { lock.withLock { stored } }
    func mutate(_ body: (inout T) -> Void) { lock.withLock { body(&stored) } }
}
