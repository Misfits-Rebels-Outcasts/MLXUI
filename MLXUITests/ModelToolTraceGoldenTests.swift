import Testing
import Foundation
@testable import MLXUI

/// MT5-1: a chat's model-tool trace → a `.cat`. The three goldens under `Fixtures/ModelTools/` are asserted byte-for-byte
/// (`CatSerializer`'s output is the truth), and every saved document must re-parse (model compared per row, F-5) and pass
/// `FlowValidator` against the curated manifests. The folder is deliberately not in the CatFlow fixture manifest.
struct ModelToolTraceGoldenTests {

    // MARK: Plumbing

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func fixture(_ name: String, _ ext: String) throws -> String {
        try String(contentsOf: repoRoot.appendingPathComponent("Fixtures/ModelTools/\(name).\(ext)"), encoding: .utf8)
    }

    private func trace(_ name: String) throws -> ModelToolTrace {
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("Fixtures/ModelTools/\(name).trace.json"))
        return try JSONDecoder().decode(ModelToolTrace.self, from: data)
    }

    private func registry() throws -> DirectoryFlowRegistry {
        try DirectoryFlowRegistry.load(directory: repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models"))
    }

    /// The saved text re-parses to the same rows and passes the validator with no `offdevice`/`code`/`network` flag.
    private func expectSavable(_ flow: ModelToolFlow, sourceLocation: SourceLocation = #_sourceLocation) throws {
        #expect(!flow.isEmpty, sourceLocation: sourceLocation)
        #expect(flow.roundTrips, "the saved text must re-parse to the same rows and models", sourceLocation: sourceLocation)
        let back = try CatParser.parse(flow.text)
        #expect(back.rows.map(\.model) .map { $0 ?? "" } == flow.document.rows.map { $0.model ?? "" }, sourceLocation: sourceLocation)
        let parsed = try CatParser.parseForValidation(flow.text)
        let issues = FlowValidator.checkFlow(parsed, registry: try registry())
        #expect(issues.isEmpty, "validator: \(issues.map(\.description))", sourceLocation: sourceLocation)
        #expect(flow.document.flags.isEmpty, sourceLocation: sourceLocation)
        #expect(flow.text.hasPrefix("mlxflow 0.8\n1."), "no capability flag on line one", sourceLocation: sourceLocation)
    }

    private func tasks(_ flow: ModelToolFlow) -> [String] { flow.document.rows.map { $0.task ?? "" } }

    // Trace builders.
    private typealias T = ModelToolTrace
    private func user(_ text: String, _ attachments: [String] = []) -> T.Event {
        T.Event(kind: .user, text: text, attachments: attachments.isEmpty ? nil : attachments)
    }
    private func assistant(_ text: String, id: String = "apple-intelligence") -> T.Event {
        T.Event(kind: .assistant, text: text, modelID: id)
    }
    private func call(_ tool: String, text: String? = nil, asset: String? = nil, out: String? = nil, ok: Bool? = nil) -> T.Event {
        T.Event(kind: .tool, text: text, tool: tool, assetArgument: asset, output: out, ok: ok)
    }
    private func attachment(_ handle: String, _ kind: String, _ name: String, text: String? = nil, missing: Bool? = nil) -> T.Asset {
        T.Asset(handle: handle, kind: kind, name: name, fileName: "\(handle).dat", text: text, missing: missing)
    }
    private func output(_ handle: String, _ kind: String, _ row: String, model: String, text: String? = nil) -> T.Asset {
        T.Asset(handle: handle, kind: kind, name: "out", fileName: "\(handle).dat", text: text, rowText: row, model: model)
    }
    private let ocr = "1. OCR   PaddleOCR-VL-1.5"
    private let speak = "1. Speak   Kokoro 82M"
    private let kokoro = "Kokoro 82M"
    private let paddle = "PaddleOCR-VL-1.5"

    // MARK: Goldens

    @Test(arguments: ["receipt-ocr-answer-speak", "meeting-audio-transcribe-answer", "text-translate-speak"])
    func goldenTraceSerializesByteForByte(name: String) throws {
        let flow = ModelToolFlowBuilder.build(try trace(name))
        let expected = try fixture(name, "cat")
        #expect(flow.text == expected)
        try expectSavable(flow)
        #expect(flow.banner.isEmpty, "\(flow.banner.map(\.line))")
    }

    @Test func receiptGoldenCopiesTheAttachmentAndDoesNotCopyOutputs() throws {
        let flow = ModelToolFlowBuilder.build(try trace("receipt-ocr-answer-speak"))
        #expect(flow.copies == [ModelToolFlow.Copy(handle: "a1", source: "a1.jpg", destination: "receipt.jpg")])
    }

    @Test func goldensSaveRowsComeLastWithRefs() throws {
        for name in ["receipt-ocr-answer-speak", "meeting-audio-transcribe-answer", "text-translate-speak"] {
            let flow = ModelToolFlowBuilder.build(try trace(name))
            let t = tasks(flow)
            let firstSave = try #require(t.firstIndex { $0.hasPrefix("Save") })
            #expect(t[firstSave...].allSatisfy { $0.hasPrefix("Save") }, "\(name): Save rows are last")
            // A Save row after another Save row cannot lean on the implicit chain (a Save outputs `status`).
            for i in (firstSave + 1)..<t.count { #expect(!flow.document.rows[i].refs.isEmpty, "\(name): row \(i + 1) needs a ref") }
        }
    }

    // MARK: Router steps

    @Test func multiTurnKeepsTheLastReplyAndCountsWhatItDropped() throws {
        let trace = T(
            events: [
                user("hello"), assistant("Hi there."),
                user("Translate to Malay: good day"),
                call("translate", text: "good day", out: "a1"),
                assistant("Selamat sejahtera."),
            ],
            assets: [output("a1", "text", "1. Translate   TranslateGemma 4B; to=\"Malay\"", model: "TranslateGemma 4B", text: "selamat sejahtera")])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(tasks(flow) == ["Template", "Translate", "Template", "Answer", "Save Text"])
        let kinds = flow.banner.map(\.kind)
        #expect(kinds == [.droppedReplies, .history])
        #expect(flow.banner.first?.count == 1)
        try expectSavable(flow)
    }

    @Test func routerTextAToolConsumedIsAnAnswerAndFoldsIntoTheFinalReply() throws {
        let trace = T(
            events: [user("say hi in French and read it"), call("speak", text: "Bonjour !", out: "a1"), assistant("Bonjour !")],
            assets: [output("a1", "audio", speak, model: kokoro)])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(tasks(flow) == ["Template", "Answer", "Speak", "Save Audio", "Save Text"])
        #expect(flow.banner.isEmpty)
        try expectSavable(flow)
    }

    @Test func appleAndMLXRoutersDisplayDifferently() throws {
        let events = [user("hi", []), call("speak", text: "Hello.", out: "a1"), assistant("Hello.", id: "mlx-community--Qwen3.5-9B-4bit")]
        let assets = [output("a1", "audio", speak, model: kokoro)]
        // A router the bridge names: displayed, not pinned.
        let bridged = ModelToolFlowBuilder.build(
            T(events: events, assets: assets),
            catalog: ModelToolFlowCatalog(routerDisplays: ["mlx-community--Qwen3.5-9B-4bit": "Qwen3.5 9B"]))
        #expect(bridged.document.rows.first { $0.task == "Answer" }?.model == "Qwen3.5 9B")
        #expect(bridged.document.models.isEmpty)
        try expectSavable(bridged)
        // The system slot: displayed as such, with no pin and no flag.
        let apple = ModelToolFlowBuilder.build(
            T(events: events.map { var e = $0; if e.kind == .assistant { e.modelID = "apple-intelligence" }; return e }, assets: assets))
        #expect(apple.document.rows.first { $0.task == "Answer" }?.model == "apple-foundation @ system")
        #expect(apple.document.models.isEmpty)
        try expectSavable(apple)
        // A chat model the bridge does not name is pinned, as the editor does (F-6).
        let unbridged = ModelToolFlowBuilder.build(
            T(events: events, assets: assets),
            catalog: ModelToolFlowCatalog(routerDisplays: ["mlx-community--Qwen3.5-9B-4bit": "Some Unbridged Chat"],
                                          hfModelIDs: ["Some Unbridged Chat": "mlx-community/Some-Unbridged-Chat-4bit"]))
        #expect(unbridged.document.models == ["Some Unbridged Chat": "mlx-community/Some-Unbridged-Chat-4bit"])
        #expect(unbridged.roundTrips)
        // No known chat model: the answer is left out, and the step that needed it.
        let unknown = ModelToolFlowBuilder.build(T(events: events, assets: assets))
        #expect(unknown.isEmpty)
        #expect(unknown.banner.contains { $0.kind == .noRouter })
    }

    // MARK: Failed, missing, unconsumed

    @Test func aFailedOrDeclinedCallMakesNoRow() throws {
        let trace = T(
            events: [user("read it", ["a1"]), call("speak", text: "x", ok: false), call("ocr_image", asset: "a1", out: "a2")],
            assets: [attachment("a1", "image", "scan.png"), output("a2", "text", ocr, model: paddle, text: "hello")])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(tasks(flow) == ["Read Image", "OCR", "Save Text"])
        #expect(flow.banner.first { $0.kind == .failedCalls }?.count == 1)
        try expectSavable(flow)
    }

    @Test func aMissingAttachmentLeavesItsStepOutWithABannerLine() throws {
        let trace = T(
            events: [user("read it", ["a1"]), call("ocr_image", asset: "a1", out: "a2")],
            assets: [attachment("a1", "image", "scan.png", missing: true), output("a2", "text", ocr, model: paddle)])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(flow.isEmpty)
        #expect(flow.banner.contains { $0.kind == .missingAttachment && $0.items == ["scan.png"] })
        #expect(flow.copies.isEmpty)
    }

    @Test func anUnconsumedAttachmentIsLeftOutWithABannerLine() throws {
        let trace = T(
            events: [user("translate: hello to Malay", ["a1"]), call("translate", text: "hello", out: "a2")],
            assets: [attachment("a1", "image", "unused.png"),
                     output("a2", "text", "1. Translate   TranslateGemma 4B; to=\"Malay\"", model: "TranslateGemma 4B", text: "helo")])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(flow.copies.isEmpty)
        #expect(flow.banner.contains { $0.kind == .unconsumedAttachment && $0.items == ["unused.png"] })
        try expectSavable(flow)
    }

    // MARK: Models a `.cat` cannot name or pin

    @Test func anUnwritableModelNameIsLeftOutAndNeverSubstituted() throws {
        let trace = T(
            events: [user("read", ["a1"]), call("ocr_image", asset: "a1", out: "a2"), call("ocr_image", asset: "a1", out: "a3")],
            assets: [attachment("a1", "image", "scan.png"),
                     output("a2", "text", "1. OCR   dots.ocr", model: "dots.ocr"),
                     output("a3", "text", "1. OCR   olmOCR-2 7B", model: "olmOCR-2 7B")])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(flow.isEmpty)
        #expect(flow.banner.first { $0.kind == .unwritableModel }?.items == ["dots.ocr", "olmOCR-2 7B"])
    }

    @Test func anUnbridgedModelIsPinnedByItsHFID() throws {
        #expect(CatalogBridge.entry(for: "Qwen3-TTS 1.7B") == nil, "the premise: no bridge row names it")
        let trace = T(
            events: [user("say it: hello"), call("speak", text: "hello", out: "a1")],
            assets: [output("a1", "audio", "1. Speak   Qwen3-TTS 1.7B", model: "Qwen3-TTS 1.7B")])
        let pinned = ModelToolFlowBuilder.build(
            trace, catalog: ModelToolFlowCatalog(hfModelIDs: ["Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"]))
        #expect(pinned.document.models == ["Qwen3-TTS 1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"])
        #expect(pinned.text.contains("Qwen3-TTS 1.7B"))
        try expectSavable(pinned)
        let unknown = ModelToolFlowBuilder.build(trace)
        #expect(unknown.isEmpty)
        #expect(unknown.banner.contains { $0.kind == .unresolvedModel })
    }

    // MARK: Text in rows

    @Test func aPlaceholderTokenInTheUsersTextLeavesItOutWithABannerLine() throws {
        let trace = T(
            events: [user("make {input} loud"), call("speak", text: "Hello.", out: "a1"), assistant("Hello.")],
            assets: [output("a1", "audio", speak, model: kokoro)])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(flow.isEmpty)
        #expect(flow.banner.contains { $0.kind == .braceToken })
        // The same for a literal source.
        let literal = T(
            events: [user("say {1} out loud"), call("speak", text: "{1}", out: "a1")],
            assets: [output("a1", "audio", speak, model: kokoro)])
        let out = ModelToolFlowBuilder.build(literal)
        #expect(out.isEmpty)
        #expect(out.banner.contains { $0.kind == .braceToken })
    }

    @Test func quotesAndNewlinesInUserTextSurviveTheRoundTrip() throws {
        let text = "Say \"hi\"\nthen \\ stop"
        let trace = T(
            events: [user(text), call("speak", text: text, out: "a1")],
            assets: [output("a1", "audio", speak, model: kokoro)])
        let flow = ModelToolFlowBuilder.build(trace)
        try expectSavable(flow)
        #expect(flow.document.rows.first?.settings == FlowSettingsEscaper.quoted(text))
    }

    // MARK: File names

    @Test func fileNamesWithSpacesAndCollisionsBecomeDistinctSafeNames() throws {
        let trace = T(
            events: [user("read both", ["a1", "a2", "a3"]),
                     call("ocr_image", asset: "a1", out: "a4"), call("ocr_image", asset: "a2", out: "a5"),
                     call("ocr_image", asset: "a3", out: "a6")],
            assets: [attachment("a1", "image", "My Receipt.JPG"), attachment("a2", "image", "my-receipt.jpg"),
                     attachment("a3", "image", "Résumé 2024!.jpeg"),
                     output("a4", "text", ocr, model: paddle), output("a5", "text", ocr, model: paddle),
                     output("a6", "text", ocr, model: paddle)])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(flow.copies.map(\.destination) == ["my-receipt.jpg", "my-receipt-2.jpg", "r-sum-2024.jpeg"])
        #expect(flow.copies.map(\.source) == ["a1.dat", "a2.dat", "a3.dat"])
        try expectSavable(flow)
    }

    @Test func aTextAttachmentIsCopiedAsItsExtractedTextWithATxtName() throws {
        let trace = T(
            events: [user("summarize this", ["a1"]), assistant("It says hello.")],
            assets: [attachment("a1", "text", "Notes.markdown", text: "hello")])
        // A text attachment the router read becomes the Answer's context.
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(tasks(flow) == ["Read Text", "Template", "Answer", "Save Text"])
        #expect(flow.copies.map(\.destination) == ["notes.txt"])
        try expectSavable(flow)
    }

    // MARK: What is never a row

    @Test func otherToolsAreNamedInTheBannerAndNeverRowsOrFlags() throws {
        let trace = T(
            events: [user("do stuff", ["a1"]),
                     call("run_shell"), call("write_file"), call("fetch_url"), call("calculator"), call("calculator"),
                     call("echo"), call("list_capabilities"), call("read_asset", asset: "a1"),
                     call("ocr_image", asset: "a1", out: "a2")],
            assets: [attachment("a1", "image", "scan.png"), output("a2", "text", ocr, model: paddle, text: "hi")])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(tasks(flow) == ["Read Image", "OCR", "Save Text"])
        let line = try #require(flow.banner.first { $0.kind == .leftOutTools })
        #expect(line.items == ["run_shell", "write_file", "fetch_url", "calculator x2", "echo"])
        #expect(flow.text.contains("code") == false && flow.text.contains("network") == false && flow.text.contains("offdevice") == false)
        try expectSavable(flow)
    }

    @Test func aTranslateRowExistsOnlyWhenTheTranslateToolRan() throws {
        let declined = T(
            events: [user("translate hello to Malay and read it"),
                     call("translate", text: "hello", ok: false), call("speak", text: "hello", out: "a1")],
            assets: [output("a1", "audio", speak, model: kokoro)])
        let flow = ModelToolFlowBuilder.build(declined)
        #expect(!tasks(flow).contains("Translate"))
        #expect(tasks(flow) == ["Template", "Speak", "Save Audio"])
        try expectSavable(flow)
    }

    @Test func twoUnusedOutputsAreSavedWithDistinctNamesAndExplicitRefs() throws {
        let trace = T(
            events: [user("say one then two"), call("speak", text: "one", out: "a1"), call("speak", text: "two", out: "a2")],
            assets: [output("a1", "audio", speak, model: kokoro), output("a2", "audio", speak, model: kokoro)])
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(tasks(flow) == ["Template", "Speak", "Template", "Speak", "Save Audio", "Save Audio"])
        #expect(flow.document.rows.suffix(2).allSatisfy { !$0.refs.isEmpty })
        #expect(flow.document.rows.suffix(2).map { $0.settings } == ["speech.wav", "speech-2.wav"])
        try expectSavable(flow)
    }
}
