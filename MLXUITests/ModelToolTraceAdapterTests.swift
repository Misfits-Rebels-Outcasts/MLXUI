import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// MT5-2: the live chat (real `TranscriptItem`s and a real `ChatAssetStore`) → `ModelToolTrace`, checked by feeding the
/// builder and comparing with the three MT5-1 golden `.cat` files.
@MainActor @Suite struct ModelToolTraceAdapterTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func golden(_ name: String) throws -> String {
        try String(contentsOf: repoRoot.appendingPathComponent("Fixtures/ModelTools/\(name).cat"), encoding: .utf8)
    }

    private struct Chat {
        let root: URL
        let store: ChatAssetStore
        func cleanup() { try? FileManager.default.removeItem(at: root) }
    }

    private func chat() -> Chat {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mt5-adapter-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return Chat(root: root, store: ChatAssetStore(conversationID: UUID(), in: root.appendingPathComponent("conversations")))
    }

    private func file(_ chat: Chat, _ name: String) throws -> URL {
        let url = chat.root.appendingPathComponent(name)
        try Data([1, 2, 3]).write(to: url)
        return url
    }

    private func provenance(_ row: String, _ model: String, _ inputs: [String]) -> ChatAssetProvenance {
        ChatAssetProvenance(rowText: row, model: model, inputs: inputs, milliseconds: 10)
    }

    private func result(_ record: ChatAssetRecord, model: String) -> String {
        ModelToolRunner.mediaResultJSON(record: record, model: model)
    }

    private func call(_ name: String, _ arguments: [String: JSONValue], result: String?, status: ToolActivity.Status = .finished) -> TranscriptItem {
        .tool(ToolActivity(name: name, arguments: arguments, status: status, result: result))
    }

    private func user(_ text: String, handles: [(String, String, String)] = []) -> TranscriptItem {
        let attachments = handles.map { ChatAttachment(name: $0.1, storedPath: "/x/\($0.1)", mode: .reference, byteCount: 3,
                                                       assetHandle: $0.0, assetKind: $0.2) }
        return .message(ChatMessage(role: .user, text: text, attachments: attachments))
    }

    private func assistant(_ text: String) -> TranscriptItem {
        .message(ChatMessage(role: .assistant, text: text, modelID: "apple-intelligence"))
    }

    // MARK: The three golden shapes, through the real record types

    @Test func receiptChatBuildsTheReceiptGolden() throws {
        let c = chat(); defer { c.cleanup() }
        let a1 = try c.store.importFile(try file(c, "receipt.jpg"), kind: "image", name: "receipt.jpg")
        let a2 = try c.store.addText("TOTAL RM 84.20", name: "OCR text", provenance: provenance("1. OCR   PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5", ["a1"]))
        let a3 = try c.store.adopt(file: try file(c, "speech.wav"), kind: "audio", name: "Speech",
                                   provenance: provenance("1. Speak   Kokoro 82M", "Kokoro 82M", []))
        let items = [
            user("What's the total? read it out", handles: [(a1.handle, "receipt.jpg", "image")]),
            call("ocr_image", ["asset": .string("a1")], result: ModelToolRunner.textResultJSON(record: a2, model: "PaddleOCR-VL-1.5", text: "TOTAL RM 84.20", budgetTokens: 500)),
            call("speak", ["text": .string("The total is RM 84.20.")], result: result(a3, model: "Kokoro 82M")),
            assistant("The total is RM 84.20."),
        ]
        let flow = ModelToolTraceAdapter.flow(items: items, store: c.store, models: [])
        #expect(flow.text == (try golden("receipt-ocr-answer-speak")))
        #expect(flow.copies == [ModelToolFlow.Copy(handle: "a1", source: "a1.jpg", destination: "receipt.jpg")])
        #expect(flow.banner.isEmpty)
    }

    @Test func meetingChatBuildsTheMeetingGolden() throws {
        let c = chat(); defer { c.cleanup() }
        let a1 = try c.store.importFile(try file(c, "meeting.m4a"), kind: "audio", name: "meeting.m4a")
        let a2 = try c.store.addText("Alice will send the budget.", name: "Transcript",
                                     provenance: provenance("1. Transcribe   Parakeet TDT 0.6B v3", "Parakeet TDT 0.6B v3", ["a1"]))
        let items = [
            user("Summarize the meeting and list the action items.", handles: [(a1.handle, "meeting.m4a", "audio")]),
            call("transcribe_audio", ["asset": .string("a1")], result: ModelToolRunner.textResultJSON(record: a2, model: "Parakeet TDT 0.6B v3", text: "Alice will send the budget.", budgetTokens: 500)),
            assistant("Alice sends the budget."),
        ]
        let flow = ModelToolTraceAdapter.flow(items: items, store: c.store, models: [])
        #expect(flow.text == (try golden("meeting-audio-transcribe-answer")))
        #expect(flow.copies.map(\.destination) == ["meeting.m4a"])
    }

    @Test func translateChatBuildsTheTranslateGolden() throws {
        let c = chat(); defer { c.cleanup() }
        let a1 = try c.store.addText("Selamat pagi, mesyuarat pada pukul sembilan.", name: "Translation",
                                     provenance: provenance("1. Translate   TranslateGemma 4B; to=\"Malay\"", "TranslateGemma 4B", []))
        let a2 = try c.store.adopt(file: try file(c, "speech.wav"), kind: "audio", name: "Speech",
                                   provenance: provenance("1. Speak   Kokoro 82M", "Kokoro 82M", ["a1"]))
        let items = [
            user("Translate this to Malay and read it out: Good morning, the meeting is at nine."),
            call("translate", ["text": .string("Good morning, the meeting is at nine."), "to": .string("Malay")],
                 result: ModelToolRunner.textResultJSON(record: a1, model: "TranslateGemma 4B", text: "Selamat pagi, mesyuarat pada pukul sembilan.", budgetTokens: 500)),
            call("speak", ["asset": .string("a1")], result: result(a2, model: "Kokoro 82M")),
        ]
        let flow = ModelToolTraceAdapter.flow(items: items, store: c.store, models: [])
        #expect(flow.text == (try golden("text-translate-speak")))
        #expect(flow.copies.isEmpty)
    }

    // MARK: Reading the chat

    @Test func aToolCalledWithAFileNameResolvesToItsHandle() throws {
        let c = chat(); defer { c.cleanup() }
        let a1 = try c.store.importFile(try file(c, "scan.png"), kind: "image", name: "scan.png")
        let a2 = try c.store.addText("hi", name: "OCR text", provenance: provenance("1. OCR   PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5", ["a1"]))
        let items = [
            user("read scan.png", handles: [(a1.handle, "scan.png", "image")]),
            call("ocr_image", ["asset": .string("scan.png")], result: ModelToolRunner.textResultJSON(record: a2, model: "PaddleOCR-VL-1.5", text: "hi", budgetTokens: 500)),
        ]
        let trace = ModelToolTraceAdapter.trace(items: items, store: c.store)
        #expect(trace.events.last?.assetArgument == "a1")
        #expect(trace.events.last?.output == "a2")
        #expect(trace.assets.first { $0.handle == "a2" }?.text == "hi")
    }

    @Test func aDeclinedFailedOrDeniedCallMakesNoRow() throws {
        let c = chat(); defer { c.cleanup() }
        let items = [
            user("translate hello to Malay"),
            call("translate", ["text": .string("hello"), "to": .string("Malay")],
                 result: "The tool did not run and produced nothing. The user declined the download."),
            call("speak", ["text": .string("hello")], result: "Error: no speech model installed."),
            call("ocr_image", ["asset": .string("a1")], result: nil, status: .failed("boom")),
            call("write_file", ["path": .string("x")], result: "ok", status: .denied),
        ]
        let trace = ModelToolTraceAdapter.trace(items: items, store: c.store)
        #expect(trace.events.filter { $0.kind == .tool }.allSatisfy { $0.ok == false })
        let flow = ModelToolFlowBuilder.build(trace)
        #expect(flow.isEmpty)
        #expect(flow.banner.first { $0.kind == .failedCalls }?.count == 4)
    }

    @Test func aFileGoneFromTheAssetsFolderIsMarkedMissing() throws {
        let c = chat(); defer { c.cleanup() }
        let a1 = try c.store.importFile(try file(c, "scan.png"), kind: "image", name: "scan.png")
        let a2 = try c.store.addText("hi", name: "OCR text", provenance: provenance("1. OCR   PaddleOCR-VL-1.5", "PaddleOCR-VL-1.5", ["a1"]))
        try FileManager.default.removeItem(at: c.store.url(for: a1))
        let items = [
            user("read it", handles: [(a1.handle, "scan.png", "image")]),
            call("ocr_image", ["asset": .string("a1")], result: ModelToolRunner.textResultJSON(record: a2, model: "PaddleOCR-VL-1.5", text: "hi", budgetTokens: 500)),
        ]
        let flow = ModelToolTraceAdapter.flow(items: items, store: c.store, models: [])
        #expect(flow.isEmpty)
        #expect(flow.banner.contains { $0.kind == .missingAttachment })
        #expect(flow.copies.isEmpty)
    }

    // MARK: Catalog

    @Test func theCatalogMapsChatModelsAndPinsFromTheBundledCatalog() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let models = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
        let catalog = ModelToolTraceAdapter.catalog(models: models)
        for entry in models where entry.runnerKind == .llm {
            #expect(catalog.routerDisplays[entry.id] == TaskModels.displayName(for: entry))
        }
        for entry in models { #expect(catalog.hfModelIDs[TaskModels.displayName(for: entry)] != nil) }
        #expect(catalog.routerDisplays["apple-intelligence"] == nil)
    }
}
