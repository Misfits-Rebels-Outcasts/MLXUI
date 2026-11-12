import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// MT5-2-FIX-1: the owner's smoke chat, built through the live record types. Turn 1 attaches a receipt and asks the total
/// (OCR answers, the router replies); turn 2 asks to read that answer aloud (`speak` of the router's text). The saved flow
/// must ask the question of the OCR text once, in the golden `Context:/Question:` shape, and speak that answer.
@MainActor @Suite struct ModelToolTwoTurnChatTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private let ocrText = "Subtotal 28.00\nTax 2.24\nTOTAL $30.24"
    private let answer = "The total amount is $30.24."

    private struct Chat {
        let root: URL
        let store: ChatAssetStore
        let items: [TranscriptItem]
    }

    /// The smoke chat, with `ModelToolRunner`'s result JSON and `ChatAttachmentAssets`' records.
    private func smokeChat() throws -> Chat {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mt5-twoturn-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = ChatAssetStore(conversationID: UUID(), in: root.appendingPathComponent("conversations"))
        let jpg = root.appendingPathComponent("receipt.jpg"); try Data([1, 2, 3]).write(to: jpg)
        let wav = root.appendingPathComponent("speech.wav"); try Data([4, 5, 6]).write(to: wav)
        let a1 = try store.importFile(jpg, kind: "image", name: "receipt.jpg")
        let a2 = try store.addText(ocrText, name: "OCR text",
                                   provenance: ChatAssetProvenance(rowText: "1. OCR   PaddleOCR-VL-1.5", model: "PaddleOCR-VL-1.5", inputs: ["a1"], milliseconds: 5100))
        let a3 = try store.adopt(file: wav, kind: "audio", name: "Speech",
                                 provenance: ChatAssetProvenance(rowText: "1. Speak   Kokoro 82M", model: "Kokoro 82M", inputs: [], milliseconds: 600))
        func tool(_ name: String, _ args: [String: JSONValue], _ result: String) -> TranscriptItem {
            .tool(ToolActivity(name: name, arguments: args, status: .finished, result: result))
        }
        let attachment = ChatAttachment(name: "receipt.jpg", storedPath: jpg.path, mode: .reference, byteCount: 3,
                                        assetHandle: "a1", assetKind: "image")
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "what's the total?", attachments: [attachment])),
            tool("ocr_image", ["asset": .string("a1")],
                 ModelToolRunner.textResultJSON(record: a2, model: "PaddleOCR-VL-1.5", text: ocrText, budgetTokens: 500)),
            .message(ChatMessage(role: .assistant, text: answer, modelID: "apple-intelligence")),
            .message(ChatMessage(role: .user, text: "read that answer aloud.")),
            tool("speak", ["text": .string(answer)], ModelToolRunner.mediaResultJSON(record: a3, model: "Kokoro 82M")),
            .message(ChatMessage(role: .assistant, text: "\(answer) Here's the audio", modelID: "apple-intelligence")),
        ]
        return Chat(root: root, store: store, items: items)
    }

    @Test func theSmokeChatBuildsTheFourthGolden() throws {
        let c = try smokeChat(); defer { try? FileManager.default.removeItem(at: c.root) }
        let flow = ModelToolTraceAdapter.flow(items: c.items, store: c.store, models: [])
        let expected = try String(contentsOf: repoRoot.appendingPathComponent("Fixtures/ModelTools/ocr-then-speak-two-turns.cat"), encoding: .utf8)
        #expect(flow.text == expected)
        #expect(flow.roundTrips)
        #expect(flow.copies == [ModelToolFlow.Copy(handle: "a1", source: "a1.jpg", destination: "receipt.jpg")])
    }

    @Test func thereIsExactlyOneAnswerAndItsTemplateCarriesTheContext() throws {
        let c = try smokeChat(); defer { try? FileManager.default.removeItem(at: c.root) }
        let rows = ModelToolTraceAdapter.flow(items: c.items, store: c.store, models: []).document.rows
        #expect(rows.compactMap(\.task) == ["Read Image", "OCR", "Template", "Answer", "Speak", "Save Audio", "Save Text"])
        #expect(rows.filter { $0.task == "Answer" }.count == 1)
        #expect(rows.filter { $0.task == "Template" }.allSatisfy { ($0.settings ?? "").contains("{input}") })
    }

    @Test func theAnswerIsAskedTheQuestionOfTheOCRText() async throws {
        let c = try smokeChat(); defer { try? FileManager.default.removeItem(at: c.root) }
        let rows = ModelToolTraceAdapter.flow(items: c.items, store: c.store, models: []).document.rows
        let template = try #require(rows.first { $0.task == "Template" })
        // The Template's input is the OCR row's text (the row above it); run the real Template tool on it.
        let input = Asset(items: [Item(kind: .text, value: ocrText, path: nil, sourceText: nil)])
        let out = try await TemplateTool(settings: template.settings).run(input) { _ in }
        let text = try #require(out.items.first?.value)
        #expect(text.contains(ocrText))
        #expect(text.contains("what's the total?"))
        #expect(!text.contains("read that answer aloud"))
        // The Answer row reads the Template directly above it and the Speak row reads the Answer.
        let kinds = rows.compactMap(\.task)
        let answerAt = try #require(kinds.firstIndex(of: "Answer"))
        #expect(kinds[answerAt - 1] == "Template")
        #expect(kinds[answerAt + 1] == "Speak")
        #expect(rows[answerAt + 1].refs.isEmpty)
    }

    @Test func aLastReplyWithItsOwnContextGetsItsOwnPair() throws {
        // Turn 2 attaches a text file and asks about it; the reply is answered from that text, not from turn 1's.
        let c = try smokeChat(); defer { try? FileManager.default.removeItem(at: c.root) }
        let notes = try c.store.addText("Call Bob at nine.", name: "notes.txt")
        var items = c.items
        items.append(.message(ChatMessage(role: .user, text: "what does the note say?", attachments: [
            ChatAttachment(name: "notes.txt", storedPath: "/x/notes.txt", mode: .inline, byteCount: 17, assetHandle: notes.handle, assetKind: "text")])))
        items.append(.message(ChatMessage(role: .assistant, text: "Call Bob at nine.", modelID: "apple-intelligence")))
        let flow = ModelToolTraceAdapter.flow(items: items, store: c.store, models: [])
        let tasks = flow.document.rows.compactMap(\.task)
        #expect(tasks.filter { $0 == "Answer" }.count == 2)
        let last = try #require(flow.document.rows.last { $0.task == "Template" })
        #expect((last.settings ?? "").contains("{input}") && (last.settings ?? "").contains("what does the note say?"))
        #expect(flow.roundTrips)
    }

    @Test func aBareFollowUpTurnAfterAToolChatAddsNoPair() throws {
        // A plain turn (no tools, no attachments) after the chat: v1 keeps today's behaviour, no history (Q-MT5-S).
        let c = try smokeChat(); defer { try? FileManager.default.removeItem(at: c.root) }
        var items = c.items
        items.append(.message(ChatMessage(role: .user, text: "thanks")))
        items.append(.message(ChatMessage(role: .assistant, text: "You're welcome.", modelID: "apple-intelligence")))
        let flow = ModelToolTraceAdapter.flow(items: items, store: c.store, models: [])
        #expect(flow.document.rows.filter { $0.task == "Answer" }.count == 1)
    }

    // MARK: The invariant

    @Test func anAnswerWithContextFedAPlainQuestionIsCaught() {
        let template = Row(task: "Template", settings: "\"read that answer aloud.\"")
        let answer = Row(task: "Answer", model: "apple-foundation @ system")
        #expect(ModelToolFlowBuilder.answersMissingContext(rows: [template, answer], contextful: [answer.id]) == 1)
        let good = Row(task: "Template", settings: "\"Context:\\n{input}\\n\\nQuestion: hi\"")
        #expect(ModelToolFlowBuilder.answersMissingContext(rows: [good, answer], contextful: [answer.id]) == 0)
        // An answer that had no context may be fed a bare question.
        #expect(ModelToolFlowBuilder.answersMissingContext(rows: [template, answer], contextful: []) == 0)
        // An Answer not directly under a Template is caught too.
        #expect(ModelToolFlowBuilder.answersMissingContext(rows: [good, Row(task: "OCR"), answer], contextful: [answer.id]) == 1)
    }
}
