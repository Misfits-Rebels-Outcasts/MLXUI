import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// MT5-2-FIX-2: a tool's inline text belongs to the reply that came BEFORE the call, never to an identical reply after
/// it; every Answer is fed its producing turn's question and context. Chats go through the live record types.
@MainActor @Suite struct ModelToolBindingTests {

    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/ModelTools")
    }

    private let ocrText = "Subtotal 28.00\nTax 2.24\nTOTAL $30.24"
    private let answer = "The total amount is $30.24."

    private struct Chat {
        let root: URL
        let store: ChatAssetStore
        var items: [TranscriptItem]
    }

    /// Turn 1: receipt + question + OCR + reply. Then `followUps` speak turns, each `speak(text: answer)` with its reply.
    private func chat(replies: [String]) throws -> Chat {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mt5-bind-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let store = ChatAssetStore(conversationID: UUID(), in: root.appendingPathComponent("conversations"))
        let jpg = root.appendingPathComponent("receipt.jpg"); try Data([1, 2, 3]).write(to: jpg)
        let a1 = try store.importFile(jpg, kind: "image", name: "receipt.jpg")
        let a2 = try store.addText(ocrText, name: "OCR text",
                                   provenance: ChatAssetProvenance(rowText: "1. OCR   PaddleOCR-VL-1.5", model: "PaddleOCR-VL-1.5", inputs: ["a1"]))
        _ = a1
        let attachment = ChatAttachment(name: "receipt.jpg", storedPath: jpg.path, mode: .reference, byteCount: 3, assetHandle: "a1", assetKind: "image")
        var items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "what's the total?", attachments: [attachment])),
            .tool(ToolActivity(name: "ocr_image", arguments: ["asset": .string("a1")], status: .finished,
                               result: ModelToolRunner.textResultJSON(record: a2, model: "PaddleOCR-VL-1.5", text: ocrText, budgetTokens: 500))),
            .message(ChatMessage(role: .assistant, text: replies[0], modelID: "apple-intelligence")),
        ]
        for reply in replies.dropFirst() {
            let wav = root.appendingPathComponent("speech-\(UUID().uuidString).wav"); try Data([4, 5, 6]).write(to: wav)
            let audio = try store.adopt(file: wav, kind: "audio", name: "Speech",
                                        provenance: ChatAssetProvenance(rowText: "1. Speak   Kokoro 82M", model: "Kokoro 82M", inputs: []))
            items.append(.message(ChatMessage(role: .user, text: "read that answer aloud.")))
            items.append(.tool(ToolActivity(name: "speak", arguments: ["text": .string(replies[0])], status: .finished,
                                            result: ModelToolRunner.mediaResultJSON(record: audio, model: "Kokoro 82M"))))
            items.append(.message(ChatMessage(role: .assistant, text: reply, modelID: "apple-intelligence")))
        }
        return Chat(root: root, store: store, items: items)
    }

    private func flow(_ c: Chat) -> ModelToolFlow {
        ModelToolTraceAdapter.flow(items: c.items, store: c.store, models: [])
    }

    private func golden(_ name: String) throws -> String {
        try String(contentsOf: fixtures.appendingPathComponent("\(name).cat"), encoding: .utf8)
    }

    private let goldenTasks = ["Read Image", "OCR", "Template", "Answer", "Speak", "Save Audio", "Save Text"]

    // MARK: The smoke chat, replies identical

    @Test func identicalRepliesBuildTheSameGoldenAsDifferingOnes() throws {
        let c = try chat(replies: [answer, answer]); defer { try? FileManager.default.removeItem(at: c.root) }
        let built = flow(c)
        #expect(built.text == (try golden("ocr-then-speak-identical-replies")))
        #expect(built.text == (try golden("ocr-then-speak-two-turns")), "the same shape as when reply 2 differed")
        #expect(built.roundTrips)
    }

    @Test func aDifferingSecondReplyStillBindsToTheFirst() throws {
        let c = try chat(replies: [answer, "\(answer) Here's the audio"]); defer { try? FileManager.default.removeItem(at: c.root) }
        #expect(flow(c).text == (try golden("ocr-then-speak-two-turns")))
    }

    @Test func threeTurnsOfIdenticalRepliesStillGiveOneAnswer() throws {
        let c = try chat(replies: [answer, answer, answer]); defer { try? FileManager.default.removeItem(at: c.root) }
        let built = flow(c)
        #expect(built.document.rows.filter { $0.task == "Answer" }.count == 1)
        #expect(built.document.rows.filter { $0.task == "Speak" }.count == 2)
        #expect(built.roundTrips)
        try expectEveryAnswerFedItsQuestion(built, question: "what's the total?")
    }

    // MARK: Order beats equality

    @Test func aToolCallBeforeItsReplyInTheSameTurnIsBoundToTheConsumingTurn() throws {
        // `speak("Bonjour !")` is called before the reply that repeats it: no earlier reply, so this turn is the producer.
        let trace = ModelToolTrace(
            events: [.init(kind: .user, text: "say hi in French and read it"),
                     .init(kind: .tool, text: "Bonjour !", tool: "speak", output: "a1"),
                     .init(kind: .assistant, text: "Bonjour !", modelID: "apple-intelligence")],
            assets: [.init(handle: "a1", kind: "audio", name: "Speech", fileName: "a1.wav", rowText: "1. Speak   Kokoro 82M", model: "Kokoro 82M")])
        let built = ModelToolFlowBuilder.build(trace)
        #expect(built.document.rows.compactMap(\.task) == ["Template", "Answer", "Speak", "Save Audio", "Save Text"])
        #expect(built.document.rows.first?.settings == FlowSettingsEscaper.quoted("say hi in French and read it"))
    }

    @Test func anEarlierIdenticalReplyWinsOverALaterOne() throws {
        // Turn 1 reply == turn 2 reply == turn 3 reply; the call in turn 3 binds turn 1's (the first that precedes it).
        // Turn 2's reply equal to turn 1's must not capture it either.
        let c = try chat(replies: [answer, "something else", answer]); defer { try? FileManager.default.removeItem(at: c.root) }
        let built = flow(c)
        let templates = built.document.rows.filter { $0.task == "Template" }
        #expect(templates.count == 1)
        #expect(templates.first?.settings == FlowSettingsEscaper.quoted("Context:\n{input}\n\nQuestion: what's the total?"))
    }

    // MARK: The audit

    private func expectEveryAnswerFedItsQuestion(_ built: ModelToolFlow, question: String) throws {
        let rows = built.document.rows
        for (i, row) in rows.enumerated() where row.task == "Answer" {
            #expect(rows[i - 1].task == "Template")
            let pattern = rows[i - 1].settings ?? ""
            #expect(pattern.contains("{input}") || pattern.contains("{1}"))
            #expect(pattern.contains(question))
        }
    }

    @Test func aMisBoundTurnIsCaughtByTheAudit() {
        let template = Row(task: "Template", settings: FlowSettingsEscaper.quoted("Context:\n{input}\n\nQuestion: read that answer aloud."))
        let answerRow = Row(task: "Answer", model: "apple-foundation @ system")
        let right = [answerRow.id: ModelToolFlowBuilder.AnswerExpectation(question: "read that answer aloud.", contextCount: 1)]
        let wrongTurn = [answerRow.id: ModelToolFlowBuilder.AnswerExpectation(question: "what's the total?", contextCount: 1)]
        let wrongContext = [answerRow.id: ModelToolFlowBuilder.AnswerExpectation(question: "read that answer aloud.", contextCount: 2)]
        #expect(ModelToolFlowBuilder.answersMisfed(rows: [template, answerRow], expected: right) == 0)
        #expect(ModelToolFlowBuilder.answersMisfed(rows: [template, answerRow], expected: wrongTurn) == 1)
        #expect(ModelToolFlowBuilder.answersMisfed(rows: [template, answerRow], expected: wrongContext) == 1)
    }

    /// Every Template that feeds an Answer, in every golden and every in-test chat, reads its upstream rows and carries
    /// the question that produced the answer.
    @Test(arguments: ["receipt-ocr-answer-speak", "meeting-audio-transcribe-answer", "text-translate-speak"])
    func everyTemplateFeedingAnAnswerCarriesItsContextInTheMapperGoldens(name: String) throws {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("\(name).trace.json"))
        let trace = try JSONDecoder().decode(ModelToolTrace.self, from: data)
        let built = ModelToolFlowBuilder.build(trace)
        try expectContext(built, trace: trace)
    }

    @Test func everyTemplateFeedingAnAnswerCarriesItsContextInTheLiveChats() throws {
        for replies in [[answer, answer], [answer, "\(answer) Here's the audio"], [answer, answer, answer], [answer, "x", answer]] {
            let c = try chat(replies: replies); defer { try? FileManager.default.removeItem(at: c.root) }
            let trace = ModelToolTraceAdapter.trace(items: c.items, store: c.store)
            try expectContext(ModelToolFlowBuilder.build(trace), trace: trace)
        }
    }

    private func expectContext(_ built: ModelToolFlow, trace: ModelToolTrace) throws {
        let rows = built.document.rows
        let users = trace.events.filter { $0.kind == .user }.compactMap(\.text)
        for (i, row) in rows.enumerated() where row.task == "Answer" {
            #expect(i > 0 && rows[i - 1].task == "Template")
            let pattern = rows[i - 1].settings ?? ""
            let hasUpstream = rows[..<(i - 1)].contains { $0.task != nil }
            if hasUpstream { #expect(pattern.contains("{input}") || pattern.contains("{1}"), "bare question feeds an Answer: \(pattern)") }
            #expect(users.contains { pattern.contains($0) }, "the Template names no user question: \(pattern)")
        }
        // The golden's own text, when saved, still round-trips.
        #expect(built.roundTrips || built.isEmpty)
    }

    // MARK: Missing file and names

    @Test func aMissingAttachmentNamesTheStepThatWasLeftOutToo() throws {
        let c = try chat(replies: [answer]); defer { try? FileManager.default.removeItem(at: c.root) }
        let a1 = try #require(c.store.record("a1"))
        try FileManager.default.removeItem(at: c.store.url(for: a1))
        let built = flow(c)
        let line = try #require(built.banner.first { $0.kind == .stepNeedsMissingFile })
        #expect(line.items == ["OCR with PaddleOCR-VL-1.5 (its file receipt.jpg is gone)"])
        #expect(built.banner.contains { $0.kind == .missingAttachment })
    }

    @Test func aSecondWorkflowWithTheSameNameGetsANumber() throws {
        typealias H = ModelToolWorkflowHandoff
        #expect(H.uniqueName("Receipt", among: []) == "Receipt")
        #expect(H.uniqueName("Receipt", among: ["receipt"]) == "Receipt 2")
        #expect(H.uniqueName("Receipt", among: ["receipt", "receipt 2"]) == "Receipt 3")

        // Through the real hand-off: a saved flow already has the name.
        let data = try Data(contentsOf: fixtures.appendingPathComponent("text-translate-speak.trace.json"))
        let flow = ModelToolFlowBuilder.build(try JSONDecoder().decode(ModelToolTrace.self, from: data))
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mt5-names-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = FlowWorkspace(root: root)
        let existing = workspace.directory(for: UUID().uuidString)
        try FileManager.default.createDirectory(at: existing, withIntermediateDirectories: true)
        try "mlxflow 0.8\n1. Template   \"x\"\n".write(to: existing.appendingPathComponent("Receipt.cat"), atomically: true, encoding: .utf8)
        let target = try H.prepare(flow: flow, chatTitle: "Receipt", assetsDirectory: root, workspace: workspace)
        #expect(target.name == "Receipt 2")
    }
}
