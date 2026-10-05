import Foundation
import MLXLMCommon
import Testing
@testable import MLXUI

/// Multiple chat conversations: the persisted model, the disk store, the title derivation,
/// and the transcript → `Chat.Message` history mapping that rehydrates `ChatSession`.
/// The list/select/delete UI is verified by build + smoke.
struct ConversationsTests {

    static func sampleItems() -> [TranscriptItem] {
        [
            .message(ChatMessage(role: .user, text: "What is 37 x 481?")),
            .tool(ToolActivity(
                name: "calculator", arguments: ["expression": .string("37*481")],
                status: .finished, result: "17797")),
            .message(ChatMessage(role: .assistant, text: "It's 17797.")),
        ]
    }

    // MARK: - Codable

    @Test func conversationRoundTripsThroughJSON() throws {
        var items = Self.sampleItems()
        // Every tool status must survive persistence, including associated values.
        for status in [ToolActivity.Status.running, .failed("boom"), .denied,
                       .unknownTool, .budgetExceeded] {
            items.append(.tool(ToolActivity(name: "t", arguments: [:], status: status)))
        }
        let conversation = Conversation(modelID: "mlx-community--Qwen3-4B-4bit", items: items)

        let data = try JSONEncoder().encode(conversation)
        let decoded = try JSONDecoder().decode(Conversation.self, from: data)
        #expect(decoded == conversation)
    }

    // MARK: - Attached folders (CW-2, C7/C8)

    /// A conversation written **before** the CW-2 keys existed must still decode, with nothing
    /// attached — `decodeIfPresent`, never `decode`, or every saved chat would vanish.
    @Test func anOldConversationJSONDecodesWithNothingAttached() throws {
        let old = """
        {
          "id": "6B29FC40-CA47-1067-B31D-00DD010662DA",
          "modelID": "mlx-community--Qwen3-4B-4bit",
          "title": "An old chat",
          "createdAt": "2026-01-02T03:04:05Z",
          "updatedAt": "2026-01-02T03:04:05Z",
          "items": []
        }
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let conversation = try decoder.decode(Conversation.self, from: Data(old.utf8))
        #expect(conversation.attachedFolders.isEmpty)
        #expect(conversation.workingFolder == nil)
        #expect(conversation.title == "An old chat")
    }

    @Test func foldersRoundTripThroughJSON() throws {
        let conversation = Conversation(
            modelID: "m",
            attachedFolders: ["/Users/me/a", "/Users/me/b"],
            workingFolder: "/Users/me/b")
        let data = try JSONEncoder().encode(conversation)
        let decoded = try JSONDecoder().decode(Conversation.self, from: data)
        #expect(decoded.attachedFolders == ["/Users/me/a", "/Users/me/b"])
        #expect(decoded.workingFolder == "/Users/me/b")
        #expect(decoded.effectiveWorkingFolder == "/Users/me/b")
    }

    /// C8: with no chosen working folder the first attached one is it.
    @Test func effectiveWorkingFolderFallsBackToTheFirstAttached() {
        let conversation = Conversation(modelID: "m", attachedFolders: ["/a", "/b"])
        #expect(conversation.effectiveWorkingFolder == "/a")
    }

    /// C8: a working folder that is no longer attached is ignored, not returned.
    @Test func effectiveWorkingFolderIgnoresADetachedChosenFolder() {
        let conversation = Conversation(
            modelID: "m", attachedFolders: ["/a"], workingFolder: "/b")
        #expect(conversation.effectiveWorkingFolder == "/a")
    }

    // MARK: - Attachments (CW-3)

    private func attachment(
        _ name: String, mode: ChatAttachmentMode = .inline
    ) -> ChatAttachment {
        ChatAttachment(
            name: name, storedPath: "/container/attachments/\(name)", mode: mode,
            byteCount: 8, inlineText: mode == .reference ? nil : "contents")
    }

    /// `chatHistory` must replay **exactly** what the model was given (C13) — that is why
    /// `inlineText` is stored rather than re-read from disk.
    @Test func chatHistoryRendersAttachmentsIntoTheUserTurn() {
        let message = ChatMessage(
            role: .user, text: "Summarise this.",
            attachments: [attachment("rates.csv")])
        let history = Conversation.chatHistory(from: [.message(message)])
        #expect(history.count == 1)
        #expect(history[0].content == """
        Summarise this.

        <attachment name="rates.csv">
        contents
        </attachment>
        """)
    }

    @Test func chatHistoryRendersAReferenceWithoutContents() {
        let message = ChatMessage(
            role: .user, text: "Look at this.",
            attachments: [attachment("big.txt", mode: .reference)])
        let history = Conversation.chatHistory(from: [.message(message)])
        #expect(history[0].content.contains("mode=\"reference\"/>"))
        #expect(!history[0].content.contains("contents"))
    }

    /// A conversation saved before CW-3 has no `attachments` key on its messages: it must decode
    /// (and not vanish from the list).
    @Test func anOldChatMessageJSONDecodesWithoutAttachments() throws {
        let old = """
        {
          "id": "6B29FC40-CA47-1067-B31D-00DD010662DA",
          "modelID": "mlx-community--Qwen3-4B-4bit",
          "title": "An old chat",
          "createdAt": "2026-01-02T03:04:05Z",
          "updatedAt": "2026-01-02T03:04:05Z",
          "items": [
            {"message": {"_0": {"id": "6B29FC40-CA47-1067-B31D-00DD010662DB",
                                "role": "user", "text": "hello"}}}
          ]
        }
        """
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let conversation = try decoder.decode(Conversation.self, from: Data(old.utf8))
        #expect(conversation.items.count == 1)
        guard case .message(let message) = conversation.items[0] else {
            Issue.record("expected a message")
            return
        }
        #expect(message.text == "hello")
        #expect(message.attachments.isEmpty)
    }

    @Test func attachmentsRoundTripThroughJSON() throws {
        let message = ChatMessage(
            role: .user, text: "see attached",
            attachments: [attachment("a.csv"), attachment("b.txt", mode: .reference)])
        let conversation = Conversation(modelID: "m", items: [.message(message)])
        let data = try JSONEncoder().encode(conversation)
        let decoded = try JSONDecoder().decode(Conversation.self, from: data)
        guard case .message(let restored) = decoded.items[0] else {
            Issue.record("expected a message")
            return
        }
        #expect(restored.attachments.count == 2)
        #expect(restored.attachments.map(\.name) == ["a.csv", "b.txt"])
        #expect(restored.attachments[1].mode == .reference)
        #expect(restored.attachments[1].inlineText == nil)
    }

    // MARK: - Store

    @Test func storeSavesLoadsPerModelAndDeletes() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversations-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        // Empty / missing directory → no conversations, no crash.
        #expect(ConversationStore.conversations(forModelID: "m", in: dir).isEmpty)

        let older = Conversation(
            modelID: "model-a", title: "older",
            updatedAt: Date(timeIntervalSinceNow: -100), items: Self.sampleItems())
        let newer = Conversation(
            modelID: "model-a", title: "newer",
            updatedAt: Date(), items: Self.sampleItems())
        let otherModel = Conversation(modelID: "model-b", items: Self.sampleItems())
        ConversationStore.save(older, in: dir)
        ConversationStore.save(newer, in: dir)
        ConversationStore.save(otherModel, in: dir)

        // Filtered to the model, newest first.
        let loaded = ConversationStore.conversations(forModelID: "model-a", in: dir)
        #expect(loaded.map(\.title) == ["newer", "older"])

        // Saving again overwrites, not duplicates.
        ConversationStore.save(newer, in: dir)
        #expect(ConversationStore.conversations(forModelID: "model-a", in: dir).count == 2)

        ConversationStore.delete(id: newer.id, in: dir)
        #expect(ConversationStore.conversations(forModelID: "model-a", in: dir)
            .map(\.title) == ["older"])
        // The other model's conversation is untouched.
        #expect(ConversationStore.conversations(forModelID: "model-b", in: dir).count == 1)
    }

    @Test func storeSkipsDamagedFiles() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversations-damaged-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        ConversationStore.save(Conversation(modelID: "m", items: Self.sampleItems()), in: dir)
        try Data("not json".utf8).write(to: dir.appendingPathComponent("junk.json"))

        #expect(ConversationStore.conversations(forModelID: "m", in: dir).count == 1)
    }

    // MARK: - Titles

    /// CW-3: `delete` removes `conversations/<uuid>/` too, so the attachment copies go with the
    /// chat (privacy: they are the user's own files, copied only to be readable after a relaunch).
    @Test func deleteRemovesTheConversationFolder() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("conversations-delete-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }

        let id = UUID()
        let conversation = Conversation(id: id, modelID: "m", items: Self.sampleItems())
        ConversationStore.save(conversation, in: dir)
        let attachments = ConversationStore.attachmentsDirectory(for: id, in: dir)
        try FileManager.default.createDirectory(at: attachments, withIntermediateDirectories: true)
        try "a,b\n".write(
            to: attachments.appendingPathComponent("1-rates.csv"),
            atomically: true, encoding: .utf8)
        #expect(FileManager.default.fileExists(atPath: attachments.path))

        ConversationStore.delete(id: id, in: dir)
        #expect(!FileManager.default.fileExists(
            atPath: ConversationStore.fileURL(for: id, in: dir).path))
        #expect(!FileManager.default.fileExists(
            atPath: ConversationStore.folderURL(for: id, in: dir).path))
    }

    @Test func titleUsesTheFirstLineTrimmedAndCapped() {
        #expect(Conversation.title(forFirstPrompt: "  What is MLX?  ") == "What is MLX?")
        #expect(Conversation.title(forFirstPrompt: "line one\nline two") == "line one")
        #expect(Conversation.title(forFirstPrompt: "   \n  ") == Conversation.untitled)
        let long = String(repeating: "a", count: 60)
        let title = Conversation.title(forFirstPrompt: long)
        #expect(title.count == 49 && title.hasSuffix("…"))
    }

    // MARK: - History rehydration

    @Test func chatHistoryMapsRolesAndFoldsToolResults() {
        let history = Conversation.chatHistory(from: Self.sampleItems())
        #expect(history.map(\.role) == [.user, .assistant])
        #expect(history[0].content == "What is 37 x 481?")
        // The tool card came before any assistant text, so it opens the assistant turn and
        // the streamed text follows it.
        #expect(history[1].content
            == "[tool calculator returned: 17797]\nIt's 17797.")
    }

    @Test func chatHistoryAppendsToolResultToPrecedingAssistantText() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "fetch it")),
            .message(ChatMessage(role: .assistant, text: "Fetching now.")),
            .tool(ToolActivity(
                name: "fetch_url", arguments: [:], status: .finished, result: "hello")),
        ]
        let history = Conversation.chatHistory(from: items)
        #expect(history.count == 2)
        #expect(history[1].content == "Fetching now.\n[tool fetch_url returned: hello]")
    }

    @Test func chatHistorySkipsUnfinishedToolsAndEmptyText() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "hi")),
            .tool(ToolActivity(name: "t", arguments: [:], status: .denied)),
            .tool(ToolActivity(name: "t", arguments: [:], status: .failed("x"), result: "x")),
            .message(ChatMessage(role: .assistant, text: "")),
        ]
        let history = Conversation.chatHistory(from: items)
        #expect(history.map(\.role) == [.user])
    }

    // MARK: - Transcript restore

    @Test func loadedTranscriptDoesNotExtendARestoredBubble() {
        var builder = TranscriptBuilder()
        builder.load(Self.sampleItems())
        #expect(builder.items.count == 3)

        // The next streamed chunk belongs to a NEW assistant bubble, not the restored one.
        builder.appendAssistantChunk("fresh")
        #expect(builder.items.count == 4)
        guard case .message(let last) = builder.items[3] else {
            Issue.record("expected a message")
            return
        }
        #expect(last.role == .assistant && last.text == "fresh")
    }
}
