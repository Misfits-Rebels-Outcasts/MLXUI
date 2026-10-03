import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CX-1 — the pure conversation exporter: Markdown shape, reasoning folding, tool cards,
/// JSON round-trip and filename sanitising. The save panel and the buttons are CX-2, verified
/// there by build + smoke; nothing here touches the filesystem.
struct ConversationExportTests {

    // MARK: - Fixtures

    /// 2026-10-03 18:41:00 in Taipei (+08:00) — the timestamp used in the backlog's Markdown
    /// sample, so the expected string below reads as the spec does.
    private static let start = Date(timeIntervalSince1970: 1_791_024_060)
    /// 2026-10-03 18:55:00 (+08:00).
    private static let updated = Date(timeIntervalSince1970: 1_791_024_900)
    /// 2026-10-03 19:02:00 (+08:00).
    private static let exported = Date(timeIntervalSince1970: 1_791_025_320)

    private static let taipei = TimeZone(secondsFromGMT: 8 * 3600) ?? .gmt

    private static func context() -> ConversationExportContext {
        ConversationExportContext(
            modelDisplayName: "Qwen3.8 27B 3-bit",
            appVersion: "0.1 (2)",
            edition: "AI Browser",
            exportedAt: exported,
            timeZone: taipei)
    }

    private static func conversation(
        title: String = "New Chat",
        items: [TranscriptItem] = ConversationsTests.sampleItems()
    ) -> Conversation {
        Conversation(
            id: UUID(uuidString: "4DCE2FCE-166D-409A-8195-C3C8ABE82BE4") ?? UUID(),
            modelID: "lukaskremla--Qwen3.8-27B-3bit-MLX",
            title: title,
            createdAt: start,
            updatedAt: updated,
            items: items)
    }

    // MARK: - Markdown

    @Test func markdownMatchesTheExpectedShape() {
        // Written out literally: this string is the CX-A review artifact. The one assistant
        // heading spans the tool card and the answer that follow it.
        let expected = """
        # New Chat

        | | |
        |---|---|
        | Model | Qwen3.8 27B 3-bit (`lukaskremla--Qwen3.8-27B-3bit-MLX`) |
        | Started | 2026-10-03 18:41 (+08:00) |
        | Last updated | 2026-10-03 18:55 (+08:00) |
        | Exported | 2026-10-03 19:02 (+08:00) · AI Browser 0.1 (2) |

        ---

        ### You

        What is 37 x 481?

        ### Qwen3.8 27B 3-bit

        > **Tool · calculator · Done**
        > Arguments: `{"expression":"37*481"}`
        >
        > ```
        > 17797
        > ```

        It's 17797.
        """

        let actual = ConversationExporter.markdown(Self.conversation(), context: Self.context())
        #expect(actual == expected + "\n")
    }

    @Test func emptyConversationStillExportsTheHeader() {
        let conversation = Self.conversation(items: [])
        let markdown = ConversationExporter.markdown(conversation, context: Self.context())

        // The metadata table alone: no section heading, and no dangling horizontal rule.
        let expected = """
        # New Chat

        | | |
        |---|---|
        | Model | Qwen3.8 27B 3-bit (`lukaskremla--Qwen3.8-27B-3bit-MLX`) |
        | Started | 2026-10-03 18:41 (+08:00) |
        | Last updated | 2026-10-03 18:55 (+08:00) |
        | Exported | 2026-10-03 19:02 (+08:00) · AI Browser 0.1 (2) |
        """
        #expect(markdown == expected + "\n")
    }

    // MARK: - Reasoning shapes

    @Test func thinkBlockWithBothTagsIsFoldedIntoReasoning() {
        let (reasoning, answer) = ConversationExporter.splitReasoning(
            "<think>2 + 2 is 4.</think>The answer is 4.")
        #expect(reasoning == "2 + 2 is 4.")
        #expect(answer == "The answer is 4.")
    }

    @Test func closingTagWithoutAnOpeningTagIsTheCommonQwen3Shape() {
        // Qwen3 templates put `<think>` in the generation prompt, so the stream starts inside
        // the block. A naive open…close match would leave all of this in the answer.
        let (reasoning, answer) = ConversationExporter.splitReasoning(
            "The user wants a sum.\nLet me add them.\n</think>\n\n\n\nIt's 17797.")
        #expect(reasoning == "The user wants a sum.\nLet me add them.")
        #expect(answer == "It's 17797.")
    }

    @Test func unclosedThinkBlockIsAllReasoningAndNoAnswer() {
        let (reasoning, answer) = ConversationExporter.splitReasoning(
            "<think>Let me check the datetime first")
        #expect(reasoning == "Let me check the datetime first")
        #expect(answer == "")
    }

    @Test func textWithoutThinkTagsIsAPlainAnswer() {
        let (reasoning, answer) = ConversationExporter.splitReasoning("Just an answer.")
        #expect(reasoning == nil)
        #expect(answer == "Just an answer.")
    }

    @Test func markdownFoldsReasoningAboveTheAnswer() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "What is 2 + 2?")),
            .message(ChatMessage(
                role: .assistant, text: "<think>Two and two.</think>It is 4.")),
        ]
        let markdown = ConversationExporter.markdown(
            Self.conversation(items: items), context: Self.context())

        #expect(markdown.contains("""
        <details><summary>Reasoning</summary>

        Two and two.

        </details>

        It is 4.
        """))
        // The raw tag never reaches the Markdown.
        #expect(!markdown.contains("<think>"))
    }

    @Test func markdownNotesAReplyCutOffMidReasoning() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "Explain the plot")),
            .message(ChatMessage(role: .assistant, text: "<think>So the story starts with")),
        ]
        let markdown = ConversationExporter.markdown(
            Self.conversation(items: items), context: Self.context())

        #expect(markdown.contains("So the story starts with"))
        #expect(markdown.contains("_(reply ended before the reasoning finished)_"))
    }

    // MARK: - Tool cards

    @Test func everyToolStatusRendersItsLabel() {
        let cases: [(ToolActivity.Status, String)] = [
            (ToolActivity.Status.running, "Running…"),
            (.finished, "Done"),
            (.failed("boom"), "Failed"),
            (.denied, "Declined"),
            (.unknownTool, "Unknown tool"),
            (.budgetExceeded, "Budget exceeded"),
        ]
        for (status, label) in cases {
            let items: [TranscriptItem] = [
                .message(ChatMessage(role: .user, text: "go")),
                .tool(ToolActivity(name: "t", arguments: [:], status: status)),
            ]
            let markdown = ConversationExporter.markdown(
                Self.conversation(items: items), context: Self.context())
            #expect(
                markdown.contains("> **Tool · t · \(label)**"),
                "status \(status) should render as \(label) — got:\n\(markdown)")
        }
    }

    @Test func failedToolShowsTheMessage() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "go")),
            .tool(ToolActivity(
                name: "fetch_url", arguments: ["url": .string("https://example.com")],
                status: .failed("boom"), result: "boom")),
        ]
        let markdown = ConversationExporter.markdown(
            Self.conversation(items: items), context: Self.context())

        #expect(markdown.contains("> **Tool · fetch_url · Failed**"))
        #expect(markdown.contains("boom"))
    }

    @Test func terminalToolWithoutArgumentsOrResultRendersTheHeaderOnly() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "rm -rf /")),
            .tool(ToolActivity(name: "shell", arguments: [:], status: .denied)),
        ]
        let markdown = ConversationExporter.markdown(
            Self.conversation(items: items), context: Self.context())

        #expect(markdown.contains("> **Tool · shell · Declined**"))
        #expect(!markdown.contains("Arguments:"))
    }

    @Test func toolResultContainingAFenceGetsALongerFence() {
        let result = """
        Here is a snippet:

        ```
        print("hi")
        ```
        """
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "show me")),
            .tool(ToolActivity(
                name: "read_file", arguments: [:], status: .finished, result: result)),
        ]
        let markdown = ConversationExporter.markdown(
            Self.conversation(items: items), context: Self.context())

        // The block's own fence is four backticks: one longer than the ``` inside the result.
        #expect(markdown.contains("> ````\n> Here is a snippet:"))
        #expect(markdown.contains("> ```\n> print(\"hi\")\n> ```"))
        #expect(markdown.contains("> ````\n"))
        // Open + close only — the result's own ``` runs never re-open the block.
        let fourRuns = markdown.components(separatedBy: "````").count - 1
        #expect(fourRuns == 2, "expected exactly two 4-backtick fences — got:\n\(markdown)")
    }

    @Test func oneAssistantHeadingSpansTextToolTextBetweenTwoUserTurns() {
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .user, text: "fetch it")),
            .message(ChatMessage(role: .assistant, text: "Fetching now.")),
            .tool(ToolActivity(
                name: "fetch_url", arguments: [:], status: .finished, result: "hello")),
            .message(ChatMessage(role: .assistant, text: "Done: hello")),
            .message(ChatMessage(role: .user, text: "thanks")),
        ]
        let markdown = ConversationExporter.markdown(
            Self.conversation(items: items), context: Self.context())

        let heading = "### Qwen3.8 27B 3-bit"
        #expect(markdown.components(separatedBy: heading).count - 1 == 1)
        #expect(markdown.components(separatedBy: "### You").count - 1 == 2)
        guard let assistantAt = markdown.range(of: heading),
              let lastUserAt = markdown.range(of: "### You", options: .backwards) else {
            Issue.record("expected both headings")
            return
        }
        #expect(assistantAt.lowerBound < lastUserAt.lowerBound)
    }

    // MARK: - JSON

    @Test func jsonCarriesTheHeaderAndTheUnchangedConversation() throws {
        let conversation = Self.conversation(title: "37 x 481")
        let data = try ConversationExporter.json(conversation, context: Self.context())

        let root = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(root["format"] as? String == "mlxui.conversation")
        #expect(root["formatVersion"] as? Int == 1)
        #expect(root["exportedAt"] as? String == "2026-10-03T11:02:00Z")

        let app = try #require(root["app"] as? [String: Any])
        #expect(app["edition"] as? String == "AI Browser")
        #expect(app["version"] as? String == "0.1 (2)")

        let model = try #require(root["model"] as? [String: Any])
        #expect(model["id"] as? String == conversation.modelID)
        #expect(model["displayName"] as? String == "Qwen3.8 27B 3-bit")

        // The embedded conversation decodes back to the input, unchanged, with ISO-8601 dates.
        let conversationData = try JSONSerialization.data(
            withJSONObject: try #require(root["conversation"]))
        let conversationObject = try #require(
            try JSONSerialization.jsonObject(with: conversationData) as? [String: Any])
        #expect(conversationObject["createdAt"] as? String == "2026-10-03T10:41:00Z")
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        #expect(try decoder.decode(Conversation.self, from: conversationData) == conversation)

        // Pretty-printed, so a human can read the file.
        let text = try #require(String(data: data, encoding: .utf8))
        #expect(text.contains("\n  \"format\""))
    }

    @Test func jsonKeepsReasoningTextVerbatim() throws {
        let raw = "<think>Two and two.</think>It is 4."
        let items: [TranscriptItem] = [
            .message(ChatMessage(role: .assistant, text: raw)),
        ]
        let conversation = Self.conversation(items: items)
        let data = try ConversationExporter.json(conversation, context: Self.context())

        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let conversationData = try JSONSerialization.data(
            withJSONObject: try #require(root["conversation"]))
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(Conversation.self, from: conversationData)

        #expect(decoded == conversation)
        #expect(decoded.items.count == 1)
        guard case .message(let message) = decoded.items[0] else {
            Issue.record("expected a message")
            return
        }
        #expect(message.text == raw)
    }

    // MARK: - The real saved-conversation shape

    /// The exact wire shape of a conversation written by the shipping app (captured 2026-10-03
    /// from `~/Library/Containers/net.connectcode.mlxui/…/conversations/4DCE2FCE-….json`, model
    /// `lukaskremla--Qwen3.8-27B-3bit-MLX`): `TranscriptItem` encodes as the case name wrapping
    /// `_0`, tool `status` as `{"finished":{}}`, and dates as ISO-8601 strings. This is the
    /// G4 "an existing saved conversation still loads" check, and it pins the **shape 2**
    /// reasoning case that conversation actually shows — an assistant text beginning inside the
    /// think block with no opening tag.
    @Test func aRealSavedConversationStillDecodesAndExports() throws {
        let json = """
        {
          "title": "do a search for latest top mlx-community modesl …",
          "id": "4DCE2FCE-166D-409A-8195-C3C8ABE82BE4",
          "createdAt": "2026-10-03T06:58:03Z",
          "updatedAt": "2026-10-03T09:08:06Z",
          "modelID": "lukaskremla--Qwen3.8-27B-3bit-MLX",
          "items": [
            {"message": {"_0": {"role": "user",
              "id": "7BDB6453-EC71-4403-A550-FBFCF257161B",
              "text": "do a search for latest top mlx-community modesl in huggingface"}}},
            {"message": {"_0": {"role": "assistant",
              "id": "970488A7-6DFA-436B-8143-2074EC4A553F",
              "text": "The user is asking me to search.\\n</think>\\n\\n\\n\\n"}}},
            {"tool": {"_0": {"result": "Fetched (HTTP 200)",
              "status": {"finished": {}},
              "id": "17817B36-124C-462A-9398-DA169A874DD2",
              "name": "fetch_url",
              "arguments": {"url": "https://huggingface.co/api/models"}}}},
            {"message": {"_0": {"role": "assistant",
              "id": "A01D390D-5D0C-446B-ACE9-2449508EEF26",
              "text": "Here are the top models."}}}
          ]
        }
        """

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let conversation = try decoder.decode(Conversation.self, from: Data(json.utf8))

        #expect(conversation.title == "do a search for latest top mlx-community modesl …")
        #expect(conversation.items.count == 4)

        // Shape 2: no opening tag, so the reasoning is everything before `</think>`.
        let (reasoning, answer) = ConversationExporter.splitReasoning(
            "The user is asking me to search.\n</think>\n\n\n\n")
        #expect(reasoning == "The user is asking me to search.")
        #expect(answer == "")

        let markdown = ConversationExporter.markdown(conversation, context: Self.context())
        #expect(markdown.contains("<details><summary>Reasoning</summary>\n\nThe user is asking me to search.\n\n</details>"))
        #expect(markdown.contains("> **Tool · fetch_url · Done**"))
        #expect(markdown.contains("Here are the top models."))
    }

    // MARK: - Filenames

    @Test func filenameUsesModelTitleAndDate() {
        let name = ConversationExporter.suggestedFilename(
            Self.conversation(title: "What is 37 x 481?"), context: Self.context(), ext: "md")
        #expect(name == "Qwen3.8 27B 3-bit - What is 37 x 481 - 2026-10-03.md")
    }

    @Test func filenameSanitisesHostileCharacters() {
        // The backlog's own example, written out exactly as the rules produce it: each hostile
        // character becomes a dash, a run of dashes collapses, the trailing `\n` + `k` leaves
        // one dash, and a dash that follows a space stays (only *adjacent* dashes collapse).
        let name = ConversationExporter.suggestedFilename(
            Self.conversation(title: "a/b: c\\d? e*f\"g<h>i|j\nk"),
            context: Self.context(), ext: "json")
        #expect(name == "Qwen3.8 27B 3-bit - a-b- c-d- e-f-g-h-i-j-k - 2026-10-03.json")
    }

    @Test func filenameKeepsSpacesAndCollapsesDashRuns() {
        // Spaces are legal in a filename, so they stay; a run of dashes becomes one dash.
        let name = ConversationExporter.suggestedFilename(
            Self.conversation(title: "spaced   out -- and--dashed"),
            context: Self.context(), ext: "md")
        #expect(name == "Qwen3.8 27B 3-bit - spaced   out - and-dashed - 2026-10-03.md")
    }

    @Test func filenameCapsALongTitle() {
        let long = String(repeating: "a", count: 200)
        let name = ConversationExporter.suggestedFilename(
            Self.conversation(title: long), context: Self.context(), ext: "md")
        let title = name
            .replacingOccurrences(of: "Qwen3.8 27B 3-bit - ", with: "")
            .replacingOccurrences(of: " - 2026-10-03.md", with: "")
        #expect(title.count == ConversationExporter.titleCharacterLimit + 1)
        #expect(title.hasSuffix("…"))
        #expect(title.dropLast() == String(long.prefix(ConversationExporter.titleCharacterLimit)))
    }

    @Test func untitledIsAValidFilenameTitle() {
        let name = ConversationExporter.suggestedFilename(
            Self.conversation(title: Conversation.untitled, items: []),
            context: Self.context(), ext: "md")
        #expect(name == "Qwen3.8 27B 3-bit - New Chat - 2026-10-03.md")
    }
}
