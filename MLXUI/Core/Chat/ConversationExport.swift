//
//  ConversationExport.swift
//  MLXUI — download a chat transcript (CX-1, `RSI/DelegateChatExportBacklog.md`).
//
//  A `Conversation` is already a persisted value type, so exporting it needs no new state:
//  `ConversationExporter` is a pure, synchronous renderer that turns one conversation into
//  Markdown (for reading and sharing) or JSON (the full-fidelity, re-importable copy).
//
//  Everything that would make the output non-deterministic — the export instant, the app
//  version, the edition and the display time zone — is **injected** through
//  `ConversationExportContext`, so tests assert exact strings without a clock or a locale.
//  No AppKit, no file I/O, no network: the caller owns the save panel and the write.
//
//  Reasoning text (`<think>…</think>`) is in the transcript as plain text, because the Qwen3
//  family streams it into the assistant bubble. Markdown folds it into a `<details>` block
//  (D2); JSON keeps the item text verbatim. Nothing here changes what is stored or shown.
//

import Foundation
import MLXLMCommon

/// Everything the exporter can't derive from the conversation itself. Injected by the caller
/// (and pinned in tests) so an export is reproducible.
struct ConversationExportContext: Sendable, Equatable {
    /// `ModelEntry.displayName`, e.g. `"Qwen3.8 27B 3-bit"`.
    let modelDisplayName: String
    /// `CFBundleShortVersionString` plus build, e.g. `"0.1 (2)"`.
    let appVersion: String
    /// `"AI Browser"` (App Store) or `"AI Browser Pro"` (Direct).
    let edition: String
    /// The export instant — injected so tests are deterministic.
    let exportedAt: Date
    /// Display zone for `Started` / `Last updated` / `Exported`, with its UTC offset.
    /// Injectable so a test can pin one.
    let timeZone: TimeZone

    init(
        modelDisplayName: String,
        appVersion: String,
        edition: String,
        exportedAt: Date,
        timeZone: TimeZone = .current
    ) {
        self.modelDisplayName = modelDisplayName
        self.appVersion = appVersion
        self.edition = edition
        self.exportedAt = exportedAt
        self.timeZone = timeZone
    }
}

/// Pure renderer for one conversation. Read-only: it never touches `ConversationStore`, the
/// live transcript, or the filesystem.
nonisolated enum ConversationExporter {

    enum ExportError: Error, Equatable {
        case encodingFailed(String)
    }

    /// How much of a title survives in the filename.
    static let titleCharacterLimit = 60

    // MARK: - Markdown

    /// The Markdown transcript: a metadata table, a horizontal rule, then one heading per
    /// speaker turn, with reasoning folded into `<details>` and tool cards as block-quoted
    /// cards. A conversation with no items stops after the table — no dangling rule.
    static func markdown(_ c: Conversation, context: ConversationExportContext) -> String {
        let body = blocks(for: c.items, context: context)
        var sections = [header(for: c, context: context)]
        if !body.isEmpty {
            sections.append("---")
            sections.append(contentsOf: body)
        }
        return sections.joined(separator: "\n\n") + "\n"
    }

    private static func header(
        for c: Conversation, context: ConversationExportContext
    ) -> String {
        """
        # \(c.title)

        | | |
        |---|---|
        | Model | \(context.modelDisplayName) (`\(c.modelID)`) |
        | Started | \(stamp(c.createdAt, in: context.timeZone)) |
        | Last updated | \(stamp(c.updatedAt, in: context.timeZone)) |
        | Exported | \(stamp(context.exportedAt, in: context.timeZone)) · \(context.edition) \(context.appVersion) |
        """
    }

    /// The conversation's items as Markdown blocks. One `### You` per user turn, and a single
    /// `### <model>` heading spanning every assistant bubble and tool card up to the next user
    /// turn — the same folding rule as `Conversation.chatHistory`.
    private static func blocks(
        for items: [TranscriptItem], context: ConversationExportContext
    ) -> [String] {
        var blocks: [String] = []
        var assistantHeadingOpen = false

        func openAssistantHeading() {
            guard !assistantHeadingOpen else { return }
            blocks.append("### \(context.modelDisplayName)")
            assistantHeadingOpen = true
        }

        for item in items {
            switch item {
            case .message(let message):
                switch message.role {
                case .user:
                    blocks.append("### You")
                    blocks.append(verbatim(message.text))
                    // CW-3: attachments by name + mode, so an exported transcript says what the
                    // model was actually given (the contents are in the bubble's text for inline).
                    for attachment in message.attachments {
                        blocks.append("📎 \(attachment.name) (\(attachment.badgeLabel))")
                    }
                    assistantHeadingOpen = false

                case .assistant:
                    let reasoning = splitReasoning(message.text)
                    guard reasoning.reasoning != nil || !reasoning.answer.isEmpty else {
                        // Nothing to show in this bubble. It does **not** close the assistant
                        // turn: only a user message does, so an empty bubble between two tool
                        // cards can't open a second heading for the same turn.
                        continue
                    }
                    openAssistantHeading()
                    if let text = reasoning.reasoning {
                        blocks.append(reasoningBlock(text, isTruncated: reasoning.isTruncated))
                    }
                    if !reasoning.answer.isEmpty {
                        blocks.append(verbatim(reasoning.answer))
                    }
                }

            case .tool(let activity):
                openAssistantHeading()
                blocks.append(toolBlock(activity))
            }
        }
        return blocks
    }

    /// Reasoning folded above the answer (D2). `<details>` is HTML inside Markdown, so no fence
    /// is needed and the raw `<think>` text stays readable.
    ///
    /// The "cut off" note is added on `isTruncated` alone — **not** whenever the answer is
    /// empty. A *closed* think block with no answer is the normal shape right before a tool
    /// call, and saying the reply was cut off there is wrong (CX-A F1).
    private static func reasoningBlock(_ reasoning: String, isTruncated: Bool) -> String {
        var block = """
        <details><summary>Reasoning</summary>

        \(verbatim(reasoning))

        </details>
        """
        if isTruncated {
            block += "\n\n_(reply ended before the reasoning finished)_"
        }
        return block
    }

    /// One tool card, as a Markdown block quote so it reads as an aside rather than as part of
    /// the model's answer. An empty result contributes nothing.
    private static func toolBlock(_ activity: ToolActivity) -> String {
        var lines = ["**Tool · \(activity.name) · \(statusLabel(activity.status))**"]

        if !activity.arguments.isEmpty, let arguments = jsonString(activity.arguments) {
            lines.append("Arguments: `\(arguments)`")
        }
        if let result = activity.result, !result.isEmpty {
            // Blank line before the fence, so the block reads as prose + code, not a table.
            lines.append("")
            lines.append(fencedBlock(result))
        }

        return lines
            .flatMap { $0.components(separatedBy: "\n") }
            .map { $0.isEmpty ? ">" : "> \($0)" }
            .joined(separator: "\n")
    }

    /// The card wording, kept in step with `RunChatView.toolStatusLabel`.
    private static func statusLabel(_ status: ToolActivity.Status) -> String {
        switch status {
        case .running: return "Running…"
        case .finished: return "Done"
        case .failed: return "Failed"
        case .denied: return "Declined"
        case .unknownTool: return "Unknown tool"
        case .budgetExceeded: return "Budget exceeded"
        }
    }

    // MARK: - Reasoning

    /// A bubble's reasoning split out from its answer. `isTruncated` is true **only** for shape 3
    /// (an opening tag with no closing tag) — it means the reply stopped mid-thought, which is
    /// the only case that earns the "ended before the reasoning finished" note (CX-A F1).
    struct ReasoningSplit: Sendable, Equatable {
        let reasoning: String?
        let answer: String
        let isTruncated: Bool
    }

    /// Split an assistant bubble into its reasoning and answer.
    ///
    /// Three shapes occur, and all three must be handled:
    /// 1. `<think>…</think>answer` — the normal case;
    /// 2. `…</think>answer` with **no opening tag** — Qwen3 templates put `<think>` in the
    ///    generation prompt, so the stream starts already inside the block;
    /// 3. `<think>…` with no closing tag — a reply cut off at the token limit; all of it is
    ///    reasoning and the answer is empty.
    ///
    /// `answer` is trimmed, so a whitespace-only bubble reads as empty and is skipped rather
    /// than appended as a blank block. No tags at all → `(nil, text, false)`.
    static func splitReasoning(_ text: String) -> ReasoningSplit {
        let open = "<think>"
        let close = "</think>"

        if let openRange = text.range(of: open) {
            let afterOpen = text[openRange.upperBound...]
            if let closeRange = afterOpen.range(of: close) {
                return ReasoningSplit(
                    reasoning: trimmed(String(afterOpen[..<closeRange.lowerBound])),
                    answer: trimmed(String(afterOpen[closeRange.upperBound...])),
                    isTruncated: false
                )
            }
            // Shape 3 — never closed.
            return ReasoningSplit(
                reasoning: trimmed(String(afterOpen)), answer: "", isTruncated: true)
        }
        if let closeRange = text.range(of: close) {
            // Shape 2 — the opening tag was in the prompt.
            return ReasoningSplit(
                reasoning: trimmed(String(text[..<closeRange.lowerBound])),
                answer: trimmed(String(text[closeRange.upperBound...])),
                isTruncated: false
            )
        }
        return ReasoningSplit(reasoning: nil, answer: trimmed(text), isTruncated: false)
    }

    // MARK: - JSON

    /// The full-fidelity export: the existing `Conversation` Codable, unchanged, wrapped in a
    /// header that identifies the file. Embedding the stored shape keeps it re-importable
    /// without a second schema.
    ///
    /// Date strategy is ISO-8601 throughout — the same strategy `ConversationStore` uses, and
    /// the reason the embedded conversation decodes back with a plain `.iso8601` decoder.
    static func json(_ c: Conversation, context: ConversationExportContext) throws -> Data {
        let root: [String: JSONValue] = [
            "format": .string("mlxui.conversation"),
            "formatVersion": .int(1),
            "exportedAt": .string(iso8601(context.exportedAt)),
            "app": .object([
                "edition": .string(context.edition),
                "version": .string(context.appVersion),
            ]),
            "model": .object([
                "id": .string(c.modelID),
                "displayName": .string(context.modelDisplayName),
            ]),
            "conversation": try jsonValue(c),
        ]

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        do {
            return try encoder.encode(root)
        } catch {
            throw ExportError.encodingFailed(error.localizedDescription)
        }
    }

    /// Route a `Codable` value through `JSONEncoder`/`JSONDecoder` to get a `JSONValue`, using
    /// the same ISO-8601 date strategy as the rest of the file.
    private static func jsonValue(_ value: some Encodable) throws -> JSONValue {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        let data = try encoder.encode(value)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(JSONValue.self, from: data)
    }

    // MARK: - Filename

    /// `<modelDisplayName> - <title> - <yyyy-MM-dd>.<ext>`, with the filesystem-hostile
    /// characters replaced so the name is usable on every volume, and the title part capped.
    static func suggestedFilename(
        _ c: Conversation, context: ConversationExportContext, ext: String
    ) -> String {
        let day = formatted(c.updatedAt, format: "yyyy-MM-dd", timeZone: context.timeZone)
        let model = sanitized(context.modelDisplayName)
        let title = capped(sanitized(c.title), at: titleCharacterLimit)
        return "\(model) - \(title) - \(day).\(ext)"
    }

    /// Replace `/ : \ ? * " < > |` and newlines with `-`, collapse each run of **adjacent**
    /// dashes, and drop trailing dashes. Spaces are kept verbatim: they are legal in a
    /// filename, and the backlog's own example (`a/b: c\d? e*f"g<h>i|j\nk`) keeps them
    /// (`a-b- c-d- e-f-g-h-i-j-k`) — collapsing spaces to dashes would erase that. So
    /// "collapse repeated spaces and dashes" is read as "collapse runs of dashes", the most
    /// conservative reading available. Implementer's call, pending owner confirmation.
    private static func sanitized(_ raw: String) -> String {
        var replaced = ""
        replaced.reserveCapacity(raw.count)
        for character in raw {
            if character.isNewline || character == "/" || character == ":" || character == "\\"
                || character == "?" || character == "*" || character == "\""
                || character == "<" || character == ">" || character == "|" {
                replaced.append("-")
            } else {
                replaced.append(character)
            }
        }

        var collapsed = ""
        collapsed.reserveCapacity(replaced.count)
        var previousWasDash = false
        for character in replaced {
            if character == "-" {
                if !previousWasDash {
                    collapsed.append(character)
                }
                previousWasDash = true
            } else {
                previousWasDash = false
                collapsed.append(character)
            }
        }

        while let last = collapsed.last, last == "-" || last.isWhitespace {
            collapsed = String(collapsed.dropLast())
        }
        return collapsed
    }

    /// Cap one filename component, adding an ellipsis when it was cut.
    private static func capped(_ text: String, at limit: Int) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(limit)) + "…"
    }

    // MARK: - Shared formatting

    /// Text with only trailing whitespace trimmed, so a bubble never leaves a run of blank
    /// lines in the middle of the document. Otherwise verbatim: model output is already
    /// Markdown, and user text is not escaped.
    private static func verbatim(_ text: String) -> String {
        var trimmed = Substring(text)
        while let last = trimmed.last, last.isWhitespace {
            trimmed = trimmed.dropLast()
        }
        return String(trimmed)
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A fenced code block whose fence is one backtick longer than the longest run of backticks
    /// inside the body, so a result containing a fence can't break out of its own block.
    private static func fencedBlock(_ text: String) -> String {
        let body = trimmed(text)
        var longestRun = 0
        var run = 0
        for character in body {
            if character == "`" {
                run += 1
                longestRun = max(longestRun, run)
            } else {
                run = 0
            }
        }
        let fence = String(repeating: "`", count: max(3, longestRun + 1))
        return "\(fence)\n\(body)\n\(fence)"
    }

    /// Compact, key-sorted JSON for a tool card's arguments.
    private static func jsonString(_ arguments: [String: JSONValue]) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes, .sortedKeys]
        guard let data = try? encoder.encode(arguments) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private static func stamp(_ date: Date, in timeZone: TimeZone) -> String {
        "\(formatted(date, format: "yyyy-MM-dd HH:mm", timeZone: timeZone)) "
            + "(\(formatted(date, format: "ZZZZZ", timeZone: timeZone)))"
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        return formatter.string(from: date)
    }

    private static func formatted(_ date: Date, format: String, timeZone: TimeZone) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = timeZone
        formatter.dateFormat = format
        return formatter.string(from: date)
    }
}
