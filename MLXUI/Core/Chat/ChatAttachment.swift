//
//  ChatAttachment.swift
//  MLXUI — chat workspace, CW-3 (`RSI/DelegateChatWorkspaceBacklog.md`, C11/C13).
//
//  One file the user attached to a message. A **text-only** chat model (the whole chat surface
//  today — `ModelRunner.run` sends vision/OCR elsewhere) can only be given text, so an attachment
//  is either inlined into the user turn, referenced by its container path, or — for an image —
//  OCR'd and then inlined. `ChatAttachment` records **exactly what was sent**, so
//  `Conversation.chatHistory` stays pure and replays the identical text after a relaunch.
//
//  Pure value type + statics, `nonisolated`: the budget decision and the rendering are unit-tested
//  without a model or a file.
//

import Foundation

/// How an attached file reached the model (C13).
nonisolated enum ChatAttachmentMode: String, Codable, Sendable, CaseIterable {
    /// Its text was put in the user message.
    case inline
    /// Only its path was given; the model reads it with `read_file`.
    case reference
    /// An image: `ocr_image` transcribed it, and the text was inlined.
    case ocr

    /// The chip badge in the composer and the export line.
    var label: String { rawValue }
}

nonisolated struct ChatAttachment: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    /// The name the user saw (`rates.csv`).
    let name: String
    /// Where the copy lives inside the container. Always a path the file tools can read, so a
    /// by-reference attachment survives a relaunch (C10).
    let storedPath: String
    let mode: ChatAttachmentMode
    let byteCount: Int
    /// Exactly the text sent for `.inline` / `.ocr`; `nil` for `.reference`.
    let inlineText: String?

    init(
        id: UUID = UUID(),
        name: String,
        storedPath: String,
        mode: ChatAttachmentMode,
        byteCount: Int,
        inlineText: String? = nil
    ) {
        self.id = id
        self.name = name
        self.storedPath = storedPath
        self.mode = mode
        self.byteCount = byteCount
        self.inlineText = inlineText
    }

    // MARK: - The budget (C11)

    /// Estimated tokens: `ceil(utf8.count / 4)`. An estimate on purpose — the app's convention for
    /// estimates is a `~`, and an exact tokenizer count is a later item.
    static func estimatedTokens(forUTF8Count count: Int) -> Int {
        (count + 3) / 4
    }

    /// A file at most this many bytes is *eligible* to be inlined, whatever the window.
    static let minimumInlineBytes = 8_000
    /// Never inline a file bigger than a quarter of the window.
    static let inlineWindowFraction = 4
    /// Never let one message's inlined files exceed a third of the window.
    static let inlineTotalFraction = 3
    /// Per-file copy cap.
    static let maxFileBytes = 50 * 1024 * 1024
    /// Files per message.
    static let maxFilesPerMessage = 10

    /// How many bytes this window allows one file to be, when inlined.
    static func inlineByteBudget(window: Int) -> Int {
        max(minimumInlineBytes, window / inlineWindowFraction * 4)
    }

    /// The mode a file of `byteCount` should use, given the window and how much of it this message
    /// has already spent on inlined text.
    static func mode(
        byteCount: Int,
        window: Int,
        inlineBytesSoFar: Int
    ) -> ChatAttachmentMode {
        guard byteCount <= inlineByteBudget(window: window) else { return .reference }
        let totalBudget = window / inlineTotalFraction * 4   // window/3 tokens → bytes
        guard inlineBytesSoFar + byteCount <= totalBudget else { return .reference }
        return .inline
    }

    // MARK: - Model-facing rendering (C13)

    /// One attachment as the model sees it, appended to the user turn.
    static func modelFacing(_ attachment: ChatAttachment) -> String {
        switch attachment.mode {
        case .inline, .ocr:
            return "<attachment name=\"\(attachment.name)\">\n\(attachment.inlineText ?? "")\n</attachment>"
        case .reference:
            return "<attachment name=\"\(attachment.name)\" path=\"\(attachment.storedPath)\" mode=\"reference\"/>"
        }
    }

    /// The whole user turn for a message: its text plus one block per attachment.
    static func modelFacing(text: String, attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return text }
        return ([text] + attachments.map(modelFacing)).joined(separator: "\n\n")
    }

    // MARK: - The environment note (C3's `attachmentsNote`)

    /// One short line per by-reference attachment, for `ChatEnvironment.instructions`. Inlined and
    /// OCR'd files are already in the user message, so they need no line.
    static func environmentNote(for attachments: [ChatAttachment]) -> String? {
        let referenced = attachments.filter { $0.mode == .reference }
        guard !referenced.isEmpty else { return nil }
        let lines = referenced.map { "Attached file \($0.name) is at \($0.storedPath); read it with read_file." }
        return lines.joined(separator: "\n")
    }
}
