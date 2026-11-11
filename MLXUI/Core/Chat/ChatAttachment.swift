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
    /// MT4-2: the handle (`a1`) this file was registered under in the chat's `ChatAssetStore`, and its `Kind` raw value,
    /// when Model Tools was on for the turn. `nil` for every CW (Model Tools off) attachment, and for a file that is not
    /// text, image or audio. Both are optional, so a conversation saved before MT4-2 decodes unchanged.
    let assetHandle: String?
    let assetKind: String?

    init(
        id: UUID = UUID(),
        name: String,
        storedPath: String,
        mode: ChatAttachmentMode,
        byteCount: Int,
        inlineText: String? = nil,
        assetHandle: String? = nil,
        assetKind: String? = nil
    ) {
        self.id = id
        self.name = name
        self.storedPath = storedPath
        self.mode = mode
        self.byteCount = byteCount
        self.inlineText = inlineText
        self.assetHandle = assetHandle
        self.assetKind = assetKind
    }

    /// This attachment, registered as asset `handle` of `kind`.
    func registered(handle: String, kind: String) -> ChatAttachment {
        ChatAttachment(id: id, name: name, storedPath: storedPath, mode: mode, byteCount: byteCount,
                       inlineText: inlineText, assetHandle: handle, assetKind: kind)
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
    /// Absolute per-file inline cap (B2). The window-based budget alone lets a 262 144-token
    /// catalog entry inline ~256 KB, and `chatHistory` replays that text on every later turn with
    /// no prompt cache — a minute-long prefill per follow-up on a 4-bit 8–9B model.
    static let maxInlineBytesPerFile = 48 * 1024
    /// Absolute per-message inline cap (B2).
    static let maxInlineBytesPerMessage = 96 * 1024
    /// Per-file copy cap.
    static let maxFileBytes = 50 * 1024 * 1024
    /// Files per message.
    static let maxFilesPerMessage = 10

    /// How many bytes one file may be inlined at: the window's share, capped absolutely (B2).
    static func inlineByteBudget(window: Int) -> Int {
        min(max(minimumInlineBytes, window / inlineWindowFraction * 4), maxInlineBytesPerFile)
    }

    /// How many inlined bytes one message may carry: the window's share, capped absolutely (B2).
    static func messageInlineByteBudget(window: Int) -> Int {
        min(window / inlineTotalFraction * 4, maxInlineBytesPerMessage)
    }

    /// The mode a file of `byteCount` should use, given the window and how much of it this message
    /// has already spent on inlined text.
    static func mode(
        byteCount: Int,
        window: Int,
        inlineBytesSoFar: Int
    ) -> ChatAttachmentMode {
        guard byteCount <= inlineByteBudget(window: window) else { return .reference }
        guard inlineBytesSoFar + byteCount <= messageInlineByteBudget(window: window) else {
            return .reference
        }
        return .inline
    }

    // MARK: - Model-facing rendering (C13)

    /// One attachment as the model sees it, appended to the user turn. The attribute values are
    /// escaped: a file called `a"b.txt` would otherwise close the tag early (B6).
    static func modelFacing(_ attachment: ChatAttachment) -> String {
        let name = escapedAttribute(attachment.name)
        // MT4-2: a registered file is named by its handle (the router passes it to a tool), and a by-reference one carries no
        // path: a model told only a handle cannot try `read_file` on a 2 MB file (≈65k tokens, minutes of prefill).
        let asset = attachment.assetHandle.map { " asset=\"\(escapedAttribute($0))\"" } ?? ""
        switch attachment.mode {
        case .inline, .ocr:
            return "<attachment name=\"\(name)\"\(asset)>\n\(attachment.inlineText ?? "")\n</attachment>"
        case .reference:
            if let handle = attachment.assetHandle {
                let kind = attachment.assetKind.map { " kind=\"\(escapedAttribute($0))\"" } ?? ""
                return "<attachment name=\"\(name)\" asset=\"\(escapedAttribute(handle))\"\(kind) mode=\"reference\"/>"
            }
            let path = escapedAttribute(attachment.storedPath)
            return "<attachment name=\"\(name)\" path=\"\(path)\" mode=\"reference\"/>"
        }
    }

    /// `&` first, then the quote — the order matters, or an `&quot;` would be re-escaped.
    static func escapedAttribute(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }

    /// The message **as the model receives it**: the user's text plus one block per attachment.
    /// `ChatAttachment.modelFacing` is the renderer; this is the name the caller should use, so
    /// "the model gets what the transcript shows" is one call and not a second decision.
    static func prompt(text: String, attachments: [ChatAttachment]) -> String {
        modelFacing(text: text, attachments: attachments)
    }

    /// The whole user turn for a message: its text plus one block per attachment.
    static func modelFacing(text: String, attachments: [ChatAttachment]) -> String {
        guard !attachments.isEmpty else { return text }
        return ([text] + attachments.map(modelFacing)).joined(separator: "\n\n")
    }

    // MARK: - The environment note (C3's `attachmentsNote`)

    /// True when this reference is an image we could not OCR. `read_file` on a PNG only ever
    /// answers "not UTF-8 text", so the model must not be told to try (B3).
    var isUnreadImage: Bool {
        mode == .reference && assetHandle == nil && Self.imageExtensions.contains(
            URL(fileURLWithPath: name).pathExtension.lowercased())
    }

    /// The chip badge: reference images that failed OCR read "no OCR" rather than "reference".
    var badgeLabel: String {
        if mode == .reference, assetHandle != nil { return "asset" }
        return isUnreadImage ? "no OCR" : mode.rawValue
    }

    static let imageExtensions: Set<String> = [
        "png", "jpg", "jpeg", "heic", "heif", "tiff", "tif", "gif", "bmp", "webp",
    ]

    /// One short line per by-reference attachment, for `ChatEnvironment.instructions`. Inlined and
    /// OCR'd files are already in the user message, so they need no line.
    static func environmentNote(for attachments: [ChatAttachment]) -> String? {
        // MT4-2: a file registered as an asset is described by `assetNote` instead (a handle, not a path).
        let referenced = attachments.filter { $0.mode == .reference && $0.assetHandle == nil }
        guard !referenced.isEmpty else { return nil }
        let lines = referenced.map { attachment -> String in
            guard attachment.isUnreadImage else {
                return "Attached file \(attachment.name) is at \(attachment.storedPath);"
                    + " read it with read_file."
            }
            return "Attached image \(attachment.name) could not be read: no OCR model is"
                + " installed. Tell the user to install one from Browse."
        }
        return lines.joined(separator: "\n")
    }

    /// MT4-2: one short line per by-reference attachment that was registered as an asset (Model Tools on). It names the
    /// handle and the way to use it; a text file is read a page at a time through `read_asset`, never whole. Inlined files
    /// are already in the message and carry their handle in the tag, so they need no line.
    static func assetNote(for attachments: [ChatAttachment], assets: ChatAssetStore?) -> String? {
        let lines = attachments.compactMap { attachment -> String? in
            guard attachment.mode == .reference, let handle = attachment.assetHandle else { return nil }
            switch attachment.assetKind {
            case "image":
                return "Attached image \(attachment.name) is asset \(handle) (image)."
                    + " Pass asset \"\(handle)\" to an image tool."
            case "audio":
                return "Attached audio \(attachment.name) is asset \(handle) (audio)."
                    + " Pass asset \"\(handle)\" to transcribe_audio."
            default:
                let words = assets?.record(handle)?.words
                let size = words.map { ", \($0) words" } ?? ""
                return "Attached file \(attachment.name) is asset \(handle) (text\(size))."
                    + " It is not in this message: call read_asset with asset \"\(handle)\", a page at a time."
            }
        }
        return lines.isEmpty ? nil : lines.joined(separator: "\n")
    }
}
