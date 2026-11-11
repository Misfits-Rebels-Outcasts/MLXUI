//
//  ChatAttachmentAssets.swift
//  MLXUI — Model Tools, MT4-2 (owner ruling I: "attachments are asset handles with no eager OCR when Model Tools is on").
//
//  Registers a file the user attached as an asset of the conversation (`a1`, `a2`, …), so a router can pass it
//  between tools by handle. Only when Model Tools is on for the turn; with it off, attachments behave exactly as
//  CW-3 made them (inline / reference / eager OCR) and nothing here runs.
//
//   - image → an `image` asset (no OCR here: `ocr_image` runs when the router asks for it);
//   - audio → an `audio` asset (`transcribe_audio` runs when the router asks for it);
//   - anything with extractable text (UTF-8 text/code/CSV/JSON/MD, text-layer PDF, RTF/docx) → a `text` asset, so a
//     file too big for the message is read a page at a time through `read_asset` instead of whole through `read_file`
//     (a 2.2 MB CSV is about 65,000 tokens, about five minutes of prefill on a 9B model);
//   - anything else (a zip, a video) is not registered and keeps its CW path.
//

import Foundation

nonisolated enum ChatAttachmentAssets {

    static let audioExtensions: Set<String> = [
        "m4a", "mp3", "wav", "aac", "flac", "ogg", "oga", "opus", "aif", "aiff", "aifc", "caf",
    ]

    /// `image` or `audio` for a file name with a known media extension; `nil` otherwise (text is decided by its content).
    static func mediaKind(forFileNamed name: String) -> String? {
        let ext = URL(fileURLWithPath: name).pathExtension.lowercased()
        if ChatAttachment.imageExtensions.contains(ext) { return "image" }
        if audioExtensions.contains(ext) { return "audio" }
        return nil
    }

    /// Register `attachment` (already copied to `storedPath`) in `store`. Returns it carrying its handle, or unchanged when it is
    /// not an asset kind or the copy fails (a failure is never a reason to lose the message).
    static func register(_ attachment: ChatAttachment, in store: ChatAssetStore) -> ChatAttachment {
        let stored = URL(fileURLWithPath: attachment.storedPath)
        if let kind = mediaKind(forFileNamed: attachment.name) {
            guard let record = try? store.importFile(stored, kind: kind, name: attachment.name) else { return attachment }
            return attachment.registered(handle: record.handle, kind: kind)
        }
        let text = attachment.inlineText ?? AttachmentImporter.extractedText(at: stored)
        guard let text, !text.isEmpty,
              let record = try? store.addText(text, name: attachment.name) else { return attachment }
        return attachment.registered(handle: record.handle, kind: "text")
    }

    /// Every attachment of the turn registered, in order.
    static func register(_ attachments: [ChatAttachment], in store: ChatAssetStore) -> [ChatAttachment] {
        attachments.map { register($0, in: store) }
    }
}
