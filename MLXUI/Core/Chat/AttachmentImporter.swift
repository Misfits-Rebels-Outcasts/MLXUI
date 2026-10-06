//
//  AttachmentImporter.swift
//  MLXUI — chat workspace, CW-3 (`RSI/DelegateChatWorkspaceBacklog.md`, C10/C12).
//
//  Turns a file the user picked (or dropped) into a `ChatAttachment`:
//
//   1. copy it into `conversations/<uuid>/attachments/<n>-<name>` — a panel-picked file gives the
//      sandbox only *temporary* access, and the container is always a root, so `read_file` can
//      re-read the copy after a relaunch (C10);
//   2. extract text where the type allows it (UTF-8 text/code/CSV/JSON/MD, PDF via PDFKit, RTF and
//      docx via `NSAttributedString`, an image via the OCR tool's backend);
//   3. apply the C11 budget: inline when it fits, otherwise reference the path.
//
//  The decision inputs (`window`, `inlineBytesSoFar`) are passed in, so the budget behaviour is
//  unit-tested without a model and the file work is unit-tested against temp directories.
//

import Foundation
import AppKit
import PDFKit
import MLXLMCommon

nonisolated enum AttachmentImporter {

    /// Why a file could not be attached at all. Model- and user-facing.
    enum Failure: Error, Equatable {
        case tooLarge(Int)
        case tooMany(Int)

        var message: String {
            switch self {
            case .tooLarge(let limit):
                return "That file is too large to attach (limit \(limit / (1024 * 1024)) MB)."
            case .tooMany(let limit):
                return "You can attach at most \(limit) files to one message."
            }
        }
    }

    /// Text an attachment's own contents can contribute, or `nil` when there is none (a by-reference
    /// binary, say). Kept separate from the file copy so it is testable and `nonisolated`.
    static func extractedText(at url: URL) -> String? {
        if let text = utf8Text(at: url) { return text }
        if let text = pdfText(at: url) { return text }
        if let text = attributedText(at: url) { return text }
        return nil
    }

    /// Plain UTF-8, with a sanity check: a binary file often decodes as *something*, so treat a NUL
    /// byte as "not text" (the same signal `file(1)` uses).
    static func utf8Text(at url: URL) -> String? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        guard !data.contains(0) else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// A text-layer PDF. A scanned PDF has none → `nil` → the file goes by reference (C12).
    static func pdfText(at url: URL) -> String? {
        guard url.pathExtension.lowercased() == "pdf" else { return nil }
        let text = PDFDocument(url: url)?.string?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (text?.isEmpty ?? true) ? nil : text
    }

    /// RTF / RTFD / docx via AppKit's attributed-string reader.
    static func attributedText(at url: URL) -> String? {
        let ext = url.pathExtension.lowercased()
        guard ["rtf", "rtfd", "docx", "doc"].contains(ext) else { return nil }
        guard let attributed = try? NSAttributedString(
            url: url, options: [:], documentAttributes: nil)
        else { return nil }
        let text = attributed.string.trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    /// Whether a file is an image we could OCR. Pure, so the branch is unit-testable.
    static func isImage(_ url: URL) -> Bool {
        let ext = url.pathExtension.lowercased()
        return ["png", "jpg", "jpeg", "heic", "heif", "tiff", "tif", "gif", "bmp", "webp"]
            .contains(ext)
    }

    /// Import one file. `inlineBytesSoFar` is the running total for this message (C11's per-message
    /// cap); the caller adds the returned attachment's inlined size before the next file.
    ///
    /// `ocr` is injected: production passes `OCRImageTool.recognizeText(in:)`, a test passes a stub,
    /// and `nil` means "no OCR model installed" → an image goes by reference with a note.
    static func importAttachment(
        from source: URL,
        into directory: URL,
        ordinal: Int,
        window: Int,
        inlineBytesSoFar: Int,
        ocr: (@Sendable (URL) async -> String?)? = nil,
        /// K1: how to copy. The default is `copy(_:into:ordinal:)`; a caller whose source needs a
        /// security scope passes a copier that wraps it (see `ModelRunner`).
        copier: ((URL, URL, Int) throws -> URL)? = nil
    ) async throws -> ChatAttachment {
        let byteCount = fileSize(at: source)
        guard byteCount <= ChatAttachment.maxFileBytes else {
            throw Failure.tooLarge(ChatAttachment.maxFileBytes)
        }
        let stored: URL
        if let copier {
            let fm = FileManager.default
            let destinationDirectory = directory
            try fm.createDirectory(at: destinationDirectory, withIntermediateDirectories: true)
            stored = try copier(source, destinationDirectory, ordinal)
        } else {
            stored = try copy(source, into: directory, ordinal: ordinal)
        }
        let name = source.lastPathComponent

        if isImage(source), let ocr {
            let text = await ocr(stored)?.trimmingCharacters(in: .whitespacesAndNewlines)
            if let text, !text.isEmpty {
                return ChatAttachment(
                    name: name, storedPath: stored.path, mode: .ocr,
                    byteCount: byteCount, inlineText: text)
            }
            return ChatAttachment(
                name: name, storedPath: stored.path, mode: .reference,
                byteCount: byteCount)
        }

        let text = extractedText(at: stored)
        let mode = ChatAttachment.mode(
            byteCount: text?.utf8.count ?? byteCount,
            window: window,
            inlineBytesSoFar: inlineBytesSoFar)
        switch (mode, text) {
        case (.inline, .some(let text)):
            return ChatAttachment(
                name: name, storedPath: stored.path, mode: .inline,
                byteCount: byteCount, inlineText: text)
        default:
            return ChatAttachment(
                name: name, storedPath: stored.path, mode: .reference,
                byteCount: byteCount)
        }
    }

    // MARK: - File work

    /// Copy the file into the conversation's attachment folder, keeping its name and avoiding a
    /// collision with an ordinal prefix.
    static func copy(_ source: URL, into directory: URL, ordinal: Int) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent("\(ordinal)-\(source.lastPathComponent)")
        if fm.fileExists(atPath: destination.path) {
            try? fm.removeItem(at: destination)
        }
        try fm.copyItem(at: source, to: destination)
        return destination
    }

    static func fileSize(at url: URL) -> Int {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return values?.fileSize ?? 0
    }
}
