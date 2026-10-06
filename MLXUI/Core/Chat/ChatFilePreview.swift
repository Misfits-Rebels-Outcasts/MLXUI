//
//  ChatFilePreview.swift
//  MLXUI — chat workspace, CW-5 (`RSI/DelegateChatWorkspaceBacklog.md`).
//
//  What the inspector's preview pane shows for one touched file, and how it reads it safely.
//
//  Two sandbox facts drive the shape:
//  * a read of a file under a grant must run inside that grant's security scope — the same
//    `withAccess` the file tools use, so a revoked folder refuses here too;
//  * Quick Look reads **lazily while it is on screen**, so its scope is held for as long as the
//    preview is visible (`FilePreviewScope`), not for the length of a read.
//

import Foundation
import AppKit

nonisolated enum ChatFilePreview {

    /// The pane's content.
    enum Content: Equatable {
        case markdown(String)
        /// Markdown longer than the cap: clipped, and the view shows the same notice as text.
        case truncatedMarkdown(String, shownBytes: Int, totalBytes: Int)
        case text(String)
        /// Text longer than the cap, with what was shown so the view can say so.
        case truncatedText(String, shownBytes: Int, totalBytes: Int)
        case image
        case quickLook
        /// Could not be read (revoked grant, missing file, no scope).
        case unavailable(String)
    }

    /// The cap on inlined text: 64 KB, with a notice rather than a silent cut.
    static let textCap = 64 * 1024

    /// The most the preview ever reads: one byte past the cap, so "there is more" is knowable
    /// without reading the rest of a 300 MB log.
    static let readLimit = textCap + 1

    /// One bounded read.
    struct FileRead {
        /// At most `readLimit` bytes, or empty when the read failed.
        let data: Data
        /// The file's real size, from `.fileSizeKey` — not `data.count`.
        let fileSize: Int
        /// A user-facing reason the read failed, if it did.
        let error: String?
    }

    /// Read at most `readLimit` bytes with a `FileHandle`, so clicking a row in a folder the model
    /// listed never loads the whole file. `injected` is the test seam (nil in the app); the real
    /// read runs through the grant's scope.
    ///
    /// `nonisolated` and `static` on purpose: the view calls it from a detached task, so a big file
    /// never blocks the main actor.
    nonisolated static func read(
        path: String,
        injected: ((URL, Int) -> FileRead)? = nil
    ) -> FileRead {
        let url = URL(fileURLWithPath: path)
        if let injected { return injected(url, readLimit) }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        let data = FolderListing.scopedData(at: url, limit: readLimit)
        guard let data else {
            return FileRead(data: Data(), fileSize: size,
                            error: "This file could not be read.")
        }
        return FileRead(data: data, fileSize: max(size, data.count), error: nil)
    }

    /// Decide the pane's content. `read` is what `read(path:)` returned, so this stays pure.
    static func content(
        for url: URL,
        presentation: SavedFilePresentation,
        read: FileRead
    ) -> Content {
        if let error = read.error { return .unavailable(error) }

        switch presentation {
        case .image:
            return .image
        case .text, .web, .table:
            guard let text = decodedText(read.data) else { return .quickLook }
            let clipped = clip(read.data)
            guard let clippedText = decodedText(clipped) else { return .quickLook }
            let isClipped = read.fileSize > textCap || read.data.count > textCap
            if isMarkdown(url) {
                return isClipped
                    ? .truncatedMarkdown(clippedText, shownBytes: clipped.count,
                                         totalBytes: read.fileSize)
                    : .markdown(clippedText)
            }
            _ = text
            return isClipped
                ? .truncatedText(clippedText, shownBytes: clipped.count, totalBytes: read.fileSize)
                : .text(clippedText)
        case .pdf, .audio, .video, .other, .folder:
            return .quickLook
        }
    }

    /// The first `textCap` bytes, backed off to a UTF-8 boundary so a cut multi-byte character does
    /// not make the whole preview fall back to Quick Look. At most 3 bytes are dropped — a UTF-8
    /// sequence is at most 4 bytes long.
    static func clip(_ data: Data) -> Data {
        guard data.count > textCap else { return data }
        var end = textCap
        var backedOff = 0
        while end > 0, backedOff < 3, isContinuationByte(data[data.startIndex + end]) {
            end -= 1
            backedOff += 1
        }
        return data.prefix(end)
    }

    /// A UTF-8 continuation byte is `10xxxxxx`.
    static func isContinuationByte(_ byte: UInt8) -> Bool {
        byte & 0b1100_0000 == 0b1000_0000
    }

    static func decodedText(_ data: Data) -> String? {
        String(data: data, encoding: .utf8)
    }

    static func isMarkdown(_ url: URL) -> Bool {
        ["md", "markdown", "mdown"].contains(url.pathExtension.lowercased())
    }

    /// A byte-count sentence for the clipped case: `64 KB of 210 KB shown`.
    static func truncationNotice(shownBytes: Int, totalBytes: Int) -> String {
        "\(kilobytes(shownBytes)) of \(kilobytes(totalBytes)) shown"
    }

    private static func kilobytes(_ bytes: Int) -> String {
        let kb = Double(bytes) / 1024
        return kb >= 1024
            ? String(format: "%.1f MB", kb / 1024)
            : String(format: "%.0f KB", kb)
    }
}

/// Holds a security-scoped URL open for as long as the object lives — Quick Look needs the scope
/// while the preview is on screen, not just while it is being read
/// (`RSI/DelegateOutputViewerBacklog.md` §7: the viewer reads lazily).
///
/// A path under the app container needs no scope and constructs as an inert value.
final class FilePreviewScope {
    private let url: URL
    private let didStart: Bool

    init?(path: String) {
        let file = URL(fileURLWithPath: path)
        let container = ModelStore.shared.baseDirectory
            .standardizedFileURL.resolvingSymlinksInPath().path
        let target = file.standardizedFileURL.resolvingSymlinksInPath().path
        if target == container || target.hasPrefix("\(container)/") {
            self.url = file
            self.didStart = false
            return
        }
        #if DIRECT_BUILD
        self.url = file
        self.didStart = false
        #else
        guard let grant = FolderGrants.grant(covering: file.path, in: FolderGrants.load()),
              !grant.bookmark.isEmpty
        else { return nil }
        var isStale = false
        guard let scoped = try? URL(
            resolvingBookmarkData: grant.bookmark,
            options: .withSecurityScope,
            relativeTo: nil,
            bookmarkDataIsStale: &isStale)
        else { return nil }
        let started = scoped.startAccessingSecurityScopedResource()
        self.url = scoped
        self.didStart = started
        #endif
    }

    /// The URL to hand to Quick Look or `NSWorkspace` — the resolved one, so the scope applies.
    var scopedURL: URL { url }

    deinit {
        #if !DIRECT_BUILD
        if didStart { url.stopAccessingSecurityScopedResource() }
        #endif
    }
}
