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

    /// Decide the pane's content for `url`. `data` is read by the caller through the scoped read so
    /// this stays pure and testable; `nil` means the read failed and `error` says why.
    static func content(
        for url: URL,
        presentation: SavedFilePresentation,
        data: Data?,
        error: String?
    ) -> Content {
        if let error { return .unavailable(error) }
        guard let data else { return .unavailable("This file could not be read.") }

        switch presentation {
        case .image:
            return .image
        case .text, .web, .table:
            guard let text = String(data: data, encoding: .utf8) else {
                return .quickLook
            }
            if isMarkdown(url) { return .markdown(text) }
            let clipped = data.prefix(textCap)
            guard let clippedText = String(data: clipped, encoding: .utf8) else {
                return .quickLook
            }
            return data.count > textCap
                ? .truncatedText(clippedText, shownBytes: clipped.count, totalBytes: data.count)
                : .text(clippedText)
        case .pdf, .audio, .video, .other, .folder:
            return .quickLook
        }
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
