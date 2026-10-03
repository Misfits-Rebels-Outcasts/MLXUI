//
//  ConversationExportPanel.swift
//  MLXUI — download a chat transcript (CX-2, `RSI/DelegateChatExportBacklog.md`).
//
//  The AppKit half of the export: the `NSSavePanel`, the clipboard, and the export context the
//  pure `ConversationExporter` needs. Kept out of `RunChatView` so the view keeps its state and
//  this stays testable in isolation. Follows the save-panel precedent in the run views
//  (`TTSRunView.save()` and the six others).
//
//  The export is user-initiated and writes only where the user points it. Nothing here touches
//  the network, so the RM-6 privacy disclosures need no change.
//

import AppKit
import Foundation
import UniformTypeIdentifiers

/// Save-panel + clipboard helpers for one conversation export.
@MainActor
enum ConversationExportPanel {

    enum ExportFailure: LocalizedError, Equatable {
        case writeFailed(String)
        case encodingFailed(String)

        var errorDescription: String? {
            switch self {
            case .writeFailed(let message): return "Couldn't save the chat: \(message)"
            case .encodingFailed(let message): return "Couldn't export the chat: \(message)"
            }
        }
    }

    /// Markdown's content type. `UTType(filenameExtension: "md")` is `.md` on macOS 14, but the
    /// initializer is optional, so fall back to plain text rather than force-unwrapping.
    private static var markdownType: UTType {
        UTType(filenameExtension: "md") ?? .plainText
    }

    /// The values the exporter can't derive from the conversation. Injected here (and never in
    /// tests) so the export is reproducible only in the sense the app wants: "when I clicked".
    static func context(
        modelDisplayName: String, exportedAt: Date = Date()
    ) -> ConversationExportContext {
        let info = Bundle.main.infoDictionary
        let version = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String
        let appVersion = build.map { "\(version) (\($0))" } ?? version
        #if APPSTORE_BUILD
        let edition = "AI Browser"
        #else
        let edition = "AI Browser Pro"
        #endif
        return ConversationExportContext(
            modelDisplayName: modelDisplayName,
            appVersion: appVersion,
            edition: edition,
            exportedAt: exportedAt
        )
    }

    /// *Save as Markdown…* — run the panel and write the transcript.
    static func saveMarkdown(_ conversation: Conversation, context: ConversationExportContext) throws {
        let text = ConversationExporter.markdown(conversation, context: context)
        try save(
            data: Data(text.utf8),
            contentType: markdownType,
            suggestedName: ConversationExporter.suggestedFilename(
                conversation, context: context, ext: "md")
        )
    }

    /// *Save as JSON…* — the full-fidelity copy, wrapping the stored `Conversation` unchanged.
    static func saveJSON(_ conversation: Conversation, context: ConversationExportContext) throws {
        let data: Data
        do {
            data = try ConversationExporter.json(conversation, context: context)
        } catch {
            throw ExportFailure.encodingFailed(error.localizedDescription)
        }
        try save(
            data: data,
            contentType: .json,
            suggestedName: ConversationExporter.suggestedFilename(
                conversation, context: context, ext: "json")
        )
    }

    /// *Copy as Markdown* — the same text the `.md` export writes.
    static func copyMarkdown(_ conversation: Conversation, context: ConversationExportContext) {
        let text = ConversationExporter.markdown(conversation, context: context)
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    /// Present the panel and write atomically. A cancelled panel is not an error.
    private static func save(
        data: Data, contentType: UTType, suggestedName: String
    ) throws {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [contentType]
        panel.nameFieldStringValue = suggestedName
        panel.canCreateDirectories = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw ExportFailure.writeFailed(error.localizedDescription)
        }
    }
}
