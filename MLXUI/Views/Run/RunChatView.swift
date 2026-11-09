import SwiftUI

/// Chat sheet for running an installed LLM locally via MLX — a fixed model, opened from Browse.
///
/// MT2-1: the body moved to `Views/Chat/ChatView.swift` (which takes the model from `ChatController`). This is the
/// sheet wrapper: it binds the bench surface, owns the sheet's size, stops the turn when the sheet goes away, and
/// keeps the statics other code and the tests call.
struct RunChatView: View {
    let model: ModelEntry

    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ChatView(surface: .bench, initialModel: model, onDone: { dismiss() })
            // CW-4-FIX-1: a plain column in the layout, sized for a sheet (see `ChatView`).
            .frame(
                minWidth: 840, idealWidth: 1100, maxWidth: .infinity,
                minHeight: 520, idealHeight: 620, maxHeight: .infinity)
            .onDisappear {
                appState.chat.stop(from: .bench)
                appState.chat.release(.bench)
            }
    }

    // The statics below stay on `RunChatView` (callers and tests name it); they live in `ChatView`.

    #if !DIRECT_BUILD
    static func grantCandidates(
        attached: [String], granted: [String] = FolderGrants.grantedPaths()
    ) -> [String] {
        ChatView.grantCandidates(attached: attached, granted: granted)
    }
    #else
    static func recentFolderCandidates(attached: [String]) -> [String] {
        ChatView.recentFolderCandidates(attached: attached)
    }
    #endif

    static func uniquelySpelled(_ paths: [String]) -> [String] { ChatView.uniquelySpelled(paths) }

    static func folderLabel(_ path: String, among paths: [String] = []) -> String {
        ChatView.folderLabel(path, among: paths)
    }

    static func restoredDraft(accepted: Bool, typed: String, files: [URL],
                              currentPrompt: String, currentFiles: [URL]) -> (prompt: String, files: [URL])? {
        ChatView.restoredDraft(accepted: accepted, typed: typed, files: files,
                               currentPrompt: currentPrompt, currentFiles: currentFiles)
    }
}
