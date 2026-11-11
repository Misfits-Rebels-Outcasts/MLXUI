//
//  AskInChatButton.swift
//  MLXUI — Model Tools, MT4-5. The one button every Run UI puts beside its result's Copy / Save.
//
//  "Ask in Chat" starts a new Chat-page conversation with the result waiting in the composer (it becomes asset `a1` when
//  the user sends), closes the Run UI's sheet and shows the page. Absent when the Chat page is hidden (`ChatGate`),
//  disabled while a chat turn is running.
//

import SwiftUI

struct AskInChatButton: View {
    /// Optional on purpose: a Run UI hosted without the app's state (a test, a preview) simply has no button.
    @Environment(AppState.self) private var appState: AppState?
    @Environment(\.dismiss) private var dismiss

    /// Evaluated on click, so a result that changes (or is encoded) costs nothing until used.
    let result: () -> AskInChatResult?
    @State private var failed = false

    init(_ result: @escaping () -> AskInChatResult?) { self.result = result }

    var body: some View {
        if let appState, ChatGate.isAvailable {
            Button {
                guard let value = result(), appState.askInChat(value) else { failed = true; return }
                failed = false
                dismiss()
            } label: {
                Label("Ask in Chat", systemImage: "bubble.left.and.text.bubble.right")
            }
            .disabled(!appState.chat.canAskInChat)
            .help(failed ? "Couldn't send this result to Chat."
                  : appState.chat.canAskInChat ? "Start a new chat with this result"
                  : "Chat is answering. Try again when it finishes.")
        }
    }
}
