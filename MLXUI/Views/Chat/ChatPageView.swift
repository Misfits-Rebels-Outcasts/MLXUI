//
//  ChatPageView.swift
//  MLXUI — Model Tools, MT2-2.
//
//  The Chat sidebar page: `ChatView` on the page surface, with the CW inspector column. Leaving the page does
//  not stop a generation (no `onDisappear` stop here; the runner and the draft live in `ChatController`).
//  The model is chosen from what is installed (`ChatPageModel`); the composer's model menu is MT2-4.
//

import SwiftUI

struct ChatPageView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        Group {
            if let model = appState.chat.model(for: .page) {
                ChatView(surface: .page, initialModel: model, onDone: nil)
            } else {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { appState.chat.refreshPageModel(installedChatModels: appState.installedChatModels) }
        .onChange(of: appState.installedModelIDs) { _, _ in
            appState.chat.refreshPageModel(installedChatModels: appState.installedChatModels)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 40)).foregroundStyle(.secondary)
            Text("No chat model installed").font(.title3)
            Text("Install a chat model to start a conversation here.")
                .font(.callout).foregroundStyle(.secondary)
            if let section = appState.visibleSections.first(where: { $0.id == "chat-text" }) ?? appState.visibleSections.first {
                Button("Browse models") { appState.selectedSection = .browse(section.id) }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }
}
