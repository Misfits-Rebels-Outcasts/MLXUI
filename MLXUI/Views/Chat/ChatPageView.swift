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
    /// CD-4: what the tools would be, shown read-only before a chat model exists (the same table as the CD-2 grid).
    @State private var capabilityStatuses: [String: ModelToolStatus] = [:]

    var body: some View {
        Group {
            if let model = appState.chat.model(for: .page) {
                ChatView(surface: .page, initialModel: model, onDone: nil)
            } else {
                emptyState
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear { appState.refreshChatPage() }
        .task(id: appState.flowReadyModelIDs) {
            capabilityStatuses = ChatCapabilityStatuses.resolve(runtime: ModelToolRuntime.live(appState: appState))
        }
        .onChange(of: appState.installedModelIDs) { _, _ in appState.refreshChatPage() }
    }

    private var emptyState: some View {
        ScrollView {
            emptyContent
                .frame(maxWidth: .infinity)
        }
    }

    private var emptyContent: some View {
        VStack(spacing: 12) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 40)).foregroundStyle(.secondary)
            Text(ChatNoModelText.title).font(.title3)
            Text(ChatNoModelText.body)
                .font(.callout).foregroundStyle(.secondary)
            // MT3-3: the smallest tool-capable chat model that fits this Mac (plain facts, no rating).
            if let pick = appState.chat.pageRecommendation {
                Text("\(pick.displayName) can use tools and fits this Mac (about \(String(format: "%.1f", pick.downloadSizeGB)) GB to download).")
                    .font(.callout).foregroundStyle(.secondary)
                Button("Download \(pick.displayName)") { appState.installModel(pick) }
                    .buttonStyle(.bordered)
            }
            if let section = appState.visibleSections.first(where: { $0.id == "chat-text" }) ?? appState.visibleSections.first {
                Button("Browse models") { appState.selectedSection = .browse(section.id) }
                    .buttonStyle(.borderedProminent)
            }
            ChatCapabilityGridView(modelName: "", canUseTools: false, statuses: capabilityStatuses,
                                   onPick: { _ in }, readOnly: true)
                .padding(.top, 12)
        }
        .padding()
    }
}
