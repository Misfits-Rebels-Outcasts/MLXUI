//
//  ChatPageView.swift
//  MLXUI — Model Tools, MT2-2.
//
//  The Chat sidebar page: `ChatView` on the page surface, with the CW inspector column. Leaving the page does
//  not stop a generation (no `onDisappear` stop here; the runner and the draft live in `ChatController`).
//  The model is chosen from what is installed (`ChatPageModel`); the composer's model menu is MT2-4.
//

import SwiftUI

/// CD-FIX-2 — what the recommended model's Download control shows for the engine's install state. Pure. The engine sets
/// `.resolving` synchronously on the click, so the control changes before the first byte.
enum ChatNoModelDownload: Equatable {
    case button(note: String?)
    /// Queued or resolving: no bytes yet.
    case starting
    case progress(fraction: Double, detail: String)
    case verifying

    /// CD-FIX-3: the engine's `total` is not final at the start. The file list from HF carries no sizes, so the engine learns each
    /// file's size as it starts (`InstallManager`: `total = done + this file + catalog sizes of the rest`, with an initial placeholder of
    /// 1), and it exposes no "list complete" signal. Until the total could be the real one the control says "Starting the
    /// download…": the total must be past the placeholder, and either near the catalog's expected size (`expectedBytes`, the
    /// weights have begun) or the download is a quarter of the way through it. Without an expected size only the placeholder is skipped.
    static let totalSettledFraction = 0.9
    static let downloadedSettledFraction = 0.25

    @MainActor static func presentation(for state: InstallState?, expectedBytes: Int64? = nil) -> ChatNoModelDownload {
        switch state {
        case nil, .idle?, .installed?: return .button(note: nil)
        case .resolving?: return .starting
        case .downloading(let progress, let downloaded, let total)?:
            guard total > 1 else { return .starting }
            if let expected = expectedBytes, expected > 0,
               Double(total) < Double(expected) * totalSettledFraction,
               Double(downloaded) < Double(expected) * downloadedSettledFraction { return .starting }
            return .progress(fraction: progress, detail: ModelInstallCard.progressText(downloaded: downloaded, total: total))
        case .verifying?: return .verifying
        case .error(let message, _)?: return .button(note: message)
        case .needsAuth(let message)?: return .button(note: message)
        }
    }
}

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

    /// The recommended model's Download button, which becomes the install progress with a Cancel the moment it is clicked.
    @ViewBuilder
    private func downloadControl(_ pick: ModelEntry) -> some View {
        switch ChatNoModelDownload.presentation(for: appState.installManager.modelStates[pick.id],
                                         expectedBytes: Int64(pick.downloadSizeGB * 1e9)) {
        case .button(let note):
            if let note { Text(note).font(.caption).foregroundStyle(.red) }
            Button("Download \(pick.displayName)") { appState.installModel(pick) }
                .buttonStyle(.bordered)
        case .starting:
            progressBlock(pick) {
                ProgressView().controlSize(.small)
                Text("Starting the download…").font(.caption).foregroundStyle(.secondary)
            }
        case .progress(let fraction, let detail):
            progressBlock(pick) {
                ProgressView(value: fraction)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        case .verifying:
            progressBlock(pick) {
                ProgressView().controlSize(.small)
                Text("Verifying…").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func progressBlock<Content: View>(_ pick: ModelEntry, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(spacing: 6) {
            content()
            Button("Cancel") { appState.cancelInstall(pick) }.controlSize(.small)
        }
        .frame(maxWidth: 280)
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
                downloadControl(pick)
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
