//
//  ChatInspectorView.swift
//  MLXUI — chat workspace, CW-4 (`RSI/DelegateChatWorkspaceBacklog.md`).
//
//  The chat's right sidebar: **Context** (what this chat can see) and **Session** (what it is
//  doing). CW-5 adds a third section, Files in this chat.
//
//  The view renders; `ChatInspectorModel` does the arithmetic, and `FolderListing` does the only
//  I/O (one level at a time, lazily). Model-facing wording is not this view's business — the
//  refusal strings it shows come from the same code the tools use.
//

import SwiftUI
import AppKit

struct ChatInspectorView: View {
    let model: ModelEntry
    let runner: ModelRunner
    /// The composer's draft, so the meter reflects what is about to be sent.
    let draft: String
    /// Files the composer is holding for the next message.
    let pendingAttachments: [String]
    /// Hands a folder to the composer's attachment list (CW-3's importer).
    let attachFile: (URL) -> Void

    @State private var expanded: Set<String> = []
    @State private var levels: [String: FolderListing.Result] = [:]

    private var canWrite: Bool {
        ChatInspectorModel.canWrite(enabledToolNames: runner.enabledToolNames)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                contextSection
                Divider()
                sessionSection
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(minWidth: 220, idealWidth: 240)
    }

    // MARK: - Context

    private var contextSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Context")
            if runner.activeAttachedFolders.isEmpty {
                Text("No folders in this chat. Use + to add one.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(runner.activeAttachedFolders, id: \.self) { path in
                    folderRow(path)
                    if expanded.contains(path) {
                        AnyView(levelView(for: path, depth: 1))
                    }
                }
            }
            attachmentsSummary
            toolsSummary
        }
    }

    private func folderRow(_ path: String) -> some View {
        let isWorking = path == runner.activeWorkingFolder
        return HStack(spacing: 4) {
            Button {
                toggle(path)
            } label: {
                Image(systemName: expanded.contains(path) ? "chevron.down" : "chevron.right")
                    .font(.caption2)
            }
            .buttonStyle(.plain)
            .help("Browse this folder")

            if isWorking {
                Image(systemName: "star.fill")
                    .font(.caption2)
                    .foregroundStyle(.yellow)
                    .help("Relative paths start here")
            }
            Image(systemName: "folder")
                .font(.caption2)
            VStack(alignment: .leading, spacing: 1) {
                Text(FileScope.displayName(for: path))
                    .font(.callout)
                    .lineLimit(1)
                Text(ChatInspectorModel.accessBadge(canWrite: canWrite))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 4)
        }
        .contextMenu {
            Button("Set as working folder") { runner.setWorkingFolder(path: path) }
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
            }
            Button("Remove from this chat") { runner.detachFolder(path: path) }
        }
        .help(path)
    }

    /// One lazily-loaded level. Rows are files only for now: CW-5 adds the file actions.
    @ViewBuilder
    private func levelView(for path: String, depth: Int) -> some View {
        let result = levels[path]
        if let error = result?.error {
            warningRow(error, path: path)
        } else if let result {
            ForEach(result.entries) { entry in
                entryRow(entry, depth: depth)
            }
            if result.hidden > 0 {
                Text("… \(result.hidden) more")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.leading, indent(depth))
            }
            if result.entries.isEmpty {
                Text("(empty folder)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.leading, indent(depth))
            }
        } else {
            ProgressView()
                .controlSize(.small)
                .padding(.leading, indent(depth))
        }
    }

    @ViewBuilder
    private func entryRow(_ entry: FolderListing.Entry, depth: Int) -> some View {
        HStack(spacing: 4) {
            if entry.isExpandable {
                Button {
                    toggle(entry.path)
                } label: {
                    Image(systemName: expanded.contains(entry.path) ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                }
                .buttonStyle(.plain)
            } else {
                Image(systemName: "doc")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(entry.name)
                .font(.caption)
                .lineLimit(1)
            Spacer(minLength: 4)
        }
        .padding(.leading, indent(depth))
        .contentShape(Rectangle())
        .onTapGesture(count: 2) {
            guard !entry.isDirectory else { return }
            attachFile(URL(fileURLWithPath: entry.path))
        }
        .contextMenu {
            if !entry.isDirectory {
                Button("Attach to message") { attachFile(URL(fileURLWithPath: entry.path)) }
            }
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.path)])
            }
        }
        .help(entry.path)

        if entry.isExpandable, expanded.contains(entry.path) {
            // `AnyView` breaks the ViewBuilder's recursive type inference; the tree is one level
            // deep per expansion, so the cost is nil.
            AnyView(levelView(for: entry.path, depth: depth + 1))
        }
    }

    /// A folder whose grant is stale or revoked: say so and offer the route to fix it (D1).
    private func warningRow(_ message: String, path: String) -> some View {
        HStack(alignment: .top, spacing: 4) {
            Image(systemName: "exclamationmark.triangle")
                .font(.caption2)
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 2) {
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Button("Re-grant…") {
                    // Direct attaches; the App Store edition grants (CW-2's rename, B6).
                    #if DIRECT_BUILD
                    FolderAccessPanel.presentAndAttach(using: runner)
                    #else
                    FolderAccessPanel.presentAndGrant(using: runner)
                    #endif
                }
                .buttonStyle(.link)
                .font(.caption2)
            }
        }
        .padding(.leading, indent(1))
    }

    @ViewBuilder
    private var attachmentsSummary: some View {
        if !pendingAttachments.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                Text("Attachments for the next message")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(pendingAttachments, id: \.self) { name in
                    Label(name, systemImage: "paperclip").font(.caption2)
                }
            }
        }
        let last = lastUserAttachments
        if !last.isEmpty {
            VStack(alignment: .leading, spacing: 2) {
                Text("Attachments in the last message")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(last) { attachment in
                    Label(attachment.name, systemImage: "paperclip")
                        .font(.caption2)
                    Text(attachment.badgeLabel)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private var toolsSummary: some View {
        let on = runner.enabledToolNames.count
        let available = runner.availableTools.count
        Text(ChatInspectorModel.toolsOnLabel(enabled: on, available: available))
            .font(.caption2)
            .foregroundStyle(.secondary)
    }

    // MARK: - Session

    private var sessionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Session")
            meterView
            LabeledContent("Tool calls") {
                Text(ChatInspectorModel.toolCallLabel(
                    used: ChatInspectorModel.toolCallsSinceLastUser(runner.transcript.items),
                    limit: runner.toolCallLimit))
                    .font(.caption)
            }
            LabeledContent("Model") {
                Text(model.displayName).font(.caption)
            }
            if runner.isRunning {
                HStack(spacing: 4) {
                    ProgressView().controlSize(.small)
                    Text("Working…").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var meterView: some View {
        let meter = ChatInspectorModel.meter(
            estimatedTokens: ChatInspectorModel.estimatedTokens(of: environmentAndHistory),
            window: ContextWindow.tokens(
                for: model,
                modelDirectory: ModelStore.shared.directory(forModelID: model.id)))
        return VStack(alignment: .leading, spacing: 3) {
            Text("Context used").font(.caption2).foregroundStyle(.secondary)
            ProgressView(value: min(1, meter.fraction))
                .tint(meter.isAmber ? .orange : .accentColor)
            Text(meter.label).font(.caption2).foregroundStyle(.secondary)
        }
    }

    /// What the next send will carry: the same environment block the model gets, plus the history
    /// and the draft. The block is rebuilt here rather than cached, so a detach shows up at once.
    private var environmentAndHistory: String {
        var parts: [String] = []
        if let block = runner.currentEnvironmentBlock() { parts.append(block) }
        parts.append(contentsOf: runner.transcript.items.compactMap { item in
            switch item {
            case .message(let message): return message.text
            case .tool(let activity): return activity.result ?? ""
            }
        })
        parts.append(draft)
        return parts.joined(separator: "\n")
    }

    private var lastUserAttachments: [ChatAttachment] {
        for item in runner.transcript.items.reversed() {
            if case .message(let message) = item, message.role == .user {
                return message.attachments
            }
        }
        return []
    }

    // MARK: - Plumbing

    private func indent(_ depth: Int) -> CGFloat { CGFloat(depth) * 12 }

    private func sectionTitle(_ title: String) -> some View {
        Text(title).font(.headline)
    }

    private func toggle(_ path: String) {
        if expanded.contains(path) {
            expanded.remove(path)
            return
        }
        expanded.insert(path)
        guard levels[path] == nil else { return }
        let result = FolderListing.listing(for: URL(fileURLWithPath: path))
        levels[path] = result
        if result.error != nil { expanded.remove(path) }
    }
}
