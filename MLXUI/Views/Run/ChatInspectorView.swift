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
    /// CW-5: set to a tool card's id to ask the transcript to scroll to it.
    @Binding var showInTranscript: UUID?

    @State private var expanded: Set<String> = []
    @State private var levels: [String: FolderListing.Result] = [:]
    /// CW-5: the file whose preview is open, and its loaded content.
    @State private var selectedPath: String?
    @State private var preview: ChatFilePreview.Content?
    /// Held while a Quick Look preview is on screen: the viewer reads lazily (sandbox trap).
    @State private var quickLookScope: FilePreviewScope?

    private var canWrite: Bool {
        ChatInspectorModel.canWrite(enabledToolNames: runner.enabledToolNames)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                contextSection
                Divider()
                filesSection
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

    // MARK: - Files in this chat (CW-5, C14)

    private var touches: FileTouchIndex.Evaluation {
        FileTouchIndex.entries(
            items: runner.transcript.items,
            workingFolder: runner.activeWorkingFolder)
    }

    /// CW-6: the row shown while the runner is blocked on the approval sheet. The sheet itself is
    /// unchanged — this only says what it is waiting for.
    private var waitingForApproval: ChatInspectorModel.Waiting? {
        guard let pending = runner.pendingApproval else { return nil }
        return ChatInspectorModel.waiting(
            toolName: pending.toolName,
            path: pending.arguments.string("path"))
    }

    private var filesSection: some View {
        let evaluation = touches
        return VStack(alignment: .leading, spacing: 8) {
            sectionTitle("Files in this chat")
            if let waiting = waitingForApproval {
                HStack(spacing: 4) {
                    ProgressView().controlSize(.small)
                    Text(waiting.label)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            if evaluation.entries.isEmpty {
                Text("Nothing yet. Tool calls that touch files show up here.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } else {
                if !evaluation.changes.isEmpty {
                    group("New · Edited", entries: evaluation.changes)
                }
                if !evaluation.reads.isEmpty {
                    group("Read", entries: evaluation.reads, collapsedLabel: "\(evaluation.reads.count) files read")
                }
                if !evaluation.declined.isEmpty {
                    group("Declined", entries: evaluation.declined)
                }
            }
            if selectedPath != nil {
                previewPane
            }
        }
    }

    @ViewBuilder
    private func group(_ title: String, entries: [FileTouchIndex.Entry], collapsedLabel: String? = nil) -> some View {
        if let collapsedLabel {
            DisclosureGroup {
                ForEach(entries) { entry in touchRow(entry) }
            } label: {
                Text(collapsedLabel).font(.caption2).foregroundStyle(.secondary)
            }
        } else {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            ForEach(entries) { entry in touchRow(entry) }
        }
    }

    private func touchRow(_ entry: FileTouchIndex.Entry) -> some View {
        HStack(spacing: 4) {
            Image(systemName: entry.kind == .declined ? "xmark.circle" : "doc.text")
                .font(.caption2)
                .foregroundStyle(entry.kind == .declined ? .secondary : .primary)
            Text(entry.name.isEmpty ? entry.path : entry.name)
                .font(.caption)
                .lineLimit(1)
            Spacer(minLength: 4)
            Text(entry.kind.label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .contentShape(Rectangle())
        .onTapGesture { select(entry.path) }
        .contextMenu {
            Button("Open") { open(entry.path) }
            Button("Reveal in Finder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: entry.path)])
            }
            Button("Show in transcript") { showInTranscript = entry.activities.last?.id }
        }
        .help(entry.path)
    }

    @ViewBuilder
    private var previewPane: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            switch preview {
            case .markdown(let text):
                ScrollView {
                    Text((try? AttributedString(markdown: text)) ?? AttributedString(text))
                        .font(.caption)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 220)
            case .text(let text):
                ScrollView {
                    Text(text)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 220)
            case .truncatedText(let text, let shown, let total):
                ScrollView {
                    Text(text)
                        .font(.system(.caption2, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 220)
                Text(ChatFilePreview.truncationNotice(shownBytes: shown, totalBytes: total))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            case .image:
                if let path = selectedPath, let image = NSImage(contentsOfFile: path) {
                    Image(nsImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 200)
                }
            case .quickLook:
                if let scope = quickLookScope {
                    QuickLookPreview(url: scope.scopedURL)
                        .frame(height: 220)
                }
            case .unavailable(let message):
                Text(message).font(.caption2).foregroundStyle(.secondary)
            case nil:
                EmptyView()
            }
            editDiffs
            previewButtons
        }
    }

    /// CW-6: every edit of the selected path, newest first, as red/green sections. **No Revert**
    /// (the owner's G3 ruling: no snapshots, no revert until it is ruled on).
    @ViewBuilder
    private var editDiffs: some View {
        let specs = selectedPath.map { touches.editActivities(for: $0) } ?? []
        if !specs.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text(specs.count == 1 ? "Edit" : "Edits (newest first)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                ForEach(ChatInspectorModel.editSpecs(specs)) { spec in
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(Array(spec.sections.enumerated()), id: \.offset) { _, section in
                            diffSection(section)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func diffSection(_ section: ApprovalPreview.Section) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(section.title)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(section.body)
                .font(.system(.caption2, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(4)
                .background(diffTint(section.style), in: RoundedRectangle(cornerRadius: 4))
        }
    }

    private func diffTint(_ style: ApprovalPreview.Style) -> Color {
        switch style {
        case .removed: return Color.red.opacity(0.12)
        case .added: return Color.green.opacity(0.12)
        case .plain, .code: return Color.secondary.opacity(0.08)
        }
    }

    @ViewBuilder
    private var previewButtons: some View {
        if let path = selectedPath {
            HStack(spacing: 8) {
                Button("Open") { open(path) }
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                }
                Button("Show in transcript") {
                    showInTranscript = touches.entry(for: path)?.activities.last?.id
                }
                Spacer()
                Button("Close") { clearPreview() }
            }
            .buttonStyle(.link)
            .font(.caption2)
        }
    }

    /// Load the preview for a path: presentation from the file itself (OV-1's rule), bytes through
    /// the scoped read, and a scope held for Quick Look when it is the reader.
    private func select(_ path: String) {
        selectedPath = path
        let url = URL(fileURLWithPath: path)
        let presentation = FlowSavedFile.presentation(url: url, task: nil)
        if presentation.allowsQuickLook, presentation != .image,
           presentation != .text, presentation != .web, presentation != .table {
            quickLookScope = FilePreviewScope(path: path)
            preview = .quickLook
            return
        }
        quickLookScope = nil
        let data = FolderListing.scopedData(at: url)
        preview = ChatFilePreview.content(
            for: url,
            presentation: presentation,
            data: data,
            error: data == nil ? "This file could not be read." : nil)
    }

    private func clearPreview() {
        selectedPath = nil
        preview = nil
        quickLookScope = nil
    }

    private func open(_ path: String) {
        // The scope must be live for the hand-off: `NSWorkspace.open` starts the app, which then
        // needs the sandbox extension to still be held.
        guard let scope = FilePreviewScope(path: path) else { return }
        NSWorkspace.shared.open(scope.scopedURL)
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
