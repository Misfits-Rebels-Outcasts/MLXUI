import SwiftUI
import UniformTypeIdentifiers
import MLXLMCommon

/// Chat sheet for running an installed LLM locally via MLX. Surfaces the streaming
/// output, loading/stop/error states, a prompt field, and — for agentic models — inline
/// tool-call cards, an approval sheet for side-effecting tools, and per-tool toggles.
struct RunChatView: View {
    let model: ModelEntry

    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @State private var prompt = ""
    /// CW-3: files chosen (or dropped) for the **next** message. Cleared when it sends.
    @State private var pendingFiles: [URL] = []
    @State private var showAuditLog = false
    @State private var conversationToDelete: Conversation?
    /// CX-2: a failed export. Its own message — `runner.errorMessage` belongs to model runs.
    @State private var exportError: String?
    /// CW-4: the right sidebar. Persisted, so it comes back the way the user left it.
    @AppStorage("chatInspectorShown") private var showsInspector = false
    /// CW-5: a tool card the inspector asked the transcript to scroll to.
    @State private var scrollTarget: UUID?

    private var runner: ModelRunner { appState.modelRunner }

    var body: some View {
        HStack(spacing: 0) {
            conversationSidebar
                .frame(width: 210)
            Divider()
            VStack(spacing: 0) {
                header
                Divider()
                transcript
                if let error = exportError {
                    errorBar(error)
                }
                if let error = runner.errorMessage {
                    errorBar(error)
                }
                Divider()
                inputBar
            }
            if showsInspector {
                Divider()
                ChatInspectorView(
                    model: model,
                    runner: runner,
                    draft: prompt,
                    pendingAttachments: pendingFiles.map(\.lastPathComponent),
                    attachFile: { url in
                        if !pendingFiles.contains(url) { pendingFiles.append(url) }
                    },
                    showInTranscript: $scrollTarget)
                .frame(width: 240)
            }
        }
        // CW-4-FIX-1: the chat's details pane is a plain HStack column, **not** `.inspector`.
        //
        // `.inspector` belongs to a window's chrome — inside a sheet it fights AppKit's sheet
        // sizing, and on macOS 14 it collapsed this one to a small, unusable box. A column in the
        // layout is deterministic in a sheet, and the sheet is free to keep one size because the
        // pane is only 240 pt of it.
        .frame(
            minWidth: 840, idealWidth: 1100, maxWidth: .infinity,
            minHeight: 520, idealHeight: 620, maxHeight: .infinity)
        .onAppear { appState.modelRunner.prepare(for: model) }
        .onDisappear { appState.stopModel() }
        .servedLoadConfirmation(runner.loadConfirmer)      // SC-5a (gate D)
        .sheet(item: Binding(
            get: { runner.pendingApproval },
            set: { if $0 == nil { runner.resolveApproval(false) } }
        )) { pending in
            approvalSheet(pending)
        }
        .sheet(isPresented: $showAuditLog) {
            AuditLogView(runner: runner)
        }
        .confirmationDialog(
            "Delete this conversation?",
            isPresented: Binding(
                get: { conversationToDelete != nil },
                set: { if !$0 { conversationToDelete = nil } }
            ),
            presenting: conversationToDelete
        ) { conversation in
            Button("Delete \"\(conversation.title)\"", role: .destructive) {
                runner.deleteConversation(conversation.id)
            }
            Button("Cancel", role: .cancel) {}
        } message: { _ in
            Text("This permanently removes the conversation from this Mac.")
        }
    }

    // MARK: - Conversation sidebar

    private var conversationSidebar: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Chats")
                    .font(.headline)
                Spacer()
                Button {
                    runner.newConversation()
                } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.plain)
                .keyboardShortcut("n", modifiers: .command)
                .help("New conversation (⌘N)")
            }
            .padding(12)
            Divider()
            List(selection: conversationSelection) {
                ForEach(runner.conversations) { conversation in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(conversation.title)
                            .font(.callout)
                            .lineLimit(1)
                        if !conversation.items.isEmpty {
                            Text(conversation.updatedAt, format: .relative(presentation: .named))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                    .tag(conversation.id)
                    .contextMenu {
                        // CX-2-FIX-1 (N1): resolve through exportSnapshot, not the ForEach's
                        // stored value — the backlog's "live vs saved copy" trap means a raw
                        // `conversation` can go stale if save timing ever changes.
                        Menu("Export") {
                            Button("Save as Markdown…") {
                                exportRow(conversation.id, as: .markdown)
                            }
                            Button("Save as JSON…") {
                                exportRow(conversation.id, as: .json)
                            }
                        }
                        .disabled(isExportDisabled(conversation))
                        Button("Delete…", role: .destructive) {
                            conversationToDelete = conversation
                        }
                    }
                }
            }
            .listStyle(.inset)
            .scrollContentBackground(.hidden)
        }
        .background(.quaternary.opacity(0.25))
    }

    private var conversationSelection: Binding<UUID?> {
        Binding(
            get: { runner.activeConversationID },
            set: { if let id = $0 { runner.selectConversation(id) } }
        )
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: model.modelType.sfSymbol)
                .foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 1) {
                Text(model.displayName)
                    .font(.headline)
                Text("Running locally · MLX")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            exportMenu
            if !runner.availableTools.isEmpty {
                toolsMenu
            }
            Button {
                showsInspector.toggle()
            } label: {
                Image(systemName: "sidebar.right")
            }
            .buttonStyle(.plain)
            .keyboardShortcut("i", modifiers: [.command, .option])
            .help("Show chat details")
            Button("Done") {
                appState.stopModel()
                dismiss()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(12)
    }

    /// CX-2 (D4a): export the **active** conversation. Placed before the tools menu and Done so
    /// the download affordance sits with the other view-level actions, not with the model's
    /// tools. The button is always drawn — D5 says the controls are **disabled**, not hidden,
    /// while a reply is streaming or when there is nothing to export — and its items act on the
    /// conversation that exists at click time.
    @ViewBuilder
    private var exportMenu: some View {
        Menu {
            Button("Save as Markdown…") { exportActive(as: .markdown) }
            Button("Save as JSON…") { exportActive(as: .json) }
            Divider()
            Button("Copy as Markdown") {
                guard let conversation = activeConversationSnapshot else { return }
                ConversationExportPanel.copyMarkdown(conversation, context: exportContext)
            }
        } label: {
            Image(systemName: "square.and.arrow.down")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Download this chat")
        .disabled(runner.isRunning || activeConversationSnapshot == nil)
    }

    /// What the header button would export right now: the live transcript of the active
    /// conversation, or `nil` when there is nothing to export (D5).
    private var activeConversationSnapshot: Conversation? {
        guard let id = runner.activeConversationID else { return nil }
        return runner.exportSnapshot(for: id)
    }

    private var exportContext: ConversationExportContext {
        ConversationExportPanel.context(modelDisplayName: model.displayName)
    }

    /// SET-4 (D4, `RSI/DelegateSettingsBacklog.md`) — the wrench is the in-session switch:
    /// per-tool on/off, approval, and the Audit Log, all one click away without leaving the
    /// chat. **Settings is the standing configuration and is complete** — the tool-call limit
    /// and folder grants live there now (SET-3), not here, so "Tool Settings…" is this menu's
    /// only way to reach them. Anything a future tool control needs beyond on/off + approval
    /// belongs in `ToolsSettingsView`, not back in this menu.
    private var toolsMenu: some View {
        Menu {
            ForEach(runner.availableTools, id: \.name) { tool in
                if tool.requiresApproval {
                    Menu(tool.name) {
                        Toggle("Enabled", isOn: enabledBinding(tool.name))
                        Divider()
                        Picker("Approval", selection: policyBinding(tool.name)) {
                            Text("Ask each time").tag(ToolApprovalPolicy.ask)
                            Text("Always allow").tag(ToolApprovalPolicy.always)
                            Text("Never allow").tag(ToolApprovalPolicy.never)
                        }
                    }
                } else {
                    Toggle(isOn: enabledBinding(tool.name)) { Text(tool.name) }
                }
            }
            Divider()
            Button("Audit Log…") { showAuditLog = true }
            SettingsOpener(pane: .agentTools) { Text("Tool Settings…") }
        } label: {
            Image(systemName: "wrench.and.screwdriver")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Tools the model may call")
    }

    private func enabledBinding(_ name: String) -> Binding<Bool> {
        Binding(
            get: { runner.enabledToolNames.contains(name) },
            set: { runner.setToolEnabled($0, for: name) }
        )
    }

    private func policyBinding(_ name: String) -> Binding<ToolApprovalPolicy> {
        Binding(
            get: { runner.toolPolicy(for: name) },
            set: { runner.setToolPolicy($0, for: name) }
        )
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if runner.transcript.isEmpty {
                        emptyState
                    }
                    ForEach(runner.transcript.items) { item in
                        transcriptRow(item)
                            .id(item.id)
                    }
                    if showsThinkingSpinner {
                        thinkingRow.id("thinking")
                    }
                }
                .padding(16)
            }
            .onChange(of: runner.transcript.items.count) { _, _ in scrollToEnd(proxy) }
            .onChange(of: runner.transcript.lastAssistantText) { _, _ in scrollToEnd(proxy) }
            .onChange(of: runner.isRunning) { _, _ in scrollToEnd(proxy) }
            .onChange(of: scrollTarget) { _, target in
                guard let target else { return }
                withAnimation { proxy.scrollTo(target, anchor: .center) }
            }
        }
    }

    /// Show a spinner while generating and no assistant text has streamed yet.
    private var showsThinkingSpinner: Bool {
        runner.isRunning && (runner.transcript.lastAssistantText?.isEmpty ?? true)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 28))
                .foregroundStyle(.secondary)
            Text("Ask \(model.displayName) something to get started.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 40)
    }

    @ViewBuilder
    private func transcriptRow(_ item: TranscriptItem) -> some View {
        switch item {
        case .message(let message):
            messageRow(message)
        case .tool(let activity):
            toolCard(activity)
        }
    }

    private func messageRow(_ message: ChatMessage) -> some View {
        let isUser = message.role == .user
        return HStack {
            if isUser { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 6) {
                Text(message.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                // CW-3: the bubble shows what was attached; the model saw the contents.
                if !message.attachments.isEmpty {
                    attachmentChips(message.attachments)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(
                isUser ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12),
                in: RoundedRectangle(cornerRadius: 10)
            )
            if !isUser { Spacer(minLength: 40) }
        }
    }

    /// Attachment chips inside a sent bubble: name plus its mode (C13).
    private func attachmentChips(_ attachments: [ChatAttachment]) -> some View {
        HStack(spacing: 6) {
            ForEach(attachments) { attachment in
                HStack(spacing: 4) {
                    Image(systemName: attachment.mode == .ocr ? "text.viewfinder" : "paperclip")
                        .font(.caption2)
                    Text(attachment.name)
                        .font(.caption2)
                        .lineLimit(1)
                    Text(attachment.badgeLabel)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.quaternary.opacity(0.6), in: Capsule())
            }
        }
    }

    private var thinkingRow: some View {
        HStack {
            ProgressView().scaleEffect(0.7)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
            Spacer(minLength: 40)
        }
    }

    // MARK: - Tool card

    private func toolCard(_ activity: ToolActivity) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: toolIcon(activity.status))
                    .foregroundStyle(toolTint(activity.status))
                Text(activity.name)
                    .font(.caption.monospaced().weight(.semibold))
                Spacer()
                Text(toolStatusLabel(activity.status))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if !activity.arguments.isEmpty, let args = argumentSummary(activity.arguments) {
                Text(args)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let result = activity.result, !result.isEmpty {
                Text(result)
                    .font(.caption)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            // CW-UX-1: the refusal says "no folder is attached"; this is the missing click. It is
            // offered only while that is still true, and clicking attaches — it never resends, so
            // the user stays the one who decides to ask again.
            if FolderSuggestion.showsOnCard(
                result: activity.result,
                attached: runner.activeAttachedFolders),
               FolderSuggestion.fileToolsOn(enabledToolNames: runner.enabledToolNames) {
                HStack(spacing: 6) {
                    if folderCandidates.isEmpty {
                        addFolderButton
                    } else {
                        folderButtons(folderCandidates)
                    }
                }
                .disabled(runner.isRunning)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(toolTint(activity.status).opacity(0.10), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(toolTint(activity.status).opacity(0.35), lineWidth: 1)
        )
    }

    private func toolIcon(_ status: ToolActivity.Status) -> String {
        switch status {
        case .running: return "gearshape.2"
        case .finished: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .denied: return "hand.raised.fill"
        case .unknownTool: return "questionmark.circle"
        case .budgetExceeded: return "hourglass"
        }
    }

    private func toolTint(_ status: ToolActivity.Status) -> Color {
        switch status {
        case .running: return .blue
        case .finished: return .green
        case .failed, .unknownTool: return .red
        case .denied, .budgetExceeded: return .orange
        }
    }

    private func toolStatusLabel(_ status: ToolActivity.Status) -> String {
        switch status {
        case .running: return "Running…"
        case .finished: return "Done"
        case .failed: return "Failed"
        case .denied: return "Declined"
        case .unknownTool: return "Unknown tool"
        case .budgetExceeded: return "Budget exceeded"
        }
    }

    private func argumentSummary(_ arguments: [String: JSONValue]) -> String? {
        guard let data = try? JSONEncoder().encode(arguments),
              let string = String(data: data, encoding: .utf8) else { return nil }
        return string
    }

    // MARK: - Export (CX-2)

    /// The two file exports a menu offers.
    private enum ExportFormat {
        case markdown
        case json
    }

    /// The header button's export: resolve the active conversation **when the item is clicked**,
    /// so the live transcript at that moment is what's written.
    private func exportActive(as format: ExportFormat) {
        guard let conversation = activeConversationSnapshot else { return }
        export(conversation, as: format)
    }

    /// The sidebar row's export: resolve **through `exportSnapshot`** when the item is clicked,
    /// never the `ForEach`'s stored value. CX-2-FIX-1 (N1) — see the "live vs saved copy" trap.
    private func exportRow(_ id: UUID, as format: ExportFormat) {
        guard let conversation = runner.exportSnapshot(for: id) else { return }
        export(conversation, as: format)
    }

    /// Run one export off the *next* main-loop turn. The panel must not open while the `Menu`
    /// that started it is still tracking, or it comes up behind this sheet — see the CX-2 traps.
    private func export(_ conversation: Conversation, as format: ExportFormat) {
        exportError = nil
        Task { @MainActor in
            await Task.yield()
            let context = exportContext
            do {
                switch format {
                case .markdown:
                    try ConversationExportPanel.saveMarkdown(conversation, context: context)
                case .json:
                    try ConversationExportPanel.saveJSON(conversation, context: context)
                }
            } catch {
                exportError = error.localizedDescription
            }
        }
    }

    /// D5: no export while a reply is streaming, and none when `exportSnapshot` has nothing to
    /// give. CX-2-FIX-1 (N1) — never read the row's stored `items` directly; that's the trap.
    private func isExportDisabled(_ conversation: Conversation) -> Bool {
        if conversation.id == runner.activeConversationID, runner.isRunning { return true }
        return runner.exportSnapshot(for: conversation.id) == nil
    }

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        let target = showsThinkingSpinner ? "thinking" : runner.transcript.items.last?.id.uuidString
        guard let target else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(target, anchor: .bottom)
        }
    }

    // MARK: - Approval sheet

    private func approvalSheet(_ pending: PendingApproval) -> some View {
        let sections = ApprovalPreview.sections(
            toolName: pending.toolName,
            arguments: pending.arguments)
        return VStack(alignment: .leading, spacing: 14) {
            Label("Allow tool call?", systemImage: "wrench.and.screwdriver")
                .font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                Text(pending.toolName)
                    .font(.callout.monospaced().weight(.semibold))
                Text(pending.toolDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if sections.isEmpty {
                Text("The model wants to run this tool. Allow it to proceed?")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                // D10: the same arguments the old sheet hid — bounded, coloured, selectable.
                Text("The model wants to run this tool with the arguments below.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(sections.indices, id: \.self) { index in
                            approvalSection(sections[index])
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 320)
            }
            HStack {
                Spacer()
                Button("Deny") { runner.resolveApproval(false) }
                    .keyboardShortcut(.cancelAction)
                Button("Allow") { runner.resolveApproval(true) }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(20)
        .frame(width: sections.isEmpty ? 360 : 560)
    }

    /// One labelled argument block: monospaced and selectable, red for a removal and green for an
    /// addition (D10).
    private func approvalSection(_ section: ApprovalPreview.Section) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(section.title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            Text(section.body)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(tint(for: section.style))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func tint(for style: ApprovalPreview.Style) -> Color {
        switch style {
        case .removed: return .red
        case .added: return .green
        case .plain, .code: return .primary
        }
    }

    // MARK: - Error

    private func errorBar(_ message: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(message)
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .foregroundStyle(.red)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.red.opacity(0.08))
    }

    // MARK: - Input

    private var inputBar: some View {
        VStack(alignment: .leading, spacing: 6) {
            folderSuggestionRow
            if !runner.activeAttachedFolders.isEmpty {
                attachedFolderChips
            }
            if !pendingFiles.isEmpty {
                pendingFileChips
            }
            HStack(spacing: 8) {
                attachMenu
                TextField("Message \(model.displayName)…", text: $prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...4)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                    .onSubmit(send)
                    .disabled(runner.isRunning)

                if runner.isRunning {
                    Button {
                        appState.stopModel()
                    } label: {
                        Image(systemName: "stop.fill")
                            .frame(width: 28, height: 28)
                    }
                    .buttonStyle(.bordered)
                    .help("Stop generating")
                } else {
                    Button {
                        send()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 22))
                    }
                    .buttonStyle(.plain)
                    .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        && pendingFiles.isEmpty)
                }
            }
            #if DIRECT_BUILD
            shellReachNote
            #endif
        }
        .padding(12)
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            // A folder attaches to the chat (CW-2); a file waits for the next message (CW-3).
            guard !runner.isRunning else { return false }
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    guard let url else { return }
                    Task { @MainActor in
                        if url.hasDirectoryPath {
                            runner.attachFolder(url)
                        } else if !pendingFiles.contains(url) {
                            pendingFiles.append(url)
                        }
                    }
                }
            }
            return true
        }
    }

    // MARK: - Composer: folders (CW-2, D1/D5)

    /// "+" — Add folder… and the folders this chat can use, one click to attach or detach (D5).
    /// **Add files…** arrives in CW-3; the v1 menu is deliberately this short.
    private var attachMenu: some View {
        Menu {
            Button("Add files…") {
                pickFiles()
            }
            Button("Add folder…") {
                #if DIRECT_BUILD
                FolderAccessPanel.presentAndAttach(using: runner)
                #else
                FolderAccessPanel.presentAndGrant(using: runner)
                #endif
            }
            Divider()
            #if DIRECT_BUILD
            let candidates = Self.recentFolderCandidates(attached: runner.activeAttachedFolders)
            if candidates.isEmpty {
                Text("No recently used folders")
            } else {
                ForEach(candidates, id: \.self) { path in
                    Toggle(isOn: attachBinding(path)) { Text(Self.folderLabel(path)) }
                }
            }
            #else
            let candidates = Self.grantCandidates(attached: runner.activeAttachedFolders)
            if candidates.isEmpty {
                Text("No folders granted yet")
            } else {
                ForEach(candidates, id: \.self) { path in
                    Toggle(isOn: attachBinding(path)) { Text(Self.folderLabel(path)) }
                }
            }
            #endif
        } label: {
            Image(systemName: "plus")
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help("Add a folder to this chat")
        .disabled(runner.isRunning)
    }

    // MARK: - CW-UX-1: one click from available to attached

    /// What this chat could use right now, in the order the "+" menu lists it.
    private var folderCandidates: [String] {
        #if DIRECT_BUILD
        Self.recentFolderCandidates(attached: runner.activeAttachedFolders)
        #else
        Self.grantCandidates(attached: runner.activeAttachedFolders)
        #endif
    }

    /// The composer's hint, above the message box. Disappears the moment a folder is attached,
    /// because the chip row replaces it (and `candidates` returns `[]`).
    @ViewBuilder
    private var folderSuggestionRow: some View {
        let candidates = FolderSuggestion.candidates(
            attached: runner.activeAttachedFolders,
            available: folderCandidates,
            fileToolsOn: FolderSuggestion.fileToolsOn(enabledToolNames: runner.enabledToolNames))
        if !candidates.isEmpty {
            HStack(spacing: 6) {
                Text("Use a folder in this chat:")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                folderButtons(candidates)
            }
            .disabled(runner.isRunning)
        } else if runner.activeAttachedFolders.isEmpty,
                  FolderSuggestion.fileToolsOn(enabledToolNames: runner.enabledToolNames),
                  folderCandidates.isEmpty {
            // Nothing is offered and nothing is attached: the only route is the picker itself.
            // App Store grants-and-attaches there; Direct attaches (no sandbox, no bookmark).
            addFolderButton
                .disabled(runner.isRunning)
        }
    }

    /// Up to `maxButtons` folders, then "More…" when there are more.
    @ViewBuilder
    private func folderButtons(_ candidates: [String]) -> some View {
        let row = FolderSuggestion.row(candidates: candidates)
        ForEach(row.shown, id: \.self) { path in
            Button {
                runner.attachFolder(path: path)
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: "plus")
                        .font(.caption2)
                    Text(Self.folderLabel(path))
                        .font(.caption2)
                        .lineLimit(1)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(.quaternary.opacity(0.6), in: Capsule())
            }
            .buttonStyle(.plain)
            .help(path)
        }
        if row.hasMore {
            // Reuses the "+" menu's own list rather than a second copy of it: the menu is the
            // established place for the full set.
            Menu("More…") {
                ForEach(candidates, id: \.self) { path in
                    Button(Self.folderLabel(path)) { runner.attachFolder(path: path) }
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize()
            .font(.caption2)
            .help("Show every folder this chat could use")
        }
    }

    private var addFolderButton: some View {
        Button {
            #if DIRECT_BUILD
            FolderAccessPanel.presentAndAttach(using: runner)
            #else
            FolderAccessPanel.presentAndGrant(using: runner, attachAfterGrant: true)
            #endif
        } label: {
            HStack(spacing: 3) {
                Image(systemName: "plus")
                    .font(.caption2)
                Text("Add folder…")
                    .font(.caption2)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(.quaternary.opacity(0.6), in: Capsule())
        }
        .buttonStyle(.plain)
    }

    /// CW-3's file picker: several files at once, no folders (Add folder… is the folder route).
    private func pickFiles() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.message = "Choose files to send with the next message."
        panel.prompt = "Attach"
        guard panel.runModal() == .OK else { return }
        pendingFiles += panel.urls
    }

    /// Files waiting to go with the next message, each with an × (CW-3).
    private var pendingFileChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(pendingFiles.enumerated()), id: \.offset) { index, url in
                    HStack(spacing: 4) {
                        Image(systemName: "paperclip")
                            .font(.caption2)
                        Text(url.lastPathComponent)
                            .font(.caption)
                            .lineLimit(1)
                        Button {
                            pendingFiles.remove(at: index)
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .help("Do not attach this file")
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.quaternary.opacity(0.6), in: Capsule())
                }
            }
        }
        .disabled(runner.isRunning)
    }

    private func attachBinding(_ path: String) -> Binding<Bool> {
        Binding(
            get: { runner.activeAttachedFolders.contains(path) },
            set: { attached in
                if attached { runner.attachFolder(path: path) }
                else { runner.detachFolder(path: path) }
            }
        )
    }

    /// Attached folders above the composer: name, a star on the working folder, an × to detach.
    private var attachedFolderChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(runner.activeAttachedFolders, id: \.self) { path in
                    HStack(spacing: 4) {
                        if path == runner.activeWorkingFolder {
                            Image(systemName: "star.fill")
                                .font(.caption2)
                                .foregroundStyle(.yellow)
                        }
                        Image(systemName: "folder")
                            .font(.caption2)
                        Text(Self.folderLabel(path))
                            .font(.caption)
                            .lineLimit(1)
                        Button {
                            runner.detachFolder(path: path)
                        } label: {
                            Image(systemName: "xmark")
                                .font(.caption2)
                        }
                        .buttonStyle(.plain)
                        .help("Remove from this chat")
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(.quaternary.opacity(0.6), in: Capsule())
                    .contextMenu {
                        Button("Set as working folder") { runner.setWorkingFolder(path: path) }
                        Button("Reveal in Finder") {
                            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
                        }
                        Button("Remove from this chat") { runner.detachFolder(path: path) }
                    }
                }
            }
        }
        .disabled(runner.isRunning)
    }

    #if DIRECT_BUILD
    /// G4 — Direct's shell can reach anywhere, so the scope is guidance. Say so plainly rather than
    /// letting the chips imply a boundary that isn't there.
    @ViewBuilder
    private var shellReachNote: some View {
        if runner.enabledToolNames.contains("run_shell"), !runner.activeAttachedFolders.isEmpty {
            Text("The shell can reach outside these folders.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
    #endif

    /// The "+" list: every granted folder (App Store), attached ones first.
    #if !DIRECT_BUILD
    static func grantCandidates(attached: [String]) -> [String] {
        let granted = FolderGrants.grantedPaths()
        let rest = granted.filter { !attached.contains($0) }
        return attached.filter { granted.contains($0) } + rest
    }
    #else
    /// The "+" list in Direct (C9): the recent-folders MRU, attached ones first.
    static func recentFolderCandidates(attached: [String]) -> [String] {
        let recent = ModelRunner.recentFolders()
        let rest = recent.filter { !attached.contains($0) }
        return attached + rest
    }
    #endif

    /// A folder's display name (its last path component).
    static func folderLabel(_ path: String) -> String {
        let name = URL(fileURLWithPath: path).lastPathComponent
        return name.isEmpty ? path : name
    }

    /// SC-5a-b: what the composer gets back after a send — exactly what was sent (text and attachment chips) when the
    /// user declined the load question, and nothing if the turn started or the composer already holds something new.
    static func restoredDraft(accepted: Bool, typed: String, files: [URL],
                              currentPrompt: String, currentFiles: [URL]) -> (prompt: String, files: [URL])? {
        guard !accepted, currentPrompt.isEmpty, currentFiles.isEmpty else { return nil }
        return (typed, files)
    }

    private func send() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let files = pendingFiles
        // A file on its own is a message: "what is in this file?" needs no typed text, and the
        // composer already shows the chip. Requiring text here made an attachment-only send a dead
        // end (CW-C-FIX-1).
        guard !text.isEmpty || !files.isEmpty, !runner.isRunning else { return }
        let typed = prompt
        prompt = ""
        pendingFiles = []
        Task {
            // SC-5a-b: Cancel on the load question means the turn never started — give back exactly what was sent,
            // unless the user has already started something new.
            let accepted = await appState.sendPrompt(text, attachments: files)
            if let draft = Self.restoredDraft(accepted: accepted, typed: typed, files: files,
                                              currentPrompt: prompt, currentFiles: pendingFiles) {
                prompt = draft.prompt
                pendingFiles = draft.files
            }
        }
    }
}

// MARK: - Audit log sheet

/// The session's tool audit trail (AG6b-2): one timestamped row per lifecycle event, newest
/// first. Takes the runner directly (not via `@Environment`) so the sheet's fresh environment
/// doesn't need a re-injection (B1). Not `private` — SET-3's Tools pane presents the same
/// sheet from `ToolsSettingsView` rather than keeping its own copy (D4).
struct AuditLogView: View {
    @Environment(\.dismiss) private var dismiss
    let runner: ModelRunner

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Label("Tool Audit Log", systemImage: "list.bullet.rectangle")
                    .font(.headline)
                Spacer()
                Button("Clear") { runner.clearAuditLog() }
                    .disabled(runner.auditLog.isEmpty)
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(12)
            Divider()
            if runner.auditLog.isEmpty {
                Text("No tool calls yet this session.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(runner.auditLog.entries.reversed()) { entry in
                            row(entry)
                        }
                    }
                    .padding(12)
                }
            }
        }
        .frame(width: 480, height: 420)
    }

    private func row(_ entry: AuditEntry) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 6) {
                Image(systemName: icon(entry.outcome))
                    .foregroundStyle(tint(entry.outcome))
                Text(entry.toolName)
                    .font(.caption.monospaced().weight(.semibold))
                Text(label(entry.outcome))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(entry.date, style: .time)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            if !entry.detail.isEmpty {
                Text(entry.detail)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
                    .textSelection(.enabled)
            }
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
    }

    private func icon(_ outcome: AuditEntry.Outcome) -> String {
        switch outcome {
        case .started: return "play.circle"
        case .finished: return "checkmark.circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .denied: return "hand.raised.fill"
        case .unknownTool: return "questionmark.circle"
        case .budgetExceeded: return "hourglass"
        }
    }

    private func tint(_ outcome: AuditEntry.Outcome) -> Color {
        switch outcome {
        case .started: return .blue
        case .finished: return .green
        case .failed, .unknownTool: return .red
        case .denied, .budgetExceeded: return .orange
        }
    }

    private func label(_ outcome: AuditEntry.Outcome) -> String {
        switch outcome {
        case .started: return "started"
        case .finished: return "finished"
        case .failed: return "failed"
        case .denied: return "declined"
        case .unknownTool: return "unknown tool"
        case .budgetExceeded: return "budget exceeded"
        }
    }
}
