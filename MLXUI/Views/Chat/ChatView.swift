import SwiftUI
import UniformTypeIdentifiers
import MLXLMCommon

/// The chat body for running an installed LLM locally via MLX (MT2-1: extracted from `RunChatView`, which is now
/// the thin sheet wrapper). Surfaces the streaming output, loading/stop/error states, a prompt field, and — for
/// agentic models — inline tool-call cards, an approval sheet for side-effecting tools, and per-tool toggles.
///
/// The model is state, not a `let`: it comes from the `ChatController` for this surface. `initialModel` only covers
/// the first render, before the controller has bound the surface.
struct ChatView: View {
    let surface: ChatSurface
    let initialModel: ModelEntry
    /// The sheet's "Done". `nil` on a surface that has nothing to dismiss.
    var onDone: (() -> Void)?

    @Environment(AppState.self) private var appState
    /// The composer's draft lives in the controller (so it can outlive this view), one per surface.
    private var prompt: String {
        get { controller.draft(for: surface).text }
        nonmutating set {
            var draft = controller.draft(for: surface)
            draft.text = newValue
            controller.setDraft(draft, for: surface)
        }
    }
    private var promptBinding: Binding<String> {
        Binding(get: { prompt }, set: { prompt = $0 })
    }
    /// CW-3: files chosen (or dropped) for the **next** message. Cleared when it sends.
    private var pendingFiles: [URL] {
        get { controller.draft(for: surface).files }
        nonmutating set {
            var draft = controller.draft(for: surface)
            draft.files = newValue
            controller.setDraft(draft, for: surface)
        }
    }
    @State private var showAuditLog = false
    @State private var conversationToDelete: Conversation?
    /// CX-2: a failed export. Its own message — `runner.errorMessage` belongs to model runs.
    @State private var exportError: String?
    /// CW-4: the right sidebar. Persisted, so it comes back the way the user left it.
    @AppStorage("chatInspectorShown") private var showsInspector = false
    /// CW-5: a tool card the inspector asked the transcript to scroll to.
    @State private var scrollTarget: UUID?
    /// MT4-7-FIX-4 (A2): presents the tool-approval question as an app-modal window.
    @State private var approvals = ApprovalPresenter(host: AppKitApprovalHost())

    private var controller: ChatController { appState.chat }
    private var runner: ModelRunner { controller.runner }
    private var model: ModelEntry { controller.model(for: surface) ?? initialModel }
    /// Whether this surface drives the runner right now (MT2-2).
    private var holdsLease: Bool { controller.holds(surface) }

    /// Shown over a surface that was refused the runner because another one is generating.
    private var waitingNotice: some View {
        VStack(spacing: 10) {
            Image(systemName: "hourglass").font(.largeTitle).foregroundStyle(.secondary)
            Text(surface == .bench ? "Chat is answering on the Chat page." : "A model's chat window is answering.")
                .font(.headline)
            Text("This chat opens as soon as it finishes.")
                .font(.callout).foregroundStyle(.secondary)
            HStack {
                Button("Stop it") { controller.stopCurrentGeneration() }
                if surface == .bench, ChatGate.isAvailable {
                    Button("Open Chat") {
                        appState.selectedSection = .chat
                        onDone?()
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }

    /// CP-4: the persisted conversations-column width, shared by the page and the sheet.
    @AppStorage("chatSidebarWidth") private var sidebarWidth: Double = ChatSidebarWidth.defaultWidth
    @State private var sidebarDragStart: Double?
    @State private var cursorPushed = false

    private func popResizeCursor() {
        if cursorPushed {
            NSCursor.pop()
            cursorPushed = false
        }
    }

    /// A ~6 pt strip over the divider: drag to resize, double-click to reset.
    private var sidebarResizeHandle: some View {
        Color.clear
            .frame(width: 6)
            .contentShape(Rectangle())
            .onHover { inside in
                if inside, !cursorPushed {
                    NSCursor.resizeLeftRight.push()
                    cursorPushed = true
                } else if !inside, sidebarDragStart == nil {
                    popResizeCursor()
                }
            }
            .gesture(
                DragGesture(minimumDistance: 1, coordinateSpace: .global)
                    .onChanged { drag in
                        let start = sidebarDragStart ?? ChatSidebarWidth.clamp(sidebarWidth)
                        sidebarDragStart = start
                        sidebarWidth = ChatSidebarWidth.clamp(start + drag.translation.width)
                    }
                    .onEnded { _ in sidebarDragStart = nil }
            )
            .onTapGesture(count: 2) { sidebarWidth = ChatSidebarWidth.defaultWidth }
            .onDisappear {
                sidebarDragStart = nil
                popResizeCursor()
            }
    }

    var body: some View {
        HStack(spacing: 0) {
            conversationSidebar
                .frame(width: ChatSidebarWidth.clamp(sidebarWidth))
            Divider()
                .overlay { sidebarResizeHandle }
            VStack(spacing: 0) {
                header
                Divider()
                transcript
                // MT4-3: a model tool waiting on a download (plan §3.8). Shown by the surface that holds the runner.
                if holdsLease {
                    ForEach(runner.installCoordinator.cards.filter { $0.conversationID == runner.activeConversationID }) { card in
                        ModelInstallCardView(
                            card: card, coordinator: runner.installCoordinator,
                            installState: appState.installManager.modelStates[card.target.id] ?? .idle)
                    }
                }
                if let error = exportError {
                    errorBar(error)
                }
                if surface == .page, let banner = controller.banner(installedChatModels: appState.installedChatModels) {
                    bannerBar(banner)
                } else if let error = runner.errorMessage {
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
                    showInTranscript: $scrollTarget,
                    showsModelTools: ChatController.modelToolsEnabled(for: surface),
                    catalogModels: appState.browserData?.domains.flatMap { $0.allModels } ?? [],
                    openWorkflowEditor: AppState.hideAutomate ? nil : { target in appState.route.append(.editor(target)) })
                .frame(width: 240)
            }
        }
        // CW-4-FIX-1: the chat's details pane is a plain HStack column, **not** `.inspector`.
        //
        // `.inspector` belongs to a window's chrome — inside a sheet it fights AppKit's sheet
        // sizing, and on macOS 14 it collapsed this one to a small, unusable box. A column in the
        // layout is deterministic in a sheet, and the sheet is free to keep one size because the
        // pane is only 240 pt of it.
        .onAppear { controller.acquire(surface, model: model) }
        // MT2-2: the bench and the page can both be in the hierarchy. The page takes the runner back when the
        // sheet lets go of it; a surface that was refused while the other generated binds itself when it ends.
        .onChange(of: controller.holder) { _, new in
            if surface == .page, new == nil { controller.acquire(.page) }
        }
        .onChange(of: runner.isRunning) { _, running in
            if !running { controller.bindWhenFree(surface) }
        }
        .overlay {
            if controller.waiting.contains(surface) { waitingNotice }
        }
        // SC-5a (gate D) — the served-load question and the approval question are presented only by the surface
        // that holds the runner. Gated inside the binding (not by a conditional modifier), so the view's identity
        // never changes when the lease moves.
        .sheet(item: Binding(
            get: { holdsLease ? runner.loadConfirmer.question : nil },
            set: { if $0 == nil, holdsLease { runner.loadConfirmer.respond(false) } }
        )) { question in
            ServedLoadConfirmSheet(message: question.message) { runner.loadConfirmer.respond($0) }
        }
        // MT4-7-FIX-4 (A2): the approval question is an app-modal window (frontmost over any sheet, Allow armed after a moment),
        // not a sheet of this view, which queued behind a Run UI sheet. Still presented only by the surface that holds the runner.
        .onChange(of: holdsLease ? runner.pendingApproval?.id : nil, initial: true) { _, _ in
            approvals.resolve = { [runner] id, approved in
                if runner.pendingApproval?.id == id { runner.resolveApproval(approved) }
            }
            approvals.sync(pending: holdsLease ? runner.pendingApproval : nil, holdsLease: holdsLease)
        }
        .onDisappear { approvals.dismissDenying() }
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
                controller.delete(conversation.id, on: surface)
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
                    controller.newConversation(on: surface)
                } label: {
                    Image(systemName: "square.and.pencil")
                }
                .buttonStyle(.plain)
                .keyboardShortcut(holdsLease ? KeyboardShortcut("n", modifiers: .command) : nil)
                .help("New conversation (⌘N)")
            }
            .padding(12)
            Divider()
            List(selection: conversationSelection) {
                let names = surface == .page ? appState.catalogDisplayNames : [:]
                ForEach(controller.conversations) { conversation in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(conversation.title)
                            .font(.callout)
                            .lineLimit(1)
                        // MT2-3: the Chat page lists every model's chats, so each row names its model.
                        if surface == .page {
                            Text(ChatConversationBadge.make(
                                modelID: conversation.modelID,
                                catalog: names.merging([AppleIntelligenceChat.modelID: AppleIntelligenceChat.displayName]) { a, _ in a },
                                installedIDs: controller.appleReadiness == .ready
                                    ? appState.installedModelIDs.union([AppleIntelligenceChat.modelID]) : appState.installedModelIDs).text)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
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
            get: { controller.activeConversationID },
            set: { if let id = $0 { controller.select(id, on: surface) } }
        )
    }

    // MARK: - Header

    private var header: some View {
        // CP-1: wide variant first (labelled "Open in Chat", title capped so its ideal width is
        // bounded); when that does not fit, the icon-only variant, whose title truncates instead.
        ViewThatFits(in: .horizontal) {
            headerRow(compact: false)
            headerRow(compact: true)
        }
    }

    private func headerRow(compact: Bool) -> some View {
        HStack(spacing: 8) {
            Image(systemName: model.modelType.sfSymbol)
                .foregroundStyle(.secondary)
                .fixedSize()
            VStack(alignment: .leading, spacing: 1) {
                Text(model.displayName)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.tail)
                Text(ChatModelLabels.headerSubtitle(modelID: model.id))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            .frame(maxWidth: compact ? .infinity : 320, alignment: .leading)
            .layoutPriority(-1)
            Spacer(minLength: 8)
            exportMenu
            if surface == .bench, onDone != nil, ChatGate.isAvailable {
                openInChatButton(compact: compact)
            }
            if !runner.availableTools.isEmpty {
                toolsMenu
            }
            Button {
                showsInspector.toggle()
            } label: {
                Image(systemName: "sidebar.right")
            }
            .buttonStyle(.plain)
            .keyboardShortcut(holdsLease ? KeyboardShortcut("i", modifiers: [.command, .option]) : nil)
            .help("Show chat details")
            if let onDone {
                Button("Done") {
                    controller.stop(from: surface)
                    onDone()
                }
                .keyboardShortcut(.cancelAction)
                .fixedSize()
            }
        }
        .padding(12)
    }

    @ViewBuilder
    private func openInChatButton(compact: Bool) -> some View {
        let button = Button {
            if appState.openBenchInChat() { onDone?() }
        } label: {
            Label("Open in Chat", systemImage: "arrow.up.forward.app")
        }
        .disabled(!controller.canOpenBenchInChat)
        if compact {
            button
                .labelStyle(.iconOnly)
                .help("Open in Chat")
                .fixedSize()
        } else {
            button
                .help("Continue this conversation on the Chat page")
                .fixedSize()
        }
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
            if let chip = modelToolChip(for: activity) {
                chip
            } else {
                toolCard(activity)
            }
        }
    }

    /// MT4-4: on the Chat page a model-tool call is a chip (and its output inline). The bench keeps the old card, and so does any
    /// card with no asset behind it (a plain-text result written before Model Tools).
    private func modelToolChip(for activity: ToolActivity) -> ModelToolChipView? {
        guard surface == .page, let conversationID = runner.activeConversationID else { return nil }
        let store = runner.assetStore(for: conversationID)
        let records = store.allRecords   // MT4-7-FIX-4 (E): a record whose file is gone still feeds the chip and the player's message
        guard let chip = ModelToolChip.make(toolName: activity.name, id: activity.id, status: activity.status,
                                            result: activity.result, records: records) else { return nil }
        let output = chip.outputHandle.flatMap { handle in
            records.first { $0.handle == handle }.map { (record: $0, url: store.url(for: $0)) }
        }
        return ModelToolChipView(chip: chip, output: output)
    }

    private func messageRow(_ message: ChatMessage) -> some View {
        let isUser = message.role == .user
        return HStack {
            if isUser { Spacer(minLength: 40) }
            VStack(alignment: .leading, spacing: 6) {
                Text(message.text)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                // MT3-FIX-1: which model wrote this answer (the Chat page lists every model's chats).
                if surface == .page, let badge = ChatModelLabels.answerBadge(
                    role: message.role, modelID: message.modelID,
                    catalog: appState.catalogDisplayNames,
                    installedIDs: controller.appleReadiness == .ready
                        ? appState.installedModelIDs.union([AppleIntelligenceChat.modelID]) : appState.installedModelIDs) {
                    Text(badge)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
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

    // MARK: - Error

    /// MT3-3: an Apple Intelligence overflow or refusal: one sentence and its buttons (plan §3.3).
    private func bannerBar(_ banner: ChatBanner) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
            Text(banner.text)
                .font(.caption)
                .frame(maxWidth: .infinity, alignment: .leading)
            ForEach(Array(banner.buttons.enumerated()), id: \.offset) { index, button in
                if case .openSettings(let pane) = button {
                    SettingsOpener(pane: pane) { Text(banner.buttonTitles[index]) }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                } else {
                    Button(banner.buttonTitles[index]) { perform(button) }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                }
            }
        }
        .foregroundStyle(.orange)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.orange.opacity(0.10))
    }

    private func perform(_ button: ChatBanner.Button) {
        switch button {
        case .continueWith(let modelID, _):
            if let chosen = appState.installedChatModels.first(where: { $0.id == modelID }) {
                Task { await controller.continueAfterOverflow(with: chosen) }
            }
        case .startNewChat:
            controller.startNewChatAfterBanner(on: surface)
        case .openSettings:
            break   // rendered as a `SettingsOpener` in `bannerBar`
        }
    }

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
                if surface == .page { modelMenu }
                TextField("Message \(model.displayName)…", text: promptBinding, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...4)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
                    .onSubmit(send)
                    .disabled(runner.isRunning)

                if runner.isRunning {
                    Button {
                        controller.stop(from: surface)
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

    /// MT2-4 (Chat page only): the model for the next turn. Installed chat models with plain facts; disabled while a
    /// turn runs, so the header never names a model that is not the one answering.
    private var modelMenu: some View {
        let installed = appState.installedChatModels
        let apple = controller.appleMenuEntry()
        let items = ChatModelMenu.items(installed: installed, selectedID: controller.model(for: surface)?.id,
                                        facts: controller.modelFacts, apple: apple,
                                        appleContextTokens: controller.modelFacts[AppleIntelligenceChat.modelID]?.usableTokens)
        return Menu {
            ForEach(items) { item in
                if item.isEnabled {
                    Button {
                        if AppleIntelligenceChat.isApple(item.id) { controller.selectPageModel(AppleIntelligenceChat.entry) }
                        else if let chosen = installed.first(where: { $0.id == item.id }) { controller.selectPageModel(chosen) }
                    } label: {
                        Text(item.isSelected ? "✓ \(item.name)" : item.name)
                        if !item.facts.isEmpty { Text(item.facts) }
                    }
                } else {
                    // Dimmed with the reason, and the one fix-it button when there is one (AFM-1's `.enableAppleIntelligence`).
                    Text("\(item.name) — \(item.note ?? "not available")")
                    if let action = item.action, let pane = settingsPane(for: action) {
                        SettingsOpener(pane: pane) { Text(AppleIntelligenceMenuEntry.turnOnTitle) }
                    }
                }
            }
            Divider()
            if let target = ChatModelMenu.getMoreTarget(visibleSectionIDs: appState.visibleSections.map(\.id)) {
                Button(ChatModelMenu.getMoreTitle) { appState.selectedSection = target }
            }
        } label: {
            Label(model.displayName, systemImage: "cpu")
                .lineLimit(1)
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .disabled(runner.isRunning)
        .help(runner.isRunning ? "The model can be changed when the reply is finished." : "Model for the next message")
        .task(id: installed.map(\.id)) {
            controller.loadModelFacts(for: installed + (apple?.isEnabled == true ? [AppleIntelligenceChat.entry] : []))
        }
    }

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
                    Toggle(isOn: attachBinding(path)) { Text(Self.folderLabel(path, among: candidates)) }
                }
            }
            #else
            let candidates = Self.grantCandidates(attached: runner.activeAttachedFolders)
            if candidates.isEmpty {
                Text("No folders granted yet")
            } else {
                ForEach(candidates, id: \.self) { path in
                    Toggle(isOn: attachBinding(path)) { Text(Self.folderLabel(path, among: candidates)) }
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
                    Text(Self.folderLabel(path, among: candidates))
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
                    Button(Self.folderLabel(path, among: candidates)) { runner.attachFolder(path: path) }
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
            get: { FolderGrants.contains(runner.activeAttachedFolders, path) },
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
                        Text(Self.folderLabel(path, among: runner.activeAttachedFolders))
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
    static func grantCandidates(
        attached: [String], granted: [String] = FolderGrants.grantedPaths()
    ) -> [String] {
        let rest = granted.filter { !FolderGrants.contains(attached, $0) }
        return uniquelySpelled(attached.filter { FolderGrants.contains(granted, $0) } + rest)
    }
    #else
    /// The "+" list in Direct (C9): the recent-folders MRU, attached ones first.
    static func recentFolderCandidates(attached: [String]) -> [String] {
        let recent = ModelRunner.recentFolders()
        let rest = recent.filter { !FolderGrants.contains(attached, $0) }
        return uniquelySpelled(attached + rest)
    }
    #endif

    /// `paths` with every later spelling of an already-listed folder dropped (CW-FIX-3).
    static func uniquelySpelled(_ paths: [String]) -> [String] {
        var kept: [String] = []
        for path in paths where !FolderGrants.contains(kept, path) { kept.append(path) }
        return kept
    }

    /// A folder's display name (its last path component).
    /// `among` is the list it is shown in: folders sharing a name there get their parent appended.
    static func folderLabel(_ path: String, among paths: [String] = []) -> String {
        FolderLabels.label(path, among: paths)
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
            let accepted = await controller.send(text, attachments: files, from: surface)
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
