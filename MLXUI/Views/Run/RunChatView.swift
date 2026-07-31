import SwiftUI
import MLXLMCommon

/// Chat sheet for running an installed LLM locally via MLX. Surfaces the streaming
/// output, loading/stop/error states, a prompt field, and — for agentic models — inline
/// tool-call cards, an approval sheet for side-effecting tools, and per-tool toggles.
struct RunChatView: View {
    let model: ModelEntry

    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @State private var prompt = ""

    private var runner: ModelRunner { appState.modelRunner }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            transcript
            if let error = runner.errorMessage {
                errorBar(error)
            }
            Divider()
            inputBar
        }
        .frame(width: 560, height: 620)
        .onAppear { appState.modelRunner.prepare(for: model) }
        .onDisappear { appState.stopModel() }
        .sheet(item: Binding(
            get: { runner.pendingApproval },
            set: { if $0 == nil { runner.resolveApproval(false) } }
        )) { pending in
            approvalSheet(pending)
        }
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
            if !runner.availableTools.isEmpty {
                toolsMenu
            }
            Button("Done") {
                appState.stopModel()
                dismiss()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(12)
    }

    /// Per-tool on/off toggles for the agent's available tools. Tools that need approval also
    /// expose a standing approval policy (ask / always / never) in a submenu.
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
            set: { on in
                if on { runner.enabledToolNames.insert(name) }
                else { runner.enabledToolNames.remove(name) }
            }
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
            Text(message.text)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(
                    isUser ? Color.accentColor.opacity(0.18) : Color.secondary.opacity(0.12),
                    in: RoundedRectangle(cornerRadius: 10)
                )
            if !isUser { Spacer(minLength: 40) }
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

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        let target = showsThinkingSpinner ? "thinking" : runner.transcript.items.last?.id.uuidString
        guard let target else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(target, anchor: .bottom)
        }
    }

    // MARK: - Approval sheet

    private func approvalSheet(_ pending: PendingApproval) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Allow tool call?", systemImage: "wrench.and.screwdriver")
                .font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                Text(pending.toolName)
                    .font(.callout.monospaced().weight(.semibold))
                Text(pending.toolDescription)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text("The model wants to run this tool. Allow it to proceed?")
                .font(.caption)
                .foregroundStyle(.secondary)
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
        .frame(width: 360)
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
        HStack(spacing: 8) {
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
                .disabled(prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(12)
    }

    private func send() {
        let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !runner.isRunning else { return }
        prompt = ""
        appState.sendPrompt(text)
    }
}
