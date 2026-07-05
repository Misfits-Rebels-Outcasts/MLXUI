import SwiftUI

/// Chat sheet for running an installed LLM locally via MLX. Surfaces the streaming
/// output, loading/stop/error states, and a prompt field so the user drives the
/// conversation (instead of the previous hardcoded prompt).
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
            Button("Done") {
                appState.stopModel()
                dismiss()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(12)
    }

    // MARK: - Transcript

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 12) {
                    if runner.transcript.isEmpty {
                        emptyState
                    }
                    ForEach(runner.transcript) { message in
                        messageRow(message)
                            .id(message.id)
                    }
                }
                .padding(16)
            }
            .onChange(of: runner.transcript.count) { _, _ in
                scrollToEnd(proxy)
            }
            .onChange(of: runner.transcript.last?.text) { _, _ in
                scrollToEnd(proxy)
            }
        }
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

    private func messageRow(_ message: ChatMessage) -> some View {
        let isUser = message.role == .user
        return HStack {
            if isUser { Spacer(minLength: 40) }
            Group {
                if message.text.isEmpty && runner.isRunning && message.role == .assistant {
                    ProgressView()
                        .scaleEffect(0.7)
                } else {
                    Text(message.text)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
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

    private func scrollToEnd(_ proxy: ScrollViewProxy) {
        guard let lastID = runner.transcript.last?.id else { return }
        withAnimation(.easeOut(duration: 0.15)) {
            proxy.scrollTo(lastID, anchor: .bottom)
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
