import SwiftUI

/// Settings → Local Server (S1-5, R12's fifth pane): on/off, the status sentence, the port, the
/// served list with un-serve (Apple Foundation Models' Serve toggle lives here — it is not listed
/// as a model anywhere else in the app, journal 2026-370), the in-memory request log, and
/// "Connect your app…". The tab itself is only built when `LocalServerGate.isAvailable`.
struct LocalServerSettingsView: View {
    @Environment(AppState.self) private var appState
    @State private var portText = ""
    @State private var showConnect = false

    private var controller: LocalServerController { appState.localServer }

    var body: some View {
        Form {
            Section("Server") {
                Toggle("Let other apps on this Mac use my models", isOn: Binding(
                    get: { controller.enabled },
                    set: { on in Task { await controller.setServerEnabled(on) } }))
                HStack {
                    Image(systemName: statusSymbol).foregroundStyle(statusColor)
                    Text(controller.statusSentence)
                        .foregroundStyle(isFailed ? Color.red : Color.primary)
                    Spacer()
                    Button("Connect your app…") { showConnect = true }
                }
                Text(LocalServerPolicy.chatOnlyNote)
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Text("Port")
                    TextField("1212", text: $portText)
                        .frame(width: 80)
                        .onSubmit { applyPort() }
                    Button("Apply") { applyPort() }
                        .disabled(Int(portText) == controller.server.settings.port || Int(portText) == nil)
                    Text("Applies on restart — a running server restarts now.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }

            Section("Served models") {
                if controller.reachableIDs.isEmpty {
                    Text("Nothing is being served. Turn on Serve on a model's page, or below.")
                        .foregroundStyle(.secondary)
                }
                if let note = controller.memoryNote {
                    Label(note, systemImage: "memorychip")
                        .font(.callout)
                        .foregroundStyle(.orange)
                }
                ForEach(controller.reachableIDs.sorted(), id: \.self) { id in
                    HStack {
                        Text(id).font(.system(.body, design: .monospaced))
                        Spacer()
                        Button("Un-serve") { Task { await controller.setServed(id, false) } }
                    }
                }
                if let readiness = AppleFoundationAvailability.currentReadiness() {
                    Toggle(isOn: Binding(
                        get: { controller.isReachable(ServedModels.appleFoundationID) },
                        set: { on in Task { await controller.setServed(ServedModels.appleFoundationID, on) } })) {
                        VStack(alignment: .leading) {
                            Text("Apple Foundation Models (on this Mac)")
                            if readiness != .ready {
                                Text(ReadinessSentence.text(for: readiness)).font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                    .help("Let other apps on this Mac use this model.")
                }
            }

            Section("Recent requests") {
                RequestLogList(log: controller.server.requestLog)
            }
        }
        .formStyle(.grouped)
        .onAppear { portText = String(controller.server.settings.port) }
        .sheet(isPresented: $showConnect) {
            ConnectAppSheet(controller: controller, initialModelID: nil)
        }
    }

    private var isFailed: Bool { if case .failed = controller.server.status { return true } else { return false } }
    private var statusSymbol: String {
        switch controller.server.status {
        case .running: return "circle.fill"
        case .starting: return "circle.dotted"
        case .failed: return "exclamationmark.triangle.fill"
        case .stopped: return "circle"
        }
    }
    private var statusColor: Color {
        switch controller.server.status {
        case .running: return .green
        case .failed: return .red
        default: return .secondary
        }
    }

    private func applyPort() {
        guard let port = Int(portText) else { return }
        Task {
            await controller.setPort(port)
            portText = String(controller.server.settings.port)      // shows the clamped value
        }
    }
}

/// The last 200 requests, newest first — time, status, model, tokens, duration, caller, and
/// "tools ignored". **Metadata only**: the log has no field for content (rule 13).
struct RequestLogList: View {
    let log: RequestLog

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            let entries = Array(log.entries.reversed())
            if entries.isEmpty {
                Text("No requests yet.").foregroundStyle(.secondary)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                            Text(Self.line(entry))
                                .font(.system(.caption, design: .monospaced))
                                .foregroundStyle(entry.status >= 400 ? Color.red : Color.primary)
                                .textSelection(.enabled)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(height: 160)
            }
        }
    }

    static func line(_ entry: RequestLogEntry) -> String {
        var parts = [entry.time.formatted(date: .omitted, time: .standard), String(entry.status),
                     entry.model ?? "—"]
        if let prompt = entry.promptTokens, let completion = entry.completionTokens {
            parts.append("\(prompt)→\(completion) tok")
        }
        parts.append(String(format: "%.1fs", entry.durationSeconds))
        if let agent = entry.userAgent { parts.append(agent) }
        if entry.toolsIgnored { parts.append("tools ignored") }
        return parts.joined(separator: "  ")
    }
}
