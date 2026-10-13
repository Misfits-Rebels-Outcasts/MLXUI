import SwiftUI

/// The Local Server screen (SC-4b, gate A): status, the models that can be served, memory, stats and the request log.
/// It renders `ServerScreenModel` and nothing else. **It never gets Run, Uninstall, Install or a catalog view** — it
/// lists only servable models, and "Serve just serves" (R51/R56): plain facts, no ratings or badges.
struct ServerView: View {
    @Environment(AppState.self) private var appState
    @State private var screen: ServerScreenModel?
    @State private var contexts: [String: Int?] = [:]

    private var controller: LocalServerController { appState.localServer }

    var body: some View {
        ScrollView {
            if let screen {
                VStack(alignment: .leading, spacing: 20) {
                    if let warning = screen.lowDiskWarning {
                        Label(warning, systemImage: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                    }
                    ServerStatusSection(screen: screen, controller: controller, refresh: refresh)
                    ServerModelsSection(screen: screen, controller: controller,
                                        systemRAMGB: appState.systemInfo.totalRAMGB, refresh: refresh)
                    ServerMemorySection(memory: screen.memory)
                    ServerStatsSection(rows: screen.stats)
                    GroupBox("Recent requests") { RequestLogList(log: controller.server.requestLog).padding(4) }
                    Text(screen.diskFact).font(.caption).foregroundStyle(.secondary)
                }
                .padding(20)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .navigationTitle("Local Server")
        .task {
            while !Task.isCancelled {
                await refresh()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    @MainActor private func refresh() async {
        let memory = await controller.memoryReading()
        let ids = controller.installedIndex().entries.filter { $0.kind == .llm }.map(\.hfModelId)
        let missing = ids.filter { contexts[$0] == nil }
        if !missing.isEmpty {
            let found = await Task.detached { missing.map { ($0, ServeModelFacts.connectFacts(hfModelID: $0).contextLength) } }.value
            for (id, tokens) in found { contexts[id] = .some(tokens) }
        }
        screen = controller.screenModel(memory: memory, freeDiskBytes: ServeDisk.freeBytes(),
                                        contextLength: { contexts[$0] ?? nil })
    }
}

private struct ServerStatusSection: View {
    let screen: ServerScreenModel
    let controller: LocalServerController
    let refresh: () async -> Void

    var body: some View {
        GroupBox("Status") {
            VStack(alignment: .leading, spacing: 8) {
                Toggle("Let other apps on this Mac use my models", isOn: Binding(
                    get: { screen.switchOn },
                    set: { on in Task { await controller.setServerEnabled(on); await refresh() } }))
                HStack {
                    Image(systemName: symbol).foregroundStyle(color)
                    Text(screen.statusSentence).foregroundStyle(screen.runState == .failed ? Color.red : Color.primary)
                    Spacer()
                    Button("Connect your app…") { controller.presentConnect(for: nil) }
                }
                if let address = screen.address {
                    Text(address).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                }
            }
            .padding(4)
        }
    }

    private var symbol: String {
        switch screen.runState {
        case .running: return "circle.fill"
        case .failed: return "exclamationmark.triangle.fill"
        case .stopped: return "circle"
        }
    }
    private var color: Color {
        switch screen.runState {
        case .running: return .green
        case .failed: return .red
        case .stopped: return .secondary
        }
    }
}

private struct ServerModelsSection: View {
    let screen: ServerScreenModel
    let controller: LocalServerController
    let systemRAMGB: Double
    let refresh: () async -> Void

    var body: some View {
        GroupBox("Models") {
            VStack(alignment: .leading, spacing: 10) {
                if screen.rows.isEmpty {
                    Text("No installed chat models. Install one from Browse to serve it.").foregroundStyle(.secondary)
                }
                ForEach(screen.rows) { row in
                    let blocked = row.isAppleFoundation ? nil : ServeMemory.blockedReason(modelRAMGB: row.ramGB, systemRAMGB: systemRAMGB)
                    HStack(alignment: .firstTextBaseline) {
                        Toggle(isOn: Binding(
                            get: { row.isServed },
                            set: { on in Task { await controller.setServed(row.id, on); await refresh() } })) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(row.name)
                                let facts = [row.sizeText, row.contextText].compactMap { $0 }.joined(separator: " · ")
                                if !facts.isEmpty { Text(facts).font(.caption).foregroundStyle(.secondary) }
                                if let line = row.stateLine { Text(line).font(.caption) }
                            }
                        }
                        .toggleStyle(.checkbox)
                        .disabled(blocked != nil && !row.isServed)
                        .help(blocked ?? "Let other apps on this Mac use this model.")
                    }
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct ServerMemorySection: View {
    let memory: ServeMemoryReading?

    var body: some View {
        GroupBox("Memory") {
            VStack(alignment: .leading, spacing: 8) {
                if let memory, memory.capacity > 0 {
                    GeometryReader { geo in
                        HStack(spacing: 0) {
                            ForEach(slices(memory), id: \.label) { slice in
                                Rectangle().fill(slice.color)
                                    .frame(width: geo.size.width * CGFloat(min(1, Double(slice.bytes) / Double(memory.capacity))))
                            }
                            Spacer(minLength: 0)
                        }
                        .background(Color.secondary.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 4))
                    }
                    .frame(height: 12)
                    ForEach(slices(memory), id: \.label) { slice in
                        Label("\(slice.label): \(Self.gb(slice.bytes))", systemImage: "square.fill")
                            .foregroundStyle(slice.color, .primary).font(.caption)
                    }
                    Text("\(Self.gb(memory.used)) of \(Self.gb(memory.capacity)) in use").font(.callout)
                    PromptCacheLine()
                } else {
                    Text("—").foregroundStyle(.secondary)
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private struct Slice { let label: String; let bytes: Int64; let color: Color }
    private func slices(_ m: ServeMemoryReading) -> [Slice] {
        [Slice(label: "Served models, kept ready", bytes: m.servedKeptReady, color: .accentColor),
         Slice(label: "Other models", bytes: m.otherModels, color: .orange),
         Slice(label: "Flow and Run engines", bytes: m.engines, color: .purple),
         Slice(label: "Prompt-cache snapshots", bytes: m.snapshots, color: .teal)]
    }
    static func gb(_ bytes: Int64) -> String { String(format: "%.1f GB", Double(bytes) / 1_073_741_824) }
}

private struct ServerStatsSection: View {
    let rows: [ServerScreenModel.StatsRow]

    var body: some View {
        GroupBox("Stats") {
            VStack(alignment: .leading, spacing: 10) {
                if rows.isEmpty { Text("No requests yet.").foregroundStyle(.secondary) }
                ForEach(rows) { row in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(row.name)
                        Text("\(row.requests) requests · \(row.unavailable) × 503 · time to first token (incl. waiting) \(row.ttft) · "
                             + "decode \(row.decode) · prompt reused \(row.reused) · last \(row.lastRequest.formatted(date: .omitted, time: .standard))")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .padding(4)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}
