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
                ServerScreenContent(screen: screen, controller: controller,
                                    systemRAMGB: appState.systemInfo.totalRAMGB, refresh: refresh)
                    .frame(maxWidth: ServerStyle.maxContentWidth)
                    .padding(ServerStyle.pagePadding)
                    .frame(maxWidth: .infinity)
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

/// One spacing scale for the whole screen (SC-4d).
enum ServerStyle {
    static let maxContentWidth: CGFloat = 820
    static let pagePadding: CGFloat = 24
    static let sectionSpacing: CGFloat = 16
    static let cardPadding: CGFloat = 16
    static let rowSpacing: CGFloat = 12
    static let corner: CGFloat = 12
}

/// The screen's content for a given model — split out of `ServerView` so it can be rendered without an `AppState`.
struct ServerScreenContent: View {
    let screen: ServerScreenModel
    let controller: LocalServerController
    let systemRAMGB: Double
    let refresh: () async -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: ServerStyle.sectionSpacing) {
            if let warning = screen.lowDiskWarning { LowDiskCallout(text: warning) }
            ServerHeader(screen: screen, controller: controller, refresh: refresh)
            ServedLoadWaitingSection()
            ServerModelsSection(screen: screen, controller: controller, systemRAMGB: systemRAMGB, refresh: refresh)
            ServerMemorySection(memory: screen.memory)
            ServerStatsSection(rows: screen.stats)
            ServerCard("Recent requests", symbol: "list.bullet.rectangle") {
                RequestLogList(log: controller.server.requestLog)
            }
            Text(screen.diskFact).font(.caption).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// A titled card: SF Symbol + title (+ an optional right-aligned detail), then the content.
private struct ServerCard<Content: View>: View {
    let title: String
    let symbol: String
    let detail: String?
    @ViewBuilder let content: Content

    init(_ title: String, symbol: String, detail: String? = nil, @ViewBuilder content: () -> Content) {
        self.title = title; self.symbol = symbol; self.detail = detail; self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: ServerStyle.rowSpacing) {
            HStack(alignment: .firstTextBaseline) {
                Label(title, systemImage: symbol).font(.headline)
                Spacer(minLength: 8)
                if let detail { Text(detail).font(.callout).monospacedDigit().foregroundStyle(.secondary) }
            }
            content
        }
        .padding(ServerStyle.cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: ServerStyle.corner))
        .overlay(RoundedRectangle(cornerRadius: ServerStyle.corner).strokeBorder(Color.primary.opacity(0.08)))
    }
}

private struct LowDiskCallout: View {
    let text: String

    var body: some View {
        Label {
            Text(text).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
        }
        .padding(ServerStyle.rowSpacing)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.14), in: RoundedRectangle(cornerRadius: 10))
        .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Color.orange.opacity(0.35)))
    }
}

/// SC-5b (gate D): flows parked on "Loading X will stop keeping Y ready…" — answerable here, never auto-answered.
private struct ServedLoadWaitingSection: View {
    private let waiting = ServedLoadWaiting.shared

    var body: some View {
        ForEach(waiting.items) { item in
            GroupBox {
                VStack(alignment: .leading, spacing: 8) {
                    Label("A flow is waiting: \(item.sentence)", systemImage: "hourglass")
                        .fixedSize(horizontal: false, vertical: true)
                    Text(item.flow).font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Button("Load anyway") { item.answer(true) }
                        Button("Cancel", role: .cancel) { item.answer(false) }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(4)
            }
        }
    }
}

private struct ServerHeader: View {
    let screen: ServerScreenModel
    let controller: LocalServerController
    let refresh: () async -> Void
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 10) {
                Circle().fill(color).frame(width: 10, height: 10)
                Text(headline)
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(screen.runState == .failed ? Color.red : Color.primary)
                    .lineLimit(2)
                if let address = screen.address {
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(address, forType: .string)
                        copied = true
                        Task { try? await Task.sleep(for: .seconds(1.5)); copied = false }
                    } label: {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .help("Copy \(address)")
                    .accessibilityLabel("Copy address")
                }
                Spacer(minLength: 12)
                Toggle("Let other apps on this Mac use my models", isOn: Binding(
                    get: { screen.switchOn },
                    set: { on in Task { await controller.setServerEnabled(on); await refresh() } }))
                    .toggleStyle(.switch)
                    .labelsHidden()
                    .help("Let other apps on this Mac use my models")
            }
            HStack {
                Text("Let other apps on this Mac use my models").font(.callout).foregroundStyle(.secondary)
                Spacer()
                Button("Connect your app…") { controller.presentConnect(for: nil) }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            }
        }
        .padding(ServerStyle.cardPadding)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: ServerStyle.corner))
        .overlay(RoundedRectangle(cornerRadius: ServerStyle.corner).strokeBorder(Color.primary.opacity(0.08)))
    }

    private var headline: String {
        switch screen.runState {
        case .running: return "Running" + (screen.address.map { " · \($0)" } ?? "")
        case .stopped: return "Stopped"
        case .failed: return screen.statusSentence
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
        ServerCard("Models", symbol: "cpu") {
            VStack(alignment: .leading, spacing: 0) {
                if screen.rows.isEmpty {
                    Text("No installed chat models. Install one from Browse to serve it.").foregroundStyle(.secondary)
                }
                ForEach(Array(screen.rows.enumerated()), id: \.element.id) { index, row in
                    if index > 0 { Divider().padding(.vertical, 8) }
                    ServerModelRow(row: row, controller: controller, systemRAMGB: systemRAMGB, refresh: refresh)
                }
            }
        }
    }
}

private struct ServerModelRow: View {
    let row: ServerScreenModel.Row
    let controller: LocalServerController
    let systemRAMGB: Double
    let refresh: () async -> Void

    var body: some View {
        let blocked = row.isAppleFoundation ? nil : ServeMemory.blockedReason(modelRAMGB: row.ramGB, systemRAMGB: systemRAMGB)
        let facts = [row.sizeText, row.contextText].compactMap { $0 }.joined(separator: " · ")
        HStack(alignment: .top, spacing: 12) {
            Toggle(isOn: Binding(
                get: { row.isServed },
                set: { on in Task { await controller.setServed(row.id, on); await refresh() } })) { EmptyView() }
                .toggleStyle(.checkbox)
                .labelsHidden()
                .disabled(blocked != nil && !row.isServed)
                .help(blocked ?? "Let other apps on this Mac use this model.")
                .accessibilityLabel("Serve \(row.name)")
            VStack(alignment: .leading, spacing: 2) {
                Text(row.name).lineLimit(1).truncationMode(.middle)
                if !facts.isEmpty { Text(facts).font(.caption).foregroundStyle(.secondary) }
            }
            Spacer(minLength: 8)
            if let line = row.stateLine {
                let state = ModelStateChip.State(line: line)
                VStack(alignment: .trailing, spacing: 2) {
                    ModelStateChip(state: state)
                    if let reason = state.reason {
                        Text(reason).font(.caption).foregroundStyle(.secondary)
                            .multilineTextAlignment(.trailing).frame(maxWidth: 260, alignment: .trailing)
                    }
                }
            }
        }
    }
}

/// The state chip: a dot (or a small spinner) and the state word. Parsed from the model's §UI state line, so the
/// wording stays owned by `ServerScreenModel.stateLine`.
private struct ModelStateChip: View {
    struct State {
        let label: String
        let reason: String?
        let color: Color
        let loading: Bool

        init(line: String) {
            let parts = line.components(separatedBy: " · ")
            label = parts[0]
            let rest = parts.dropFirst().joined(separator: " · ")
            reason = rest.isEmpty ? nil : rest
            switch parts[0] {
            case "Ready": color = .green; loading = false
            case "Loading…": color = .orange; loading = true
            case "Couldn't load": color = .red; loading = false
            default: color = .secondary; loading = false
            }
        }
    }
    let state: State

    var body: some View {
        HStack(spacing: 6) {
            if state.loading { ProgressView().controlSize(.mini) }
            else { Circle().fill(state.color).frame(width: 8, height: 8) }
            Text(state.label).font(.callout)
        }
        .padding(.horizontal, 10).padding(.vertical, 3)
        .background(state.color.opacity(0.12), in: Capsule())
    }
}

private struct ServerMemorySection: View {
    let memory: ServeMemoryReading?

    var body: some View {
        ServerCard("Memory", symbol: "memorychip",
                   detail: memory.flatMap { $0.capacity > 0 ? "\(Self.gb($0.used)) of \(Self.gb($0.capacity))" : nil }) {
            if let memory, memory.capacity > 0 {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.primary.opacity(0.10))
                        HStack(spacing: 1) {
                            ForEach(slices(memory), id: \.label) { slice in
                                Rectangle().fill(slice.color)
                                    .frame(width: geo.size.width * CGFloat(min(1, Double(slice.bytes) / Double(memory.capacity))))
                            }
                        }
                        .clipShape(Capsule())
                    }
                }
                .frame(height: 16)
                FlowLayoutRow(spacing: 16) {
                    ForEach(slices(memory), id: \.label) { slice in
                        HStack(spacing: 6) {
                            Circle().fill(slice.color).frame(width: 8, height: 8)
                            Text(slice.label).foregroundStyle(.secondary)
                            Text(Self.gb(slice.bytes)).monospacedDigit()
                        }
                        .font(.caption)
                    }
                }
                PromptCacheLine()
            } else {
                Text("—").foregroundStyle(.secondary)
            }
        }
    }

    private struct Slice { let label: String; let bytes: Int64; let color: Color }
    private func slices(_ m: ServeMemoryReading) -> [Slice] {
        [Slice(label: "Served models, kept ready", bytes: m.servedKeptReady, color: .accentColor),
         Slice(label: "Other models", bytes: m.otherModels, color: .orange),
         Slice(label: "Flow, Run windows and other", bytes: m.engines, color: .purple),
         Slice(label: "Prompt-cache snapshots", bytes: m.snapshots, color: .teal)]
    }
    static func gb(_ bytes: Int64) -> String { String(format: "%.1f GB", Double(bytes) / 1_073_741_824) }
}

/// A row that wraps onto more lines when the window is narrow.
private struct FlowLayoutRow: Layout {
    var spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        arrange(width: proposal.width ?? .infinity, subviews: subviews).size
    }
    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        let result = arrange(width: bounds.width, subviews: subviews)
        for (index, origin) in result.origins.enumerated() {
            subviews[index].place(at: CGPoint(x: bounds.minX + origin.x, y: bounds.minY + origin.y), proposal: .unspecified)
        }
    }
    private func arrange(width: CGFloat, subviews: Subviews) -> (size: CGSize, origins: [CGPoint]) {
        var origins: [CGPoint] = []
        var x: CGFloat = 0, y: CGFloat = 0, lineHeight: CGFloat = 0, maxX: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width { x = 0; y += lineHeight + 6; lineHeight = 0 }
            origins.append(CGPoint(x: x, y: y))
            x += size.width + spacing; maxX = max(maxX, x - spacing)
            lineHeight = max(lineHeight, size.height)
        }
        return (CGSize(width: maxX, height: y + lineHeight), origins)
    }
}

private struct ServerStatsSection: View {
    let rows: [ServerScreenModel.StatsRow]

    var body: some View {
        ServerCard("Stats", symbol: "chart.bar") {
            if rows.isEmpty {
                Text("No requests yet.").foregroundStyle(.secondary)
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    Grid(alignment: .trailing, horizontalSpacing: 18, verticalSpacing: 8) {
                        GridRow {
                            Text("Model").gridColumnAlignment(.leading)
                            Text("Requests"); Text("503s"); Text("First token"); Text("Decode"); Text("Reused"); Text("Last")
                        }
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                        Divider().gridCellUnsizedAxes(.horizontal)
                        ForEach(rows) { row in
                            GridRow {
                                Text(row.name).lineLimit(1).truncationMode(.middle).frame(maxWidth: 220, alignment: .leading)
                                Text("\(row.requests)")
                                Text("\(row.unavailable)")
                                Text(row.ttft)
                                Text(row.decode)
                                Text(row.reused)
                                Text(row.lastRequest.formatted(date: .omitted, time: .standard)).foregroundStyle(.secondary)
                            }
                            .font(.callout).monospacedDigit()
                        }
                    }
                }
            }
        }
    }
}
