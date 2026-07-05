import SwiftUI

struct ModelCard: View {
    let model: ModelEntry
    var showCompare = true

    @Environment(AppState.self) private var appState
    @State private var isHovered = false

    private static let cardHeight: CGFloat = 244

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            topBar
            iconArea
            nameArea
            summaryArea
            specsArea
            Spacer(minLength: 0)
            actionButton
        }
        .frame(height: Self.cardHeight)
        .padding(10)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(appState.isInComparison(model) ? Color.blue : Color.clear, lineWidth: 2)
        )
        .opacity(model.exceedsRAM(appState.filterRAMLimitGB) ? 0.4 : 1.0)
        .onHover { isHovered = $0 }
    }

    // MARK: - Components

    private var topBar: some View {
        HStack {
            if showCompare, isHovered || appState.isInComparison(model) {
                Button {
                    appState.toggleComparison(model)
                } label: {
                    Image(systemName: appState.isInComparison(model)
                          ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(appState.isInComparison(model) ? .blue : .secondary)
                }
                .buttonStyle(.plain)
            }
            Spacer()
            SourceBadge(source: model.source)
        }
        .frame(height: 18)
    }

    private var iconArea: some View {
        Image(systemName: model.modelType.sfSymbol)
            .font(.system(size: 28))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 6)
    }

    private var nameArea: some View {
        VStack(spacing: 1) {
            Text(model.displayName)
                .font(.headline)
                .lineLimit(1)
                .frame(maxWidth: .infinity, alignment: .leading)
            Text(model.paramSize)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .frame(height: 32)
    }

    private var summaryArea: some View {
        Text(model.displaySummary)
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(2, reservesSpace: true)
            .multilineTextAlignment(.leading)
            .frame(maxWidth: .infinity, alignment: .topLeading)
            .padding(.vertical, 3)
            .help(model.displaySummary)
    }

    private var specsArea: some View {
        VStack(alignment: .leading, spacing: 3) {
            specRow("memorychip",
                    model.exceedsRAM(appState.systemInfo.totalRAMGB)
                    ? "—" : "\(String(format: "%.1f", model.ramGB)) GB RAM",
                    color: model.exceedsRAM(appState.systemInfo.totalRAMGB) ? .red : .primary)
            specRow("gauge.with.dots.needle.33percent",
                    model.speedTokensPerSec.map { "~\(Int($0)) tok/s" } ?? "—")
            specRow("rectangle.expand.vertical",
                    model.contextWindow.map { $0 >= 1000 ? "\($0 / 1000)K ctx" : "\($0) ctx" } ?? "—")
        }
    }

    private var actionButton: some View {
        let state = appState.installManager.modelStates[model.id] ?? .idle

        return Group {
            switch state {
            case .downloading(let progress, let downloaded, let total):
                VStack(spacing: 3) {
                    ProgressView(value: progress)
                        .progressViewStyle(.linear)
                    Text("\(formatBytes(downloaded)) / \(formatBytes(total))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Button {
                        appState.cancelInstall(model.id)
                    } label: {
                        Text("Cancel")
                            .font(.caption2)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.red)
                }
            case .installed:
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 5)
                    .background(.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                    .onTapGesture {
                        appState.runModel(model)
                    }
            case .error(let msg, let canRetry):
                VStack(spacing: 2) {
                    Text(msg).font(.caption2).foregroundStyle(.red).lineLimit(1)
                    if canRetry {
                        Button("Retry") { appState.installModel(model) }
                            .buttonStyle(.plain).font(.caption2).foregroundStyle(.blue)
                    }
                }
            case .needsAuth:
                VStack(spacing: 2) {
                    Text("Token needed").font(.caption2).foregroundStyle(.orange).lineLimit(1)
                    Button("Open Settings") { appState.showSettings = true }
                        .buttonStyle(.plain).font(.caption2).foregroundStyle(.blue)
                }
            case .resolving:
                HStack(spacing: 4) {
                    ProgressView().scaleEffect(0.6)
                    Text("Resolving...").font(.caption2).foregroundStyle(.secondary)
                }
            case .verifying:
                Text("Verifying...").font(.caption2).foregroundStyle(.secondary)
            default:
                if appState.installedModelIDs.contains(model.id) || appState.installManager.isInstalled(model.id) {
                    Label("Installed", systemImage: "checkmark.circle.fill")
                        .font(.caption).foregroundStyle(.green)
                        .frame(maxWidth: .infinity).padding(.vertical, 5)
                        .background(.green.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                        .onTapGesture {
                            appState.runModel(model)
                        }
                } else if model.exceedsRAM(appState.systemInfo.totalRAMGB) {
                    Label("Needs \(String(format: "%.0f", model.ramGB)) GB", systemImage: "xmark.circle")
                        .font(.caption2).foregroundStyle(.red)
                        .frame(maxWidth: .infinity).padding(.vertical, 5)
                        .background(.red.opacity(0.1), in: RoundedRectangle(cornerRadius: 6))
                } else {
                    Button {
                        appState.installModel(model)
                    } label: {
                        Label("Install", systemImage: "arrow.down.circle")
                            .font(.caption).fontWeight(.medium).frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent).controlSize(.small)
                }
            }
        }
        .frame(height: 30)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        if bytes >= 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes >= 1_000_000 { return String(format: "%.1f MB", Double(bytes) / 1e6) }
        if bytes >= 1_000 { return String(format: "%.0f KB", Double(bytes) / 1e3) }
        return "\(bytes) B"
    }

    private func specRow(_ icon: String, _ label: String, color: Color = .primary) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .frame(width: 14)
                .foregroundStyle(color)
            Text(label)
                .font(.caption2)
                .foregroundStyle(color)
        }
    }
}
