import SwiftUI

struct ModelDetailView: View {
    let model: ModelEntry
    @Environment(AppState.self) private var appState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Hero
                HStack(alignment: .top, spacing: 16) {
                    Image(systemName: model.modelType.sfSymbol)
                        .font(.system(size: 48))
                        .foregroundStyle(.secondary)
                        .frame(width: 80, height: 80)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))

                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text(model.displayName)
                                .font(.title)
                                .fontWeight(.bold)
                            SourceBadge(source: model.source)
                            if let score = model.qualityScore {
                                StarRating(rating: score)
                            }
                        }
                        Text(model.paramSize)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        Text(model.displaySummary)
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .padding(.top, 4)
                    }
                }

                // Action buttons — kept directly under the header/description so Run and
                // Uninstall are reachable without scrolling past the model's details.
                actionButtons

                // Flag models whose architecture has no MLX runner yet (see ModelSupport /
                // backlog § "Model support gaps"): install works, but Run will error.
                if let reason = ModelSupport.unsupportedReason(for: model) {
                    Label(reason, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
                }

                Divider()

                // Quick Info — columns are `.leading` so short tiles (RAM, License, Updated)
                // don't get centered in their column (which left a gap on the left).
                LazyVGrid(
                    columns: [
                        GridItem(.flexible(), alignment: .leading),
                        GridItem(.flexible(), alignment: .leading),
                    ],
                    spacing: 12
                ) {
                    if let ctx = model.contextWindow {
                        infoTile("Context Window", "\(ctx >= 1000 ? "\(ctx / 1000)K" : "\(ctx)") tokens")
                    }
                    infoTile("RAM", "\(String(format: "%.1f", model.ramGB)) GB")
                    if let speed = model.speedTokensPerSec {
                        infoTile("Speed", "~\(Int(speed)) tok/s")
                    }
                    infoTile("Download", "\(String(format: "%.1f", model.downloadSizeGB)) GB")
                    infoTile("License", DataNormalizer.normalizeLicense(model.license))
                    if let arch = model.architecture {
                        infoTile("Architecture", DataNormalizer.normalizeArchitecture(arch))
                    }
                    if let date = model.lastUpdated {
                        infoTile("Updated", date)
                    }
                    infoTile("Downloads", model.formattedDownloads())
                }

                // About (full model-card description)
                if let desc = model.description,
                   !desc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                   desc != model.displaySummary {
                    Divider()
                    VStack(alignment: .leading, spacing: 8) {
                        Text("About")
                            .font(.headline)
                        Text(descriptionMarkdown(desc))
                            .font(.body)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        if let src = model.descriptionSource,
                           src != "generated", src != "curated" {
                            Text("Source: \(src)")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                        }
                    }
                }

                Divider()

                // Variant
                if let variant = model.variants?.first {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Variant")
                            .font(.headline)
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Text(variant.quantization)
                                .fontWeight(.medium)
                            Text("· \(String(format: "%.1f", variant.ramGB)) GB · \(String(format: "%.1f", variant.downloadSizeGB)) GB")
                                .foregroundStyle(.secondary)
                        }
                        .font(.subheadline)
                    }
                }

                Divider()

                // Benchmarks
                VStack(alignment: .leading, spacing: 8) {
                    Text("Benchmarks")
                        .font(.headline)
                    if let b = model.benchmarks {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                            benchmarkItem("MMLU", b.mmlu)
                            benchmarkItem("HumanEval", b.humanEval)
                            benchmarkItem("GSM8K", b.gsm8k)
                            benchmarkItem("HellaSwag", b.hellaswag)
                            benchmarkItem("ARC", b.arc)
                            benchmarkItem("TruthfulQA", b.truthfulQA)
                        }
                    } else {
                        Text("Benchmark data is not yet available for this model.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(24)
        }
    }

    /// Install / Run / Uninstall controls, driven by the model's current install state.
    @ViewBuilder private var actionButtons: some View {
        let state = appState.installManager.modelStates[model.id] ?? .idle
        HStack(spacing: 12) {
            switch state {
            case .downloading(let progress, let downloaded, let total):
                VStack(alignment: .leading, spacing: 4) {
                    ProgressView(value: progress)
                        .frame(width: 200)
                    Text("\(formatBytes(downloaded)) / \(formatBytes(total))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Cancel") { appState.cancelInstall(model.id) }
                        .buttonStyle(.plain)
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            case .installed:
                HStack(spacing: 12) {
                    Button { appState.runModel(model) } label: {
                        Label("Run", systemImage: "play.fill").frame(minWidth: 100)
                    }
                    .buttonStyle(.borderedProminent)
                    Button {
                        appState.uninstallModel(model.id)
                    } label: {
                        Label("Uninstall", systemImage: "trash").frame(minWidth: 100)
                    }
                    .buttonStyle(.bordered)
                }
            case .error(let msg, let canRetry):
                VStack(alignment: .leading, spacing: 4) {
                    Text(msg).font(.caption).foregroundStyle(.red)
                    if canRetry {
                        Button("Retry") { appState.installModel(model) }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                    }
                }
            case .needsAuth(let msg):
                VStack(alignment: .leading, spacing: 6) {
                    Label(msg, systemImage: "lock.fill")
                        .font(.caption).foregroundStyle(.orange)
                    HStack(spacing: 8) {
                        Button("Open Settings") { appState.showSettings = true }
                            .buttonStyle(.borderedProminent).controlSize(.small)
                        Button("Retry") { appState.installModel(model) }
                            .buttonStyle(.bordered).controlSize(.small)
                    }
                }
            case .resolving:
                HStack { ProgressView().scaleEffect(0.8); Text("Resolving...").font(.caption) }
            case .verifying:
                Text("Verifying...").font(.caption).foregroundStyle(.secondary)
            default:
                if appState.installedModelIDs.contains(model.id) || appState.installManager.isInstalled(model.id) {
                    HStack(spacing: 12) {
                        Button { appState.runModel(model) } label: {
                            Label("Run", systemImage: "play.fill").frame(minWidth: 100)
                        }
                        .buttonStyle(.borderedProminent)
                        Button { appState.uninstallModel(model.id) } label: {
                            Label("Uninstall", systemImage: "trash").frame(minWidth: 100)
                        }
                        .buttonStyle(.bordered)
                    }
                } else if model.exceedsRAM(appState.systemInfo.totalRAMGB) {
                    Button {} label: {
                        Label("Needs \(String(format: "%.0f", model.ramGB)) GB RAM", systemImage: "xmark.circle")
                            .frame(minWidth: 120)
                    }
                    .buttonStyle(.bordered).disabled(true)
                } else {
                    Button { appState.installModel(model) } label: {
                        Label("Install", systemImage: "arrow.down.circle").frame(minWidth: 120)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            Spacer(minLength: 0)
        }
    }

    /// Renders the model-card description: strips block heading markers (#) so
    /// they read as plain lines, and interprets inline markdown (bold, links)
    /// while preserving paragraph breaks.
    private func descriptionMarkdown(_ raw: String) -> AttributedString {
        let cleaned = raw.replacingOccurrences(
            of: "(?m)^#{1,6}[ \t]*", with: "", options: .regularExpression)
        return (try? AttributedString(
            markdown: cleaned,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)))
            ?? AttributedString(cleaned)
    }

    private func infoTile(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
        }
    }

    private func benchmarkItem(_ name: String, _ value: Double?) -> some View {
        HStack(spacing: 4) {
            Text(name)
                .font(.caption)
                .foregroundStyle(.secondary)
            if let v = value {
                Text(String(format: "%.1f", v))
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .monospacedDigit()
            } else {
                Text("—")
                    .font(.subheadline)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(8)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
    }

    private func formatBytes(_ bytes: Int64) -> String {
        if bytes >= 1_000_000_000 { return String(format: "%.1f GB", Double(bytes) / 1e9) }
        if bytes >= 1_000_000 { return String(format: "%.1f MB", Double(bytes) / 1e6) }
        if bytes >= 1_000 { return String(format: "%.0f KB", Double(bytes) / 1e3) }
        return "\(bytes) B"
    }
}
