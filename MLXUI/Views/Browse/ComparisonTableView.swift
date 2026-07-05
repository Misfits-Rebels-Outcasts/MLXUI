import SwiftUI

struct ComparisonTableView: View {
    let models: [ModelEntry]
    @Environment(\.dismiss) private var dismiss
    @Environment(AppState.self) private var appState

    private var rows: [(String, (ModelEntry) -> String, Bool)] {
        [
            ("Family", { $0.family }, false),
            ("Size", { $0.paramSize }, false),
            ("RAM", { String(format: "%.1f GB", $0.ramGB) }, true),
            ("Speed", { $0.speedTokensPerSec.map { "~\(Int($0)) tok/s" } ?? "—" }, false),
            ("Context", { $0.contextWindow.map { $0 >= 1000 ? "\($0/1000)K" : "\($0)" } ?? "—" }, false),
            ("Download", { String(format: "%.1f GB", $0.downloadSizeGB) }, true),
            ("License", { DataNormalizer.normalizeLicense($0.license) }, false),
            ("Downloads", { $0.formattedDownloads() }, false),
            ("Source", { $0.source.rawValue.uppercased() }, false),
        ]
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Compare (\(models.count) models)")
                    .font(.headline)
                Spacer()
                Button("Done") { dismiss() }
                    .buttonStyle(.bordered)
            }
            .padding()

            ScrollView([.horizontal, .vertical]) {
                Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                    // Header row
                    GridRow {
                        Text("")
                            .frame(width: 100)
                            .gridCellColumns(1)
                        ForEach(models) { model in
                            VStack(spacing: 4) {
                                Text(model.displayName)
                                    .font(.subheadline)
                                    .fontWeight(.bold)
                                Text(model.paramSize)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Button {
                                    appState.toggleComparison(model)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                .buttonStyle(.plain)
                            }
                            .frame(width: 120)
                            .padding(.vertical, 8)
                            .background(.quaternary.opacity(0.3))
                        }
                    }

                    Divider()

                    // Data rows
                    ForEach(rows, id: \.0) { row in
                        GridRow {
                            Text(row.0)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .frame(width: 100, alignment: .leading)
                                .padding(.vertical, 6)
                                .padding(.horizontal, 8)

                            ForEach(models) { model in
                                let value = row.1(model)
                                let isBest = row.2 && isBestValue(row, model: model)
                                Text(value)
                                    .font(.caption)
                                    .fontWeight(isBest ? .bold : .regular)
                                    .foregroundStyle(isBest ? .green : .primary)
                                    .frame(width: 120)
                                    .padding(.vertical, 6)
                                    .background(isBest ? Color.green.opacity(0.08) : Color.clear)
                            }
                        }
                        Divider()
                    }
                }
                .padding(.horizontal)
            }

            if models.count < 5 {
                Text("Check compare boxes on model cards to add more (max 5).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding()
            }
        }
        .frame(minWidth: 500, minHeight: 400)
    }

    private func isBestValue(_ row: (String, (ModelEntry) -> String, Bool), model: ModelEntry) -> Bool {
        guard models.count > 1 else { return false }
        let name = row.0
        switch name {
        case "RAM":
            return model.ramGB == models.map(\.ramGB).min()
        case "Download":
            return model.downloadSizeGB == models.map(\.downloadSizeGB).min()
        default:
            return false
        }
    }
}
