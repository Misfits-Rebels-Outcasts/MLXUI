import SwiftUI

struct CommandPaletteView: View {
    @Environment(AppState.self) private var appState
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""
    @FocusState private var isFocused: Bool

    private var results: [ModelEntry] {
        guard let data = appState.browserData, !query.isEmpty else { return [] }
        let q = query.lowercased()
        return data.domains
            .flatMap { $0.allModels }
            .filter { model in
                model.displayName.localizedCaseInsensitiveContains(q) ||
                model.family.localizedCaseInsensitiveContains(q) ||
                (model.summary ?? "").localizedCaseInsensitiveContains(q) ||
                (model.description ?? "").localizedCaseInsensitiveContains(q)
            }
            .prefix(20)
            .map { $0 }
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search models...", text: $query)
                    .textFieldStyle(.plain)
                    .font(.title3)
                    .focused($isFocused)
                if !query.isEmpty {
                    Button { query = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(12)

            if !results.isEmpty {
                Divider()
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(results) { model in
                            Button {
                                dismiss()
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                    appState.selectedModel = model
                                }
                            } label: {
                                HStack(spacing: 10) {
                                    Image(systemName: model.modelType.sfSymbol)
                                        .foregroundStyle(.secondary)
                                        .frame(width: 24)
                                    VStack(alignment: .leading, spacing: 1) {
                                        Text(model.displayName)
                                            .font(.body)
                                        Text("\(model.family) · \(model.paramSize)")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    SourceBadge(source: model.source)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                                .contentShape(Rectangle())
                            }
                            .buttonStyle(.plain)
                            Divider().padding(.leading, 46)
                        }
                    }
                }
            } else if !query.isEmpty {
                Divider()
                VStack(spacing: 8) {
                    Image(systemName: "magnifyingglass")
                        .font(.title)
                        .foregroundStyle(.secondary)
                        .padding(.top, 40)
                    Text("No models found for \"\(query)\"")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity)
            }
        }
        .frame(width: 500, height: 400)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .onAppear { isFocused = true }
        .onKeyPress(.escape) {
            dismiss()
            return .handled
        }
    }
}
