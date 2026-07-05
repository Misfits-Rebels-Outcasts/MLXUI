import SwiftUI

struct SidebarView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        List(selection: Binding(
            get: { appState.selectedSection },
            set: { new in
                appState.selectedSection = new
                appState.resetFilters()
            }
        )) {
            Section {
                Label("Home", systemImage: "house.fill")
                    .tag(SidebarItem.home)
            }

            if appState.browserData != nil {
                Section("Browse") {
                    ForEach(appState.visibleSections) { section in
                        Label(section.name, systemImage: section.sfSymbol)
                            .tag(SidebarItem.browse(section.id))
                    }
                }
            }

            Section("Installed") {
                if appState.installedModelIDs.isEmpty {
                    Label("None installed", systemImage: "tray")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(appState.installedModelIDs).sorted(), id: \.self) { id in
                        Button {
                            if let model = installedModel(for: id) {
                                appState.selectedModel = model
                            }
                        } label: {
                            Label(displayName(for: id),
                                  systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button {
                                appState.uninstallModel(id)
                            } label: {
                                Label("Uninstall", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .frame(minWidth: 200)
        .toolbar(removing: .sidebarToggle)
    }

    private func installedModel(for modelId: String) -> ModelEntry? {
        appState.browserData?.domains
            .flatMap { $0.allModels }
            .first(where: { $0.id == modelId })
    }

    private func displayName(for modelId: String) -> String {
        installedModel(for: modelId)?.displayName
            ?? modelId.replacingOccurrences(of: "mlx-community--", with: "")
    }
}
