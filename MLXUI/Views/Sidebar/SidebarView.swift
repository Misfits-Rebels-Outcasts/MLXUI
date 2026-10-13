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

            if !AppState.hideFlows, !AppState.hideAutomate, !appState.galleryEntries.isEmpty {
                Section("Automate") {
                    Label("Overview", systemImage: "info.circle")
                        .tag(SidebarItem.overview)
                    Label("AI Workflows", systemImage: "flowchart")
                        .tag(SidebarItem.aiWorkflows)
                }
            }

            // SC-4b: only while serving is enabled (the option on); `surfacesVisible` is the observed `isEnabled`.
            if appState.localServer.surfacesVisible {
                Section("Serve") {
                    Label("Local Server", systemImage: "server.rack")
                        .tag(SidebarItem.server)
                }
            }

            Section("Installed") {
                if appState.installedModelIDs.isEmpty {
                    Label("None installed", systemImage: "tray")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(appState.installedModelIDs).sorted(), id: \.self) { id in
                        Button {
                            // WA-5: a sidebar pick is a lateral jump, not a push deeper into
                            // whatever was on the stack — replace it, matching the old
                            // single-optional behavior of always showing this one frame.
                            if let model = installedModel(for: id) {
                                appState.route = [.model(model)]
                            }
                        } label: {
                            Label(displayName(for: id),
                                  systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button {
                                if let model = installedModel(for: id) {
                                    appState.uninstallModel(model)
                                }
                            } label: {
                                Label("Uninstall", systemImage: "trash")
                            }
                        }
                    }
                }
            }
        }
        .onChange(of: appState.localServer.surfacesVisible) { _, visible in
            appState.selectedSection = LocalServerPolicy.resolvedSidebarItem(appState.selectedSection, serving: visible)
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
