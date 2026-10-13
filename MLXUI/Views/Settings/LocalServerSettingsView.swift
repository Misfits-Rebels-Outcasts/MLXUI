import SwiftUI
import AppKit

/// Settings → Local Server (S1-5, R12's fifth pane; slimmed by SC-4b): the "Enable Local Server" option (gate J),
/// the port, and a link to the Local Server screen. Status, the running switch, the models, stats and the request log
/// live on that screen. The tab itself is only built when `LocalServerGate.isAvailable`; the rest of the pane shows
/// only when serving is enabled.
struct LocalServerSettingsView: View {
    @Environment(AppState.self) private var appState
    @State private var portText = ""

    private var controller: LocalServerController { appState.localServer }

    var body: some View {
        Form {
            // SC-4a (gate J): the first control. Everything below exists only while it is on.
            Section {
                Toggle("Enable Local Server", isOn: Binding(
                    get: { controller.optionEnabled },
                    set: { on in Task { await controller.setOptionEnabled(on) } }))
                Text("Lets apps on this Mac, like OpenCode or Xcode, use your installed models. Adds Local Server to the sidebar.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if controller.surfacesVisible {
                Section {
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
                    if case .failed(let reason) = controller.server.status {
                        Label(reason, systemImage: "exclamationmark.triangle.fill").foregroundStyle(.red)
                    }
                    Button("Open Local Server") { openScreen() }
                }
            }
        }
        .formStyle(.grouped)
        .onAppear { portText = String(controller.server.settings.port) }
    }

    private func applyPort() {
        guard let port = Int(portText) else { return }
        Task {
            await controller.setPort(port)
            portText = String(controller.server.settings.port)      // shows the clamped value
        }
    }

    /// Selects the screen in the main window and brings that window forward (Settings is its own window).
    private func openScreen() {
        appState.selectedSection = .server
        appState.route = []
        NSApp.windows.first { $0.canBecomeMain && $0.title != "Settings" && !($0 is NSPanel) }?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}
