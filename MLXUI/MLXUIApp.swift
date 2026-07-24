import SwiftUI

@main
struct MLXUIApp: App {
    //Terminate app when window closed
    //https://developer.apple.com/forums/thread/710376
    class AppDelegate: NSObject, NSApplicationDelegate {
        func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
            return true
        }
    }
    @NSApplicationDelegateAdaptor private var appDelegate: AppDelegate

    
    @State private var appState = AppState()

    var body: some Scene {
        WindowGroup {
            Group {
                if let error = appState.loadError {
                    ErrorView(message: error, retry: { appState.loadBrowserData() })
                } else if appState.browserData == nil {
                    ProgressView("Loading catalog…")
                        .onAppear { appState.loadBrowserData() }
                } else {
                    NavigationSplitView {
                        SidebarView()
                    } detail: {
                        NavigationStack {
                            Group {
                                if appState.selectedSection.isHome {
                                    HomeView()
                                } else {
                                    BrowseView()
                                }
                            }
                            // Selecting a model (from the Installed sidebar list or the
                            // command palette) pushes its detail page onto the stack.
                            .navigationDestination(item: $appState.selectedModel) { model in
                                ModelDetailView(model: model)
                            }
                        }
                    }
                }
            }
            .environment(appState)
            // Sheets present in a fresh environment, so re-inject AppState into each
            // content closure — otherwise @Environment(AppState.self) lookups crash (B1).
            .sheet(isPresented: $appState.showCommandPalette) {
                CommandPaletteView()
                    .environment(appState)
            }
            .sheet(item: $appState.runningModel) { model in
                RunChatView(model: model)
                    .environment(appState)
            }
            // Registry-driven run sheet for non-LLM models (e.g. Whisper ASR): resolve the
            // model to its module and present the module's run view; fall back to a
            // graceful "unsupported" view if resolution fails.
            .sheet(item: $appState.asrRunModel) { model in
                Group {
                    if let resolved = appState.registry.bestModule(for: model),
                       let stage = try? resolved.sdk.makeStage(for: model, config: .default) {
                        resolved.ui.makeRunView(for: model, stage: stage)
                    } else {
                        UnsupportedModelView(model: model)
                    }
                }
                .environment(appState)   // B1 re-inject
            }
            .sheet(isPresented: $appState.showSettings) {
                SettingsView()
                    .environment(appState)
            }
            .onChange(of: appState.filterSource) { _, _ in appState.saveFilters() }
            .onChange(of: appState.sortOrder) { _, _ in appState.saveFilters() }
        }
        .windowResizability(.contentSize)
        .defaultSize(width: 1400, height: 900)
        .commands {
            CommandGroup(replacing: .newItem) {}
            CommandGroup(replacing: .appSettings) {
                Button("Settings…") {
                    appState.showSettings = true
                }
                .keyboardShortcut(",", modifiers: .command)
            }
            CommandMenu("Find") {
                Button("Find Models...") {
                    appState.showCommandPalette = true
                }
                .keyboardShortcut("k", modifiers: .command)
            }
            CommandGroup(replacing: .help) {
                Link("AI Browser Help", destination: URL(string: "https://connectcode.net/mlxui_local_llm_ai_browser.html")!)
            }
        }
    }
}
