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
                                } else if case .overview = appState.selectedSection, !AppState.hideAutomate {
                                    OverviewView()
                                } else if case .aiWorkflows = appState.selectedSection, !AppState.hideAutomate {
                                    FlowGalleryView()
                                } else if let opened = appState.openedCatFlow {
                                    OpenedFlowView(opened: opened)
                                        .id(opened.url.path)
                                } else {
                                    BrowseView()
                                }
                            }
                            // Selecting a model (from the Installed sidebar list or the
                            // command palette) pushes its detail page onto the stack.
                            .navigationDestination(item: $appState.selectedModel) { model in
                                ModelDetailView(model: model)
                            }
                            // Selecting a badge in the "AI Workflows" gallery pushes that
                            // flow's detail (title, rows, inspector) with a back button.
                            .navigationDestination(item: $appState.selectedFlow) { selection in
                                FlowListView(flowID: selection.flowID,
                                             source: selection.isUserFlow ? .user : .gallery)
                                    .id(selection.id)
                            }
                            // CFM-R8/R11-0: the flow editor — a fresh flow (nil document) or
                            // an edited copy of an existing one (Duplicate & Edit / Edit copy).
                            .navigationDestination(item: $appState.editingFlow) { target in
                                FlowEditorView(flowID: target.flowID,
                                               name: target.name,
                                               document: target.document,
                                               savedText: target.savedText,
                                               workspace: target.workspace)
                            }
                            // CFM-R17-3: a workspace page — the flows in it, its shared
                            // files, Reveal in Finder on the one directory they share.
                            .navigationDestination(item: $appState.selectedWorkspace) { ws in
                                WorkspaceListView(workspace: ws)
                                    .id(ws.id)
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
            .alert("Couldn't Open This Flow", isPresented: Binding(
                get: { appState.openCatFlowError != nil },
                set: { if !$0 { appState.openCatFlowError = nil } }
            )) {
                Button("OK", role: .cancel) { appState.openCatFlowError = nil }
            } message: {
                Text(appState.openCatFlowError ?? "")
            }
            .alert("Couldn't Remove This Flow", isPresented: Binding(
                get: { appState.flowRemoveError != nil },
                set: { if !$0 { appState.flowRemoveError = nil } }
            )) {
                Button("OK", role: .cancel) { appState.flowRemoveError = nil }
            } message: {
                Text(appState.flowRemoveError ?? "")
            }
            .alert("Couldn't Import This Flow", isPresented: Binding(
                get: { appState.flowImportError != nil },
                set: { if !$0 { appState.flowImportError = nil } }
            )) {
                Button("OK", role: .cancel) { appState.flowImportError = nil }
            } message: {
                Text(appState.flowImportError ?? "")
            }
            .alert("Couldn't Export This Flow", isPresented: Binding(
                get: { appState.flowExportError != nil },
                set: { if !$0 { appState.flowExportError = nil } }
            )) {
                Button("OK", role: .cancel) { appState.flowExportError = nil }
            } message: {
                Text(appState.flowExportError ?? "")
            }
            .alert("Couldn't Import This Workspace", isPresented: Binding(
                get: { appState.workspaceImportError != nil },
                set: { if !$0 { appState.workspaceImportError = nil } }
            )) {
                Button("OK", role: .cancel) { appState.workspaceImportError = nil }
            } message: {
                Text(appState.workspaceImportError ?? "")
            }
            .alert("Couldn't Remove This Workspace", isPresented: Binding(
                get: { appState.workspaceRemoveError != nil },
                set: { if !$0 { appState.workspaceRemoveError = nil } }
            )) {
                Button("OK", role: .cancel) { appState.workspaceRemoveError = nil }
            } message: {
                Text(appState.workspaceRemoveError ?? "")
            }
            .onChange(of: appState.filterSource) { _, _ in appState.saveFilters() }
            .onChange(of: appState.sortOrder) { _, _ in appState.saveFilters() }
        }
        .windowResizability(.contentSize)
        .defaultSize(width: 1400, height: 900)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Open .cat…") {
                    appState.presentOpenCatPanel()
                }
                .keyboardShortcut("o", modifiers: .command)
            }
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
                /*
                Link("Help build MLXUI", destination: URL(string: "https://www.connectcode.net/mlxui_local_llm_ai_browser.html")!)
                 */
            }
        }
    }
}
