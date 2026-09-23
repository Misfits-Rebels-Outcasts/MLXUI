import Foundation
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
    @State private var columnVisibility: NavigationSplitViewVisibility = .all

    /// AFM-FOLLOWUP-1: the **one** place the real app opts into reading the actual
    /// `SystemLanguageModel` — everywhere else sees a deterministic "absent"
    /// `AppleFoundationAvailability` by default, "mock by default" applied to the OS itself.
    ///
    /// **`MLXUIApp.init()` is *not* a safe "only in a real launch" signal on its own** — a
    /// macOS app's unit test bundle runs *inside* the app process as its test host, so
    /// `@main` fires and this initializer runs even under `xcodebuild test`. Guarded on
    /// `TestEnvironment.isRunningTests` (the KEY review's request: one named place for this
    /// check, not a raw environment lookup repeated at every call site).
    init() {
        if !TestEnvironment.isRunningTests {
            AppleFoundationAvailability.useRealSystem = true
        }
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if let error = appState.loadError {
                    ErrorView(message: error, retry: { appState.loadBrowserData() })
                } else if appState.browserData == nil {
                    ProgressView("Loading catalog…")
                        .onAppear { appState.loadBrowserData() }
                } else {
                    NavigationSplitView(columnVisibility: $columnVisibility) {
                        SidebarView()
                    } detail: {
                        // WA-5 (owner ruling Q2, 2026-09-19): one `NavigationStack` over one
                        // `[FlowRoute]` path, replacing four sibling `navigationDestination
                        // (item:)` modifiers that all presented from this same root — a
                        // workspace push followed by a flow-editor push used to leave both
                        // bindings non-nil with the stack stuck at depth 1, so Back from a
                        // workspace flow skipped the workspace (`RSI
                        // /DelegateWorkspaceAdvisoryBacklog.md`, root cause 4).
                        NavigationStack(path: $appState.route) {
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
                            .navigationDestination(for: FlowRoute.self) { route in
                                switch route {
                                case .model(let model):
                                    // Selecting a model (Installed sidebar list or the
                                    // command palette) pushes its detail page.
                                    ModelDetailView(model: model)
                                case .flow(let selection):
                                    // Selecting a badge in the "AI Workflows" gallery pushes
                                    // that flow's detail (title, rows, inspector).
                                    FlowListView(flowID: selection.flowID,
                                                 source: selection.isUserFlow ? .user : .gallery,
                                                 workspace: selection.workspace,
                                                 autoRun: selection.autoRun)
                                        .id(selection.id)
                                case .editor(let target):
                                    // CFM-R8/R11-0: the flow editor — a fresh flow (nil
                                    // document) or an edited copy of an existing one
                                    // (Duplicate & Edit / Edit copy). `fileURL` (FH-6) seeds
                                    // `savedURL` for a plain My Workflows flow opened via this
                                    // route — dropping it would resurrect the "Save the flow
                                    // first" bug FH-6 fixed for this call site.
                                    FlowEditorView(flowID: target.flowID,
                                                   name: target.name,
                                                   document: target.document,
                                                   savedText: target.savedText,
                                                   workspace: target.workspace,
                                                   fileURL: target.fileURL)
                                case .workspace(let ws):
                                    // CFM-R17-3: a workspace page — the flows in it, its
                                    // shared files, Reveal in Finder on the shared directory.
                                    WorkspaceListView(workspace: ws)
                                        .id(ws.id)
                                }
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
            CommandMenu("Find") {
                Button("Find Models...") {
                    appState.showCommandPalette = true
                }
                .keyboardShortcut("k", modifiers: .command)
            }
            CommandGroup(replacing: .help) {
                Link("AI Browser Help", destination: URL(string: "https://connectcode.net/mlxui_local_llm_ai_browser.html")!)
                Link("AI Workflows (mlx-workflow)", destination: URL(string: "https://www.connectcode.net/mlx-workflow.html")!)
                /*
                Link("Help build MLXUI", destination: URL(string: "https://www.connectcode.net/mlxui_local_llm_ai_browser.html")!)
                 */
            }
        }

        // SET-1 (`RSI/DelegateSettingsBacklog.md`) — a real `Settings` scene replaces the old
        // 460×560 sheet. The scene itself supplies the app menu's "Settings…" item and ⌘,;
        // `CommandGroup(replacing: .appSettings)` above is gone, so there is exactly one.
        Settings {
            SettingsRootView()
                .environment(appState)
        }
    }
}
