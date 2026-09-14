import AppKit

#if !DIRECT_BUILD
/// SET-3 (`RSI/DelegateSettingsBacklog.md`, D4) — the one place the folder-grant
/// `NSOpenPanel` is built. Both the wrench (`RunChatView`) and the Tools pane
/// (`ToolsSettingsView`) call this instead of each keeping their own copy — two copies of the
/// same panel is how D4 (two homes for the same settings) happened in the first place.
enum FolderAccessPanel {
    /// Presents the panel; a chosen folder is granted via `runner.grantFolderAccess`.
    @MainActor
    static func presentAndGrant(using runner: ModelRunner) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose a folder the model's file tools may access."
        panel.prompt = "Grant Access"
        if panel.runModal() == .OK, let url = panel.url {
            runner.grantFolderAccess(to: url)
        }
    }
}
#endif
