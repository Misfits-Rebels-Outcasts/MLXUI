import AppKit

/// SET-3 (`RSI/DelegateSettingsBacklog.md`, D4) — the one place the folder-grant `NSOpenPanel` is
/// built. Settings → Tools and, since CW-2, the chat's "+" menu both call this instead of each
/// keeping their own copy — two copies of the same panel is how D4 (two homes for the same
/// settings) happened in the first place.
///
/// App Store: a picked folder must be granted (the panel's selection is what authorizes creating a
/// security-scoped bookmark), so the picker *grants and attaches* in one step (plan D1). Direct:
/// there is no sandbox and nothing to grant, so the picker only chooses a path.
enum FolderAccessPanel {

    /// Present the panel alone and return the chosen folder. `nil` when the user cancelled.
    @MainActor
    static func presentPicker(message: String, prompt: String) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = message
        panel.prompt = prompt
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    #if !DIRECT_BUILD
    /// Grant a folder the user chose in Settings, where granting is the point.
    @MainActor
    static func presentAndGrant(using runner: ModelRunner) {
        guard let url = presentPicker(
            message: "Choose a folder the model's file tools may access.",
            prompt: "Grant Access")
        else { return }
        runner.grantFolderAccess(to: url)
    }
    #else
    /// Direct's counterpart (CW-2). Kept so the chat's "+" menu is one call site in both editions;
    /// with no bookmarks, choosing a folder is all there is to do.
    @MainActor
    static func presentAndGrant(using runner: ModelRunner) {
        guard let url = presentPicker(
            message: "Choose a folder this chat may use.",
            prompt: "Use Folder")
        else { return }
        runner.attachFolder(url)
    }
    #endif
}
