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
    ///
    /// `startingAt` (M2) points the panel at a folder the user already knows — Re-grant… opens at
    /// the stale folder's parent so the user lands where they were instead of somewhere generic.
    /// A path that no longer exists is passed through anyway: AppKit opens the nearest parent it
    /// can, which is the same convenience.
    @MainActor
    static func presentPicker(
        message: String, prompt: String, startingAt: String? = nil
    ) -> URL? {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.message = message
        panel.prompt = prompt
        if let startingAt, !startingAt.isEmpty {
            panel.directoryURL = URL(fileURLWithPath: startingAt, isDirectory: true)
        }
        guard panel.runModal() == .OK else { return nil }
        return panel.url
    }

    #if !DIRECT_BUILD
    /// Grant a folder the user chose in Settings, where granting is the point.
    ///
    /// `attachAfterGrant` is for the one caller that needs both: the inspector's **Re-grant…**,
    /// which exists to make a folder this chat has already attached work again. Granting alone
    /// leaves the chat without it (the scope is the intersection of attached and granted), so the
    /// row would vanish and the user would have to add the folder again with "+". Settings and the
    /// "+" menu keep the default, because granting is a standing permission and attaching is a
    /// per-chat choice (D1).
    @MainActor
    static func presentAndGrant(
        using runner: ModelRunner, attachAfterGrant: Bool = false, startingAt: String? = nil
    ) {
        guard let url = presentPicker(
            message: "Choose a folder the model's file tools may access.",
            prompt: "Grant Access",
            startingAt: startingAt)
        else { return }
        runner.grantFolderAccess(to: url)
        guard attachAfterGrant else { return }
        // Only if the grant took: `attachFolder` would put a chip on a folder it could not read.
        guard FolderGrants.grant(covering: url.path, in: FolderGrants.load()) != nil else { return }
        runner.attachFolder(path: url.path)
    }
    #else
    /// Direct's counterpart (CW-2). Named for what it does — there is nothing to grant without a
    /// sandbox, so it **attaches**; one name with two meanings is how D4 happened (B6).
    @MainActor
    static func presentAndAttach(using runner: ModelRunner) {
        guard let url = presentPicker(
            message: "Choose a folder this chat may use.",
            prompt: "Use Folder")
        else { return }
        runner.attachFolder(url)
    }
    #endif
}
