import Foundation

/// CFM-R17-3 — points a flow surface (`FlowListView` / `FlowEditorView`) at a flow that
/// lives **inside a workspace**: its own file name, and the workspace directory every flow
/// in that workspace shares. `nil` on those views means a plain `flows/` flow, unchanged.
///
/// The `FlowScope` this builds has `identity` = the flow's own stem (run seed, `On Flow`
/// matching, run records) and `locationID` = the workspace id (where `.cat` paths resolve),
/// so two flows in one workspace resolve the same relative path to the same file — the whole
/// point of a workspace (CFM-R17-1).
nonisolated struct WorkspaceRef: Hashable, Sendable {
    /// `workspaces/<workspaceID>/`.
    let workspaceID: String
    /// The flow's own file, e.g. `AskYourDocs.cat`.
    let flowFile: String

    var flowStem: String { (flowFile as NSString).deletingPathExtension }

    var workspace: FlowWorkspace { FlowWorkspace(root: ModelStore.shared.workspacesDirectory) }
    var directory: URL { workspace.directory(for: workspaceID) }
    var fileURL: URL { directory.appendingPathComponent(flowFile) }

    /// The run/edit scope for this flow. `text` is the flow's own `.cat` text (for the run
    /// seed); `nil` falls through to `0`, the same as a `flows/` user flow.
    func scope(text: String?) -> FlowScope {
        FlowScope(identity: flowStem, workspace: workspace, locationID: workspaceID,
                  flowText: text, selfFile: fileURL)
    }
}
