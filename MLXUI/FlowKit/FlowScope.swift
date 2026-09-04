import Foundation

/// CFM-R17-1: `flowID` was doing two jobs, and a workspace separates them.
///
/// - **identity** (`identity`) — the run seed source, `On Flow` matching, and run records.
///   Always the flow's own id.
/// - **location** (`workspace` + `locationID`) — where a `.cat`'s unqualified paths resolve:
///   the id `FlowWorkspace.resolve` is keyed on, and the id every tool's `(workspace, flowID)`
///   pair lands under. For a plain flow this is `flows/<identity>/`; for a workspace flow it
///   is `workspaces/<workspace-id>/`, shared by every flow in that workspace.
///
/// For a plain flow the two ids are equal and the workspace is rooted at `flows/` — the
/// `.plain` factory. Nothing about `FlowWorkspace.resolve` or its escape checks changes;
/// only the id it is keyed on does.
nonisolated struct FlowScope: Sendable {
    /// The flow's own id — the run seed source, `On Flow` matching, run records.
    let identity: String
    /// The root the `.cat`'s paths resolve against (the security boundary).
    let workspace: FlowWorkspace
    /// The directory id inside `workspace` this flow's paths resolve under. Equal to
    /// `identity` for a plain flow; the workspace id for a workspace flow.
    let locationID: String
    /// The flow's own `.cat` text. `FlowSeed.runSeed` is a pure function of it, so a
    /// workspace flow — whose text is **not** in `Bundle.main` — still derives the right
    /// seed. `nil` falls back to the bundled gallery text keyed on `identity` (the
    /// pre-R17 behaviour for every `flows/` flow).
    let flowText: String?

    init(identity: String, workspace: FlowWorkspace, locationID: String, flowText: String? = nil) {
        self.identity = identity
        self.workspace = workspace
        self.locationID = locationID
        self.flowText = flowText
    }

    /// A plain `flows/`-rooted flow: identity and location are the same id, and the run
    /// seed is derived exactly as before — from the bundled gallery text keyed on the id.
    static func plain(_ flowID: String,
                      root: URL = ModelStore.shared.flowsDirectory) -> FlowScope {
        FlowScope(identity: flowID, workspace: FlowWorkspace(root: root), locationID: flowID)
    }

    /// This flow's working directory — `workspace/<locationID>/`.
    var directory: URL { workspace.directory(for: locationID) }

    /// The deterministic run-level seed for `_with_seed` (CFM-R3-2): derived from the
    /// flow's own text when known, else the bundled gallery text keyed on `identity`,
    /// else `0` — the unchanged fallback.
    var runSeed: UInt32 {
        let text = flowText ?? (try? GalleryLoader.rawCatText(flowID: identity))
        return text.map { FlowSeed.runSeed(for: $0) } ?? 0
    }
}
