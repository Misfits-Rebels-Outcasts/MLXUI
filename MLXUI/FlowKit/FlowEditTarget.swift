import Foundation

/// CFM-R11-0 — a flow the editor is asked to edit: a fresh flow (nil document), a bundled
/// flow's copy, or a user flow. The editor is routed to it via `AppState.editingFlow`.
nonisolated struct FlowEditTarget: Hashable, Identifiable {
    let flowID: String
    let name: String
    let document: FlowDocument?
    /// The canonical text the file was last written with — a copied flow opens clean (not
    /// dirty) until the user edits it.
    let savedText: String?
    /// CFM-R17-3: set when the flow being edited lives inside a workspace — the editor then
    /// saves into and resolves against the shared workspace directory.
    var workspace: WorkspaceRef? = nil
    /// FH-6: the file this (non-workspace) flow's document already lives at on disk — set when
    /// opening an existing, already-saved `flows/<id>/` flow, or a just-written Duplicate &
    /// Edit / Edit-opened-copy. `workspace?.fileURL` is preferred when both are set; this is
    /// the plain-flow equivalent KW-1-1 only wired for workspace flows. nil for a brand-new
    /// flow that has never been written.
    var fileURL: URL? = nil

    var id: String {
        if let workspace { return "w-\(workspace.workspaceID)/\(workspace.flowFile)" }
        return flowID
    }

    /// Identity is the flow folder (or, in a workspace, the folder + file); the document
    /// payload is deliberately excluded so the navigation destination doesn't invalidate on
    /// every edit.
    static func == (lhs: FlowEditTarget, rhs: FlowEditTarget) -> Bool {
        lhs.id == rhs.id
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}

/// CFM-R11-0 — the two routes into the editor. Each **copies the flow into the user's flow
/// folder first** (a bundled flow's own folder is app-bundle read-only), writes the `.cat`,
/// and returns the edit target. Pure enough to test without the views.
nonisolated enum FlowEditRoute {
    /// Duplicate & Edit for a bundled gallery flow: copy its `.cat` and input assets into a
    /// fresh flow folder, return the target that opens the editor on the copy.
    static func duplicateAndEdit(flowID: String, title: String, document: FlowDocument,
                                 workspace: FlowWorkspace, sourceDir: URL) throws -> FlowEditTarget {
        let newID = UUID().uuidString
        let dir = workspace.directory(for: newID)
        let fm = FileManager.default
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        try workspace.prepare(flowID: newID, sourceDir: sourceDir,
                              bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let text = CatSerializer.serialize(document)
        let filename = "\(FlowEditorModel.sanitizedFileName(title)).\(FlowEditorModel.fileExtension(for: document.fileKind))"
        let fileURL = dir.appendingPathComponent(filename)
        try text.write(to: fileURL, atomically: true, encoding: .utf8)
        return FlowEditTarget(flowID: newID, name: title, document: document, savedText: text,
                              fileURL: fileURL)
    }

    /// Edit for an opened `.cat` (R5-5): copy it into the user's flow folder and return the
    /// target — the original file is never written. The copy opens clean (canonical text) so
    /// the editor only reports real changes as dirty.
    static func editOpenedCopy(displayName: String, parsed: ParsedFlow,
                               workspace: FlowWorkspace) throws -> FlowEditTarget {
        let newID = UUID().uuidString
        let dir = workspace.directory(for: newID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let document = try CatParser.resolveDocument(parsed)
        let text = CatSerializer.serialize(document)
        let filename = "\(FlowEditorModel.sanitizedFileName(displayName)).\(FlowEditorModel.fileExtension(for: document.fileKind))"
        let fileURL = dir.appendingPathComponent(filename)
        try text.write(to: fileURL, atomically: true, encoding: .utf8)
        return FlowEditTarget(flowID: newID, name: displayName, document: document, savedText: text,
                              fileURL: fileURL)
    }

    /// KW-3-1 (Q1 — "Copy flow here" on the workspace page): copy a bundled gallery flow's
    /// `.cat` and input assets into an **existing** `workspaces/<workspaceID>/` directory,
    /// rather than minting a fresh single-flow folder under `flows/`. `duplicateAndEdit`
    /// cannot be reused for this — its `newID` is generated inside, so passing a different
    /// `workspace:` there produces a brand-new pseudo-workspace *inside* `workspaces/`, not a
    /// file inside the workspace the user meant.
    ///
    /// The collision rule matches `KW-2-1`'s: `WorkspaceStore.firstFreeFlowName` picks the
    /// first free name rather than overwriting a sibling. `workspace.prepare` is already
    /// idempotent about a destination that exists, so a shared folder's assets another flow
    /// already owns are never overwritten — only what's missing is copied.
    static func copyIntoWorkspace(flowID: String, title: String, document: FlowDocument,
                                  workspaceID: String, workspace: FlowWorkspace,
                                  sourceDir: URL) throws -> FlowEditTarget {
        guard document.fileKind != .catpipeline else {
            throw WorkspaceStoreError.catpipelineNotSupportedInWorkspace(title)
        }
        let dir = workspace.directory(for: workspaceID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try workspace.prepare(flowID: workspaceID, sourceDir: sourceDir,
                              bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let filename = WorkspaceStore.firstFreeFlowName(
            stem: FlowEditorModel.sanitizedFileName(title), in: dir)
        let text = CatSerializer.serialize(document)
        try text.write(to: dir.appendingPathComponent(filename), atomically: true, encoding: .utf8)
        let ref = WorkspaceRef(workspaceID: workspaceID, flowFile: filename)
        return FlowEditTarget(flowID: workspaceID, name: ref.flowStem, document: document,
                              savedText: text, workspace: ref)
    }
}
