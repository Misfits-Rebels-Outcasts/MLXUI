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
        try text.write(to: dir.appendingPathComponent(filename), atomically: true, encoding: .utf8)
        return FlowEditTarget(flowID: newID, name: title, document: document, savedText: text)
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
        try text.write(to: dir.appendingPathComponent(filename), atomically: true, encoding: .utf8)
        return FlowEditTarget(flowID: newID, name: displayName, document: document, savedText: text)
    }
}
