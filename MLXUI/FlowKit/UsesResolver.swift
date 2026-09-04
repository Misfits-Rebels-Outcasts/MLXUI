import Foundation

/// CFM-R17-5 — resolves a flow's `uses:` section into the `[String: FlowInterpreter.UsedFlow]`
/// graph the interpreter expands, reading the sibling `.cat` files off disk. The **execution**
/// counterpart of `FlowValidator`'s `resolveUsesLevel` (which builds a validation-only
/// structure): a port of the clean path of `catflow-mlx/src/catflow/core/uses.py::_resolve_level`.
///
/// This is **best-effort** — an entry that escapes the workspace (E114), closes a cycle
/// (E115), is missing, or doesn't parse (E116) is simply left out of the graph. Reporting
/// those is `FlowValidator.checkFlow`'s job, and `FlowRunner.canRun` refuses the run when an
/// entry a row names is not in the returned graph. So a flow that reaches the interpreter has
/// already had every `uses:` entry resolved and checked.
nonisolated enum UsesResolver {

    /// Resolve `doc`'s `uses:` entries against `workspace`/`flowID` (for a workspace flow,
    /// `flowID` is the workspace id, so `./X.cat` resolves to a sibling in the shared
    /// directory). `selfFile` is the calling flow's own `.cat` path — the first frame of the
    /// cycle stack, so `A → B → A` is caught. Recurses for nested `uses:`.
    static func resolve(_ doc: FlowDocument, workspace: FlowWorkspace, flowID: String,
                        selfFile: URL? = nil) -> [String: FlowInterpreter.UsedFlow] {
        resolveLevel(doc, workspace: workspace, flowID: flowID,
                     stack: selfFile.map { [$0.resolvingSymlinksInPath()] } ?? [])
    }

    private static func resolveLevel(_ doc: FlowDocument, workspace: FlowWorkspace, flowID: String,
                                     stack: [URL]) -> [String: FlowInterpreter.UsedFlow] {
        var out: [String: FlowInterpreter.UsedFlow] = [:]
        for (name, rawPath) in doc.uses {
            // E114 territory — `workspace.resolve` throws for an absolute path or any escape.
            guard let candidate = try? workspace.resolve(rawPath, flowID: flowID) else { continue }
            let resolved = candidate.resolvingSymlinksInPath()
            // E115 — this file is already an ancestor: the edge that closes the loop.
            if stack.contains(resolved) { continue }
            // E116 — must be a readable, parseable file.
            var isDir: ObjCBool = false
            guard FileManager.default.fileExists(atPath: candidate.path, isDirectory: &isDir),
                  !isDir.boolValue,
                  let text = try? String(contentsOf: candidate, encoding: .utf8),
                  let used = try? CatParser.parse(text) else { continue }
            let nested = resolveLevel(used, workspace: workspace, flowID: flowID,
                                      stack: stack + [resolved])
            out[name] = FlowInterpreter.UsedFlow(
                params: used.params, rows: used.rows,
                definitions: used.definitions, presets: used.presets,
                nested: nested, sourceText: text)
        }
        return out
    }
}
