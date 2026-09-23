import Foundation

/// FH-4 (`RSI/DelegateOutputViewerBacklog.md`): why one `CapabilityFlag` is ticked, unticked,
/// or fixed the way it is — never a bare checkbox. A flag is a *declaration of what a flow
/// does*, not a switch that grants anything, so every row the Flow tab renders carries a
/// reason alongside its checkbox.
nonisolated enum FlowFlagStatus: Sendable, Equatable {
    /// A row genuinely needs this flag, and the header declares it. Ticked, editable — Q1
    /// (owner ruling, already recorded in the backlog): unticking a flag a flow genuinely
    /// needs is allowed; it just fails `check` with a defined error, kept on Save. Never
    /// silently refused, never silently fixed.
    case requiredAndDeclared(rowNumber: String, task: String)
    /// A row needs this flag and the header doesn't declare it — a real check error today
    /// (E109/E118/E120/E604 — **not** `network`'s E103, which has no raise site anywhere in
    /// `FlowValidator`; an unticked-but-used `network` reads `.declaredNotRequired` instead,
    /// since this case's whole meaning is "Save is blocked until this is fixed" and nothing
    /// currently blocks on `network`). Unticked; ticking it runs the existing one-click repair
    /// (`FlowEditorModel.applyHeaderRepair`) — there is no second repair path.
    case requiredButMissing(code: String, message: String)
    /// No row needs this flag right now, whatever the header currently says (fact 20: there
    /// is no "declared but unused" validator error, so unticking one that's declared is
    /// always safe). The checkbox's own tick state — not this case — reflects whether it's
    /// actually declared.
    case declaredNotRequired
    /// A `uses:` sibling (at any depth) declares this flag; this flow does not. Ticked,
    /// disabled — the flag lives in that file, not this one (fact 21).
    case inherited(path: String)
    /// This edition's channel refuses the flag outright — `code`/`improvise` on the App
    /// Store tier — regardless of need. Shown, disabled, matching the judgement
    /// `FlowEditorModel.headerRepair(for:)` already applies (fact 22): never offered there,
    /// never offered here either.
    case refusedByChannel
}

/// Computes `FlowFlagStatus` for each of the five `CapabilityFlag` cases against a real
/// document. Pure and stateless — the caller (`FlowInspectorPane`'s Flow tab) re-derives it
/// on every render, the same way `FlowEditorModel.structuralIssues()` re-derives issues.
nonisolated enum FlowFlagInventory {

    /// `network`, `improvise`, `code`, `offdevice` can be inherited through a `uses:` chain
    /// (`FlowValidator`'s own E117 propagation set, `capabilityFlags` — `events` is
    /// deliberately absent there: a trigger's placement is row-1-specific to the flow that
    /// declares it, never inherited).
    private static let inheritableFlags: Set<String> = ["network", "improvise", "code", "offdevice"]

    static func status(of flag: CapabilityFlag, in doc: FlowDocument,
                       workspace: FlowWorkspace, flowID: String) -> FlowFlagStatus {
        if CapabilityGate.isAppStoreBuild, CapabilityGate.appStoreRefusedFlags.contains(flag.rawValue) {
            return .refusedByChannel
        }
        if !doc.flags.contains(flag), inheritableFlags.contains(flag.rawValue),
           let path = inheritedFrom(flag, doc: doc, workspace: workspace, flowID: flowID) {
            return .inherited(path: path)
        }
        guard let parsed = try? CatParser.parseForValidation(CatSerializer.serialize(doc)) else {
            return .declaredNotRequired
        }
        if doc.flags.contains(flag) {
            if let (path, task) = requiringRow(for: flag, parsed: parsed) {
                return .requiredAndDeclared(rowNumber: path, task: task)
            }
            return .declaredNotRequired
        }
        // Not declared. FIX (found in owner smoke-testing FH-S2, 2026-09-22): an earlier
        // version of this function also special-cased `network` here, synthesizing a
        // `.requiredButMissing` status from E103's dormant `ErrorCatalog` template. That was
        // wrong to ship: `requiredButMissing`'s whole meaning is "Save is blocked until this
        // is fixed" (owner ruling Q1), and E103 has no raise site anywhere in
        // `FlowValidator.checkFlow` (`FlowHeaderRepair.swift`'s own comment, a prior,
        // deliberate decision — trap 4 of that phase's own backlog forbids adding one) — so
        // unticking `network` on a flow that uses it doesn't block anything, and the Flow tab
        // must not claim it does. Wiring a real E103 check into the shared validator would be
        // a parity-relevant change to what `mlxflow check` reports, outside this item's scope
        // and this implementer's standing authority; the honest, conservative fix is for the
        // Flow tab to stop overclaiming instead. `network`'s `requiringRow` predicate still
        // drives `requiredAndDeclared` above (a true statement — "this row does use it" — with
        // no enforcement claim attached), it is only the missing-and-unenforced case that
        // falls through to `declaredNotRequired` below like every other unenforced flag would.
        let issues = FlowValidator.checkFlow(parsed, workspace: workspace, flowID: flowID)
        if let issue = issues.first(where: { FlowHeaderRepair.flagForCode[$0.code] == flag }) {
            return .requiredButMissing(code: issue.code, message: issue.message)
        }
        return .declaredNotRequired
    }

    /// The row (path, task) that needs `flag` right now, or nil when nothing does. Mirrors
    /// the condition each real check (`checkOffdeviceFlag`/`checkEventsFlag`/E109/E118) tests
    /// for firing, reusing their own primitives (`isTriggerRow`/`isRemoteRow`/`namesAModel`/
    /// `iterFlowRows`) — never duplicating their logic, only orchestrating it — so this stays
    /// correct as those checks evolve. `network`'s own predicate (the four network-reaching
    /// tasks) has no validator equivalent to reuse; it mirrors `CLAUDE.md`'s own "what leaves
    /// the machine" table instead.
    private static func requiringRow(for flag: CapabilityFlag, parsed: ParsedFlow) -> (path: String, task: String)? {
        switch flag {
        case .network:
            let networkTasks: Set<String> = ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File"]
            return FlowValidator.iterFlowRows(parsed.rows)
                .first { networkTasks.contains($0.1.task ?? "") }
                .map { (path: $0.0, task: $0.1.task ?? "") }
        case .improvise:
            return FlowValidator.iterFlowRows(parsed.rows)
                .first { $0.1.task == "Improvise" }
                .map { (path: $0.0, task: $0.1.task ?? "Improvise") }
        case .code:
            guard !parsed.transforms.isEmpty else { return nil }
            if let hit = FlowValidator.iterFlowRows(parsed.rows)
                .first(where: { parsed.transforms[$0.1.task ?? ""] != nil }) {
                return (path: hit.0, task: hit.1.task ?? "")
            }
            // `transforms:` declared but no row calls one yet — E118 itself still fires,
            // citing row "1" generically; mirror that rather than citing nothing.
            return (path: "1", task: parsed.rows.first?.task ?? "transforms:")
        case .events:
            guard let first = parsed.rows.first, FlowValidator.isTriggerRow(first) else { return nil }
            return (path: "1", task: first.task ?? "")
        case .offdevice:
            for (path, row) in FlowValidator.iterFlowRows(parsed.rows) {
                if row.task == "Web Search" { return (path: path, task: row.task ?? "") }
                if FlowValidator.namesAModel(row), row.model != nil, FlowValidator.isRemoteRow(row) {
                    return (path: path, task: row.task ?? "")
                }
            }
            return nil
        }
    }

    /// The `uses:` path (at any depth) whose own header declares `flag`, or nil. Mirrors
    /// `CapabilityGate.effectiveFlags`'s walk, but returns *where* the flag was found instead
    /// of only accumulating the flat set — the Flow tab needs to say "declared in {path}."
    private static func inheritedFrom(_ flag: CapabilityFlag, doc: FlowDocument,
                                      workspace: FlowWorkspace, flowID: String) -> String? {
        guard !doc.uses.isEmpty else { return nil }
        var visited: Set<String> = []
        func reach(_ path: String) -> String? {
            guard !visited.contains(path) else { return nil }
            visited.insert(path)
            guard let url = try? workspace.resolve(path, flowID: flowID),
                  let text = try? String(contentsOf: url, encoding: .utf8),
                  let used = try? CatParser.parseForValidation(text) else { return nil }
            if used.flags.contains(flag.rawValue) { return path }
            for sub in used.uses.values {
                if let found = reach(sub) { return found }
            }
            return nil
        }
        for path in doc.uses.values {
            if let found = reach(path) { return found }
        }
        return nil
    }
}
