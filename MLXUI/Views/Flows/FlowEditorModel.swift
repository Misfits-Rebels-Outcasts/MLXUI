import Foundation
import Observation
import SwiftUI

/// CFM-R8 — the flow editor's model: a small layer over the immutable `FlowDocument` value.
///
/// The identity contract (CFM-R1-3) earns its keep here: every edit addresses rows by
/// `Row.id`, never by number, so delete and reorder can never silently re-aim a reference
/// (QR8 — "no reference is ever silently re-aimed").
///
/// **Validity contract (CFM-R8-FIX-1):** the save gate and the input picker consult
/// `FlowValidator.checkFlow` — the same authority the read-only Open path uses and the one
/// `catflow check` runs — via a serialize→reparse round trip, so the editor refuses every
/// file the Python rejects (E203 forward/deleted refs, E204, E205 arity, E401 `(input:N)`,
/// E403 cross-scope). E104 (unpinned model display names) is *excluded*: the app resolves
/// those through `CatalogBridge` at run time, and the from-scratch flow's defaults carry no
/// `models:` section yet.
///
/// The yellow row (answers `c5`/`c9`/`l`): a row with a ref issue (from the validator), a
/// model-class row with no runnable model (answer `h3`), or a no-ref row whose auto-chain
/// upstream is incompatible.
@Observable
final class FlowEditorModel {

    // MARK: - State

    /// The flow's display name (the saved file's stem).
    var name: String
    /// Stable id for the flow's workspace folder (`…/flows/<flowID>/`).
    let flowID: String
    /// Where Save writes — injectable so tests never touch real user data.
    var workspace: FlowWorkspace
    /// The immutable document — replaced wholesale on every edit.
    private(set) var document: FlowDocument
    /// Undo/redo as whole-state value snapshots (answer `c11`): document + selection +
    /// tombstones, so a no-op edit is not an undo unit (CFM-R8-FIX-8).
    private(set) var undoStack: [Snapshot] = []
    private(set) var redoStack: [Snapshot] = []
    /// Deleted row id → the 1-based number it held, so a broken reference renders `(?N)`.
    private(set) var tombstones: [UUID: Int] = [:]
    /// Decide edges (and goto/fork targets) that pointed at a now-deleted row: decider row id
    /// → edge slots. They keep the deleted row's number but turn the row yellow and block
    /// Save — the QR9 ruling: a clause target is never silently re-aimed at whatever now
    /// sits in the dead row's slot.
    private(set) var staleClauseTargets: [UUID: Set<Int>] = [:]
    /// INPUTS-1 — ids `setReference` minted as "nothing chosen for this slot yet", never a
    /// real row. Reuses `Ref.rowRef` (never a new case to teach the parser/serializer/
    /// interpreter about) with an id that can never resolve, and gives `structuralIssues()`
    /// and the pickers a way to tell "never bound" apart from "the target row was deleted" —
    /// both dangle the same way structurally, but only one of them is honestly "None". Not
    /// part of `Snapshot`: an id undo strands here (the ref it marked is gone from
    /// `document.rows` again) is just an unreferenced UUID, not a correctness issue.
    private var unsetSlotIDs: Set<UUID> = []
    /// The row the step picker and Add-below target; nil = append at the end.
    var selectedRowID: UUID?
    /// The URL the flow was last saved to, nil until the first save.
    private(set) var savedURL: URL?
    /// Whether this flow's folder may hold sibling flow files that this editor did not open —
    /// true for a flow opened from a shared `workspaces/<id>/` folder, where `save()`'s
    /// stale-sibling guard must leave untouched flows alone (KW-1-1). Defaults to comparing the
    /// injected `workspace.root` against the live app's workspaces directory, matching every
    /// production call site; tests that stand in for a workspace with a temp root can set it
    /// directly, since a temp path never equals that singleton. KW-1-FIX-3: `private(set)` —
    /// `workspace` is never reassigned after init anywhere in `MLXUI/`, so this is set once and
    /// never goes stale; nothing outside init has a reason to flip save semantics at runtime.
    private(set) var isSharedWorkspaceFolder: Bool
    /// The text last written to disk — the dirty check compares the current document to it.
    private(set) var savedText: String?
    var saveError: String?
    /// A failed sample-seed copy (CFM-R11-0b) — the row is still added, honestly unseeded.
    private(set) var seedError: String?
    /// CFM-R14-2 — the flat catalog + the registry's claim table, so `add()` seeds a row's
    /// default from the **derived** pool (never a hand-maintained display-name table). Empty
    /// in tests that don't care about seeding.
    var modelCatalog: [ModelEntry] = []
    var claimableModelIDs: Set<String> = []

    /// One undo/redo unit — the whole editor state, so selection survives undo (FIX-8).
    struct Snapshot: Equatable {
        var document: FlowDocument
        var selectedRowID: UUID?
        var tombstones: [UUID: Int]
        var staleClauseTargets: [UUID: Set<Int>]
    }

    static let maxUndoDepth = 50

    // MARK: - Init

    /// Where the bundled samples live (`Resources/Samples/` in the built app). Injectable so
    /// tests can point at a temp dir; nil means no seeding (CFM-R11-0b).
    var sampleSourceDir: URL?

    init(name: String, flowID: String = UUID().uuidString, document: FlowDocument? = nil,
         workspace: FlowWorkspace = .shared, sampleSourceDir: URL? = Bundle.main.resourceURL,
         savedText: String? = nil, isSharedWorkspaceFolder: Bool? = nil, savedURL: URL? = nil) {
        self.name = name
        self.flowID = flowID
        self.workspace = workspace
        self.document = document ?? FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [])
        self.sampleSourceDir = sampleSourceDir
        self.savedText = savedText
        self.savedURL = savedURL
        self.isSharedWorkspaceFolder = isSharedWorkspaceFolder
            ?? (workspace.root == ModelStore.shared.workspacesDirectory)
        // Opening an existing flow selects its first row so the inspector pane is up and
        // showing the Properties tab (CFM — a click on a My Workflows flow opens Edit).
        if let first = self.document.rows.first {
            self.selectedRowID = first.id
        }
    }

    /// KW-1-FIX-3: the production decision behind seeding `savedURL` on open, extracted out of
    /// `FlowEditorView.init` so it's testable on its own rather than only through a SwiftUI
    /// view's `@State` init. A workspace flow whose document already exists on disk (a
    /// `WorkspaceRef` paired with a non-nil `document`) seeds from its real file URL. FH-6
    /// widened this: a plain `flows/` flow (`workspace == nil`) now seeds from the caller's own
    /// `fileURL` — the on-disk location `FlowGalleryView.openUserFlow` / `FlowListView`'s Edit
    /// button / `FlowEditRoute.duplicateAndEdit`/`.editOpenedCopy` already know, since the file
    /// genuinely exists there. A `WorkspaceRef` with no document (the unparseable-flow path,
    /// which routes to `FlowListView`, never the editor) still seeds nothing, and a brand-new
    /// flow that has never been written passes `fileURL: nil` too.
    nonisolated static func seedSavedURL(document: FlowDocument?, workspace: WorkspaceRef?,
                                         fileURL: URL? = nil) -> URL? {
        guard document != nil else { return nil }
        return workspace?.fileURL ?? fileURL
    }

    // MARK: - Step picker (CFM-R8-1, CFM-R8-FIX-2/7)

    /// `catflow tasks --accepts K` generalized to the actual output shape of the selected
    /// row. A `tupleOf` accepts (Diff, Retrieve, …) is offerable when the given can fill
    /// its **first** slot — the row is then added and the user assigns Input 1/Input 2
    /// (CFM-R8-FIX-2). `SameAsInput` widens to `.anyKind`. `includeAdvanced` (the "Add from
    /// Full Catalog" escape hatch, answer `l`) reveals the hidden primitives **and** every
    /// shape-mismatched task — a row added that way turns yellow until its input is pointed.
    nonisolated static func stepsAccepting(_ output: Shape?, includeAdvanced: Bool = false) -> [TaskDescriptor] {
        let advanced: Set<String> = ["Generate", "Decide"]
        let given = output.map { $0 == .sameAsInput ? Shape.anyKind : $0 } ?? .anyKind
        return TaskCatalog.allTasks().filter { desc in
            if !includeAdvanced && advanced.contains(desc.name) { return false }
            if includeAdvanced { return true }
            return acceptsFirstSlot(desc.accepts, given: given, rk: desc.refKind)
        }
    }

    /// Whether a single given shape can feed a task's accepts — a `tupleOf` is satisfied by
    /// its first slot (the rest come from the Input pickers), everything else by
    /// `singleCompatible`.
    private nonisolated static func acceptsFirstSlot(_ accepts: Shape, given: Shape, rk: RefKind?) -> Bool {
        if case .tupleOf(let kinds) = accepts, let first = kinds.first {
            return Shape.singleCompatible(accepts: .single(first), given: given, rk: rk)
        }
        return Shape.singleCompatible(accepts: accepts, given: given, rk: rk)
    }

    /// Starting nodes for an empty flow (answer `c8`): the file-reading source tasks, the
    /// triggers, `Template`, and anyKind-accepting tasks — restricted to what `canRun`
    /// accepts and what can usefully sit at row 1 (CFM-R8-FIX-7): the hidden primitives and
    /// the `human`/`staged`/`net`/`agent`/`trigger` classes are out.
    nonisolated static func startingNodes() -> [TaskDescriptor] {
        let advanced: Set<String> = ["Generate", "Decide"]
        return TaskCatalog.allTasks().filter { desc in
            if advanced.contains(desc.name) { return false }
            // HR-2 (`RSI/DelegateHumanRowBacklog.md`) — a single-task exception to the
            // `.human` class refusal just below, admitted by name rather than by widening the
            // class: `TaskCatalog.swift:234`/`:235`'s signature difference is the reason —
            // `Ask Human` gives `.sameAsInput` (a passthrough that fires a tag and cannot
            // source content at row 1), `Human Input` gives `t(.text)` (it produces text of
            // its own). `FlowRunner.rowClassRefusal` (`:233`) already returns `nil` for the
            // whole `.human` class (`:217` — "model / human / trigger / staged … in scope"),
            // so this function's own doc comment ("restricted to what `canRun` accepts")
            // still holds for this one addition.
            if desc.name == "Human Input" { return true }
            switch desc.taskClass {
            case .human, .staged, .net, .agent, .trigger: return false
            case .instant, .model: break
            }
            switch desc.accepts {
            case .single(.file), .single(.folder), .single(.occurrence), .listOf(.file):
                return true
            case .anyKind:
                return true
            default:
                return desc.name == "Template"
            }
        }
    }

    /// `models_for_task` — a task's usual model pool (display names), ported from
    /// `catalog/models.py::TASK_MODELS` (the table now lives in `TaskModels`, shared with
    /// `TaskAvailability` — CFM-R12-FIX-12).
    nonisolated static func models(forTask task: String) -> [String] {
        TaskModels.models(forTask: task)
    }

    /// Every model task's seeded default — for the FIX-3 invariant test that each default
    /// resolves through the derived pool (the build-can-run-it authority, CFM-R14-2). The
    /// catalog + claim table are required inputs — there is no catalog-free default anymore
    /// (CFM-R14-FIX-5: the old zero-returning stub made the invariant test vacuous).
    nonisolated static func allDefaultModels(for catalog: [ModelEntry],
                                             claimableModelIDs: Set<String>) -> [(task: String, model: String?)] {
        TaskCatalog.allTasks()
            .filter { $0.taskClass == .model }
            .map { ($0.name, Self.defaultModel(forTask: $0.name, catalog: catalog, claimableModelIDs: claimableModelIDs)) }
    }

    /// The default model for a model-class row — the **first pool-named derived candidate**
    /// (`TaskModels.defaultModel`, CFM-R14-2). Since CFM-R14-1 all four `Transcribe` pool
    /// entries are derived, so it seeds the first — `Whisper Tiny` (0.11 GB), matching the
    /// Python's `_ASR_MODELS` order. `Segment` seeds **`SAM Base`** since CFM-R15-1: the owner
    /// ruled hazard H2 / `CFM-R13-6` option (1) 2026-08-27, the executor serves
    /// `engines.diffusion.segment`, and `SAM Base` bridges onto the derived sam3 entry
    /// (`.substitute`). Nil = the row shows its "needs a model" warning.
    nonisolated static func defaultModel(forTask task: String,
                                         catalog: [ModelEntry] = [],
                                         claimableModelIDs: Set<String> = []) -> String? {
        TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimableModelIDs)
    }

    // MARK: - CFM-R12-3: rows inside a block are rows

    /// CFM-R12-3 item 5 — "Keep the steps": remove a block header but unwrap its children
    /// into the block's position.
    func unwrapBlock(_ blockID: UUID) {
        commitChange {
            let before = document.rows
            guard let idx = document.rows.firstIndex(where: { $0.id == blockID }) else { return }
            let block = document.rows.remove(at: idx)
            document.rows.insert(contentsOf: block.children, at: idx)
            if selectedRowID == blockID { selectedRowID = block.children.first?.id }
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// The enclosing block's display name for a row (the toolbar's "Add step inside `name`").
    func enclosingBlockName(for rowID: UUID) -> String? {
        for block in document.rows where block.blockKind != nil {
            if Self.findRow(rowID, in: block.children) != nil {
                return block.blockName ?? "<\(block.blockKind?.rawValue ?? "block")>"
            }
        }
        return nil
    }

    /// The block a row is a direct child of, plus its index among the block's children —
    /// the R12-3 move-inside-scope and Add-inside targeting.
    func childIndexOf(_ id: UUID) -> (blockID: UUID, index: Int)? {
        for block in document.rows where block.blockKind != nil {
            if let idx = block.children.firstIndex(where: { $0.id == id }) {
                return (block.id, idx)
            }
        }
        return nil
    }

    /// A block's child count.
    func childCount(of blockID: UUID) -> Int {
        document.rows.first(where: { $0.id == blockID })?.children.count ?? 0
    }

    // MARK: - Editing ops (CFM-R8-2/3/4, CFM-R8-FIX-6)

    /// Add a task row below the selected row — into a block when a block child is selected
    /// (CFM-R8-FIX-6), else the top level — or append when nothing is selected. A
    /// model-class row gets the default model; no explicit ref (a compatible upstream
    /// auto-chains; an incompatible one turns yellow per answer `l`). A seeded `Read *` row
    /// gets a bundled sample copied into the flow folder and its settings set to the bare
    /// file name (CFM-R11-0b), so the row runs with no further input.
    func add(task name: String) {
        let seedValue = seedSample(for: name) ?? Self.defaultSettings(forTask: name)
        commitChange {
            let before = document.rows
            let desc = TaskCatalog.get(name)
            let model = (desc?.taskClass == .model)
                ? Self.defaultModel(forTask: name, catalog: modelCatalog, claimableModelIDs: claimableModelIDs)
                : nil
            let newRow = Row(id: UUID(), task: name, model: model, settings: seedValue, refs: [])
            if let selectedRowID, let (blockID, _) = childIndex(selectedRowID) {
                insertChild(newRow, into: blockID, below: selectedRowID)
            } else if let selectedRowID,
                      let selected = row(withID: selectedRowID), selected.blockKind != nil {
                // CFM-R12-FIX-4: a block *header* selected — "Add step inside" must actually
                // append into the block as its last child, not insert a top-level row.
                insertChild(newRow, into: selectedRowID, below: nil)
            } else if let selectedRowID,
                      let idx = document.rows.firstIndex(where: { $0.id == selectedRowID }) {
                document.rows.insert(newRow, at: idx + 1)
            } else {
                document.rows.append(newRow)
            }
            selectedRowID = newRow.id
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// Insert a block row (`<each>` / `<parallel>` / `<list>`) below the selection with one
    /// `Template` child (CFM-R8-FIX-6) — a block's body is entered by adding inside it.
    func insertBlock(kind: BlockKind, name blockName: String, after selectedID: UUID?) {
        commitChange {
            let before = document.rows
            let child = Row(id: UUID(), task: "Template", settings: "\"\"", refs: [])
            let block = Row(id: UUID(), task: nil, blockKind: kind, blockName: blockName,
                            children: [child])
            if let selectedID,
               let idx = document.rows.firstIndex(where: { $0.id == selectedID }) {
                document.rows.insert(block, at: idx + 1)
            } else {
                document.rows.append(block)
            }
            selectedRowID = child.id
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// Add a child task into a block, below the selected child if one is selected inside.
    func addChild(task name: String, into blockID: UUID) {
        let seedValue = seedSample(for: name) ?? Self.defaultSettings(forTask: name)
        commitChange {
            let before = document.rows
            let desc = TaskCatalog.get(name)
            let model = (desc?.taskClass == .model)
                ? Self.defaultModel(forTask: name, catalog: modelCatalog, claimableModelIDs: claimableModelIDs)
                : nil
            let newRow = Row(id: UUID(), task: name, model: model, settings: seedValue, refs: [])
            insertChild(newRow, into: blockID, below: selectedRowID)
            selectedRowID = newRow.id
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// HU-1 — a freshly added human row would otherwise show **E501** (`FlowValidator
    /// .checkHumanRows`) the moment it's added, with no field on it yet to fix that from
    /// (`knownSettingKeys` had no case for either human task until HU-3). `wait=forever` is
    /// the right seed because the interpreter already treats a GUI run this way
    /// (`FlowInterpreter.swift:257` parks a `timeout=` row "like `wait=forever`" since a GUI
    /// has a person in front of it) — seeding it explicitly just makes that assumption
    /// visible and editable instead of implicit. Kept separate from `seedSample` (which
    /// exists to copy **sample assets**, not to answer a validator check) and merged with it
    /// at the call sites, per the backlog's instruction not to overload it.
    ///
    /// HR-3 (`RSI/DelegateHumanRowBacklog.md`) — `Human Input` additionally gets a seeded
    /// quoted criterion, so a freshly added row has something to ask instead of leaving
    /// `FlowHumanPromptView` to render a blank line (root cause 3). `Ask Human` is left as
    /// `wait=forever` alone, unchanged: the backlog's recommended seed is `Human Input`-only,
    /// and this stays optional either way — a row with no criterion is still legal in the
    /// language (E501 polices the waiting policy, never the question).
    nonisolated static func defaultSettings(forTask task: String) -> String? {
        switch task {
        case "Human Input": return "\(FlowSettingsEditor.quote("What should I use?")); wait=forever"
        case "Ask Human": return "wait=forever"
        default: return nil
        }
    }

    /// CFM-R11-0b: copy a bundled sample into the flow folder and return the bare path the
    /// new row's settings should hold, or nil when the task is unseeded. Reuses
    /// `FlowWorkspace.prepare` (idempotent — a file the user has since replaced is never
    /// clobbered, and a second seeded row of the same task points at the same file). A failed
    /// copy is recorded in `seedError` and returns nil: the row is still added, honestly
    /// unseeded, rather than pretending it has a file.
    private func seedSample(for task: String) -> String? {
        seedError = nil
        guard let value = SampleSeed.seedValue(for: task),
              let assets = SampleSeed.seedAssets(for: task),
              let sourceDir = sampleSourceDir else { return nil }
        do {
            try workspace.prepare(flowID: flowID, sourceDir: sourceDir, bundledAssets: assets)
            return value
        } catch {
            seedError = "Couldn't copy the sample for \(task) into the flow's folder — the row was added without a file."
            return nil
        }
    }

    /// Wrap `rowID` in a new block with the row as its only child (CFM-R8-FIX-6 `wrap`).
    func wrapInBlock(_ rowID: UUID, kind: BlockKind, name blockName: String) {
        commitChange {
            let before = document.rows
            guard let idx = document.rows.firstIndex(where: { $0.id == rowID }) else { return }
            var child = document.rows.remove(at: idx)
            child.chainBreak = false
            child.refs = []
            let block = Row(id: UUID(), task: nil, blockKind: kind, blockName: blockName,
                            children: [child])
            document.rows.insert(block, at: idx)
            selectedRowID = block.id
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// HR-5 (`RSI/DelegateHumanRowBacklog.md`, Q3 ruled (c), 2026-09-20) — "Give this row a
    /// default value…": inserts an empty `Template` row directly above `rowID` and selects it,
    /// so a `Human Input` row's `default=unchanged` has something of its own to resolve to
    /// (Q2's ruling: the default value *is* the row above, never a new setting). The inserted
    /// row is an ordinary `Template` row — same shape `insertBlock`'s own child seed uses
    /// (`settings: "\"\""`, an empty pattern) — with no marker comment or hidden token, so the
    /// `.cat` output is byte-identical to a user typing the two rows by hand. Deliberately does
    /// **not** touch `rowID`'s own settings (it does not add `default=unchanged` itself): that
    /// stays the user's own edit via the existing waiting-policy control, same as any other
    /// row's settings. One level of block nesting, matching `childIndexOf`'s own scope (the
    /// same one `moveMenu`'s context-menu actions already assume).
    func addDefaultValueRow(above rowID: UUID) {
        commitChange {
            let before = document.rows
            let template = Row(id: UUID(), task: "Template", settings: "\"\"", refs: [])
            if let (blockID, idx) = childIndexOf(rowID) {
                replaceRow(id: blockID) { block in
                    block.children.insert(template, at: idx)
                }
            } else if let idx = document.rows.firstIndex(where: { $0.id == rowID }) {
                document.rows.insert(template, at: idx)
            } else {
                return
            }
            selectedRowID = template.id
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// Duplicate `rowID` (a new id, new child ids — CFM-R8-FIX-6; the duplicate-keys trap in
    /// `FlowDocument` is fixed in the same commit).
    func duplicate(_ rowID: UUID) {
        commitChange {
            let before = document.rows
            guard let idx = document.rows.firstIndex(where: { $0.id == rowID }) else { return }
            let copy = Self.reminted(document.rows[idx])
            document.rows.insert(copy, at: idx + 1)
            selectedRowID = copy.id
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// Set/clear a row's chain break (the blank line).
    func setChainBreak(_ enabled: Bool, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { $0.chainBreak = enabled }
        }
    }

    /// Remove a row by id (top-level or any depth). The removed id's number is remembered
    /// (`tombstones`) so a downstream reference renders `(?N)` (answers c5/c9). Clause
    /// targets pointing at it keep the number and go stale (QR9) — never re-aimed.
    func remove(_ id: UUID) {
        commitChange {
            let before = document.rows
            if let path = Self.displayPath(id, rows: document.rows) {
                tombstones[id] = Self.numericValue(of: path)
            }
            removeById(id, from: &document.rows)
            if selectedRowID == id { selectedRowID = nil }
            reaimClauseTargets(before: before, after: document.rows, removedIDs: [id])
        }
    }

    /// Reorder the top-level rows (`.onMove`). Always allowed (answer `c12`); id-based refs
    /// are position-independent, so this never re-aims one — it can only turn a row yellow
    /// (a now-forward reference, or an incompatible auto-chain). Clause targets are
    /// renumbered by identity (QR9), so reordering never re-aims a decide edge either.
    func move(from source: IndexSet, to destination: Int) {
        commitChange {
            let before = document.rows
            document.rows.move(fromOffsets: source, toOffset: destination)
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// Reorder a block's children (CFM-R8-FIX-6 "reordered within"). CFM-R12-FIX-4: reaims
    /// clause targets by identity, exactly like the top-level `move` — reordering inside a
    /// block must never silently re-aim a decide edge (the QR9 ruling, half-applied before).
    func moveInside(blockID: UUID, from source: IndexSet, to destination: Int) {
        commitChange {
            let before = document.rows
            replaceRow(id: blockID) { row in
                row.children.move(fromOffsets: source, toOffset: destination)
            }
            reaimClauseTargets(before: before, after: document.rows)
        }
    }

    /// Set a row's k-th reference (1-based) to `targetID`, or nil to clear that slot —
    /// ordered-input support for bundle-accepting tasks like `Diff` (CFM-R8-FIX-2).
    ///
    /// INPUTS-1 fixes two hazards this used to have. First, clearing slot `k` used to
    /// `refs.remove(at:)`, which shifts every later ref down a position — clearing Input 1
    /// of a two-input row silently re-pointed `{2}` (Template) or a frame's own
    /// `{input[1]}` (Revise, Verify, …) at whatever Input 2 named, with no warning. A slot
    /// is now cleared **in place** — every other bound ref keeps exactly the position it
    /// had. Second, growing past the bound prefix (jumping to slot 3 with only slot 1 set,
    /// or setting slot 2 before slot 1 — Diff's two pickers are independent, so this is
    /// reachable in stock order) used to pad the gap with `.inputRef(n)`, a *real* reference
    /// to the row's own positional block input — inventing a binding the author never chose,
    /// and one that silently "works" (with the wrong data) for any row inside a block with
    /// that many inputs. Both cases now fill the gap with `markUnsetSlot()` instead: a
    /// `.rowRef` that can never resolve, so the picker reads it as "None" (`isSlotUnset`)
    /// and `structuralIssues()` blocks Save with an honest message, never a silent success
    /// or a confusing "(input:N) only makes sense inside a block" parser error.
    func setReference(to targetID: UUID?, slot: Int = 1, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                var refs = row.refs
                while refs.count < slot - 1 { refs.append(.rowRef(markUnsetSlot())) }
                let value: Ref = .rowRef(targetID ?? markUnsetSlot())
                if refs.count < slot {
                    refs.append(value)
                } else {
                    refs[slot - 1] = value
                }
                // Trailing unset slots carry no information (nothing bound after them to
                // preserve the position of) — drop them so `refs.count` keeps meaning "how
                // many slots are meaningfully occupied," the way `slotsToDraw` relies on.
                while let last = refs.last, isUnsetSlot(last) { refs.removeLast() }
                row.refs = refs
            }
        }
    }

    /// A fresh id `setReference` can hand to a slot with nothing chosen yet — see the
    /// `unsetSlotIDs` doc comment.
    private func markUnsetSlot() -> UUID {
        let id = UUID()
        unsetSlotIDs.insert(id)
        return id
    }

    private func isUnsetSlot(_ ref: Ref) -> Bool {
        if case .rowRef(let id) = ref { return unsetSlotIDs.contains(id) }
        return false
    }

    /// Whether `rowID`'s `slot`-th ref is genuinely unbound (never chosen, or cleared in
    /// place) rather than pointing at a row that once existed and was deleted — the two
    /// look identical structurally (a `.rowRef` to an id `document.rows` doesn't have), but
    /// only one of them is honestly "None" (`FlowRowInspectorView.currentInputLabel`).
    func isSlotUnset(for rowID: UUID, slot: Int) -> Bool {
        guard let row = row(withID: rowID), slot >= 1, slot <= row.refs.count,
              case .rowRef(let id) = row.refs[slot - 1] else { return false }
        return unsetSlotIDs.contains(id)
    }

    // MARK: - CFM-R9 — the row inspector's edits (byte-preserving settings round-trip)

    /// Set a row's declared signature (a block's `text -> [text, text, text]`), or nil.
    func setDeclaredSignature(_ signature: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { $0.declaredSignature = signature }
        }
    }

    /// Set a row's model display name (nil clears it). Also keeps the document's `models:`
    /// block honest (CFM-R14-4): a display no bridge entry resolves — an R14-2 unbridged pick,
    /// whose display **is** its `hfModelId` — must be pinned there, or the saved `.cat` names a
    /// model no other runtime could resolve and E104 flags it on reload. Bridged displays never
    /// need a pin (the bridge resolves them at runtime). An entry whose display no row uses
    /// anymore is dropped, so the block never goes stale.
    func setModel(_ display: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { $0.model = display }
            reconcileModelsBlock()
        }
    }

    /// CFM-R14-4 + FIX-7 — sync `document.models`/`modelsOrder` with the rows' model choices.
    /// An unbridged pick (display = the catalog `displayName`, no bridge row) is pinned as
    /// `displayName = hfModelId` — two different strings, which is the whole point of the
    /// `models:` block (a readable row plus a real pin). Bridged displays need no pin (the
    /// bridge resolves them). A pin whose display no row uses is dropped. Runs inside the same
    /// `commitChange` as the model edit, so undo restores the block with the row.
    private func reconcileModelsBlock() {
        var models = document.models
        var order = document.modelsOrder
        for row in allRows() {
            guard let display = row.model, CatalogBridge.entry(for: display) == nil else { continue }
            guard let entry = modelCatalog.first(where: {
                $0.displayName == display || $0.hfModelId == display
            }) else { continue }
            models[display] = entry.hfModelId
            if !order.contains(display) { order.append(display) }
        }
        let used = Set(allRows().compactMap(\.model))
        for key in models.keys where !used.contains(key) {
            models.removeValue(forKey: key)
            order.removeAll { $0 == key }
        }
        document.models = models
        document.modelsOrder = order
    }

    /// Edit one `key=value` setting, splicing only that token (the QR9 round-trip: the rest
    /// of the settings string is byte-identical).
    func setSetting(key: String, value: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                row.settings = FlowSettingsEditor.replace(key: key, value: value, in: row.settings)
            }
        }
    }

    /// HU-2 — a human row's waiting policy (check 6, E501): either park until answered
    /// (`wait=forever`) or give up after a duration and proceed as a declared default
    /// (`timeout=…`, `default=…`). The two are mutually exclusive on the wire, so writing one
    /// clears the other here, in one `commitChange`, so switching the picker is one undo step.
    nonisolated enum HumanWaitPolicy: Equatable {
        case waitForever
        case giveUpAfter(timeout: String, default: String)
    }

    func setHumanWaitPolicy(_ policy: HumanWaitPolicy, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                switch policy {
                case .waitForever:
                    row.settings = FlowSettingsEditor.replace(key: "wait", value: "forever", in: row.settings)
                    row.settings = FlowSettingsEditor.replace(key: "timeout", value: nil, in: row.settings)
                    row.settings = FlowSettingsEditor.replace(key: "default", value: nil, in: row.settings)
                case .giveUpAfter(let timeout, let dflt):
                    row.settings = FlowSettingsEditor.replace(key: "wait", value: nil, in: row.settings)
                    row.settings = FlowSettingsEditor.replace(key: "timeout", value: timeout, in: row.settings)
                    row.settings = FlowSettingsEditor.replace(key: "default", value: dflt, in: row.settings)
                }
            }
        }
    }

    /// HU-2's App Store read-only line, and HU-3's repair-path invariant: this reads what the
    /// row **actually says**, never assuming `wait=forever` — an imported flow authored on the
    /// Direct build with `timeout=`/`default=` must keep displaying that policy, not have it
    /// silently overridden by the fresh-row default (Ruling 3 restricts *authoring* in the App
    /// Store build, never *reading*).
    ///
    /// SP-3 (SPEC-Q226, owner ruling 2026-09-14): a row whose text still carries a bare,
    /// valueless `timeout=` (SP-2 stops the picker writing one, but a hand-edited or
    /// pre-SP-2 `.cat` may already have one) now falls straight through to the
    /// "No waiting policy is set" line instead of rendering an empty duration as a real
    /// policy — `FlowSettings.value(for: "timeout")` reads that token as absent, matching
    /// `FlowValidator.parseSettingsKV`'s reading, which is what raises E501 on the same row.
    nonisolated static func waitPolicyDescription(settings raw: String?) -> String {
        let settings = FlowSettings(raw)
        if let timeout = settings.value(for: "timeout"), let dflt = settings.value(for: "default") {
            return "Gives up after \(timeout) and proceeds as \"\(dflt)\"."
        }
        if settings.value(for: "wait") == "forever" {
            return "Waits until you answer."
        }
        return "No waiting policy is set — add `wait=forever` or `timeout=`/`default=` in Settings below."
    }

    /// Set the row's instruction — the quoted "What should it do?" text (answer `a5`).
    func setInstruction(_ text: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                row.settings = FlowSettingsEditor.replaceInstruction(text, in: row.settings)
            }
        }
    }

    /// RT-1 — set a `Template` row's pattern, the row's entire settings string (fact 11), via
    /// the whole-string writer rather than `setInstruction`'s first-quoted-token splice — a
    /// second token in the settings string would otherwise desync what this box shows from
    /// what the runtime renders (§6 Q1). `nil` (or empty) clears the row's settings entirely.
    func setPattern(_ text: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                row.settings = FlowSettingsEditor.replaceWholeSettings(text ?? "", in: row.settings)
            }
        }
    }

    /// Set a row's path token — the settings' `path=` or first bare token (CFM-R11-0b's
    /// "the chooser replaces the file by copying it in and the row still says a bare
    /// filename": the inspector writes the copied-in file's bare name here).
    func setPath(_ path: String, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                row.settings = FlowSettingsEditor.replacePath(path, in: row.settings)
            }
        }
    }

    /// RT-5 — the no-dead-ends backstop's raw "Raw settings" box (FV-1: relabelled from "Row
    /// text"). Unlike every other setter here,
    /// this replaces the row's settings string outright (like `setPattern`) but additionally
    /// **refuses** a commit that makes things worse: if the edit either makes the whole
    /// document unparseable, or gives this specific row a validator issue it didn't already
    /// have, the mutation is rolled back before this returns — inside the same `commitChange`
    /// call, so a refused edit is a true no-op (`currentSnapshot` unchanged) and never reaches
    /// the undo stack. Returns `nil` on a no-op or an accepted edit, the refusing issue's own
    /// sentence otherwise — the caller (the box) is responsible for leaving what the user
    /// typed on screen ("left dirty"), never silently reverting the visible text along with
    /// the data.
    @discardableResult
    func setRowText(_ text: String, for rowID: UUID) -> String? {
        guard let originalSettings = row(withID: rowID)?.settings else { return nil }
        let newSettings = text.isEmpty ? nil : text
        guard newSettings != originalSettings else { return nil }

        let path = displayNumber(of: rowID) ?? ""
        let beforeAll = structuralIssues()
        let beforeUnparseable = beforeAll.contains { $0.code == "E100" }
        let beforeRowCodes = Set(beforeAll.filter { $0.row == path }.map(\.code))

        var refusal: String?
        commitChange {
            replaceRow(id: rowID) { $0.settings = newSettings }
            let afterAll = structuralIssues()
            if !beforeUnparseable, let parseFailure = afterAll.first(where: { $0.code == "E100" }) {
                refusal = parseFailure.message
            } else {
                let afterRowCodes = afterAll.filter { $0.row == path }
                if let newIssue = afterRowCodes.first(where: { !beforeRowCodes.contains($0.code) }) {
                    refusal = newIssue.message
                }
            }
            if refusal != nil {
                replaceRow(id: rowID) { $0.settings = originalSettings }
            }
        }
        return refusal
    }

    /// Set a decider's declared tags — keeping the decide clause's edge tags in sync (FIX-4:
    /// the two halves must never drift).
    func setTags(_ tags: [String], for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                row.tags = tags.isEmpty ? nil : tags
                if let edges = row.clause?.edges {
                    row.clause = .decide(edges: zip(edges, tags).map { edge, tag in
                        ClauseEdge(tag: tag, target: edge.target)
                    })
                }
            }
        }
    }

    /// Set `max_visits` (nil clears it).
    func setVisitsLeq(_ n: Int?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { $0.visitsLeq = n }
        }
    }

    /// Set `on_budget` (nil clears it).
    func setOnBudget(_ value: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { $0.onBudget = value }
        }
    }

    /// Edit one `[tag | target]` decide-clause edge at `slot` (0-based); a nil tag removes
    /// the edge. `target` is the clause-target code: 0 = `done`, -1 = `resume`, else a row
    /// number (FIX-5 — the inspector's `done`/`resume` buttons produce real targets). The
    /// row's declared tags follow the edges (FIX-4).
    func setClauseEdge(tag: String?, target: Int?, at slot: Int, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                var edges = row.clause?.edges ?? []
                let targetClause: ClauseTarget?
                switch target {
                case 0: targetClause = .done
                case -1: targetClause = .resume
                case let n?: targetClause = .row(number: n)
                case nil: targetClause = .done
                }
                if slot < edges.count {
                    if let tag {
                        edges[slot] = ClauseEdge(tag: tag, target: targetClause ?? .done)
                    } else {
                        edges.remove(at: slot)
                    }
                } else if let tag {
                    edges.append(ClauseEdge(tag: tag, target: targetClause ?? .done))
                }
                row.clause = edges.isEmpty ? nil : .decide(edges: edges)
                row.tags = edges.isEmpty ? nil : edges.map(\.tag)
                staleClauseTargets[row.id]?.remove(slot)
                if staleClauseTargets[row.id]?.isEmpty == true { staleClauseTargets[row.id] = nil }
            }
        }
    }

    /// A decide clause's `[tag | target]` pairs.
    func deciderEdges(for rowID: UUID) -> [(tag: String, target: Int)] {
        guard let row = row(withID: rowID), case .decide(let edges)? = row.clause else { return [] }
        return edges.map { edge in
            let target: Int
            switch edge.target {
            case .row(let n), .call(let n): target = n
            case .done: target = 0
            case .resume: target = -1
            }
            return (edge.tag, target)
        }
    }

    /// The decision slots whose target row was deleted (QR9): they keep the dead number but
    /// render `(?N)` and block Save until re-pointed.
    func staleClauseSlots(for rowID: UUID) -> Set<Int> {
        staleClauseTargets[rowID] ?? []
    }

    /// The models that can serve `task` in this build — the **derived pool** (CFM-R14-2): the
    /// catalog entries whose corrected `runnerKind` serves the task, that the registry can
    /// claim, with no `ModelSupport` gap, plus (MS-2) any system/provider slots the task
    /// serves. RAM-sorted for the cataloged entries that have a RAM figure at all — a
    /// non-cataloged slot sorts after them, in registry order (inert today: both registries
    /// are empty, so this is always every entry, RAM-sorted, exactly as before MS).
    nonisolated static func candidateModels(for task: String,
                                            catalog: [ModelEntry],
                                            claimableModelIDs: Set<String>) -> [ModelSlot] {
        let derived = TaskModels.derivedModels(for: task, catalog: catalog,
                                               claimableModelIDs: claimableModelIDs)
        let withRAM = derived.filter { $0.modelEntry != nil }
            .sorted { $0.modelEntry!.ramGB < $1.modelEntry!.ramGB }
        let withoutRAM = derived.filter { $0.modelEntry == nil }
        return withRAM + withoutRAM
    }

    /// CFM-R14-3 + MS-2 — the Model menu's sections. The install state partitions the
    /// RAM-sorted cataloged candidates: **Installed** first (catalog ids already on disk),
    /// then **Available to download** with the total `downloadSizeGB` across the remaining.
    /// MS-2 adds **Built in** between them for non-cataloged slots (`.system`/`.provider`) —
    /// empty until Phase AFM/RM, so it changes no visible behaviour today; a section with no
    /// members is never rendered (`FlowRowInspectorView.swift`'s existing pattern).
    nonisolated static func sectionedModelCandidates(
        for task: String,
        catalog: [ModelEntry],
        installedModelIDs: Set<String>,
        claimableModelIDs: Set<String>
    ) -> (installed: [ModelSlot], builtIn: [ModelSlot], available: [ModelSlot], availableTotalGB: Double) {
        let all = candidateModels(for: task, catalog: catalog, claimableModelIDs: claimableModelIDs)
        let installed = all.filter { slot in
            guard let entry = slot.modelEntry else { return false }
            return installedModelIDs.contains(entry.id)
        }
        let builtIn = all.filter { $0.modelEntry == nil }
        let available = all.filter { slot in
            guard let entry = slot.modelEntry else { return false }
            return !installedModelIDs.contains(entry.id)
        }
        let total = available.reduce(0.0) { $0 + ($1.modelEntry?.downloadSizeGB ?? 0) }
        return (installed, builtIn, available, total)
    }

    /// Per-task suggested instruction defaults (answer `a5`, the docs' "You write" table).
    nonisolated static func suggestedInstructions(for task: String) -> [String] {
        switch task {
        case "Summarize": return ["TL;DR in 3 bullets", "one page"]
        case "Draft": return ["a launch announcement, 200 words", "a short email"]
        case "Rewrite": return ["formal", "half the length"]
        case "Title": return ["filename-safe", "a headline"]
        case "Translate": return ["to=French", "to=Spanish"]
        case "Ask": return ["10 factual questions"]
        case "Critique": return ["what's weak and how to fix it"]
        case "Gate": return ["Is this ready to ship? Answer with one tag."]
        case "Classify": return ["Which category does this belong to?"]
        case "Score": return ["Rate this out of 5."]
        case "Judge": return ["Which candidate is better?"]
        default: return []
        }
    }

    /// The row's output shape (its `gives`), or nil when unknown.
    func outputShape(for rowID: UUID) -> Shape? {
        guard let r = row(withID: rowID) else { return nil }
        return Self.signature(of: r, context: signatureContext)?.gives
    }

    // MARK: - The input picker (CFM-R8-FIX-1/2)

    /// The rows a given input slot of `rowID` may point at — same scope, **strictly
    /// earlier** (the validator's E203 forward-reference rule), shape-compatible for that
    /// slot. Ordered by display path. The save gate (not this menu) is the final authority
    /// on whether the whole document is valid.
    func validInputs(for rowID: UUID, slot: Int = 1) -> [(rowID: UUID, number: String, task: String)] {
        let context = signatureContext
        guard let r = row(withID: rowID), let accepts = Self.signature(of: r, context: context)?.accepts else { return [] }
        let slotShape: Shape
        if case .tupleOf(let kinds) = accepts {
            guard slot >= 1, slot <= kinds.count else { return [] }
            slotShape = .single(kinds[slot - 1])
        } else {
            // INPUTS-1: every non-tuple shape's compatibility rule is the same regardless
            // of position — `listOf(K)` needs every member to be `K`, `.frame` needs every
            // member to be text (`singleCompatible` ignores `accepts` entirely when
            // `rk == .frame`), and `.anyKind` accepts anything — so `slot == 1` was never a
            // correctness requirement, only an artifact of the panel not offering slot > 1.
            guard slot >= 1 else { return [] }
            slotShape = accepts
        }
        let scope = scopeRows(containing: rowID)
        guard let index = scope.firstIndex(where: { $0.id == rowID }), index > 0 else { return [] }
        return scope[0..<index].compactMap { candidate in
            guard let gives = Self.signature(of: candidate, context: context)?.gives else { return nil }
            let given: Shape = gives == .sameAsInput ? .anyKind : gives
            let refKind = r.task.flatMap { TaskCatalog.get($0)?.refKind }
            guard Shape.singleCompatible(accepts: slotShape, given: given, rk: refKind) else {
                return nil
            }
            return (candidate.id, displayNumber(of: candidate.id) ?? "?", Self.displayTaskName(candidate))
        }
    }

    /// INPUTS-1 — `inputSlotCount` retired. It answered one question ("how many pickers")
    /// from the task's declared *signature* alone, which is only ever right for `tupleOf`
    /// (a fixed arity) and a non-frame `single` (always 1) — `Shape.bundleCompatible`
    /// (`Core/Shape.swift:123`) is what actually governs how many refs a row may bind, and
    /// it has three more cases (`listOf`, `.frame`, `.anyKind`) where the signature says
    /// nothing about ref count at all. Replaced with the two questions that function was
    /// conflating: `declaredSlotFloor` (the fixed part, unchanged) plus `slotsToDraw` (what
    /// to actually show, `RSI/backlog.md` INPUTS-1) for "how many pickers", and
    /// `canAddInput` for "may this row take another one".

    /// The task's declared floor — `tupleOf`'s fixed arity, or 1 for every other shape.
    /// This alone was the old `inputSlotCount`'s whole answer; it's still exactly right for
    /// `tupleOf` (Diff, Retrieve, Store Index, Edit Image, Contact Sheet, …), whose slots are
    /// fixed, always shown, and never grow — but it's only ever a lower bound for everything
    /// else, which `slotsToDraw` accounts for.
    func declaredSlotFloor(for rowID: UUID) -> Int {
        guard let r = row(withID: rowID), let accepts = Self.signature(of: r, context: signatureContext)?.accepts else { return 1 }
        if case .tupleOf(let kinds) = accepts { return kinds.count }
        return 1
    }

    /// How many input pickers to draw for `rowID` right now: however many refs are actually
    /// bound, or the declared floor, whichever is larger. A `tupleOf` task's floor already
    /// covers its whole (fixed) slot count, so this is unchanged for it; every other shape —
    /// `listOf`, `.frame`, `.anyKind` — now draws one picker per **bound** ref instead of
    /// always exactly 1, so a hand-authored row loaded with several refs (33 rows across 28
    /// bundled flows, `RSI/backlog.md` INPUTS-1) shows all of them, not just the first. A row
    /// with nothing bound yet still shows exactly 1 empty picker — the simple case is
    /// pixel-identical to before.
    func slotsToDraw(for rowID: UUID) -> Int {
        let bound = row(withID: rowID)?.refs.count ?? 0
        return max(bound, declaredSlotFloor(for: rowID), 1)
    }

    /// Whether `rowID` may bind one more reference than `slotsToDraw` already shows —
    /// mirrors `Shape.bundleCompatible`'s branch order exactly, so the two can never
    /// disagree about what's legal: a `.frame` task ignores its declared `accepts` for
    /// ref-count purposes entirely (checked first, same as `bundleCompatible`), so it always
    /// takes another one regardless of what it declares; otherwise a `listOf`/`.anyKind`
    /// accepts takes any number, while `tupleOf` (a fixed arity) and a non-frame
    /// `single`/`unionOf`/`sameAsInput` (exactly one) never do.
    func canAddInput(for rowID: UUID) -> Bool {
        guard let r = row(withID: rowID) else { return false }
        if let task = r.task, TaskCatalog.get(task)?.refKind == .frame { return true }
        guard let accepts = Self.signature(of: r, context: signatureContext)?.accepts else { return false }
        switch accepts {
        case .listOf, .anyKind: return true
        case .tupleOf, .single, .unionOf, .sameAsInput: return false
        }
    }

    /// RT-1 — whether `id` sits inside an `<each>` block at any depth. Governs the Pattern
    /// box's `{index}`/`{item}` chips: those placeholders are only legal there
    /// (`FlowValidator.checkTemplatePlaceholders`'s `inEach`, propagated the same way —
    /// `inEach || row.blockKind == .each` — down through nested blocks).
    func isInsideEach(_ id: UUID) -> Bool {
        Self.isInsideEach(id, rows: document.rows, inEach: false)
    }

    private nonisolated static func isInsideEach(_ id: UUID, rows: [Row], inEach: Bool) -> Bool {
        for row in rows {
            if row.id == id { return inEach }
            let childInEach = inEach || row.blockKind == .each
            if isInsideEach(id, rows: row.children, inEach: childInEach) { return true }
        }
        return false
    }

    // MARK: - Integrity (the yellow row; CFM-R8-FIX-1/3)

    /// The one-line warning under a row, or nil when fine: a ref issue from the validator
    /// (E203/E204/E205/E401/E403…), a model-class row with no runnable model (h3), or a
    /// no-ref row whose auto-chain upstream is incompatible.
    func warning(for rowID: UUID) -> String? {
        guard let r = row(withID: rowID) else { return nil }
        let path = displayNumber(of: rowID) ?? "?"

        // WA-4 fallout: the validator's E104 is no longer filtered out (the registry now
        // resolves a real display name, so a genuine unresolvable one is a real issue —
        // `structuralIssues()`/`saveBlockReason` still see it and still block Save). But for
        // a model-class row with **no runnable model at all**, the check just below already
        // produces the GUI-appropriate sentence ("needs a model — pick one that runs on this
        // Mac", pointing at the Model menu); E104's own wording ("pin an id … add the line
        // yourself") is CLI-authoring advice that doesn't apply to this editor. Skip E104
        // here only in that case and let the check below speak instead — every other
        // validator issue, and E104 on a row whose model the app CAN otherwise run (a real
        // registry-coverage gap), still surfaces as-is.
        let ownIssues = issues(for: rowID)
        let isUnresolvableUnrunnableModel = ownIssues.first?.code == "E104"
            && r.task != nil && r.blockKind == nil
            && TaskCatalog.get(r.task ?? "")?.taskClass == .model
            && !isRunnableModel(r.model, for: r.task ?? "")
        if let first = ownIssues.first, !isUnresolvableUnrunnableModel {
            return "\(first.message)"
        }

        if r.task != nil, r.blockKind == nil, let desc = TaskCatalog.get(r.task ?? "") {
            if desc.taskClass == .model, !isRunnableModel(r.model, for: r.task ?? "") {
                // SPEC-Q214 (DA-6): a model-optional task (`Text to Table`) with **no** model
                // named is legitimate — its deterministic fast path needs none. Still warn
                // when a model IS named but doesn't resolve.
                if !(r.model == nil && TaskModels.isModelOptional(r.task ?? "")) {
                    return "Row \(path) needs a model — pick one that runs on this Mac."
                }
            }
            // HU: `.human` used to be in this "can't complete" bucket, but CFM-R10-Human
            // (47929d1) gave Ask Human/Human Input a real runtime path (park, or resolve to a
            // declared default on timeout) without this stale check being told — every fresh
            // human row showed a false "can't complete" warning underneath, on top of E501.
            // Same drift hit `.net`: CFM-R12-9 + WS-2 gave `RealExecutor.runNet` a real path
            // for all five networked tools (Web Fetch/Search/Fetch Feed/Download File/HTTP
            // Get), and `FlowRunner.rowClassRefusal` already lets `.net` run — this check
            // wasn't told either, so every net row showed the same false warning.
            if desc.taskClass != .instant, desc.taskClass != .model, desc.taskClass != .human,
               desc.taskClass != .net {
                return "Row \(path) is a \(desc.taskClass.rawValue) row, which this version of Flows can't complete."
            }
            // ES-UI-1: `Extract Structured`'s settings string *is* its column list. An empty or
            // unparseable one fails only at run time (`parseSchema` raises) — flag it here so
            // clearing the editor's "Columns to extract" field is never a silent break.
            if desc.refName == "engines.llm.extract_structured",
               (try? ExtractStructuredStage.parseSchema(r.settings)) == nil {
                return "Row \(path) needs a column list — e.g. \"merchant, date, total\"."
            }
        }

        // WA-2: mirrors `FlowValidator.checkRows`' E201 gate (`FlowValidator.swift:1067`,
        // ported verbatim from `core/validator.py:784-791`) — the gate raises E201 only
        // when `row.settings == nil` **and** `position not in edgeTargets`, on top of
        // `row.refs.isEmpty`. A row that carries its own settings sources itself (`Read
        // Index kb.index`, `Human Input "…"`); a row a decider jumps to is entered by the
        // jump, not by the row above it. Both make the row above's output shape
        // irrelevant, so neither is a real incompatibility.
        if r.refs.isEmpty, r.settings == nil, let previous = previousRow(before: rowID) {
            let scope = scopeRows(containing: rowID)
            let position = (scope.firstIndex(where: { $0.id == rowID }) ?? -1) + 1
            let edgeTargets = Set(scope.flatMap { FlowValidator.clauseTargets($0.clause) }.compactMap { target -> Int? in
                switch target {
                case .row(let n), .call(let n): return n
                case .resume, .done: return nil
                }
            })
            if !edgeTargets.contains(position),
               !Self.isCompatible(from: previous, to: r, context: signatureContext) {
                return "Row \(path) can't take row \(displayNumber(of: previous.id) ?? "?")'s output — add an input reference."
            }
        }
        // HR-1 (`RSI/DelegateHumanRowBacklog.md`) — a `Human Input` row at row 1 that carries
        // its own settings sources itself from a person, exactly as `Read Index`/`Read Text`
        // source themselves from a filename in their own settings; this mirrors the general
        // principle behind E201's `row.settings == nil` clause (`FlowValidator.swift
        // :1067-1068`: `row.refs.isEmpty && effectiveAccepts == nil && row.settings == nil` —
        // a settings-bearing row is never treated as unfed), the same exemption WA-2 already
        // gave the auto-chain branch above. Scoped to the task by name, not the `.human`
        // class: `Ask Human` shares the class but not the shape (`TaskCatalog.swift:234` gives
        // it `.sameAsInput`, a passthrough that cannot source content at row 1 — only `Human
        // Input`'s `t(.text)`, `:235`, produces one), so `Ask Human` never gets this exemption.
        //
        // **Not folded into the generic row-1 branch below, and evaluated first, deliberately
        // (HR-2 hazard):** HR-2 adds `Human Input` to `startingNodes()` so the step picker can
        // offer it, but that branch's own guard is `!startingNodes().contains(task)` — once
        // `Human Input` is a starting node, that guard goes false for it and the whole branch,
        // including its own settings-based read, would silently stop running for `Human Input`
        // specifically. That would mean a `timeout=`-without-`wait=forever` row 1 — the exact
        // case HR-1's new sentence exists for — goes back to showing nothing, regressing Q1's
        // owner-ruled split. Reading `Human Input` here, ahead of and independent of the
        // `startingNodes()` gate, is what keeps HR-1's distinction alive after HR-2 lands.
        if r.task == "Human Input", r.refs.isEmpty, r.blockKind == nil,
           previousRow(before: rowID) == nil, r.settings != nil {
            // Root cause 2: `timeout=` at row 1 is a real defect — Spec §10.1's
            // `default=unchanged` has no input to pass through here, so an unanswered row
            // silently emits an empty asset (`FlowInterpreter.swift:1182-1200`). A truthful,
            // different sentence about the missing fallback, not the false one about a
            // missing input; it blocks nothing (no `saveBlockReason` change).
            if FlowInterpreter.hasTimeout(r), !FlowInterpreter.waitsForever(r) {
                return "Row 1 has nothing to fall back on — if nobody answers, this row produces nothing."
            }
            return nil
        }
        if r.refs.isEmpty, r.blockKind == nil,
           let task = r.task, !Self.startingNodes().contains(where: { $0.name == task }),
           previousRow(before: rowID) == nil {
            // WA-3: this branch is right for a flow being built from scratch and wrong for
            // a **used** flow, whose row 1 is fed by its caller — restricted to the flow's
            // own row 1 (not a block's first child, which is a different `blockInputs`
            // question this check never modeled).
            let isFlowRowOne = document.rows.first?.id == rowID
            if isFlowRowOne && isCalledByWorkspaceSibling {
                return nil
            }
            return "Row \(path) needs an input — nothing feeds it."
        }
        return nil
    }

    /// FIX-1 — the flag a one-click repair would add for this row's own issue
    /// (E103/E109/E118/E120/E604), or nil when there's no such issue, no such repair, or this
    /// edition refuses to run the flag (`CapabilityGate.appStoreRefusedFlags`): offering
    /// "add `code`" in the App Store build would produce a flow that same build then refuses
    /// to run — a fix that creates the next refusal. Driven from the issue's code, not a
    /// hard-coded button per error, so all five share one path (rule 11: never parse the
    /// message back apart to find the flag).
    func headerRepair(for rowID: UUID) -> CapabilityFlag? {
        guard let issue = issues(for: rowID).first,
              let flag = FlowHeaderRepair.flag(forCode: issue.code) else { return nil }
        if CapabilityGate.isAppStoreBuild, CapabilityGate.appStoreRefusedFlags.contains(flag.rawValue) {
            return nil
        }
        return flag
    }

    /// FIX-1 — apply a header-flag repair through the editor's own change path
    /// (`commitChange`, the same wrapper `setHumanWaitPolicy` uses), so it joins the undo
    /// stack like any other edit and a second application is a no-op (`FlowHeaderRepair.apply`
    /// is idempotent).
    func applyHeaderRepair(_ flag: CapabilityFlag) {
        commitChange {
            document = FlowHeaderRepair.apply(flag, to: document)
        }
    }

    /// FH-4 — the Flow tab's untick action, the mirror of `applyHeaderRepair` through the
    /// same `commitChange` wrapper, so it joins undo/redo like any other edit. Untick is
    /// always offered (owner ruling Q1, already recorded): it cannot make the flow *unsafe* —
    /// at worst it fails `check` with the same defined error `applyHeaderRepair` fixes, and
    /// Save stays blocked until it's fixed or the flag is re-added.
    func removeHeaderFlag(_ flag: CapabilityFlag) {
        commitChange {
            document = FlowHeaderRepair.remove(flag, from: document)
        }
    }

    /// The `(?N)` reference label for a row (a broken ref renders `(?N)`, the row is yellow).
    func referenceLabel(for rowID: UUID) -> String? {
        guard let row = row(withID: rowID), !row.refs.isEmpty else { return nil }
        let parts = row.refs.map { ref in
            switch ref {
            case .rowRef(let id):
                if let number = displayNumber(of: id) { return number }
                if unsetSlotIDs.contains(id) { return "_" }
                return "?" + String(tombstones[id] ?? 0)
            case .inputRef(let position):
                return "input:\(position)"
            case .paramRef(let name):
                return "param:\(name)"
            }
        }
        return "(" + parts.joined(separator: ",") + ")"
    }

    /// The sentence blocking Save, or nil when saveable (CFM-R8-FIX-1).
    var saveBlockReason: String? {
        structuralIssues().first?.message
    }

    var canSave: Bool { saveBlockReason == nil }

    /// Whether the in-memory document differs from what's on disk (CFM-R8-FIX-4).
    var isDirty: Bool {
        guard let savedText else { return true }
        return catText != savedText
    }

    // MARK: - Validator bridge (CFM-R8-FIX-1)

    /// The document's issues from `FlowValidator.checkFlow` (serialize→reparse round trip,
    /// the same file `catflow check` sees), **excluding E104** (unpinned model display names
    /// — the app resolves those via `CatalogBridge` at run time), **plus** a local dead-ref
    /// issue for any reference to a deleted row (the file can't represent one — it would
    /// serialize as `(-1)`, which the validator can't flag). **FIX-3:** an unparseable
    /// document is itself a blocking issue — the gate fails *closed*, never open. Cached.
    func structuralIssues() -> [FlowIssue] {
        if let cached = issueCache, cached.document == document { return cached.issues }
        var issues: [FlowIssue] = []
        // INPUTS-1: an unset-slot marker (`markUnsetSlot`) has to be checked, and added,
        // *before* the round-trip validator runs — `saveBlockReason`/`warning(for:)` both
        // take `issues.first`, so a row that also happens to carry some unrelated validator
        // issue must still show the honest "isn't set yet" first, not lose to whatever the
        // validator says.
        for row in allRows() {
            for (index, ref) in row.refs.enumerated() {
                guard case .rowRef(let targetID) = ref, unsetSlotIDs.contains(targetID) else { continue }
                guard let path = Self.displayPath(row.id, rows: document.rows) else { continue }
                issues.append(FlowIssue(row: path, code: "E203",
                                        message: "Row \(path)'s input \(index + 1) isn't set yet — pick one or remove the slot, then save."))
            }
        }
        do {
            let parsed = try CatParser.parseForValidation(catText)
            // WA-4: the registry now resolves a display name the way `mlxflow check` does
            // (`CuratedManifest.installedFlowRegistry`), including inside a nested `uses:`
            // check (`resolveUsesLevel` passes this same `registry` down) — the blanket
            // `.filter { $0.code != "E104" }` this line used to carry is gone with it; a
            // display the registry can't resolve now genuinely means E104.
            issues += FlowValidator.checkFlow(parsed, registry: CuratedManifest.installedFlowRegistry(),
                                             workspace: workspace, flowID: flowID)
        } catch {
            issues.append(FlowIssue(row: "run", code: "E100",
                                message: "This flow doesn't parse as a valid .cat right now — fix or undo the last edit, then save."))
        }
        for row in allRows() {
            for ref in row.refs {
                guard case .rowRef(let targetID) = ref, !unsetSlotIDs.contains(targetID) else { continue }
                if Self.findRow(targetID, in: document.rows) == nil, let path = Self.displayPath(row.id, rows: document.rows) {
                    issues.append(FlowIssue(row: path, code: "E203",
                                            message: "Row \(path) references a row that was deleted — fix the reference or undo, then save."))
                }
            }
        }
        for (deciderID, slots) in staleClauseTargets {
            guard let path = Self.displayPath(deciderID, rows: document.rows) else { continue }
            for slot in slots.sorted() {
                issues.append(FlowIssue(row: path, code: "E203",
                                        message: "Row \(path)'s decision \(slot + 1) targets a row that was deleted — re-point it or undo, then save."))
            }
        }
        // CFM-R12-3 item 5: a block emptied to zero children blocks Save — same honesty as a
        // broken reference, never a silent deletion of the block.
        for row in allRows() where row.blockKind != nil && row.children.isEmpty {
            if let path = Self.displayPath(row.id, rows: document.rows) {
                issues.append(FlowIssue(row: path, code: "E2xx",
                                        message: "Row \(path) is a block with no steps — add one inside it, or remove the block."))
            }
        }
        issueCache = (document, issues)
        return issues
    }

    /// The issues for one row (dotted-path filtered).
    func issues(for rowID: UUID) -> [FlowIssue] {
        let path = displayNumber(of: rowID) ?? ""
        return structuralIssues().filter { $0.row == path }
    }

    private var issueCache: (document: FlowDocument, issues: [FlowIssue])?

    // MARK: - Save (CFM-R8-5)

    /// The canonical `.cat` bytes for the current document (the flow list *is* the file).
    var catText: String {
        CatSerializer.serialize(document)
    }

    /// Write the canonical `.cat` into the flow's workspace folder. Refuses while the
    /// validator rejects the file (UI-only ids and broken refs never reach it — answer `e`).
    /// CFM-R11-3: the extension follows the document's kind — a flow opened from a
    /// `.catpipeline` saves back as one, a `.cat` as `.cat`, a new flow as `.cat`.
    func save() throws {
        if let reason = saveBlockReason {
            throw FlowEditingError.refusingToSave(reason)
        }
        let fm = FileManager.default
        let dir = workspace.directory(for: flowID)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(Self.sanitizedFileName(name)).\(Self.fileExtension(for: document.fileKind))")

        // Clear stale flow files *before* writing. A rename, a `.cat` ↔ `.catpipeline` kind
        // change, or a reopened editor (which starts with `savedURL == nil`, so the old file
        // isn't tracked) all leave the previous file behind, and `UserFlowStore.scan` shows
        // whichever the filesystem lists first. A case-only rename is worse: `write(to:)`
        // reuses the existing directory entry on a case-insensitive volume, so the old casing
        // sticks unless the file is removed first. Comparisons are by name — `contentsOf
        // Directory` can hand back `/private/var/…` where `url` is `/var/…`.
        let siblings = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []

        // KW-1-FIX-2: in a *shared* workspace folder, a sibling that exactly occupies the
        // write target's name and isn't the file this editor already owns (`savedURL`) is a
        // *different* flow — writing would silently destroy it. The loop below already skips
        // deleting it (`name != url.lastPathComponent` below), which is right, but nothing
        // previously stopped the write itself from overwriting it anyway. Refuse instead of
        // guessing a fix. A case-only match (the `Extract table data...` vs `EXTRACT TABLE
        // DATA...` case) is this same file re-cased, not a collision, so it isn't caught here.
        // Restricted to shared folders: a plain `flows/<flowID>/` folder holds exactly one
        // file by construction, so a pre-existing file at the target name there is always this
        // editor's own (e.g. a freshly-opened model with no seeded `savedURL` resaving itself)
        // — never someone else's.
        if isSharedWorkspaceFolder, let collision = siblings.first(where: {
            ($0.pathExtension == "cat" || $0.pathExtension == "catpipeline")
                && $0.lastPathComponent == url.lastPathComponent
                && savedURL?.lastPathComponent != $0.lastPathComponent
        }) {
            throw FlowEditingError.refusingToSave(
                "'\(collision.lastPathComponent)' already exists in this workspace — pick a different name.")
        }

        for f in siblings where f.pathExtension == "cat" || f.pathExtension == "catpipeline" {
            let name = f.lastPathComponent
            guard name != url.lastPathComponent else { continue }        // never the exact target
            let caseOnlyDuplicate = name.compare(url.lastPathComponent, options: .caseInsensitive) == .orderedSame
            let trackedPrevious = savedURL.map { name == $0.lastPathComponent } ?? false
            if !isSharedWorkspaceFolder || caseOnlyDuplicate || trackedPrevious {
                try? fm.removeItem(at: f)
            }
        }

        try catText.write(to: url, atomically: true, encoding: .utf8)
        savedURL = url
        savedText = catText
        saveError = nil
    }

    /// The file extension for a kind — R11-3: `.catpipeline` round-trips as `.catpipeline`.
    nonisolated static func fileExtension(for kind: FileKind) -> String {
        kind == .catpipeline ? "catpipeline" : "cat"
    }

    /// A valid `.cat` filename from the display name (answer `h4`: a valid file name).
    nonisolated static func sanitizedFileName(_ name: String) -> String {
        let cleaned = name.replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
            .replacingOccurrences(of: "\\", with: "-")
        return cleaned.isEmpty ? "Untitled" : cleaned
    }

    // MARK: - Undo/redo (CFM-R8-4, CFM-R8-FIX-8)

    private var currentSnapshot: Snapshot {
        Snapshot(document: document, selectedRowID: selectedRowID, tombstones: tombstones,
                 staleClauseTargets: staleClauseTargets)
    }

    private func restore(_ snapshot: Snapshot) {
        document = snapshot.document
        selectedRowID = snapshot.selectedRowID
        tombstones = snapshot.tombstones
        staleClauseTargets = snapshot.staleClauseTargets
    }

    func undo() {
        guard let previous = undoStack.popLast() else { return }
        redoStack.append(currentSnapshot)
        restore(previous)
    }

    func redo() {
        guard let next = redoStack.popLast() else { return }
        undoStack.append(currentSnapshot)
        restore(next)
    }

    /// Reset to a blank flow (the "New Flow" action).
    func reset() {
        undoStack.removeAll()
        redoStack.removeAll()
        tombstones.removeAll()
        staleClauseTargets.removeAll()
        selectedRowID = nil
        savedURL = nil
        savedText = nil
        saveError = nil
        document = FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [])
    }

    // MARK: - Run

    var runnability: Runnability { FlowRunner.canRun(document) }

    // MARK: - Helpers

    /// Snapshot the pre-edit state, mutate, and only then record the undo unit — a no-op
    /// edit (e.g. dropping a row back where it was) neither pushes undo nor clears redo
    /// (CFM-R8-FIX-8).
    private func commitChange(_ mutate: () -> Void) {
        let before = currentSnapshot
        mutate()
        guard currentSnapshot != before else { return }
        undoStack.append(before)
        if undoStack.count > Self.maxUndoDepth {
            undoStack.removeFirst(undoStack.count - Self.maxUndoDepth)
        }
        redoStack.removeAll()
    }

    /// The row with `id` (top-level or any depth).
    func row(withID id: UUID) -> Row? {
        Self.findRow(id, in: document.rows)
    }

    private func allRows() -> [Row] {
        Self.flatten(document.rows)
    }

    private nonisolated static func flatten(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + flatten($0.children) }
    }

    private nonisolated static func findRow(_ id: UUID, in rows: [Row]) -> Row? {
        for row in rows {
            if row.id == id { return row }
            if let found = findRow(id, in: row.children) { return found }
        }
        return nil
    }

    /// `(blockID, childIndex)` when `id` is a block's child, at any depth.
    private func childIndex(_ id: UUID) -> (UUID, Int)? {
        for block in document.rows {
            if let idx = block.children.firstIndex(where: { $0.id == id }) {
                return (block.id, idx)
            }
            for grandchild in block.children {
                if let idx = grandchild.children.firstIndex(where: { $0.id == id }) {
                    return (grandchild.id, idx)
                }
            }
        }
        return nil
    }

    private func insertChild(_ child: Row, into blockID: UUID, below selectedID: UUID?) {
        replaceRow(id: blockID) { block in
            if let selectedID,
               let idx = block.children.firstIndex(where: { $0.id == selectedID }) {
                block.children.insert(child, at: idx + 1)
            } else {
                block.children.append(child)
            }
        }
    }

    private func removeById(_ id: UUID, from rows: inout [Row]) {
        if let idx = rows.firstIndex(where: { $0.id == id }) {
            rows.remove(at: idx)
            return
        }
        for i in rows.indices {
            removeById(id, from: &rows[i].children)
        }
    }

    /// Replace the row with `id` (any depth) via `transform`.
    private func replaceRow(id: UUID, transform: (inout Row) -> Void) {
        func replace(_ rows: inout [Row]) {
            for i in rows.indices where rows[i].id == id {
                transform(&rows[i])
                return
            }
            for i in rows.indices {
                replace(&rows[i].children)
            }
        }
        replace(&document.rows)
    }

    /// The QR9 ClauseTarget ruling: renumber every `.row(number:)` clause target (decide
    /// edges, `-> N`, `fork`) so it keeps pointing at the **same row** (by identity) after a
    /// structural edit — reordering/inserting/duplicating never silently re-aims a target at
    /// whatever now occupies the old slot. Targets that pointed at a deleted row keep that
    /// row's tombstoned number and are recorded as stale (yellow + Save blocked), exactly
    /// like a broken reference. Only top-level numbers are meaningful (FIX-5: nested rows
    /// have no numeric clause-target form).
    private func reaimClauseTargets(before: [Row], after: [Row], removedIDs: Set<UUID> = []) {
        let beforeNumbers: [Int: UUID] = Dictionary(uniqueKeysWithValues: before.enumerated().map { ($0.offset + 1, $0.element.id) })
        let afterIDs: [UUID: Int] = Dictionary(uniqueKeysWithValues: after.enumerated().map { ($0.element.id, $0.offset + 1) })
        var newlyStale: [UUID: Set<Int>] = [:]

        for row in Self.flatten(document.rows) {
            guard let clause = row.clause else { continue }
            let rewritten = Self.rewrittenClause(clause, beforeNumbers: beforeNumbers,
                                                 afterIDs: afterIDs, rowID: row.id,
                                                 staleSlots: &newlyStale)
            if rewritten != clause {
                replaceRow(id: row.id) { $0.clause = rewritten }
            }
        }
        if !newlyStale.isEmpty {
            for (decider, slots) in newlyStale {
                staleClauseTargets[decider, default: []].formUnion(slots)
            }
        }
    }

    /// Rewrite one clause's `.row(number:)` targets against the identity maps; stale slots
    /// (target row deleted) are recorded into `staleSlots`.
    private nonisolated static func rewrittenClause(
        _ clause: Clause, beforeNumbers: [Int: UUID], afterIDs: [UUID: Int],
        rowID: UUID, staleSlots: inout [UUID: Set<Int>]
    ) -> Clause {
        func rewrite(_ target: ClauseTarget, slot: Int?) -> ClauseTarget {
            guard case .row(let number) = target, let targetID = beforeNumbers[number] else {
                return target
            }
            if let newNumber = afterIDs[targetID] {
                return newNumber == number ? target : .row(number: newNumber)
            }
            if let slot {
                staleSlots[rowID, default: []].insert(slot)
            }
            return target   // keep the tombstoned number; the editor flags it as stale
        }
        switch clause {
        case .goto(let target):
            return .goto(target: rewrite(target, slot: nil))
        case .fork(let targets):
            return .fork(targets: targets.map { rewrite($0, slot: nil) })
        case .decide(let edges):
            var out = edges
            for i in out.indices {
                out[i] = ClauseEdge(tag: out[i].tag, target: rewrite(out[i].target, slot: i))
            }
            return .decide(edges: out)
        case .call, .resume:
            return clause
        }
    }

    /// Remint every id in a subtree (for `duplicate`).
    private nonisolated static func reminted(_ row: Row) -> Row {
        Row(id: UUID(), task: row.task, blockKind: row.blockKind, blockName: row.blockName,
            model: row.model, settings: row.settings, refs: [], chainBreak: row.chainBreak,
            children: row.children.map(reminted), clause: row.clause, tags: row.tags,
            visitsLeq: row.visitsLeq, onBudget: row.onBudget,
            declaredSignature: row.declaredSignature, comment: row.comment,
            leadingComments: row.leadingComments)
    }

    /// The dotted display path of `id` (e.g. `"3"`, `"2.1"`) — nested rows included
    /// (CFM-R8-FIX-5).
    func displayNumber(of id: UUID) -> String? {
        Self.displayPath(id, rows: document.rows)
    }

    private nonisolated static func displayPath(_ id: UUID, rows: [Row], prefix: String = "") -> String? {
        for (i, row) in rows.enumerated() {
            let path = prefix.isEmpty ? "\(i + 1)" : "\(prefix).\(i + 1)"
            if row.id == id { return path }
            if let found = displayPath(id, rows: row.children, prefix: path) { return found }
        }
        return nil
    }

    /// The numeric part of a dotted path ("2.1" → 21) — the tombstone's `(?N)` display.
    private nonisolated static func numericValue(of path: String) -> Int {
        Int(path.split(separator: ".").map(String.init).joined()) ?? 0
    }

    /// The rows that contain `rowID` as a top-level member — its scope (the flow's own rows
    /// or a block's children).
    private func scopeRows(containing rowID: UUID) -> [Row] {
        if document.rows.contains(where: { $0.id == rowID }) { return document.rows }
        for block in document.rows {
            if block.children.contains(where: { $0.id == rowID }) { return block.children }
        }
        return document.rows
    }

    /// The previous row in the same scope (auto-chain source), honoring `chainBreak`
    /// (CFM-R8-FIX-6): a blank line breaks the chain, so no upstream.
    private func previousRow(before id: UUID) -> Row? {
        let scope = scopeRows(containing: id)
        guard let index = scope.firstIndex(where: { $0.id == id }), index > 0 else { return nil }
        let previous = scope[index - 1]
        return previous.chainBreak ? nil : previous
    }

    /// Whether `target`'s output can feed `row`'s accepts as a bundle (CFM-R8-FIX-2).
    /// WA-1: a `nil` signature on either side means "the editor cannot type this row" — an
    /// **unknown** shape, not an incompatible one (`FlowValidator.checkRows`' E201 gate
    /// never fires on a `uses:`/`definitions:`/`transforms:` row precisely because it
    /// always resolves a signature for one, `FlowValidator.swift:983-999`). Returning
    /// `false` here for a task the editor genuinely can't resolve produced a false yellow
    /// row on every task next to a `uses:` call; a wrong warning is worse than a missing
    /// one.
    private nonisolated static func isCompatible(from target: Row, to row: Row, context: FlowSignatureContext) -> Bool {
        guard let toSig = signature(of: row, context: context),
              let fromSig = signature(of: target, context: context) else { return true }
        let given: Shape = fromSig.gives == .sameAsInput ? .anyKind : fromSig.gives
        let refKind = row.task.flatMap { TaskCatalog.get($0)?.refKind }
        return Shape.bundleCompatible(accepts: toSig.accepts, given: [given], rk: refKind)
    }

    /// Whether a display model name can run **the row's task** — membership in that task's
    /// derived pool (CFM-R14-2 + FIX-6). Asking "is this display any claimable catalog entry?"
    /// would let a `Transcribe` row name `Kokoro 82M` and pass; the pool for the row's task is
    /// the single authority. `nil` (no model) is false. With no catalog wired yet, the answer
    /// is **false** (the warning shows), never a different authority: the bridge table is not a
    /// second source of truth for "can this run".
    private func isRunnableModel(_ display: String?, for task: String) -> Bool {
        guard let display, !modelCatalog.isEmpty else { return false }
        return TaskModels.derivedModels(for: task, catalog: modelCatalog,
                                        claimableModelIDs: claimableModelIDs).contains { slot in
            slot.displayName == display || slot.modelEntry?.hfModelId == display
        }
    }

    /// A row's (accepts, gives), mirroring `Shape.signature`'s block inference — the same
    /// resolution the interpreter and auto-chain use. (A block's *declared* signature is the
    /// validator's concern — `structuralIssues()` runs `checkFlow`, which compares it to the
    /// resolved type and flags a mismatch. Declared signatures do NOT change auto-chain.)
    ///
    /// WA-1: a task outside `TaskCatalog` is not necessarily unknown — it may name a
    /// `definitions:` composite, a `uses:` entry, or a `transforms:` entry, exactly the
    /// three cases `FlowValidator.checkRows` resolves before falling through to
    /// unknown-task (`FlowValidator.swift:980`-`:1000`, ported verbatim below). Only a task
    /// in none of those four places stays unresolved.
    private nonisolated static func signature(of row: Row, context: FlowSignatureContext) -> (accepts: Shape, gives: Shape)? {
        if let blockKind = row.blockKind {
            return inferredBlockSignature(row, blockKind: blockKind, context: context)
        }
        guard let task = row.task else { return nil }
        if let desc = TaskCatalog.get(task) {
            return (desc.accepts, desc.gives)
        }
        if let comp = context.definitions[task] {
            let declared = comp.signature.flatMap(FlowValidator.parseDeclaredSignature)
            return (declared?.0 ?? .anyKind, declared?.1 ?? .anyKind)
        }
        if context.uses[task] != nil {
            return (.anyKind, .anyKind)
        }
        if let transform = context.transforms[task] {
            let declared = transform.signature.flatMap(FlowValidator.parseDeclaredSignature)
            return (declared?.0 ?? .anyKind, declared?.1 ?? .anyKind)
        }
        return nil
    }

    private nonisolated static func inferredBlockSignature(_ row: Row, blockKind: BlockKind, context: FlowSignatureContext) -> (accepts: Shape, gives: Shape)? {
        guard let first = row.children.first, let last = row.children.last,
              let firstSig = signature(of: first, context: context),
              let lastSig = signature(of: last, context: context) else {
            return nil
        }
        var accepts = firstSig.accepts
        var gives = lastSig.gives
        if blockKind == .each {
            if case .single(let k) = accepts { accepts = .listOf(k) }
            if case .single(let k) = gives { gives = .listOf(k) }
        }
        return (accepts, gives)
    }

    /// WA-1 — the `uses:`/`definitions:`/`transforms:` sections `signature(of:)` needs to
    /// resolve a non-catalog row the way `FlowValidator.checkRows` does. A value type
    /// (rather than making `signature(of:)` an instance method) so the function stays
    /// `nonisolated static` — `validInputs`, `outputShape`, `inputSlotCount` and
    /// `isCompatible` all call it off the main actor in tests.
    private nonisolated struct FlowSignatureContext {
        var uses: [String: String]
        var definitions: [String: CompositeDef]
        var transforms: [String: TransformDef]
    }

    private var signatureContext: FlowSignatureContext {
        FlowSignatureContext(uses: document.uses, definitions: document.definitions, transforms: document.transforms)
    }

    /// WA-3 — whether any sibling flow in this workspace calls this flow via `uses:`,
    /// the same fact `KW-2-FIX-1`'s rename/delete confirmation already computes
    /// (`WorkspaceStore.callers(of:in:ws:)`, `WorkspaceStore.swift:93`). Computed once per
    /// model rather than per render: `warning(for:)` evaluates every row on every render,
    /// and the underlying scan touches disk. Never invalidated — there is no file watcher
    /// on `workspaces/` (`CLAUDE.md`), so the sibling set is stable for an editing session.
    /// A plain memoized function, not `lazy var`: `@Observable`'s macro expansion can't
    /// generate an init accessor for a `lazy` stored property.
    private var isCalledByWorkspaceSiblingCache: Bool?

    private var isCalledByWorkspaceSibling: Bool {
        if let cached = isCalledByWorkspaceSiblingCache { return cached }
        let result: Bool
        if isSharedWorkspaceFolder,
           let ws = WorkspaceStore.scan(workspace: workspace).first(where: { $0.url == workspace.directory(for: flowID) }) {
            let dir = workspace.directory(for: flowID)
            let target = savedURL ?? dir.appendingPathComponent(
                "\(Self.sanitizedFileName(name)).\(Self.fileExtension(for: document.fileKind))")
            result = !WorkspaceStore.callers(of: target, in: ws, ws: workspace).isEmpty
        } else {
            result = false
        }
        isCalledByWorkspaceSiblingCache = result
        return result
    }

    private nonisolated static func displayTaskName(_ row: Row) -> String {
        if let task = row.task { return task }
        return "<\(row.blockKind?.rawValue ?? "block")>"
    }
}

/// CFM-R8-5 save-gate failures.
nonisolated enum FlowEditingError: Error, CustomStringConvertible, Equatable {
    case refusingToSave(String)

    var description: String {
        switch self {
        case .refusingToSave(let reason): return reason
        }
    }
}
