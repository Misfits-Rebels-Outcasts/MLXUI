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
    /// The row the step picker and Add-below target; nil = append at the end.
    var selectedRowID: UUID?
    /// The URL the flow was last saved to, nil until the first save.
    private(set) var savedURL: URL?
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
         savedText: String? = nil) {
        self.name = name
        self.flowID = flowID
        self.workspace = workspace
        self.document = document ?? FlowDocument(version: "0.8", headerKeyword: "mlxflow", rows: [])
        self.sampleSourceDir = sampleSourceDir
        self.savedText = savedText
        // Opening an existing flow selects its first row so the inspector pane is up and
        // showing the Properties tab (CFM — a click on a My Workflows flow opens Edit).
        if let first = self.document.rows.first {
            self.selectedRowID = first.id
        }
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
        let seedValue = seedSample(for: name)
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
        let seedValue = seedSample(for: name)
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
    func setReference(to targetID: UUID?, slot: Int = 1, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                var refs = row.refs
                while refs.count < slot { refs.append(.inputRef(refs.count + 1)) }
                if let targetID {
                    refs[slot - 1] = .rowRef(targetID)
                } else {
                    refs.remove(at: slot - 1)
                }
                row.refs = refs
            }
        }
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

    /// Set the row's instruction — the quoted "What should it do?" text (answer `a5`).
    func setInstruction(_ text: String?, for rowID: UUID) {
        commitChange {
            replaceRow(id: rowID) { row in
                row.settings = FlowSettingsEditor.replaceInstruction(text, in: row.settings)
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
    /// claim, with no `ModelSupport` gap. Paired with the `.cat` display name (`TaskModels.
    /// displayName` — the bridge name when the model is bridged, else the raw `hfModelId`),
    /// RAM-sorted. The inspector shows the display name and dims the ones that don't fit.
    nonisolated static func candidateModels(for task: String,
                                            catalog: [ModelEntry],
                                            claimableModelIDs: Set<String>) -> [(display: String, model: ModelEntry)] {
        let derived = TaskModels.derivedModels(for: task, catalog: catalog,
                                               claimableModelIDs: claimableModelIDs)
        return derived
            .map { (TaskModels.displayName(for: $0), $0) }
            .sorted { $0.model.ramGB < $1.model.ramGB }
    }

    /// CFM-R14-3 — the Model menu's two sections. The install state partitions the
    /// RAM-sorted derived candidates: **Installed** first (catalog ids already on disk), then
    /// **Available to download** with the total `downloadSizeGB` across the remaining. The
    /// section header shows the total so a user sees the download cost before the arm sheet.
    nonisolated static func sectionedModelCandidates(
        for task: String,
        catalog: [ModelEntry],
        installedModelIDs: Set<String>,
        claimableModelIDs: Set<String>
    ) -> (installed: [(display: String, model: ModelEntry)], available: [(display: String, model: ModelEntry)], availableTotalGB: Double) {
        let all = candidateModels(for: task, catalog: catalog, claimableModelIDs: claimableModelIDs)
        let installed = all.filter { installedModelIDs.contains($0.model.id) }
        let available = all.filter { !installedModelIDs.contains($0.model.id) }
        let total = available.reduce(0.0) { $0 + $1.model.downloadSizeGB }
        return (installed, available, total)
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
        return Self.signature(of: r)?.gives
    }

    // MARK: - The input picker (CFM-R8-FIX-1/2)

    /// The rows a given input slot of `rowID` may point at — same scope, **strictly
    /// earlier** (the validator's E203 forward-reference rule), shape-compatible for that
    /// slot. Ordered by display path. The save gate (not this menu) is the final authority
    /// on whether the whole document is valid.
    func validInputs(for rowID: UUID, slot: Int = 1) -> [(rowID: UUID, number: String, task: String)] {
        guard let r = row(withID: rowID), let accepts = Self.signature(of: r)?.accepts else { return [] }
        let slotShape: Shape
        if case .tupleOf(let kinds) = accepts {
            guard slot >= 1, slot <= kinds.count else { return [] }
            slotShape = .single(kinds[slot - 1])
        } else {
            guard slot == 1 else { return [] }
            slotShape = accepts
        }
        let scope = scopeRows(containing: rowID)
        guard let index = scope.firstIndex(where: { $0.id == rowID }), index > 0 else { return [] }
        return scope[0..<index].compactMap { candidate in
            guard let gives = Self.signature(of: candidate)?.gives else { return nil }
            let given: Shape = gives == .sameAsInput ? .anyKind : gives
            let refKind = r.task.flatMap { TaskCatalog.get($0)?.refKind }
            guard Shape.singleCompatible(accepts: slotShape, given: given, rk: refKind) else {
                return nil
            }
            return (candidate.id, displayNumber(of: candidate.id) ?? "?", Self.displayTaskName(candidate))
        }
    }

    /// The number of input slots a row needs (1 for ordinary accepts; `tupleOf` arity).
    func inputSlotCount(for rowID: UUID) -> Int {
        guard let r = row(withID: rowID), let accepts = Self.signature(of: r)?.accepts else { return 1 }
        if case .tupleOf(let kinds) = accepts { return kinds.count }
        return 1
    }

    // MARK: - Integrity (the yellow row; CFM-R8-FIX-1/3)

    /// The one-line warning under a row, or nil when fine: a ref issue from the validator
    /// (E203/E204/E205/E401/E403…), a model-class row with no runnable model (h3), or a
    /// no-ref row whose auto-chain upstream is incompatible.
    func warning(for rowID: UUID) -> String? {
        guard let r = row(withID: rowID) else { return nil }
        let path = displayNumber(of: rowID) ?? "?"

        let ownIssues = issues(for: rowID)
        if let first = ownIssues.first {
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
            if desc.taskClass != .instant, desc.taskClass != .model {
                return "Row \(path) is a \(desc.taskClass.rawValue) row, which this version of Flows can't complete."
            }
        }

        if r.refs.isEmpty, let previous = previousRow(before: rowID) {
            if !Self.isCompatible(from: previous, to: r) {
                return "Row \(path) can't take row \(displayNumber(of: previous.id) ?? "?")'s output — add an input reference."
            }
        }
        if r.refs.isEmpty, r.blockKind == nil,
           let task = r.task, !Self.startingNodes().contains(where: { $0.name == task }),
           previousRow(before: rowID) == nil {
            return "Row \(path) needs an input — nothing feeds it."
        }
        return nil
    }

    /// The `(?N)` reference label for a row (a broken ref renders `(?N)`, the row is yellow).
    func referenceLabel(for rowID: UUID) -> String? {
        guard let row = row(withID: rowID), !row.refs.isEmpty else { return nil }
        let parts = row.refs.map { ref in
            switch ref {
            case .rowRef(let id):
                if let number = displayNumber(of: id) { return number }
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
        do {
            let parsed = try CatParser.parseForValidation(catText)
            issues = FlowValidator.checkFlow(parsed, workspace: workspace, flowID: flowID)
                .filter { $0.code != "E104" }
        } catch {
            issues = [FlowIssue(row: "run", code: "E100",
                                message: "This flow doesn't parse as a valid .cat right now — fix or undo the last edit, then save.")]
        }
        for row in allRows() {
            for ref in row.refs {
                guard case .rowRef(let targetID) = ref else { continue }
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
        let isSharedWorkspaceFolder = workspace.root == ModelStore.shared.workspacesDirectory
        let siblings = (try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
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
    private nonisolated static func isCompatible(from target: Row, to row: Row) -> Bool {
        guard let accepts = signature(of: row)?.accepts,
              let gives = signature(of: target)?.gives else { return false }
        let given: Shape = gives == .sameAsInput ? .anyKind : gives
        let refKind = row.task.flatMap { TaskCatalog.get($0)?.refKind }
        return Shape.bundleCompatible(accepts: accepts, given: [given], rk: refKind)
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
                                        claimableModelIDs: claimableModelIDs).contains { entry in
            TaskModels.displayName(for: entry) == display || entry.hfModelId == display
        }
    }

    /// A row's (accepts, gives), mirroring `Shape.signature`'s block inference — the same
    /// resolution the interpreter and auto-chain use. (A block's *declared* signature is the
    /// validator's concern — `structuralIssues()` runs `checkFlow`, which compares it to the
    /// resolved type and flags a mismatch. Declared signatures do NOT change auto-chain.)
    private nonisolated static func signature(of row: Row) -> (accepts: Shape, gives: Shape)? {
        if let blockKind = row.blockKind {
            return inferredBlockSignature(row, blockKind: blockKind)
        }
        guard let task = row.task, let desc = TaskCatalog.get(task) else { return nil }
        return (desc.accepts, desc.gives)
    }

    private nonisolated static func inferredBlockSignature(_ row: Row, blockKind: BlockKind) -> (accepts: Shape, gives: Shape)? {
        guard let first = row.children.first, let last = row.children.last,
              let firstSig = signature(of: first), let lastSig = signature(of: last) else {
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
