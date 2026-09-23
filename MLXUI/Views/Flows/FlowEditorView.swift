import SwiftUI

/// CFM-R8 — the flow editor: assemble a flow from the step picker, add/remove/reorder rows,
/// undo/redo by value snapshot, save as `.cat`, and run — all without touching text.
///
/// - Empty flow → a big Add that shows the **starting nodes** (answer `c8`).
/// - Add opens the **step picker** filtered by the selected row's output shape, else the
///   last row's (answer `c4`); "Add from Full Catalog" is the escape hatch (answer `l`).
/// - Add inserts **below** the selected row (answer `c7`); a row that can't take what's
///   upstream turns **yellow** with a one-line warning (answers `l`, `h5`).
/// - Remove is swipe-native + a menu; reorder is drag (`.onMove`, scoped to the row's own
///   level) + Move up/down in the menu (answers `c10`, `c12`). A deleted row's references
///   render `(?N)` and stay yellow — never silently re-aimed (answers `c5`, `c9`); there is
///   no "fix for me" (answer `h3`).
/// - Undo/redo are whole-document value snapshots (answer `c11`).
/// - Save writes the canonical `.cat` into the flow's folder (answer `e`); it refuses while
///   any reference is broken. Run works like every gallery flow (preflight → run dots).
struct FlowEditorView: View {
    @Environment(AppState.self) private var appState
    @State private var model: FlowEditorModel
    @State private var showPicker = false
    @State private var showFullCatalog = false
    @State private var showInstallSheet = false
    @State private var session = FlowRunSession()
    /// FH-7: owned here (not `FlowInspectorPane`'s local state) so a run's completion can force
    /// it to `.output` from `run()`'s completion handler below.
    @State private var inspectorTab: FlowInspectorTab = .step
    /// FIP-2 — a `Read *` row whose file isn't there yet, or nil. Warn only (owner ruling):
    /// renders as a banner, never disables Run.
    @State private var inputAdvisory: FlowPreflight.RowAdvisory?

    /// CFM-R17-3: non-nil when the flow being edited lives inside a workspace — the model
    /// then saves into and resolves against the shared workspace directory.
    private let workspaceRef: WorkspaceRef?

    init(flowID: String = UUID().uuidString, name: String = "Untitled Flow",
         document: FlowDocument? = nil, savedText: String? = nil,
         workspace: WorkspaceRef? = nil, fileURL: URL? = nil) {
        self.workspaceRef = workspace
        // KW-1-1: a workspace flow opened from disk (a `WorkspaceRef` naming an existing
        // document) already has a file at `workspace.fileURL` — seed `savedURL` so a rename's
        // stale-sibling guard recognizes that file as this editor's own and clears it, instead
        // of leaving it behind as an untracked duplicate. KW-1-FIX-3: the decision itself is
        // `FlowEditorModel.seedSavedURL`, tested on its own. FH-6: the plain-`flows/`
        // equivalent — `fileURL` is the caller's own on-disk URL for a non-workspace flow that
        // already exists (opening it from My Workflows, or a just-written Duplicate & Edit /
        // Edit-opened-copy) — falls back only when there's no workspace ref to seed from.
        _model = State(initialValue: FlowEditorModel(
            name: workspace?.flowStem ?? name,
            flowID: workspace?.workspaceID ?? flowID,
            document: document,
            workspace: workspace?.workspace ?? .shared,
            savedText: savedText,
            savedURL: FlowEditorModel.seedSavedURL(document: document, workspace: workspace, fileURL: fileURL)))
    }

    /// The run scope for this flow — workspace-rooted when it lives in one (CFM-R17-1).
    private func makeScope() -> FlowScope {
        workspaceRef?.scope(text: model.catText) ?? .plain(model.flowID)
    }

    /// The flow list *is* the file: canonical lines, with a deleted row's references
    /// rendered `(?N)` from the editor's tombstones.
    private var serialized: (lines: [String], lineRanges: [UUID: Range<Int>], clauseRanges: [UUID: Range<Int>]) {
        CatSerializer.serializeLines(model.document, deadRefNumbers: model.tombstones)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
            // FH-1: the save/seed error, moved out of the header's fixed-width toolbar row
            // (where it was the only unbounded child) into a banner here — same shape as the
            // input advisory just below, red instead of blue. Shown first when both are
            // present, since a save-blocking error outranks a non-blocking setup note.
            if let notice = model.saveError ?? model.seedError {
                HStack(alignment: .top, spacing: 8) {
                    Label(notice, systemImage: "exclamationmark.circle")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
            // FIP-2 — same non-blocking shape `FlowListView`'s setup advisory renders: names
            // the row and the missing file, never disables Run below.
            if let advisory = inputAdvisory {
                HStack(alignment: .top, spacing: 8) {
                    Label(advisory.reason, systemImage: "exclamationmark.circle")
                        .font(.callout)
                        .foregroundStyle(.blue)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
            if model.document.rows.isEmpty {
                emptyState
            } else {
                rowList
                // CFM-R11-1: same per-run peak-memory record as the gallery detail, so a
                // cold run in the editor is measurable too.
                if !session.isRunning && !session.metrics.rowSamples.isEmpty {
                    Text(session.metrics.summary())
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 6)
                }
            }
        }
        .frame(minWidth: 620, minHeight: 420)
        .navigationTitle(model.name.isEmpty ? "Untitled Flow" : model.name)
        // CFM-R12-3 item 5: removing a block header asks — never delete N rows on one
        // keystroke without saying so.
        .confirmationDialog("Remove this block?", isPresented: Binding(
            get: { blockPendingRemoval != nil },
            set: { if !$0 { blockPendingRemoval = nil } }
        ), titleVisibility: .visible) {
            Button("Remove all", role: .destructive) {
                if let id = blockPendingRemoval { model.remove(id) }
                blockPendingRemoval = nil
            }
            Button("Keep the steps") {
                if let id = blockPendingRemoval { model.unwrapBlock(id) }
                blockPendingRemoval = nil
            }
            Button("Cancel", role: .cancel) { blockPendingRemoval = nil }
        } message: {
            if let id = blockPendingRemoval {
                Text("Remove '\(model.row(withID: id)?.blockName ?? "block")' and its \(model.childCount(of: id)) steps, or keep the steps by unwrapping them to its place?")
            } else {
                Text("")
            }
        }
        .sheet(isPresented: $showPicker) {
            FlowStepPickerView(steps: pickerSteps,
                               showFullCatalog: $showFullCatalog,
                               onPick: { addStep($0) },
                               onCancel: { showPicker = false },
                               catalog: appState.browserData?.domains.flatMap { $0.allModels } ?? [],
                               claimableModelIDs: appState.claimableModelIDs)
        }
        .sheet(isPresented: $showInstallSheet) {
            if let result = session.preflight {
                installSheet(result)
            }
        }
        // CFM-R10-Human: a `wait=forever` human row parked the run — ask the person.
        // Q2 (`RSI/DelegateWorkspaceRunBacklog.md`, owner ruling 2026-09-22): dismissing this
        // sheet — Esc, click outside, or the Stop button inside it — is one operation, not
        // two. All three now route through `cancel()`.
        .sheet(isPresented: Binding(
            get: { session.parked != nil },
            set: { if !$0 { session.cancel() } }
        )) {
            if let parked = session.parked, let row = model.row(withID: parked.rowID) {
                FlowHumanPromptView(
                    parked: parked,
                    row: row,
                    doc: model.document,
                    session: session,
                    runner: FlowRunner(),
                    context: AppFlowExecutorFactory.cachingContext(scope: makeScope(), appState: appState, transforms: model.document.transforms))
            }
        }
        // Editing invalidates run results; preflight follows the document.
        .onChange(of: model.document) { _, _ in
            session.clearRun(doc: model.document)
            prepareInstall()
        }
        // FH-7: a run that reaches the end un-cancelled, un-parked, and without a hard failure
        // — select the last row and show its Output tab (the convenience of "go look at what
        // just ran" without a manual click).
        .onChange(of: session.completedRunToken) { _, _ in
            if let last = model.document.rows.last {
                model.selectedRowID = last.id
                inspectorTab = .output
            }
        }
        // CFM-R14-2: seed defaults from the derived pool, not a hand table.
        .onAppear {
            model.modelCatalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
            model.claimableModelIDs = appState.claimableModelIDs
            prepareInstall()
        }
    }

    // MARK: - Header (FH-5: three rungs, widest that fits — never a second row)

    /// `ViewThatFits` measures each rung's ideal size and keeps the first that fits — wide,
    /// then medium, then narrow, in priority order. Not a second row: that would spend
    /// vertical space permanently for a problem that only exists at small widths, and would
    /// move controls on every resize (rejected in the backlog, recorded there so it isn't
    /// re-proposed).
    private var header: some View {
        ViewThatFits(in: .horizontal) {
            wideHeader
            mediumHeader
            narrowHeader
        }
        .padding(12)
    }

    // FH-3: identity (name, header keyword/version, extension) moved to the Flow tab —
    // renaming happens there now. What's left here is just enough to say which flow this is,
    // truncating rather than claiming a fixed width the way the old `TextField` did.
    private var flowTitle: some View {
        HStack(spacing: 4) {
            if model.isDirty {
                Text("•")
                    .font(.headline)
                    .foregroundStyle(.secondary)
            }
            Text(model.name)
                .font(.headline)
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }

    private var undoButton: some View {
        Button {
            model.undo()
        } label: {
            Label("Undo", systemImage: "arrow.uturn.backward")
        }
        .disabled(model.undoStack.isEmpty)
        .keyboardShortcut("z", modifiers: .command)
        .help("Undo the last edit")
    }

    private var redoButton: some View {
        Button {
            model.redo()
        } label: {
            Label("Redo", systemImage: "arrow.uturn.forward")
        }
        .disabled(model.redoStack.isEmpty)
        .keyboardShortcut("z", modifiers: [.command, .shift])
        .help("Redo the last undo")
    }

    /// The dynamic Add-step label (`addStepLabel`) can read "Add step inside {block name}" —
    /// unbounded, since a block name is free text. Reserving width off a hidden, generously
    /// long placeholder (a `ZStack`, not a `.frame` on the visible label — a `.background`
    /// wouldn't grow the parent) keeps the wide rung's ideal size stable across selection
    /// changes for any realistically-named block, so `ViewThatFits` doesn't flip rungs purely
    /// because the user selected a different row — only an extraordinarily long block name
    /// could still do that, an accepted edge case rather than an unbounded reservation.
    private var addStepButtonLabel: some View {
        ZStack(alignment: .leading) {
            Label("Add step inside a reasonably long block name", systemImage: "plus")
                .lineLimit(1)
                .hidden()
            Label(addStepLabel, systemImage: "plus")
                .lineLimit(1)
        }
    }

    // CFM-R12-2: Add/Remove in the toolbar — the visible path to building and pruning a flow
    // without ever opening a context menu. Add targets below the selected row (or the end
    // when nothing is selected), the same call the context menu makes.
    private var addStepButton: some View {
        Button {
            showFullCatalog = false
            showPicker = true
        } label: {
            addStepButtonLabel
        }
        .keyboardShortcut(.return, modifiers: .command)
        .help(addStepHelp)
    }

    private var removeButton: some View {
        Button {
            if let selected = model.selectedRowID {
                removeRow(model.row(withID: selected) ?? Row(id: selected, task: nil))
            }
        } label: {
            Label("Remove", systemImage: "minus")
        }
        .disabled(model.selectedRowID == nil)
        .keyboardShortcut(.delete, modifiers: [])
        .help(model.selectedRowID == nil ? "Select a row to remove it" : "Remove the selected row")
    }

    /// Run never loses its word, at any width — the one button the backlog names explicitly.
    /// A flow opened here needing a download (a My Workflows flow, most often one Duplicate &
    /// Edit copied, or one whose row Properties just picked an "Available to download" model)
    /// used to leave Run disabled with no way to reach the install sheet — `run()` only opens
    /// it from inside the body a disabled button can't call. Install Required Models fills the
    /// same slot instead, mirroring `FlowListView`'s header exactly.
    @ViewBuilder
    private var runOrCancelButton: some View {
        if session.isRunning {
            Button {
                session.cancel()
            } label: {
                Label("Cancel", systemImage: "stop.fill")
            }
        } else if session.isInstalling {
            Button {} label: {
                Label {
                    Text("Installing Models")
                } icon: {
                    ProgressView()
                        .controlSize(.small)
                }
            }
            .disabled(true)
        } else if session.needsInstall {
            Button {
                showInstallSheet = true
            } label: {
                Label("Install Required Models", systemImage: "arrow.down.circle")
            }
        } else {
            Button {
                run()
            } label: {
                Label("Run", systemImage: "play.fill")
            }
            .buttonStyle(.borderedProminent)
            .disabled(!canRun)
            .help(runDisabledHelp)
        }
    }

    private var saveButton: some View {
        Button {
            save()
        } label: {
            Label("Save", systemImage: "square.and.arrow.down")
        }
        .disabled(!model.canSave)
        .keyboardShortcut("s", modifiers: .command)
        .help(model.saveBlockReason ?? "Save the flow as a .cat file")
    }

    // CACHE-Q: the same ⋯ menu the gallery detail carries — a working My Workflows flow opens
    // here, and DA-5's determinism check / DA-10's skip test need a way to force a cold
    // re-run of an *unchanged* flow. FH-2: Reveal in Finder joins it too. At the narrow rung
    // (FH-5), Save folds in alongside Reveal — a file-system-adjacent action, not one of the
    // structural edits the narrow rung's Edit menu groups.
    private func maintenanceMenu(includeSave: Bool) -> some View {
        FlowMaintenanceMenu(session: session, doc: model.document,
                            workspace: model.workspace, flowID: model.flowID) {
            Divider()
            Button("Reveal in Finder", systemImage: "folder") {
                reveal()
            }
            .disabled(model.savedURL == nil)
            if includeSave {
                saveButton
            }
        }
    }

    private var wideHeader: some View {
        HStack(spacing: 10) {
            flowTitle
            Spacer()
            undoButton
            redoButton
            Divider().frame(height: 20)
            addStepButton
            removeButton
            runOrCancelButton
            saveButton
            maintenanceMenu(includeSave: false)
        }
    }

    /// Icons only, except Run — the toolbar's second-widest rung.
    private var mediumHeader: some View {
        HStack(spacing: 10) {
            flowTitle
            Spacer()
            undoButton.labelStyle(.iconOnly)
            redoButton.labelStyle(.iconOnly)
            Divider().frame(height: 20)
            addStepButton.labelStyle(.iconOnly)
            removeButton.labelStyle(.iconOnly)
            runOrCancelButton
            saveButton.labelStyle(.iconOnly)
            maintenanceMenu(includeSave: false)
        }
    }

    /// Undo/Redo/Add/Remove fold into one "Edit" menu; Run keeps its word; Save joins the ⋯
    /// menu. Each button keeps its own `.help()`/`.keyboardShortcut` inside the menu — a
    /// SwiftUI keyboard shortcut fires wherever its view sits in the active hierarchy, menu
    /// item or not.
    private var narrowHeader: some View {
        HStack(spacing: 10) {
            flowTitle
            Spacer()
            Menu("Edit") {
                undoButton
                redoButton
                Divider()
                addStepButton
                removeButton
            }
            runOrCancelButton
            maintenanceMenu(includeSave: true)
        }
    }

    /// CFM-R8-FIX-4: Run is gated on the same integrity check as Save — a document Save
    /// refuses must not run (and the interpreter's errors must surface, never vanish).
    private var canRun: Bool {
        guard model.savedURL != nil else { return false }
        guard model.canSave else { return false }
        return session.canRun
    }

    private var runDisabledHelp: String {
        if model.savedURL == nil { return "Save the flow first." }
        if let reason = model.saveBlockReason { return reason }
        if session.isInstalling { return "Installing the required models…" }
        if case .notRunnable(let reason)? = session.runnability { return reason }
        return session.runDisabledReason ?? "Run the flow"
    }

    // MARK: - Empty state (answer c8)

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "flowchart")
                .font(.system(size: 44))
                .foregroundStyle(.secondary)
            Text("Start with a step")
                .font(.title3.weight(.semibold))
            Text("Pick something that produces output with no input — a file, a folder,\na prompt, a template, or a trigger. Then keep adding steps that fit what came before.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Add a step…") {
                showFullCatalog = false
                showPicker = true
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Row list (CFM-R12-3: one entry per row, children included)

    private var rowList: some View {
        HStack(spacing: 0) {
            List {
                ForEach(visibleRows) { display in
                    editorRow(display.row, depth: display.depth)
                }
                // CFM-R12-FIX-4: drag reorder is back, scoped to the row's own level — a
                // child moves among its siblings (moveInside), a top-level row among
                // top-level rows (move).
                .onMove { source, destination in
                    handleMove(from: source, to: destination)
                }
            }
            .listStyle(.plain)
            if model.selectedRowID != nil {
                Divider()
                FlowInspectorPane(
                    flowInfo: flowTabInfo,
                    output: selectedRowOutput,
                    rowTitle: selectedRowTitle,
                    substitutionNote: selectedSubstitutionNote,
                    savedFile: selectedSavedFile,
                    savedKind: selectedSavedKind,
                    savedPresentation: selectedSavedPresentation,
                    tab: $inspectorTab
                ) {
                    FlowRowInspectorView(model: model,
                                         rowID: model.selectedRowID ?? UUID(),
                                         catalog: appState.browserData?.domains.flatMap { $0.allModels } ?? [],
                                         totalRAMGB: appState.systemInfo.totalRAMGB,
                                         installedModelIDs: appState.installedModelIDs,
                                         claimableModelIDs: appState.claimableModelIDs,
                                         resolvePromptSupport: { appState.registry.bestModule(for: $0)?.sdk.promptSupport ?? .none },
                                         editable: true)
                }
            }
        }
    }

    /// CFM-R12-FIX-4: map a flattened-list drag onto the model's scope-aware move. The
    /// destination offset counts same-scope rows before the drop point (the convention
    /// `Array.move(fromOffsets:toOffset:)` expects in the pre-move list).
    private func handleMove(from source: IndexSet, to destination: Int) {
        let rows = visibleRows
        guard let from = source.first, source.count == 1 else { return }
        let rowID = rows[from].row.id
        if let (blockID, childIndex) = model.childIndexOf(rowID) {
            let beforeDest = rows[..<max(destination, 0)].filter {
                model.childIndexOf($0.row.id)?.blockID == blockID
            }.count
            model.moveInside(blockID: blockID, from: [childIndex], to: beforeDest)
        } else if let topIndex = model.document.rows.firstIndex(where: { $0.id == rowID }) {
            let beforeDest = rows[..<max(destination, 0)].filter {
                model.childIndexOf($0.row.id) == nil
            }.count
            model.move(from: [topIndex], to: beforeDest)
        }
    }

    // MARK: - Inspector Output tab data (the editor can run, then inspect row outputs)

    /// The selected row's last finished output — the Output tab's content source.
    private var selectedRowOutput: Asset? {
        model.selectedRowID.flatMap { session.outputs[$0] }
    }

    private var selectedRowTitle: String {
        guard let id = model.selectedRowID, let row = model.row(withID: id) else { return "" }
        return FlowRowSummary.taskName(for: row)
    }

    /// FH-3: the Flow tab's identity, bound live to the model — `name` is `$model.name`
    /// itself, so a rename typed there reaches `FlowEditorModel.save()`'s stale-sibling guard
    /// exactly as the old toolbar `TextField` did (§7 trap 8).
    private var flowTabInfo: FlowTabInfo {
        FlowTabInfo(name: $model.name,
                    headerKeyword: model.document.headerKeyword,
                    version: model.document.version,
                    fileExtension: FlowEditorModel.fileExtension(for: model.document.fileKind),
                    savedURL: model.savedURL,
                    rowCount: model.document.rows.count,
                    document: model.document,
                    workspace: model.workspace,
                    flowID: model.flowID,
                    toggleFlag: { flag, declared in
                        if declared {
                            model.applyHeaderRepair(flag)
                        } else {
                            model.removeHeaderFlag(flag)
                        }
                    },
                    editable: true)
    }

    private var selectedSubstitutionNote: String? {
        model.selectedRowID.flatMap { session.substitutionNotes[$0] }
    }

    /// The file a selected `Save *` row wrote, resolved against the flow's folder.
    private var selectedSavedFile: URL? {
        guard let id = model.selectedRowID, let row = model.row(withID: id) else { return nil }
        return FlowSavedFile.resolved(row: row, flowID: model.flowID, workspace: model.workspace)
    }

    /// The saved file's kind, driving how the Output tab presents it.
    private var selectedSavedKind: Kind? {
        guard let id = model.selectedRowID, let row = model.row(withID: id) else { return nil }
        return FlowSavedFile.kind(forTask: row.task)
    }

    /// OV-1: the saved file's presentation (by extension, task as fallback) — drives whether
    /// the Output tab offers a Quick Look button (OV-2).
    private var selectedSavedPresentation: SavedFilePresentation? {
        guard let id = model.selectedRowID, let row = model.row(withID: id),
              let url = selectedSavedFile else { return nil }
        return FlowSavedFile.presentation(url: url, task: row.task)
    }

    /// R12-3: every row flattened with depth; a collapsed block hides its children.
    private var visibleRows: [FlowRowFlatten.Entry] {
        FlowRowFlatten.flatten(model.document.rows, collapsed: collapsedBlocks)
    }

    @State private var collapsedBlocks: Set<UUID> = []
    /// CFM-R12-3 item 5: the block header awaiting a Remove all / Keep the steps decision.
    @State private var blockPendingRemoval: UUID?

    private func editorRow(_ row: Row, depth: Int) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .top, spacing: 8) {
                if row.blockKind != nil {
                    Button {
                        if collapsedBlocks.contains(row.id) {
                            collapsedBlocks.remove(row.id)
                        } else {
                            collapsedBlocks.insert(row.id)
                        }
                    } label: {
                        Image(systemName: collapsedBlocks.contains(row.id) ? "chevron.right" : "chevron.down")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .frame(width: 12)
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 5)
                }
                FlowStatusDot(status: status(for: row))
                    .padding(.top, 4)
                if let range = serialized.lineRanges[row.id] {
                    ScrollView(.horizontal, showsIndicators: false) {
                        // QR12R2-1: a block's clause line is drawn after its header.
                        let clause = serialized.clauseRanges[row.id].map { serialized.lines[$0] } ?? []
                        Text((Array(serialized.lines[range]) + clause).joined(separator: "\n"))
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                    }
                    .onTapGesture { model.selectedRowID = row.id }
                }
            }
            .padding(.vertical, 3)
            .padding(.leading, CGFloat(depth) * 20)
            .contextMenu { contextMenu(for: row) }
            .swipeActions(edge: .trailing) {
                Button(role: .destructive) {
                    removeRow(row)
                } label: {
                    Label("Remove", systemImage: "trash")
                }
            }
            // CFM-R12-10: a `<parallel>` block says plainly that it runs one chain at a time.
            if row.blockKind == .parallel {
                Text("semantic parallel — runs one chain at a time, not faster")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.leading, 28 + CGFloat(depth) * 20)
            }
            if let warning = model.warning(for: row.id) {
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.leading, 28 + CGFloat(depth) * 20)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .listRowBackground(rowBackground(for: row))
    }

    /// R12-3 item 5: removing a block header asks — Remove all / Keep the steps / Cancel.
    private func removeRow(_ row: Row) {
        if row.blockKind != nil {
            blockPendingRemoval = row.id
        } else {
            model.remove(row.id)
        }
    }

    @ViewBuilder
    private func contextMenu(for row: Row) -> some View {
        Button("Add step below…") {
            model.selectedRowID = row.id
            showFullCatalog = false
            showPicker = true
        }
        if row.blockKind != nil {
            Button("Add step inside…") {
                model.selectedRowID = row.children.first?.id ?? row.id
                showFullCatalog = false
                showPicker = true
            }
            Menu("Wrap…") {
                Button("<each>") { model.wrapInBlock(row.id, kind: .each, name: "each_group") }
                Button("<parallel>") { model.wrapInBlock(row.id, kind: .parallel, name: "parallel_group") }
                Button("<list>") { model.wrapInBlock(row.id, kind: .list, name: "list_group") }
            }
        }
        // HR-5 (Q3 ruled (c), 2026-09-20): `Human Input` only — `Ask Human`'s `default=` names
        // a tag, not a row-above value, so the pattern this inserts has nothing to give it.
        if row.task == "Human Input" {
            Button("Give this row a default value…") {
                model.addDefaultValueRow(above: row.id)
            }
        }
        Menu("Block…") {
            Button("<each>") { model.insertBlock(kind: .each, name: "each_group", after: row.id) }
            Button("<parallel>") { model.insertBlock(kind: .parallel, name: "parallel_group", after: row.id) }
            Button("<list>") { model.insertBlock(kind: .list, name: "list_group", after: row.id) }
        }
        // CFM-R12-3 item 6: reorder stays inside its scope — top-level via `move`, a block
        // child among its siblings via `moveInside`.
        moveMenu(for: row)
        // INPUTS-1: one submenu per bound ref, plus one more open, empty slot when the
        // task's shape allows another (`canAddInput`) — picking from that last one binds
        // it and the next menu-open offers a fresh one, so the context menu never needs
        // its own persisted "+" state the way the Properties panel does.
        let slots = model.slotsToDraw(for: row.id) + (model.canAddInput(for: row.id) ? 1 : 0)
        ForEach(1...slots, id: \.self) { slot in
            inputMenu(for: row, slot: slot)
        }
        if !row.refs.isEmpty {
            Button("Clear input") {
                model.setReference(to: nil, slot: 1, for: row.id)
            }
        }
        Button(row.chainBreak ? "Clear chain break" : "Set chain break") {
            model.setChainBreak(!row.chainBreak, for: row.id)
        }
        Button("Duplicate") {
            model.duplicate(row.id)
        }
        Button("Remove", role: .destructive) {
            removeRow(row)
        }
    }

    /// R12-3 item 6: Move up/down within the row's own scope (top-level or a block's
    /// children) — dragging a child out of a block, or a row into one, is out of scope.
    @ViewBuilder
    private func moveMenu(for row: Row) -> some View {
        if let (blockID, childIndex) = model.childIndexOf(row.id) {
            Menu("Move") {
                if childIndex > 0 {
                    Button("Move up") {
                        model.moveInside(blockID: blockID, from: [childIndex], to: childIndex - 1)
                    }
                }
                if childIndex < model.childCount(of: blockID) - 1 {
                    Button("Move down") {
                        model.moveInside(blockID: blockID, from: [childIndex], to: childIndex + 2)
                    }
                }
            }
        } else if let index = model.document.rows.firstIndex(where: { $0.id == row.id }) {
            Menu("Move") {
                if index > 0 {
                    Button("Move up") {
                        model.move(from: [index], to: index - 1)
                    }
                }
                if index < model.document.rows.count - 1 {
                    Button("Move down") {
                        model.move(from: [index], to: index + 2)
                    }
                }
            }
        }
    }

    /// One input slot's "Choose input…" menu — `Diff` shows Input 1 and Input 2 (FIX-2).
    @ViewBuilder
    private func inputMenu(for row: Row, slot: Int) -> some View {
        Menu(slotLabel(for: row, slot: slot)) {
            let inputs = model.validInputs(for: row.id, slot: slot)
            if inputs.isEmpty {
                Button("No compatible rows") {}
                    .disabled(true)
            } else {
                ForEach(inputs, id: \.rowID) { input in
                    Button("\(input.number) — \(input.task)") {
                        model.setReference(to: input.rowID, slot: slot, for: row.id)
                    }
                }
            }
        }
    }

    private func slotLabel(for row: Row, slot: Int) -> String {
        let count = model.slotsToDraw(for: row.id) + (model.canAddInput(for: row.id) ? 1 : 0)
        return count > 1 ? "Input \(slot)…" : "Choose input…"
    }

    private func rowBackground(for row: Row) -> Color {
        if model.selectedRowID == row.id {
            return Color.accentColor.opacity(0.12)
        }
        if model.warning(for: row.id) != nil {
            return Color.orange.opacity(0.07)
        }
        return Color.clear
    }

    /// The row's dot: the run's own status once a run has happened, else the editing
    /// state — yellow `△` for a row that needs attention (answer c5/c9/l).
    /// R12-3 item 4: the Add button names where it will insert — inside the enclosing block,
    /// below the selected row, or at the end.
    private var addStepLabel: String {
        if let id = model.selectedRowID, let name = model.enclosingBlockName(for: id) {
            return "Add step inside \(name)"
        }
        if let id = model.selectedRowID {
            return model.row(withID: id)?.blockKind != nil ? "Add step inside" : "Add step below"
        }
        return "Add step"
    }

    private var addStepHelp: String {
        if model.selectedRowID == nil { return "Add a step at the end" }
        if model.enclosingBlockName(for: model.selectedRowID!) != nil { return "Add a step inside the selected block" }
        return "Add a step below the selected row"
    }

    private func status(for row: Row) -> FlowStatus {
        let run = session.status(for: row.id)
        if run != .notRun { return run }
        return model.warning(for: row.id) != nil ? .needsAttention : .notRun
    }

    // MARK: - Step picker (CFM-R8-1)

    /// The picker's task list: full catalog when that toggle is on (answer `l`); starting
    /// nodes on an empty flow (answer `c8`); else the tasks that can accept the selected
    /// row's output — the last row's when nothing is selected (answer `c4`).
    private var pickerSteps: [TaskDescriptor] {
        if showFullCatalog {
            return FlowEditorModel.stepsAccepting(filterOutput, includeAdvanced: true)
        }
        if model.document.rows.isEmpty {
            return FlowEditorModel.startingNodes()
        }
        return FlowEditorModel.stepsAccepting(filterOutput)
    }

    private var filterOutput: Shape? {
        if let id = model.selectedRowID, model.row(withID: id) != nil {
            return model.outputShape(for: id)
        }
        if let last = model.document.rows.last {
            return model.outputShape(for: last.id)
        }
        return nil
    }

    private func addStep(_ name: String) {
        model.add(task: name)
        showPicker = false
    }

    // MARK: - Save (CFM-R8-5)

    private func save() {
        do {
            try model.save()
            model.saveError = nil
        } catch {
            model.saveError = (error as CustomStringConvertible).description
        }
        prepareInstall()
    }

    private func reveal() {
        guard let url = model.savedURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    // MARK: - Run (reuses the gallery's session + preflight machinery)

    private func run() {
        // FIX-4: never run a document the save gate refuses (the button is disabled, this
        // is defense in depth).
        guard model.canSave else {
            model.saveError = model.saveBlockReason
            return
        }
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        let result = FlowPreflight.run(model.document, catalog: catalog,
                                       installedModelIDs: appState.installedModelIDs,
                                       totalRAMGB: appState.systemInfo.totalRAMGB,
                                       claimableModelIDs: appState.claimableModelIDs)
        session.prepareInstall(result, doc: model.document, scope: workspaceRef != nil ? makeScope() : nil)
        guard !result.toDownload.isEmpty else {
            startRun()
            return
        }
        showInstallSheet = true
    }

    private func startRun() {
        let context = AppFlowExecutorFactory.cachingContext(scope: makeScope(), appState: appState, transforms: model.document.transforms)
        session.start(doc: model.document, runner: FlowRunner(), context: context,
                      resume: session.hasRunResults)
    }

    private func prepareInstall() {
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        session.prepareInstall(FlowPreflight.run(model.document, catalog: catalog,
                                                 installedModelIDs: appState.installedModelIDs,
                                                 totalRAMGB: appState.systemInfo.totalRAMGB,
                                                 claimableModelIDs: appState.claimableModelIDs),
                               doc: model.document, scope: workspaceRef != nil ? makeScope() : nil)
        // FIP-2: `makeScope()` already falls back to `.plain(model.flowID)` for a non-workspace
        // flow, so this is safe to call unconditionally (unlike the line above, which only
        // needs a scope at all for a workspace flow's `uses:` resolution).
        inputAdvisory = FlowInputAdvisory.advisory(for: model.document, scope: makeScope())
    }

    private func installSheet(_ result: FlowPreflight.Result) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Install models to run this flow", systemImage: "arrow.down.circle")
                .font(.headline)
            VStack(alignment: .leading, spacing: 6) {
                ForEach(result.downloadSet, id: \.id) { model in
                    HStack {
                        Text(model.displayName)
                        Spacer()
                        Text(String(format: "%.1f GB", model.downloadSizeGB))
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))
            HStack {
                Spacer()
                Button("Cancel") { showInstallSheet = false }
                    .keyboardShortcut(.cancelAction)
                Button("Install") {
                    showInstallSheet = false
                    installModels(Array(result.downloadSet))
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    private func installModels(_ models: [ModelEntry]) {
        session.isInstalling = true
        Task {
            for model in models {
                appState.installModel(model)
                _ = await InstallPoller.awaitInstalled(model: model,
                                                       installManager: appState.installManager)
            }
            session.isInstalling = false
            prepareInstall()
        }
    }
}
