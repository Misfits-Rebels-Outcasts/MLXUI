import SwiftUI

/// CFM-R8 — the flow editor: assemble a flow from the step picker, add/remove/reorder rows,
/// undo/redo by value snapshot, save as `.cat`, and run — all without touching text.
///
/// - Empty flow → a big Add that shows the **starting nodes** (answer `c8`).
/// - Add opens the **step picker** filtered by the selected row's output shape, else the
///   last row's (answer `c4`); "Add from Full Catalog" is the escape hatch (answer `l`).
/// - Add inserts **below** the selected row (answer `c7`); a row that can't take what's
///   upstream turns **yellow** with a one-line warning (answers `l`, `h5`).
/// - Remove is swipe-native + a menu; reorder is `.onMove`, always allowed (answers `c10`,
///   `c12`). A deleted row's references render `(?N)` and stay yellow — never silently
///   re-aimed (answers `c5`, `c9`); there is no "fix for me" (answer `h3`).
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

    init(flowID: String = UUID().uuidString, name: String = "Untitled Flow",
         document: FlowDocument? = nil, savedText: String? = nil) {
        _model = State(initialValue: FlowEditorModel(name: name, flowID: flowID,
                                                     document: document, savedText: savedText))
    }

    /// The flow list *is* the file: canonical lines, with a deleted row's references
    /// rendered `(?N)` from the editor's tombstones.
    private var serialized: (lines: [String], lineRanges: [UUID: Range<Int>]) {
        CatSerializer.serializeLines(model.document, deadRefNumbers: model.tombstones)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            header
            Divider()
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
        .sheet(isPresented: $showPicker) {
            FlowStepPickerView(steps: pickerSteps,
                               showFullCatalog: $showFullCatalog,
                               onPick: { addStep($0) },
                               onCancel: { showPicker = false })
        }
        .sheet(isPresented: $showInstallSheet) {
            if let result = session.preflight {
                installSheet(result)
            }
        }
        // CFM-R10-Human: a `wait=forever` human row parked the run — ask the person.
        .sheet(isPresented: Binding(
            get: { session.parked != nil },
            set: { if !$0 { session.clearParked() } }
        )) {
            if let parked = session.parked, let row = model.row(withID: parked.rowID) {
                FlowHumanPromptView(
                    parked: parked,
                    row: row,
                    doc: model.document,
                    session: session,
                    runner: FlowRunner(),
                    context: AppFlowExecutorFactory.cachingContext(flowID: model.flowID, appState: appState, transforms: model.document.transforms))
            }
        }
        // Editing invalidates run results; preflight follows the document.
        .onChange(of: model.document) { _, _ in
            session.clearRun(doc: model.document)
            prepareInstall()
        }
        .onAppear(perform: prepareInstall)
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 10) {
            HStack(spacing: 4) {
                if model.isDirty {
                    Text("•")
                        .font(.headline)
                        .foregroundStyle(.secondary)
                }
                TextField("Flow name", text: $model.name)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 200)
            }
            Text("catflow 0.8")
                .font(.caption.monospaced())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
            // CFM-R11-3: the saved file's kind — a `.catpipeline` opened in the editor saves
            // back as `.catpipeline` (a `.cat` as `.cat`).
            Text("." + FlowEditorModel.fileExtension(for: model.document.fileKind))
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            Spacer()
            if let notice = model.saveError ?? model.seedError {
                Text(notice)
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            Button {
                model.undo()
            } label: {
                Label("Undo", systemImage: "arrow.uturn.backward")
            }
            .disabled(model.undoStack.isEmpty)
            .keyboardShortcut("z", modifiers: .command)
            .help("Undo the last edit")
            Button {
                model.redo()
            } label: {
                Label("Redo", systemImage: "arrow.uturn.forward")
            }
            .disabled(model.redoStack.isEmpty)
            .keyboardShortcut("z", modifiers: [.command, .shift])
            .help("Redo the last undo")
            Divider().frame(height: 20)
            // CFM-R12-2: Add/Remove in the toolbar — the visible path to building and
            // pruning a flow without ever opening a context menu. Add targets below the
            // selected row (or the end when nothing is selected), the same call the context
            // menu makes.
            Button {
                showFullCatalog = false
                showPicker = true
            } label: {
                Label("Add step", systemImage: "plus")
            }
            .keyboardShortcut(.return, modifiers: .command)
            .help("Add a step below the selected row (or at the end)")
            Button {
                if let selected = model.selectedRowID {
                    model.remove(selected)
                }
            } label: {
                Label("Remove", systemImage: "minus")
            }
            .disabled(model.selectedRowID == nil)
            .keyboardShortcut(.delete, modifiers: [])
            .help(model.selectedRowID == nil ? "Select a row to remove it" : "Remove the selected row")
            if session.isRunning {
                Button {
                    session.cancel()
                } label: {
                    Label("Cancel", systemImage: "stop.fill")
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
            Button {
                save()
            } label: {
                Label("Save", systemImage: "square.and.arrow.down")
            }
            .disabled(!model.canSave)
            .keyboardShortcut("s", modifiers: .command)
            .help(model.saveBlockReason ?? "Save the flow as a .cat file")
            Button {
                reveal()
            } label: {
                Label("Reveal in Finder", systemImage: "folder")
            }
            .disabled(model.savedURL == nil)
        }
        .padding(12)
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

    // MARK: - Row list

    private var rowList: some View {
        HStack(spacing: 0) {
            List {
                ForEach(model.document.rows) { row in
                    editorRow(row)
                }
                .onMove { source, destination in
                    model.move(from: source, to: destination)
                }
            }
            .listStyle(.plain)
            if model.selectedRowID != nil {
                Divider()
                FlowRowInspectorView(model: model,
                                     rowID: model.selectedRowID ?? UUID(),
                                     catalog: appState.browserData?.domains.flatMap { $0.allModels } ?? [],
                                     totalRAMGB: appState.systemInfo.totalRAMGB)
            }
        }
    }

    private func editorRow(_ row: Row) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .top, spacing: 8) {
                FlowStatusDot(status: status(for: row))
                    .padding(.top, 4)
                if let range = serialized.lineRanges[row.id] {
                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(serialized.lines[range].joined(separator: "\n"))
                            .font(.system(.body, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .contentShape(Rectangle())
                    }
                    .onTapGesture { model.selectedRowID = row.id }
                }
            }
            .padding(.vertical, 3)
            .contextMenu { contextMenu(for: row) }
            .swipeActions(edge: .trailing) {
                Button(role: .destructive) {
                    model.remove(row.id)
                } label: {
                    Label("Remove", systemImage: "trash")
                }
            }
            if let warning = model.warning(for: row.id) {
                Text(warning)
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .padding(.leading, 28)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .listRowBackground(rowBackground(for: row))
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
        Menu("Block…") {
            Button("<each>") { model.insertBlock(kind: .each, name: "each_group", after: row.id) }
            Button("<parallel>") { model.insertBlock(kind: .parallel, name: "parallel_group", after: row.id) }
            Button("<list>") { model.insertBlock(kind: .list, name: "list_group", after: row.id) }
        }
        let slots = model.inputSlotCount(for: row.id)
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
            model.remove(row.id)
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
        let count = model.inputSlotCount(for: row.id)
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
            model.saveError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
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
                                       totalRAMGB: appState.systemInfo.totalRAMGB)
        session.prepareInstall(result, doc: model.document)
        guard !result.toDownload.isEmpty else {
            startRun()
            return
        }
        showInstallSheet = true
    }

    private func startRun() {
        let context = AppFlowExecutorFactory.cachingContext(flowID: model.flowID, appState: appState, transforms: model.document.transforms)
        session.start(doc: model.document, runner: FlowRunner(), context: context,
                      resume: session.hasRunResults)
    }

    private func prepareInstall() {
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        session.prepareInstall(FlowPreflight.run(model.document, catalog: catalog,
                                                 installedModelIDs: appState.installedModelIDs,
                                                 totalRAMGB: appState.systemInfo.totalRAMGB),
                               doc: model.document)
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
                _ = await InstallPoller.awaitInstalled(modelID: model.id,
                                                       installManager: appState.installManager)
            }
            session.isInstalling = false
            prepareInstall()
        }
    }
}
