import SwiftUI
import UniformTypeIdentifiers

/// CFM-R9 — the row inspector: edit a row's details without touching text. Bound to one
/// selected row of the flow editor.
///
/// - **Model picker** (R9-2): the task's pool of runnable models, RAM-sorted, the ones that
///   exceed this Mac's RAM dimmed; the raw id and the `CatalogBridge` substitution note sit
///   behind an Advanced disclosure (answer `a4`).
/// - **Instruction textbox** (R9-3): the quoted "What should it do?" text, multiline, with
///   per-task suggested defaults that fill the box (answer `a5`).
/// - **Input pickers** (R9-4): one per input slot, listing the actual rows whose output
///   matches — "2 Transcribe — text" — never bare numbers (answers `d2`, `d4`).
/// - **Settings** (R9-1): the row's existing `key=value` pairs as fields, spliced back
///   byte-preservingly (the QR9 round-trip); unknown keys addable from a short list.
/// - **Decisions & budget** (R9-5): `[tag | target]` pairs plus `max_visits` and `on_budget`
///   (answer `g1`).
struct FlowRowInspectorView: View {
    let model: FlowEditorModel
    let rowID: UUID
    let catalog: [ModelEntry]
    let totalRAMGB: Double
    /// CFM-R14-3 — catalog ids already installed on disk. The Model menu sections on this:
    /// Installed first, then Available to download (with the total size). Passed at both call
    /// sites from `AppState.installedModelIDs` (`@Observable`, so an install re-renders live).
    var installedModelIDs: Set<String> = []
    /// CFM-R14-2 — catalog ids the registry can claim; the Model menu's derived pool filter.
    var claimableModelIDs: Set<String> = []
    /// OCP-2-2 — how the row's named model's stage varies with a per-run prompt. **Unlike the
    /// Run UI** (`2026-229`), the inspector has registry access, so `FlowEditorView` passes a
    /// resolver that goes `ModelEntry → SDK → promptSupport`. Default `.none` keeps the
    /// read-only / preview call sites and previews working without a registry.
    var resolvePromptSupport: @MainActor (ModelEntry) -> PromptSupport = { _ in .none }
    /// Whether the row's details are editable. The flow editor edits in place; the read-only
    /// flow list passes `false`, so the properties tab is browsable but never mutable.
    var editable: Bool = true
    /// Whether the pane is "frozen" — a bundled gallery flow's properties are read-only
    /// *and* dimmed with a lock badge, while a user flow's stay at full opacity.
    var isFrozen: Bool = false

    private var row: Row? { model.row(withID: rowID) }

    @State private var showAdvanced = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let row {
                header(row)
                if let warning = model.warning(for: rowID) {
                    Label(warning, systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 6))
                }
                Divider()
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        if let task = row.task, isModelClass(task) {
                            modelPicker(task, row: row)
                        }
                        if let warning = strandedPromptWarning(row) {
                            Label(warning, systemImage: "exclamationmark.triangle")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if let task = row.task {
                            switch promptControl(task: task, row: row) {
                            case .instructionBox:
                                instructionBox(task, row: row)
                            case .modePicker(let values, let defaultValue):
                                modePicker(row: row, values: values, defaultValue: defaultValue)
                            case .schemaEditor:
                                schemaEditor(row: row)
                            case .none:
                                EmptyView()
                            }
                        }
                        // CFM — a `Save *` row's filename is its path token; let the user
                        // type it (or a subfolder) instead of hunting in the file list.
                        if let task = row.task, task.hasPrefix("Save") {
                            saveFilenameField(row)
                        }
                        inputs(row)
                        settingsSection(row)
                        if let task = row.task, hasPathSetting(task) {
                            chooseFileButton(task: task)
                        }
                        decisionsSection(row)
                    }
                    .padding(4)
                }
                // Read-only (the flow list): every control is disabled — the pane becomes a
                // browse-only inspection, the row's values still fully visible. The Advanced
                // disclosure stays toggleable (it only holds informational labels).
                .disabled(!editable)
            } else {
                Text(editable ? "Select a row to edit it." : "Select a row to inspect its properties.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .opacity(isFrozen ? 0.85 : 1)
    }

    // MARK: - Header

    private func header(_ row: Row) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "slider.horizontal.3")
                .foregroundStyle(.secondary)
            Text(model.displayNumber(of: rowID) ?? "?")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
            Text(FlowRowSummary.taskName(for: row))
                .font(.headline)
            if isFrozen {
                Image(systemName: "lock.fill")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("This flow's properties are read-only — Duplicate & Edit to change them")
            }
            Spacer()
        }
    }

    // MARK: - R9-2 model picker

    private func modelPicker(_ task: String, row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Model")
                .font(.subheadline.weight(.semibold))
            let sections = FlowEditorModel.sectionedModelCandidates(
                for: task, catalog: catalog, installedModelIDs: installedModelIDs,
                claimableModelIDs: claimableModelIDs)
            Menu {
                Button("None") { model.setModel(nil, for: rowID) }
                Divider()
                if !sections.installed.isEmpty {
                    Section("Installed") {
                        ForEach(sections.installed, id: \.model.id) { candidate in
                            modelButton(candidate, row: row)
                        }
                    }
                }
                if !sections.available.isEmpty {
                    Section("Available to download — \(String(format: "%.1f GB", sections.availableTotalGB))") {
                        ForEach(sections.available, id: \.model.id) { candidate in
                            modelButton(candidate, row: row)
                        }
                    }
                }
            } label: {
                HStack {
                    Text(row.model ?? "No model")
                        .foregroundStyle(row.model == nil ? .secondary : .primary)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
            if let display = row.model {
                DisclosureGroup("Advanced", isExpanded: $showAdvanced) {
                    VStack(alignment: .leading, spacing: 4) {
                        // FIX-7: "Catalog id" is the pinned HF id, not the friendly name.
                        Text("Catalog id: \(CatalogBridge.entry(for: display)?.pinnedID ?? display)")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.secondary)
                        if let note = substitutionNote(for: display) {
                            Label(note, systemImage: "arrow.triangle.swap")
                                .font(.caption2)
                                .foregroundStyle(.orange)
                        }
                    }
                    .padding(.vertical, 4)
                }
                .font(.caption)
                // The Advanced disclosure holds only informational labels — it stays
                // toggleable even in read-only mode, so its note is browsable.
                .disabled(false)
            }
        }
    }

    /// One row of the Model menu: the friendly name, the RAM figure (orange when it exceeds
    /// this Mac's RAM), disabled when it doesn't fit. Shared by both sections (CFM-R14-3).
    private func modelButton(_ candidate: (display: String, model: ModelEntry), row: Row) -> some View {
        Button {
            model.setModel(candidate.display, for: rowID)
        } label: {
            HStack {
                Text(candidate.display)
                Spacer()
                Text(String(format: "%.1f GB", candidate.model.ramGB))
                    .foregroundStyle(candidate.model.ramGB > totalRAMGB ? .orange : .secondary)
            }
        }
        .disabled(candidate.model.ramGB > totalRAMGB)
    }

    private func substitutionNote(for display: String) -> String? {
        guard let entry = CatalogBridge.entry(for: display) else { return nil }
        for candidate in entry.candidates where catalog.contains(where: { $0.hfModelId == candidate }) {
            return entry.equivalence.note(display: display, substitutedID: candidate)
        }
        return nil
    }

    // MARK: - R9-3 instruction textbox

    private func instructionBox(_ task: String, row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("What should it do?")
                .font(.subheadline.weight(.semibold))
            let instruction = quotedInstruction(row.settings)
            TextField("…", text: Binding(
                get: { instruction ?? "" },
                set: { model.setInstruction($0.isEmpty ? nil : $0, for: rowID) }
            ), axis: .vertical)
            .lineLimit(2...5)
            .textFieldStyle(.roundedBorder)
            let suggestions = FlowEditorModel.suggestedInstructions(for: task)
            if !suggestions.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Suggested")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    ForEach(suggestions, id: \.self) { suggestion in
                        Button {
                            model.setInstruction(suggestion, for: rowID)
                        } label: {
                            Text(suggestion)
                                .font(.caption)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.quaternary.opacity(0.5), in: Capsule())
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    // MARK: - ES-UI-1: Extract Structured column list

    /// The `Extract Structured` schema editor. The row's settings string is a bare quoted
    /// column list (`"merchant, date, total"`) — the same first-quoted-span `setInstruction`
    /// reads and writes — so this reuses those helpers with a label that fits a column list.
    /// Clearing writes an empty settings string; `FlowEditorModel.warning(for:)` then flags the
    /// row so the failure surfaces in the editor, not only when `parseSchema` raises at run time.
    private func schemaEditor(row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Columns to extract")
                .font(.subheadline.weight(.semibold))
            let columns = quotedInstruction(row.settings)
            TextField("merchant, date, total, category", text: Binding(
                get: { columns ?? "" },
                set: { model.setInstruction($0.isEmpty ? nil : $0, for: rowID) }
            ), axis: .vertical)
            .lineLimit(1...3)
            .textFieldStyle(.roundedBorder)
            Text("Comma-separated — one name per field. A column name can't contain a comma.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - CFM: Save-row filename

    /// A `Save *` row's output file name, edited in place as the path token (a `path=` value
    /// or the first bare token). Clearing the box leaves the current name alone — an empty
    /// path would make the row's save fail, so nothing is written instead.
    private func saveFilenameField(_ row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("File name")
                .font(.subheadline.weight(.semibold))
            TextField("output.txt", text: Binding(
                get: { FlowSettings(row.settings).pathValue() ?? "" },
                set: { newValue in
                    guard !newValue.isEmpty else { return }
                    model.setPath(newValue, for: rowID)
                }
            ))
            .textFieldStyle(.roundedBorder)
        }
    }

    // MARK: - R9-4 input pickers

    @ViewBuilder
    private func inputs(_ row: Row) -> some View {
        let slots = model.inputSlotCount(for: rowID)
        if slots > 0 {
            VStack(alignment: .leading, spacing: 6) {
                Text(slots > 1 ? "Inputs" : "Input")
                    .font(.subheadline.weight(.semibold))
                ForEach(1...slots, id: \.self) { slot in
                    inputPicker(slot: slot)
                }
            }
        }
    }

    private func inputPicker(slot: Int) -> some View {
        Menu {
            Button("None") { model.setReference(to: nil, slot: slot, for: rowID) }
            Divider()
            ForEach(model.validInputs(for: rowID, slot: slot), id: \.rowID) { input in
                Button("\(input.number) — \(input.task)") {
                    model.setReference(to: input.rowID, slot: slot, for: rowID)
                }
            }
        } label: {
            HStack {
                Text(inputLabel(slot: slot))
                    .foregroundStyle(.secondary)
                Spacer()
                Text(currentInputLabel(slot: slot))
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(8)
            .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
        }
    }

    private func inputLabel(slot: Int) -> String {
        let count = model.inputSlotCount(for: rowID)
        return count > 1 ? "Input \(slot)" : "Input"
    }

    private func currentInputLabel(slot: Int) -> String {
        guard let row = row, slot >= 1, slot <= row.refs.count,
              case .rowRef(let id) = row.refs[slot - 1] else {
            return "None"
        }
        guard let target = model.row(withID: id) else { return "(?)" }
        let number = model.displayNumber(of: id) ?? "?"
        return "\(number) — \(FlowRowSummary.taskName(for: target))"
    }

    // MARK: - R9-1 settings fields

    private func settingsSection(_ row: Row) -> some View {
        let settings = FlowSettings(row.settings)
        let keys = settings.testKeys.sorted()
        return VStack(alignment: .leading, spacing: 6) {
            Text("Settings")
                .font(.subheadline.weight(.semibold))
            ForEach(keys, id: \.self) { key in
                HStack(spacing: 6) {
                    Text(key)
                        .font(.caption.monospaced())
                        .frame(width: 90, alignment: .trailing)
                    TextField("value", text: Binding(
                        // FIX-7: write the value even when empty — clearing to nil deletes the
                        // key, so select-all-and-retype would make the field vanish mid-edit.
                        get: { settings.value(for: key) ?? "" },
                        set: { model.setSetting(key: key, value: $0, for: rowID) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                }
            }
            if keys.isEmpty {
                Text("No settings on this row yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            addSettingMenu(row: row)
        }
    }

    private func addSettingMenu(row: Row) -> some View {
        let known = Self.knownSettingKeys(for: row.task ?? "")
        return Menu {
            ForEach(known, id: \.self) { key in
                Button(key) { model.setSetting(key: key, value: "", for: rowID) }
            }
        } label: {
            Label("Add a setting…", systemImage: "plus")
                .font(.caption)
        }
        .disabled(known.isEmpty)
    }

    /// The common per-task settings keys (the engines' own parsed keys — there is no
    /// machine-readable catalog for them, so this is a small curated table).
    nonisolated static func knownSettingKeys(for task: String) -> [String] {
        switch task {
        case "Transcribe": return ["lang", "timestamps"]
        case "Split": return ["by", "chunk_size", "overlap"]
        case "Speak": return ["voice", "lang", "speed"]
        case "Retrieve": return ["query", "top_k"]
        case "Rerank": return ["query", "top_k"]
        case "Filter": return ["by"]
        case "Sort": return ["by", "reverse"]
        case "Dedupe": return ["by"]
        case "Extract": return ["by"]
        case "Count": return ["group_by"]
        case "Compare": return ["mode"]
        case "Template": return []
        default: return []
        }
    }

    // MARK: - CFM-R11-0b: the file chooser (copy in, never bookmark)

    /// Whether this row's task takes a path (a `Read *`/`Save *` of a file or folder).
    private func hasPathSetting(_ task: String) -> Bool {
        guard let desc = TaskCatalog.get(task) else { return false }
        switch desc.accepts {
        case .single(.file), .single(.folder): return true
        default: return false
        }
    }

    private var currentPathLabel: String {
        guard let row, let path = FlowSettings(row.settings).pathValue() else {
            return "Choose a file…"
        }
        return path
    }

    private func chooseFileButton(task: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("File")
                .font(.subheadline.weight(.semibold))
            Button {
                chooseFile(task: task)
            } label: {
                HStack {
                    Text(currentPathLabel)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Image(systemName: "folder")
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
        }
    }

    /// R11-0b's rule: the chooser **copies in** — it stores a bare filename, never a
    /// security-scoped bookmark or an absolute path, so the flow folder stays a
    /// self-contained thing that runs after a zip-and-send.
    private func chooseFile(task: String) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = (task == "Read Images" || task == "Read Files")
        panel.allowsMultipleSelection = false
        if let types = Self.utTypes(for: task) {
            panel.allowedContentTypes = types
        }
        // FILE-1: when the row already names a file, they're adjusting something already
        // copied in — open the panel at the flow's own folder. On first setup the row has no
        // path and the images live wherever the user keeps them (Desktop, …); opening inside
        // the app container would make them navigate out every time, so leave the panel where
        // AppKit puts it.
        if let row, FlowSettings(row.settings).pathValue() != nil {
            panel.directoryURL = model.workspace.directory(for: model.flowID)
        }
        guard panel.runModal() == .OK, let chosen = panel.url else { return }
        copyInAndSetPath(chosen)
    }

    private func copyInAndSetPath(_ chosen: URL) {
        let flowDir = model.workspace.directory(for: model.flowID)
        do {
            let stored = try Self.copyIn(chosen, toFlowDir: flowDir)
            model.setPath(stored, for: rowID)
        } catch {
            model.saveError = "Couldn't copy '\(chosen.lastPathComponent)' into the flow's folder — pick a file you can read."
        }
    }

    /// Copy `chosen` into `flowDir` (R11-0b — a bare name that survives a zip-and-send) and
    /// return the path token to store in the row's settings.
    ///
    /// **FILE-1:** when `chosen` is *already inside* `flowDir` (the Properties button now opens
    /// there), copy nothing — the per-item `removeItem` + `copyItem` loop would delete each
    /// file and then copy the file it just deleted. Store its **flow-relative** path instead:
    /// slash-separated (`FlowWorkspace.resolve` accepts that and rejects only absolute / `..`),
    /// with a directory's trailing `/` kept so `CatParser.splitModelSettings` reads
    /// `receipts/jan/` as a path, not an HF repo id (SPEC-Q96). *(A nested file with no
    /// extension — `receipts/scan1` — still parses as a model: DA-8's territory, not fixed
    /// here.)*
    ///
    /// `nonisolated static` so the copy/no-copy decision is unit-testable without a View.
    nonisolated static func copyIn(_ chosen: URL, toFlowDir flowDir: URL) throws -> String {
        let fm = FileManager.default
        try fm.createDirectory(at: flowDir, withIntermediateDirectories: true)

        if let relative = relativePath(of: chosen, under: flowDir) {
            return relative
        }

        if chosen.hasDirectoryPath {
            let destDir = flowDir.appendingPathComponent(chosen.lastPathComponent)
            try fm.createDirectory(at: destDir, withIntermediateDirectories: true)
            for url in try fm.contentsOfDirectory(at: chosen, includingPropertiesForKeys: nil) {
                let dest = destDir.appendingPathComponent(url.lastPathComponent)
                if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
                try fm.copyItem(at: url, to: dest)
            }
            return chosen.lastPathComponent
        }
        let dest = flowDir.appendingPathComponent(chosen.lastPathComponent)
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.copyItem(at: chosen, to: dest)
        return chosen.lastPathComponent
    }

    /// `url`'s slash-separated path relative to `base`, or nil when `url` is not inside it.
    /// Both sides go through `resolvingSymlinksInPath()` — the app-container path is symlinked
    /// (`/var` ⇄ `/private/var`), the same reason `FlowWorkspace.resolve` does it. A directory
    /// keeps its trailing `/`.
    nonisolated static func relativePath(of url: URL, under base: URL) -> String? {
        let target = url.resolvingSymlinksInPath().pathComponents
        let root = base.resolvingSymlinksInPath().pathComponents
        guard target.count > root.count, Array(target.prefix(root.count)) == root else {
            return nil
        }
        let relative = target.dropFirst(root.count).joined(separator: "/")
        return url.hasDirectoryPath ? relative + "/" : relative
    }

    /// The open panel's allowed types for a task's sample-replaceable file (nil = any).
    nonisolated static func utTypes(for task: String) -> [UTType]? {
        switch task {
        case "Read Audio": return [.audio]
        case "Read Text": return [.plainText, .text]
        case "Read Image", "Read Images": return [.image]
        case "Read PDF": return [.pdf]
        case "Read CSV": return [.commaSeparatedText]
        case "Read JSON": return [.json]
        default: return nil
        }
    }

    // MARK: - R9-5 decisions & budget

    @ViewBuilder
    private func decisionsSection(_ row: Row) -> some View {
        let edges = model.deciderEdges(for: rowID)
        if !edges.isEmpty || isDecider(row) {
            VStack(alignment: .leading, spacing: 6) {
                Text("Decisions")
                    .font(.subheadline.weight(.semibold))
                ForEach(Array(edges.enumerated()), id: \.offset) { index, edge in
                    HStack(spacing: 6) {
                        TextField("tag", text: Binding(
                            get: { edge.tag },
                            set: { model.setClauseEdge(tag: $0.isEmpty ? nil : $0, target: edge.target, at: index, for: rowID) }
                        ))
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        .frame(width: 90)
                        Text("→")
                            .foregroundStyle(.secondary)
                        targetPicker(slot: index)
                        Button {
                            model.setClauseEdge(tag: nil, target: nil, at: index, for: rowID)
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                if edges.isEmpty {
                    Text("No tags yet — add one to branch this row.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Button {
                    model.setClauseEdge(tag: "tag", target: nil, at: edges.count, for: rowID)
                } label: {
                    Label("Add tag", systemImage: "plus")
                        .font(.caption)
                }
                .disabled(!isDecider(row))
            }
        }
        let hasBudget = row.visitsLeq != nil || row.onBudget != nil
        if hasBudget || isDecider(row) {
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                Text("Budget")
                    .font(.subheadline.weight(.semibold))
                HStack(spacing: 8) {
                    Text("max_visits")
                        .font(.caption)
                    TextField("∞", text: Binding(
                        get: { row.visitsLeq.map(String.init) ?? "" },
                        set: { model.setVisitsLeq(Int($0), for: rowID) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .frame(width: 60)
                    Text("on_budget")
                        .font(.caption)
                    TextField("fail", text: Binding(
                        get: { row.onBudget ?? "" },
                        set: { model.setOnBudget($0.isEmpty ? nil : $0, for: rowID) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .frame(width: 80)
                }
            }
        }
    }

    private func targetPicker(slot: Int) -> some View {
        // FIX-5: the *slot's own* tag, never `.first` (editing the second edge's target must
        // not rewrite the first's tag into a duplicate).
        let edges = model.deciderEdges(for: rowID)
        let currentTag = slot < edges.count ? edges[slot].tag : ""
        return Menu {
            Button("done") {
                model.setClauseEdge(tag: currentTag, target: 0, at: slot, for: rowID)
            }
            Button("resume") {
                model.setClauseEdge(tag: currentTag, target: -1, at: slot, for: rowID)
            }
            Divider()
            // FIX-5: only top-level rows are offerable — clause targets are Int row numbers,
            // and a nested row's dotted path ("2.1") has no numeric form.
            ForEach(model.document.rows, id: \.id) { candidate in
                let number = model.displayNumber(of: candidate.id).flatMap(Int.init) ?? 0
                Button("Row \(number) — \(FlowRowSummary.taskName(for: candidate))") {
                    model.setClauseEdge(tag: currentTag, target: number, at: slot, for: rowID)
                }
            }
        } label: {
            Text(targetLabel(targetPickerTarget(slot: slot), slot: slot))
                .font(.caption)
                .padding(6)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 5))
        }
    }

    private func targetPickerTarget(slot: Int) -> Int {
        let edges = model.deciderEdges(for: rowID)
        guard slot < edges.count else { return 0 }
        return edges[slot].target
    }

    /// QR9: a stale slot (its target row was deleted) renders `(?N)` — the number it still
    /// carries is not an honest label.
    private func targetLabel(_ target: Int, slot: Int) -> String {
        if model.staleClauseSlots(for: rowID).contains(slot) { return "(?)" }
        switch target {
        case 0: return "done"
        case -1: return "resume"
        default: return "Row \(target)"
        }
    }

    // MARK: - Helpers

    private func isModelClass(_ task: String) -> Bool {
        TaskCatalog.get(task)?.taskClass == .model
    }

    /// OCP-2-2 — which per-run prompt control (if any) this row gets.
    enum PromptControl: Equatable {
        case instructionBox
        case modePicker(values: [String], defaultValue: String)
        /// ES-UI-1 — `Extract Structured`'s whole settings string *is* its column list, a bare
        /// quoted span with no `key=` label, so neither the instruction box nor the generic
        /// settings section offers it. This is an instruction-box-shaped editor with the right
        /// label for a column list.
        case schemaEditor
        case none
    }

    /// A frame-backed row keeps its instruction box (FIX-6). `engines.vlm.describe_image`
    /// now gets one too — a prompt is its whole point, and OCP-2-1 makes it reach the model.
    /// An `engines.vlm.ocr` row is driven by its **named model's** `promptSupport`:
    /// `.freeText` → the instruction box, `.modes` → a picker, `.none` → nothing. Keyed on
    /// the capability, never on a model name (`CatalogBridge` is the one table where names
    /// live).
    private func promptControl(task: String, row: Row) -> PromptControl {
        guard let desc = TaskCatalog.get(task) else { return .none }
        if desc.refName.hasPrefix("frames/") { return .instructionBox }
        if desc.refName == "engines.vlm.describe_image" { return .instructionBox }
        if desc.refName == "engines.llm.extract_structured" { return .schemaEditor }   // ES-UI-1
        if desc.refName == "engines.vlm.ocr" {
            switch modelPromptSupport(for: row) {
            case .freeText: return .instructionBox
            case .modes(let values, let def): return .modePicker(values: values, defaultValue: def)
            case .none: return .none
            }
        }
        return .none
    }

    /// The `promptSupport` of the row's named model (`.none` when it names nothing, or nothing
    /// runnable). Resolved via `CatalogBridge` → the injected registry resolver.
    private func modelPromptSupport(for row: Row) -> PromptSupport {
        guard let display = row.model,
              case let .runnable(model, _, _) = CatalogBridge.resolve(display, catalog: catalog)
        else { return .none }
        return resolvePromptSupport(model)
    }

    /// OCP-2-2 — recognition-mode picker for a `.modes` OCR model (PaddleOCR-VL). Writes the
    /// mode as the row's first quoted token (`OCR PaddleOCR-VL; "table"`), the same token the
    /// runtime reads via `firstBare()`; the default mode writes **nothing** so `firstBare()`
    /// is nil and the SDK's own default applies.
    private func modePicker(row: Row, values: [String], defaultValue: String) -> some View {
        let current = FlowSettings(row.settings).firstBare() ?? defaultValue
        return VStack(alignment: .leading, spacing: 6) {
            Text("Recognition mode")
                .font(.subheadline.weight(.semibold))
            Picker("", selection: Binding(
                get: { values.contains(current) ? current : defaultValue },
                set: { model.setInstruction($0 == defaultValue ? nil : $0, for: rowID) }
            )) {
                ForEach(values, id: \.self) { Text(Self.modeLabel($0)).tag($0) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
        }
    }

    private static func modeLabel(_ id: String) -> String {
        switch id {
        case "ocr": return "Text"
        case "table": return "Table"
        case "formula": return "Formula"
        case "chart": return "Chart"
        default: return id.capitalized
        }
    }

    /// OCP-2-3 (RULED: validator warns, runtime ignores). The row carries a prompt token its
    /// named model cannot consume — a `.none` model that ignores it, or a `.modes` value
    /// outside the model's set. The flow still runs (the runtime drops the value); this is
    /// the warning that makes the hidden field honest. Surfaced like the `CatalogBridge`
    /// substitution note.
    private func strandedPromptWarning(_ row: Row) -> String? {
        guard let task = row.task, let desc = TaskCatalog.get(task),
              desc.refName == "engines.vlm.ocr",
              let token = FlowSettings(row.settings).firstBare(), !token.isEmpty,
              let display = row.model
        else { return nil }
        return Self.strandedPromptMessage(token: token, display: display,
                                          support: modelPromptSupport(for: row))
    }

    /// The pure decision behind `strandedPromptWarning` — `nil` when the token is fine for the
    /// model, else the sentence. `nonisolated static` so it is unit-testable without a View.
    nonisolated static func strandedPromptMessage(token: String, display: String,
                                                  support: PromptSupport) -> String? {
        switch support {
        case .freeText:
            return nil
        case .none:
            return "\(display) transcribes with a fixed prompt — \"\(token)\" won't reach it. Remove it, or pick a model that takes an instruction."
        case .modes(let values, _):
            guard !values.contains(token) else { return nil }
            return "\"\(token)\" isn't a recognition mode \(display) understands (\(values.joined(separator: ", "))) — this row will run in the default mode."
        }
    }

    private func isDecider(_ row: Row) -> Bool {
        guard let task = row.task else { return false }
        return TaskCatalog.deciderTasks[task] != nil
    }

    /// The quoted instruction in a settings string — the first quoted span's unquoted text,
    /// the same token `setInstruction` writes (FIX-6).
    private func quotedInstruction(_ settings: String?) -> String? {
        guard let token = FlowSettingsEditor.firstQuotedToken(in: settings ?? "") else { return nil }
        return FlowSettings.unquote(token.token)
    }
}
