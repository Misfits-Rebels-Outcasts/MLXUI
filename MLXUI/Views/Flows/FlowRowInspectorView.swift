import SwiftUI
import UniformTypeIdentifiers

/// FILE-2 — "Create a folder here"'s failure modes. Kept separate from `FlowError` (that one
/// names a *row*; this names a *user action* in the inspector, before any row runs).
nonisolated enum FlowFolderCreateError: Error, CustomStringConvertible, Equatable {
    case invalidName
    case alreadyExists(String)

    var description: String {
        switch self {
        case .invalidName:
            return "Give the folder a name."
        case .alreadyExists(let name):
            return "A folder named '\(name)' already exists in this flow."
        }
    }
}

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
    /// RT-1 — the Pattern box's "Edit…" sheet.
    @State private var showPatternSheet = false
    /// RT-5 — the "Row text" box's own draft, decoupled from `row.settings` while the user is
    /// mid-edit (and left showing the rejected text — "dirty" — after a refusal). `nil` means
    /// "not being edited right now; show the committed value."
    @State private var rowTextDraft: String?
    @State private var rowTextRefusal: String?
    @FocusState private var rowTextFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if let row {
                header(row)
                if let warning = model.warning(for: rowID) {
                    VStack(alignment: .leading, spacing: 6) {
                        Label(warning, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption)
                            .foregroundStyle(.orange)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        // FIX-1 — the message names a fix ("fmt will do this for you"); this
                        // app has no `fmt`, so offer the one-flag repair directly, only where
                        // the document is writable (`editable`) and the flag can run in this
                        // edition (`headerRepair` already excludes App-Store-refused flags).
                        if editable, let flag = model.headerRepair(for: rowID) {
                            Button("Add `\(flag.rawValue)` to line one") {
                                model.applyHeaderRepair(flag)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                        }
                    }
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
                            case .patternEditor:
                                patternEditor(row: row)
                            case .singleLineField(let label):
                                singleLineField(label: label, row: row)
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
                        if let task = row.task, isHumanClass(task) {
                            humanWaitPolicySection(task, row: row)
                        }
                        // RT-1 (§6 Q1): a `Template` row's Pattern box owns the *entire*
                        // settings string (fact 11 — the runtime reads it whole, not as
                        // `key=value` pairs), so the generic Settings list — and its Add menu,
                        // which would otherwise offer a way to silently corrupt the pattern
                        // with an unrelated token — is hidden for `Template` rows.
                        if row.task != "Template" {
                            if let warning = shadowingWarning(row) {
                                Label(warning, systemImage: "exclamationmark.triangle")
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            settingsSection(row)
                        }
                        if let task = row.task, hasPathSetting(task) {
                            chooseFileButton(task: task)
                        }
                        decisionsSection(row)
                        // RT-5 — the no-dead-ends backstop: after this, no row in any flow,
                        // present or future, is uneditable in the UI. A frozen gallery flow's
                        // raw text is already visible in the row list itself; the box adds
                        // nothing there but a false invitation to edit read-only bytes, so
                        // it's hidden (not just disabled) when `isFrozen`.
                        if !isFrozen {
                            rowTextDisclosure(row)
                        }
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
                        ForEach(sections.installed, id: \.id) { slot in
                            modelButton(slot, row: row)
                        }
                    }
                }
                // MS-2: empty until Phase AFM/RM populate a system/provider slot — never
                // rendered today, so this changes nothing visible.
                if !sections.builtIn.isEmpty {
                    Section("Built in") {
                        ForEach(sections.builtIn, id: \.id) { slot in
                            modelButton(slot, row: row)
                        }
                    }
                }
                if !sections.available.isEmpty {
                    Section("Available to download — \(String(format: "%.1f GB", sections.availableTotalGB))") {
                        ForEach(sections.available, id: \.id) { slot in
                            modelButton(slot, row: row)
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

    /// One row of the Model menu: the friendly name, the resource note (a cataloged model's
    /// RAM figure, orange when it exceeds this Mac's RAM), disabled when `isModelButtonEnabled`
    /// says no. Shared by every section (CFM-R14-3 + MS-2).
    private func modelButton(_ slot: ModelSlot, row: Row) -> some View {
        let overRAM = (slot.modelEntry?.ramGB ?? 0) > totalRAMGB
        return Button {
            model.setModel(slot.displayName, for: rowID)
        } label: {
            HStack {
                Text(slot.displayName)
                Spacer()
                if let note = slot.resourceNote {
                    Text(note)
                        .foregroundStyle(overRAM ? .orange : .secondary)
                }
            }
        }
        .disabled(!Self.isModelButtonEnabled(slot, installedModelIDs: installedModelIDs, totalRAMGB: totalRAMGB))
    }

    /// MS-4 — whether a Model-menu row for `slot` should be selectable. `.unavailable`
    /// disables it; **`.needsSetup` stays selectable** — a flow may legitimately name a model
    /// whose key you're about to paste in, and refusing at pick time would be a lie. A
    /// cataloged model that doesn't fit this Mac's RAM disables too (CFM-R14-3, unchanged).
    /// `nonisolated static` so it's unit-testable with a fixture slot, without a View.
    nonisolated static func isModelButtonEnabled(_ slot: ModelSlot, installedModelIDs: Set<String>,
                                                 totalRAMGB: Double) -> Bool {
        if case .unavailable = slot.readiness(installedModelIDs: installedModelIDs) { return false }
        if (slot.modelEntry?.ramGB ?? 0) > totalRAMGB { return false }
        return true
    }

    private func substitutionNote(for display: String) -> String? {
        guard let entry = CatalogBridge.entry(for: display) else { return nil }
        for candidate in entry.candidates where catalog.contains(where: { $0.hfModelId == candidate }) {
            return entry.equivalence.note(display: display, substitutedID: candidate)
        }
        return nil
    }

    // MARK: - R9-3 instruction textbox

    /// RT-2/RT-3 (§8 copy table) — the instruction-box labels that aren't "What should it do?":
    /// the human rows ask a person a question, not a model; the diffusion rows' whole point is
    /// the prompt.
    private static func instructionBoxLabel(for task: String) -> String {
        switch task {
        case "Ask Human", "Human Input": return "What should the person be asked?"
        case "Generate Image", "Generate Sound", "Generate Video": return "Prompt"
        default: return "What should it do?"
        }
    }

    private func instructionBox(_ task: String, row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(Self.instructionBoxLabel(for: task))
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

    // MARK: - RT-3: a short, structured value (not prose)

    /// RT-3 (§3 rule 4) — `Compare`'s criterion today. Reads and writes the same first-quoted
    /// token as `instructionBox` (`quotedInstruction`/`setInstruction`); the only difference is
    /// a true single-line field, since a multi-line box would misrepresent a short expression
    /// like `"> 50"` as prose.
    private func singleLineField(label: String, row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label)
                .font(.subheadline.weight(.semibold))
            TextField("…", text: Binding(
                get: { quotedInstruction(row.settings) ?? "" },
                set: { model.setInstruction($0.isEmpty ? nil : $0, for: rowID) }
            ))
            .textFieldStyle(.roundedBorder)
        }
    }

    // MARK: - RT-1: Template's Pattern box

    /// RT-1 — a `Template` row's pattern *is* its entire settings string (fact 11:
    /// `TextTools.unquoteWhole`, not a single spliced token), so this reads and writes the
    /// whole thing through `FlowEditorModel.setPattern` / `FlowSettingsEditor
    /// .replaceWholeSettings`, never `setInstruction`'s first-quoted-token splice — a second
    /// token would desync what the box shows from what the runtime renders (§6 Q1). The
    /// generic Settings list is hidden for `Template` rows for the same reason (see `body`).
    /// SPEC-Q229 — built to the backlog's own recommended reading (whole string); no owner
    /// quote on record for this specific call, so implementer's call, pending confirmation.
    private func patternEditor(row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Pattern")
                .font(.subheadline.weight(.semibold))
            TextField("…", text: Binding(
                get: { TextTools.unquoteWhole(row.settings ?? "") },
                set: { model.setPattern($0.isEmpty ? nil : $0, for: rowID) }
            ), axis: .vertical)
            .lineLimit(4...12)
            .font(.caption.monospaced())
            .textFieldStyle(.roundedBorder)
            Text("Placeholders pull in this row's inputs. {1} is the first input, {2} the second, {input} all of them.")
                .font(.caption2)
                .foregroundStyle(.secondary)
            placeholderChips(row: row)
            Button("Edit…") { showPatternSheet = true }
                .buttonStyle(.bordered)
                .controlSize(.small)
        }
        .sheet(isPresented: $showPatternSheet) {
            patternEditSheet(row: row)
        }
    }

    /// The full-size editor — the pane itself is 260–320 pt (fact 15) and a real pattern runs
    /// 600+ characters, too cramped for the inline box alone.
    private func patternEditSheet(row: Row) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Edit pattern")
                .font(.headline)
            TextEditor(text: Binding(
                get: { TextTools.unquoteWhole(model.row(withID: rowID)?.settings ?? "") },
                set: { model.setPattern($0.isEmpty ? nil : $0, for: rowID) }
            ))
            .font(.body.monospaced())
            .frame(minWidth: 420, minHeight: 320)
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            // E207 (fact 13) reaches this sheet the same way it reaches the header banner
            // underneath it — both read `model.warning(for:)`, which recomputes whenever
            // `setPattern` mutates the document. The sheet occludes that banner, so it's
            // repeated here rather than left invisible until the sheet closes.
            if let warning = model.warning(for: rowID) {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            HStack {
                Spacer()
                Button("Done") { showPatternSheet = false }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding()
        .frame(minWidth: 460)
    }

    /// `{1}`, `{2}`, … for `model.inputSlotCount(for:)`, `{input}`, and — only inside an
    /// `<each>` block, where they're legal (`FlowValidator.reTemplateToken` / `checkTemplate
    /// Placeholders`'s `inEach`, hazard 9) — `{index}`/`{item}`. Tapping one appends it to the
    /// pattern: this box has no reliable cursor-position API at the app's macOS 14.0 floor, so
    /// "insert" here means "append", not a mid-string splice.
    private func placeholderChips(row: Row) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 4) {
                ForEach(placeholderTokens(row: row), id: \.self) { token in
                    Button {
                        let current = TextTools.unquoteWhole(row.settings ?? "")
                        model.setPattern(current + token, for: rowID)
                    } label: {
                        Text(token)
                            .font(.caption2.monospaced())
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.quaternary.opacity(0.5), in: Capsule())
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private func placeholderTokens(row: Row) -> [String] {
        let slots = model.inputSlotCount(for: rowID)
        var tokens = slots >= 1 ? (1...slots).map { "{\($0)}" } : []
        tokens.append("{input}")
        if model.isInsideEach(rowID) {
            tokens.append(contentsOf: ["{index}", "{item}"])
        }
        return tokens
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
        // RT-4: `orderedPairs` (source order), not the test-only `testKeys.sorted()` — a
        // `Set`'s iteration order isn't source order, and driving production UI off a
        // `test`-prefixed accessor was never the point of that accessor's existing.
        let pairs = FlowSettings(row.settings).orderedPairs
        return VStack(alignment: .leading, spacing: 6) {
            Text("Settings")
                .font(.subheadline.weight(.semibold))
            ForEach(pairs, id: \.key) { pair in
                settingField(key: pair.key, row: row)
            }
            if pairs.isEmpty {
                Text("No settings on this row yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            addSettingMenu(row: row)
        }
    }

    /// RT-4 — the per-key control shape: a prose key gets a short multi-line field, a
    /// `separator` gets the escape-visible field (below), everything else stays single-line.
    @ViewBuilder
    private func settingField(key: String, row: Row) -> some View {
        if key == "separator" {
            separatorField(row: row)
        } else {
            let settings = FlowSettings(row.settings)
            HStack(alignment: .top, spacing: 6) {
                Text(key)
                    .font(.caption.monospaced())
                    .frame(width: 90, alignment: .trailing)
                if Self.proseSettingKeys.contains(key) {
                    TextField("value", text: Binding(
                        // FIX-7: write the value even when empty — clearing to nil deletes
                        // the key, so select-all-and-retype would make the field vanish
                        // mid-edit.
                        get: { settings.value(for: key) ?? "" },
                        set: { model.setSetting(key: key, value: $0, for: rowID) }
                    ), axis: .vertical)
                    .lineLimit(1...4)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                } else {
                    TextField("value", text: Binding(
                        get: { settings.value(for: key) ?? "" },
                        set: { model.setSetting(key: key, value: $0, for: rowID) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                }
            }
        }
    }

    /// RT-4 (§3 rule 4) — a string value, not a mode or number: `query`/`contains`/`expected`/
    /// `text`/`naming` get room to wrap rather than a cramped one-line field. `separator` is
    /// excluded (`settingField` routes it to `separatorField` instead) — its value is a short
    /// symbolic string, not prose, and needs to stay visibly escaped (below), which a real
    /// multi-line box would defeat by rendering `\n` as an actual, easy-to-miss blank line.
    private static let proseSettingKeys: Set<String> = ["query", "contains", "expected", "text", "naming"]

    /// RT-4 (escape-visibility) — `separator`'s value round-trips through the same `\n`/`\t`
    /// escapes a quoted settings value does (`FlowSettingsEditor.quote` / `FlowSettings
    /// .unquote`), so it's shown and edited as that escaped literal — `\n`, two characters —
    /// rather than the real, invisible control character `FlowSettings.value(for:)` already
    /// decoded it to. "Do not silently normalise": what's on screen is what's on disk.
    private func separatorField(row: Row) -> some View {
        let settings = FlowSettings(row.settings)
        return VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .top, spacing: 6) {
                Text("separator")
                    .font(.caption.monospaced())
                    .frame(width: 90, alignment: .trailing)
                TextField("value", text: Binding(
                    get: { Self.escapedForDisplay(settings.value(for: "separator") ?? "") },
                    set: { model.setSetting(key: "separator", value: Self.unescapedFromDisplay($0), for: rowID) }
                ))
                .textFieldStyle(.roundedBorder)
                .font(.caption.monospaced())
            }
            Text("Shown escaped — \\n is a newline, \\t is a tab.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .padding(.leading, 96)
        }
    }

    /// The exact inverse pair `FlowSettingsEditor.quote`/`FlowSettings.unquote` already
    /// implement, reused rather than re-derived — stripped of (added back for) the surrounding
    /// quote marks those two assume, since a settings *value* (not a whole quoted token) has
    /// none. Not `private`: `CatFlowSettingsListEditorTests` asserts against these directly.
    static func escapedForDisplay(_ value: String) -> String {
        String(FlowSettingsEditor.quote(value).dropFirst().dropLast())
    }

    static func unescapedFromDisplay(_ text: String) -> String {
        FlowSettings.unquote("\"\(text)\"")
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
        // HU-3: lets the generic "Add a setting…" menu repair an imported flow that already
        // carries `timeout=`/`default=` (or add `wait=forever` by hand), on either edition —
        // the dedicated picker above is Direct-only, but this fallback text editor isn't.
        case "Ask Human", "Human Input": return ["wait", "timeout", "default"]
        case "Transcribe": return ["lang", "timestamps"]
        case "Split": return ["by", "chunk_size", "overlap"]
        case "Speak": return ["voice", "lang", "speed"]
        case "Retrieve": return ["query", "top_k"]
        case "Rerank": return ["query", "top_k"]
        // WS-1: ported verbatim from `net.py:318-330` so a `.cat` moves between runtimes
        // unchanged; `provider` is WS-2's own addition (an explicit override of the
        // Tavily-first selection), not part of the reference's settings surface.
        case "Web Search": return ["query", "top_k", "site", "recency", "timeout", "provider"]
        case "Filter": return ["by", "contains"]              // TextTools.swift:169
        case "Sort": return ["by", "reverse"]
        case "Dedupe": return ["by"]
        case "Extract": return ["by"]
        case "Count": return ["group_by"]
        case "Compare": return ["mode"]
        case "Template": return []
        // RT-4 (§4 Tier 2) — the rest of the keys the executors already read that
        // `knownSettingKeys` never offered, each cited at the line that reads it.
        case "Join Text": return ["separator"]                // TextTools.swift:418
        case "Text to Table": return ["expected"]             // TextToTableStage.swift:26
        case "Save Images": return ["naming"]                 // SaveImageTools.swift:95
        case "Web Fetch", "HTTP Get", "Fetch Feed", "Download File":
            return ["url"]                                    // NetTools.swift:220 (resolveURL)
        default:
            // The diffusion four — every `engines.diffusion.*` task shares
            // `RealExecutor.stageConfig`'s one `seed`/`width`/`height`/`steps` reader
            // (`RealExecutor.swift:918-921`), so this is keyed on the refName prefix rather
            // than naming each of the dozen-plus diffusion tasks by hand.
            if TaskCatalog.get(task)?.refName.hasPrefix("engines.diffusion.") == true {
                return ["seed", "width", "height", "steps"]
            }
            return []
        }
    }

    // MARK: - FILE-2: the File section (an in-flow list, a panel only for something new)

    /// Whether this row's task takes a path (a `Read *`/`Save *` of a file or folder).
    private func hasPathSetting(_ task: String) -> Bool {
        guard let desc = TaskCatalog.get(task) else { return false }
        switch desc.accepts {
        case .single(.file), .single(.folder): return true
        default: return false
        }
    }

    @State private var creatingFolder = false
    @State private var newFolderName = ""

    /// FILE-2 (owner-ruled 2026-09-11, widened 2026-09-12): the "In this flow" list is never
    /// shown *empty*, but it is no longer gated on "something already chosen" — it shows
    /// whenever there is something real to offer: a current pick (even a flagged one) **or**
    /// the flow folder already holding a matching candidate. A genuinely fresh row in a
    /// genuinely empty flow still shows nothing to list, just "Create a folder here" (+ "Add
    /// from my Mac…" for a `.file` task).
    ///
    /// **Why the widening:** the original two-state design left a real row-2 flow with no way
    /// to point a *second*, still-unset `Read Images` row at a folder `Read Images` row 1
    /// already created — "Create a folder here" only makes a *new* one, and there was no list
    /// to pick the existing one from until the row already had *a* path. Owner-reported
    /// 2026-09-12; fixed by widening the list's gate rather than adding a separate affordance.
    ///
    /// **A `.folder` task drops "Add from my Mac…" entirely** (owner, 2026-09-12): Create a
    /// folder here + the in-flow list's per-row Reveal in Finder already cover bringing
    /// content in (create, then drag files into the revealed folder), and the panel's own
    /// "browse an existing folder in from elsewhere" motion doesn't fit that model as
    /// cleanly. A `.file` task keeps the panel — there's no folder/create equivalent for a
    /// single new file, so removing it there would leave no way to add one.
    private func chooseFileButton(task: String) -> some View {
        let wantsFolder = (task == "Read Images" || task == "Read Files")
        let flowDir = model.workspace.directory(for: model.flowID)
        let currentToken = row.flatMap { FlowSettings($0.settings).pathValue() }
        let entries = Self.inFlowEntries(task: task, flowDir: flowDir, wantsFolder: wantsFolder)

        return VStack(alignment: .leading, spacing: 6) {
            Text("File")
                .font(.subheadline.weight(.semibold))

            if Self.shouldShowInFlowList(currentToken: currentToken, entries: entries) {
                inFlowList(entries: entries, currentToken: currentToken, flowDir: flowDir, task: task)
            }

            if wantsFolder {
                createFolderControl(flowDir: flowDir)
            } else {
                Button {
                    chooseFile(task: task)
                } label: {
                    Label(Self.addFromMacLabel(for: task), systemImage: "folder.badge.plus")
                }
                .buttonStyle(.plain)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    /// The "In this flow" box: one row per flat entry the task accepts, plus — per the
    /// owner's ruling on a stored path that isn't one of them (a nested path from before FILE-2,
    /// or a since-deleted one) — the current token shown anyway, flagged, never silently
    /// dropped. Never rendered with zero rows: the caller only shows it when `currentToken` is
    /// set (always ≥1 row, in the list or flagged) or `entries` is non-empty (≥1 real
    /// candidate, none marked current — a fresh row picking its first path from what's already
    /// there).
    ///
    /// Matching is **slash-insensitive** (`Self.tokensMatch`): the `Read Images`/`Read Files`
    /// sample seed (and any flow written before FILE-2) can carry a bare folder name with no
    /// trailing `/`, which is the same folder a listed entry names *with* one — comparing the
    /// raw strings would wrongly flag it as "not in this list" (owner-reported, 2026-09-11).
    private func inFlowList(entries: [FlowRowInspectorView.InFlowEntry], currentToken: String?,
                            flowDir: URL, task: String) -> some View {
        var rows = entries
        if let currentToken {
            let currentIsListed = entries.contains { Self.tokensMatch($0.token, currentToken) }
            if !currentIsListed {
                // Owner ruling, 2026-09-11 (open question 2): a stored path that isn't one of
                // the flat entries — nested from before FILE-2, or since-deleted — is still
                // shown, as its own row, flagged, never silently dropped.
                let status = Self.currentPickStatus(token: currentToken, entries: entries, flowDir: flowDir)
                let note = (status == .missing) ? "missing" : "not in this list"
                rows.insert(.init(token: currentToken, isDirectory: currentToken.hasSuffix("/"),
                                  count: nil, note: note), at: 0)
            }
        }

        return VStack(alignment: .leading, spacing: 4) {
            Text("In this flow")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(rows, id: \.token) { entry in
                inFlowRow(entry, isCurrent: currentToken.map { Self.tokensMatch(entry.token, $0) } ?? false,
                         flowDir: flowDir, task: task)
            }
        }
        .padding(8)
        .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 6))
    }

    /// One row of the "In this flow" list. The current pick is disabled — tapping it again is
    /// a no-op — everything else is a one-tap switch (no confirmation: it's a settings edit
    /// like any other in this pane, not a destructive one). Every **folder** row (owner,
    /// 2026-09-11) carries its own "Reveal in Finder" — the container is awkward to reach any
    /// other way, so every folder gets the same escape hatch a freshly-created one does, not
    /// only the current one when it happens to be empty.
    private func inFlowRow(_ entry: FlowRowInspectorView.InFlowEntry, isCurrent: Bool,
                           flowDir: URL, task: String) -> some View {
        HStack(spacing: 6) {
            Button {
                model.setPath(entry.token, for: rowID)
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isCurrent ? "largecircle.fill.circle" : "circle")
                        .foregroundStyle(isCurrent ? Color.accentColor : .secondary)
                    Text(entry.token)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let count = entry.count {
                        Text("\(count) \(Self.countNoun(task: task, count: count))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if let note = entry.note {
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
            }
            .buttonStyle(.plain)
            .disabled(isCurrent)
            Spacer()
            // A "missing" flagged row names nothing real to reveal; every other folder does.
            if Self.canReveal(entry) {
                Button("Reveal in Finder") {
                    NSWorkspace.shared.activateFileViewerSelecting([flowDir.appendingPathComponent(entry.token)])
                }
                .buttonStyle(.link)
                .font(.caption)
            }
        }
    }

    /// A directory token compares equal regardless of a trailing `/` — the `Read Images`/
    /// `Read Files` sample seed predates FILE-1's trailing-slash convention, and any flow
    /// written before FILE-2 could carry the same bare form. `nonisolated` so it's testable.
    nonisolated static func tokensMatch(_ a: String, _ b: String) -> Bool {
        func strip(_ s: String) -> String { s.hasSuffix("/") ? String(s.dropLast()) : s }
        return strip(a) == strip(b)
    }

    /// Whether an "In this flow" row gets a "Reveal in Finder" link (owner, 2026-09-11: every
    /// folder row, not only the current one when it's empty). A `"missing"`-flagged row names
    /// nothing real on disk to reveal.
    nonisolated static func canReveal(_ entry: InFlowEntry) -> Bool {
        entry.isDirectory && entry.note != "missing"
    }

    /// Whether the "In this flow" box renders at all. Widened 2026-09-12 (owner-reported gap):
    /// a fresh row — no `currentToken` yet — still shows the box when the flow folder already
    /// holds a real candidate, so a *second* `Read Images` row can pick the folder a first one
    /// already created instead of only being offered "Create a folder here" (which would make
    /// a colliding new one). A genuinely fresh row in a genuinely empty flow still shows
    /// nothing — the box is never rendered with zero rows either way.
    nonisolated static func shouldShowInFlowList(currentToken: String?, entries: [InFlowEntry]) -> Bool {
        currentToken != nil || !entries.isEmpty
    }

    /// "Create a folder here" (owner approved 2026-09-11): an empty subfolder in the flow,
    /// the row's path set to it with a trailing `/`, and a nudge to Reveal in Finder once it's
    /// picked (`inFlowList`'s empty-folder hint) — the container is awkward to reach by hand,
    /// so dragging files in via Finder is the honest way to fill it.
    private func createFolderControl(flowDir: URL) -> some View {
        Group {
            if creatingFolder {
                HStack(spacing: 6) {
                    TextField("Folder name", text: $newFolderName)
                        .textFieldStyle(.roundedBorder)
                        .font(.caption)
                        .onSubmit { createFolder(flowDir: flowDir) }
                    Button("Create") { createFolder(flowDir: flowDir) }
                        .disabled(newFolderName.trimmingCharacters(in: .whitespaces).isEmpty)
                    Button("Cancel") {
                        creatingFolder = false
                        newFolderName = ""
                    }
                }
                .font(.caption)
            } else {
                Button {
                    creatingFolder = true
                } label: {
                    Label("Create a folder here", systemImage: "folder.badge.plus")
                }
                .buttonStyle(.plain)
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
            }
        }
    }

    private func createFolder(flowDir: URL) {
        do {
            let token = try Self.createFolder(named: newFolderName, in: flowDir)
            model.setPath(token, for: rowID)
            creatingFolder = false
            newFolderName = ""
        } catch {
            model.saveError = (error as CustomStringConvertible).description
        }
    }

    /// R11-0b's rule: the chooser **copies in** — it stores a bare filename, never a
    /// security-scoped bookmark or an absolute path, so the flow folder stays a
    /// self-contained thing that runs after a zip-and-send.
    ///
    /// **FILE-2:** no `directoryURL` is set. FILE-1-FIX-1 confirmed a sandboxed `NSOpenPanel`
    /// cannot be pointed inside the app's container — it runs out of process in Powerbox
    /// (`com.apple.appkit.xpc.openAndSavePanelService`), which has no access to it. This panel
    /// is now only for bringing in something *new*, so AppKit's own default location (where the
    /// user's files actually are) is exactly right, unmodified.
    private func chooseFile(task: String) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = (task == "Read Images" || task == "Read Files")
        panel.allowsMultipleSelection = false
        if let types = Self.utTypes(for: task) {
            panel.allowedContentTypes = types
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

    /// A short button label naming the kind of thing the panel is for, per task.
    nonisolated static func addFromMacLabel(for task: String) -> String {
        switch task {
        case "Read Images": return "Add images from my Mac…"
        case "Read Files": return "Add files from my Mac…"
        case "Read Image": return "Add an image from my Mac…"
        case "Read Audio": return "Add an audio file from my Mac…"
        case "Read Text": return "Add a text file from my Mac…"
        case "Read PDF": return "Add a PDF from my Mac…"
        case "Read CSV": return "Add a CSV file from my Mac…"
        case "Read JSON": return "Add a JSON file from my Mac…"
        default: return "Add from my Mac…"
        }
    }

    private static func countNoun(task: String, count: Int) -> String {
        if task == "Read Images" { return count == 1 ? "image" : "images" }
        return count == 1 ? "file" : "files"
    }

    // MARK: - FILE-2: the flow-folder list (flat, task-filtered, no macOS involved)

    /// One entry the in-flow list offers: its flow-relative token — the exact one FILE-1's
    /// guard writes, trailing `/` on a directory — plus a count for directories and, only for
    /// the synthesized "current but not listed" row, a flagging note.
    nonisolated struct InFlowEntry: Equatable {
        let token: String
        let isDirectory: Bool
        let count: Int?
        var note: String?

        init(token: String, isDirectory: Bool, count: Int?, note: String? = nil) {
            self.token = token
            self.isDirectory = isDirectory
            self.count = count
            self.note = note
        }
    }

    /// Where a row's stored path stands relative to the flat list — resolves the owner's
    /// ruling that a stored path never goes unshown, even when it isn't one of the flat picks
    /// (a nested path from before FILE-2, or a folder/file since deleted).
    enum CurrentPickStatus: Equatable {
        case inList
        case notInList
        case missing
    }

    /// Names FILE-2 always skips regardless of task: the flow's own `.cat` (a flow folder
    /// holds exactly one), dotfiles/dirs (`.blobs`, `.trash`, `.improvise`, …), and the
    /// `uses:` snapshot directory.
    nonisolated static func isReservedFlowFolderName(_ name: String) -> Bool {
        name.hasPrefix(".") || name.hasSuffix(".cat") || name == "used"
    }

    /// FILE-2's in-flow list contents: directories for a `.folder` task (Read Images / Read
    /// Files), matching files for a `.file` task — **flat**, no recursion into subfolders
    /// (owner's ruling; `FlowWorkspace.resolve` supports nesting so this can grow later).
    /// `nonisolated static` so the filter is testable without a View.
    nonisolated static func inFlowEntries(task: String, flowDir: URL, wantsFolder: Bool) -> [InFlowEntry] {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: flowDir, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]
        ) else { return [] }
        let types = utTypes(for: task)
        var entries: [InFlowEntry] = []
        for url in contents {
            let name = url.lastPathComponent
            guard !isReservedFlowFolderName(name) else { continue }
            let isDir = (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
            if wantsFolder {
                guard isDir else { continue }
                entries.append(InFlowEntry(token: name + "/", isDirectory: true,
                                           count: countMatchingFiles(in: url, task: task)))
            } else {
                guard !isDir else { continue }
                if let types, !matches(url, types) { continue }
                entries.append(InFlowEntry(token: name, isDirectory: false, count: nil))
            }
        }
        return entries.sorted { $0.token.localizedStandardCompare($1.token) == .orderedAscending }
    }

    private nonisolated static func matches(_ url: URL, _ types: [UTType]) -> Bool {
        guard let fileType = UTType(filenameExtension: url.pathExtension) else { return false }
        return types.contains { fileType.conforms(to: $0) }
    }

    /// The count shown beside a directory entry — an estimate for display, not the row's own
    /// matching logic (a `pattern=` setting on the actual row isn't consulted here).
    private nonisolated static func countMatchingFiles(in folder: URL, task: String) -> Int {
        let fm = FileManager.default
        guard let contents = try? fm.contentsOfDirectory(
            at: folder, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles]
        ) else { return 0 }
        let files = contents.filter { !$0.hasDirectoryPath }
        guard task == "Read Images" else { return files.count }
        let imageExts: Set<String> = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"]
        return files.filter { imageExts.contains($0.pathExtension.lowercased()) }.count
    }

    nonisolated static func currentPickStatus(token: String, entries: [InFlowEntry],
                                              flowDir: URL) -> CurrentPickStatus {
        if entries.contains(where: { tokensMatch($0.token, token) }) { return .inList }
        let url = flowDir.appendingPathComponent(token)
        return FileManager.default.fileExists(atPath: url.path) ? .notInList : .missing
    }

    /// "Create a folder here": a fresh, empty subfolder in the flow, named by the user.
    /// `nonisolated static` so name validation + the create-vs-refuse decision are testable
    /// without a View. Refuses a name that collides, rather than silently disambiguating one.
    nonisolated static func createFolder(named rawName: String, in flowDir: URL) throws -> String {
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !name.contains("/"), name != ".", name != ".." else {
            throw FlowFolderCreateError.invalidName
        }
        let fm = FileManager.default
        try fm.createDirectory(at: flowDir, withIntermediateDirectories: true)
        let dest = flowDir.appendingPathComponent(name, isDirectory: true)
        guard !fm.fileExists(atPath: dest.path) else {
            throw FlowFolderCreateError.alreadyExists(name)
        }
        try fm.createDirectory(at: dest, withIntermediateDirectories: false)
        return name + "/"
    }

    // MARK: - RT-5: the no-dead-ends backstop

    /// The row's raw settings string, one monospaced `TextEditor`, committed on blur — not
    /// live per-keystroke like every other box here, because a commit re-validates the whole
    /// row, and validating mid-keystroke against an unterminated quote would refuse constantly
    /// while the user is still typing. Read-only when `!editable`: the outer `ScrollView`'s
    /// `.disabled(!editable)` already covers this (a disabled `TextEditor` can't be typed into
    /// or focused, so the commit-on-blur path simply never fires). Hidden entirely when
    /// `isFrozen` (the call site, `body`).
    private func rowTextDisclosure(_ row: Row) -> some View {
        DisclosureGroup("Row text") {
            VStack(alignment: .leading, spacing: 4) {
                TextEditor(text: Binding(
                    get: { rowTextDraft ?? row.settings ?? "" },
                    set: { rowTextDraft = $0 }
                ))
                .font(.caption.monospaced())
                .frame(minHeight: 60, maxHeight: 160)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
                .focused($rowTextFocused)
                .onChange(of: rowTextFocused) { _, isFocused in
                    if !isFocused { commitRowText(row) }
                }
                if let rowTextRefusal {
                    // §8 copy, verbatim — not invented at the call site.
                    Label("That change wouldn't check: \(rowTextRefusal). The row's text is unchanged until it does.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                }
            }
            .padding(.top, 4)
        }
        .font(.caption)
    }

    /// A refusal leaves `rowTextDraft` exactly as typed ("left dirty" — never silently
    /// dropped back to the last-committed value); acceptance clears it, so the box falls back
    /// to reading `row.settings` (now equal to the draft) directly.
    private func commitRowText(_ row: Row) {
        guard let draft = rowTextDraft else { return }
        if let refusal = model.setRowText(draft, for: rowID) {
            rowTextRefusal = refusal
        } else {
            rowTextRefusal = nil
            rowTextDraft = nil
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

    private func isHumanClass(_ task: String) -> Bool {
        TaskCatalog.get(task)?.taskClass == .human
    }

    // MARK: - HU-2: "If nobody answers"

    /// A human row's waiting policy, above the generic Settings section. The App Store build
    /// (Ruling 3, 2026-09-11: "Wait-forever only in App Store") gets a single read-only line —
    /// there is nothing to author, but an imported flow's own `timeout=`/`default=` still
    /// displays honestly (`FlowEditorModel.waitPolicyDescription`, HU-3). The Direct build gets
    /// a picker between parking forever and giving up after a duration.
    private func humanWaitPolicySection(_ task: String, row: Row) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("If nobody answers")
                .font(.subheadline.weight(.semibold))
            if CapabilityGate.isAppStoreBuild {
                Text(FlowEditorModel.waitPolicyDescription(settings: row.settings))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                directHumanWaitPolicyControl(task, row: row)
            }
        }
    }

    /// Direct-build-only: a segmented picker between the two policies, mutually exclusive
    /// (`FlowEditorModel.setHumanWaitPolicy` clears the other in the same commit). Switching
    /// to "Give up after…" reveals a duration field and, for `Ask Human`, a **picker over the
    /// row's declared tags** (never free text — a typed default outside the tag set is exactly
    /// what E502 exists to catch). `Human Input` doesn't branch, so the spec fixes its default
    /// to `unchanged` (§10.1) and this shows it as a fixed label, not a field.
    private func directHumanWaitPolicyControl(_ task: String, row: Row) -> some View {
        let settings = FlowSettings(row.settings)
        let isWaitForever = settings.value(for: "wait") == "forever"
        let tags = row.tags ?? []
        return VStack(alignment: .leading, spacing: 6) {
            Picker("", selection: Binding<Bool>(
                get: { isWaitForever },
                set: { waitForever in
                    if waitForever {
                        model.setHumanWaitPolicy(.waitForever, for: rowID)
                    } else {
                        let dflt = (task == "Human Input") ? "unchanged" : (tags.first ?? "")
                        model.setHumanWaitPolicy(
                            .giveUpAfter(timeout: settings.value(for: "timeout") ?? "", default: dflt),
                            for: rowID)
                    }
                }
            )) {
                Text("Wait until answered").tag(true)
                Text("Give up after…").tag(false)
            }
            .pickerStyle(.segmented)
            .labelsHidden()

            if !isWaitForever {
                HStack(spacing: 6) {
                    Text("timeout")
                        .font(.caption.monospaced())
                        .frame(width: 90, alignment: .trailing)
                    TextField("e.g. 10m", text: Binding(
                        get: { settings.value(for: "timeout") ?? "" },
                        set: { model.setSetting(key: "timeout", value: $0, for: rowID) }
                    ))
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                }
                HStack(spacing: 6) {
                    Text("default")
                        .font(.caption.monospaced())
                        .frame(width: 90, alignment: .trailing)
                    if task == "Human Input" {
                        Text("unchanged")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("", selection: Binding(
                            get: { settings.value(for: "default") ?? tags.first ?? "" },
                            set: { model.setSetting(key: "default", value: $0, for: rowID) }
                        )) {
                            ForEach(tags, id: \.self) { tag in
                                Text(tag).tag(tag)
                            }
                        }
                        .labelsHidden()
                        .disabled(tags.isEmpty)
                    }
                }
            }
        }
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
        /// RT-1 — `Template`'s whole settings string *is* its pattern (fact 11), the same
        /// "bare quoted span, no `key=`" shape as `schemaEditor`, with its own multi-line box
        /// and placeholder chips instead of a column-list label.
        case patternEditor
        /// RT-3 (§3 rule 4) — a short, structured authored value that isn't prose: `Compare`'s
        /// criterion (`"> 50"`) today. Same first-quoted-token read/write as `.instructionBox`,
        /// but a true single-line field — a multi-line box would misrepresent the value's type.
        case singleLineField(label: String)
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
        if desc.refName == "tools.text.template" { return .patternEditor }             // RT-1
        // RT-2 — the question a person is actually asked at run time. Written with the same
        // `setInstruction` (first quoted token) as every other `.instructionBox`: these rows
        // also carry `timeout=`/`default=`/`wait=`, which the splice already leaves alone.
        if desc.refName == "tools.human.ask_human" { return .instructionBox }
        if desc.refName == "tools.human.human_input" { return .instructionBox }
        // RT-3 — the diffusion prompt, read the same way `RealExecutor.runModel` reads it
        // (`RealExecutor.swift:469-479`, `firstBare()`) when the row has no input: the same
        // `engines.diffusion.generate_*` prefix, so a future fourth generator inherits the
        // box the day it lands, matching that call site instead of re-deriving its own list.
        if desc.refName.hasPrefix("engines.diffusion.generate_") { return .instructionBox }
        if desc.refName == "tools.compare.compare" { return .singleLineField(label: "Criterion") }  // RT-3
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
              case let .runnable(slot, _, _) = CatalogBridge.resolve(display, catalog: catalog),
              let model = slot.modelEntry
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

    /// RT-4 (fact 23, Tier-2 "key/bare shadowing") — `Web Search`/`Rerank` read
    /// `value(for: "query") ?? firstBare()` (`Tools/WebSearchTool.swift:154`,
    /// `RealExecutor.swift`'s `engines.rerank.` branch): when a row carries **both**, the key
    /// silently wins and the pane said nothing. Same amber `Label` shape `strandedPromptWarning`
    /// uses. Copy from §8, not invented here.
    ///
    /// Fact 23 names `Retrieve` too, but `RetrieveTool.run` (`Tools/IndexStoreTools.swift:285`)
    /// never reads a `query` setting at all — its query is the bound **vector** input, not
    /// text. `knownSettingKeys("Retrieve")` already (pre-RT-4) offers `query` in the Add menu
    /// regardless; that's a separate, pre-existing loose end this phase doesn't touch. Scoped
    /// here to the two tasks actually verified to read it this way.
    private func shadowingWarning(_ row: Row) -> String? {
        guard let task = row.task, task == "Web Search" || task == "Rerank"
        else { return nil }
        let settings = FlowSettings(row.settings)
        guard settings.value(for: "query") != nil,
              let bare = settings.firstBare(), !bare.isEmpty
        else { return nil }
        return Self.shadowingMessage(key: "query")
    }

    /// The pure decision behind `shadowingWarning` — `nonisolated static` so it's
    /// unit-testable without a View, same reasoning as `strandedPromptMessage`.
    nonisolated static func shadowingMessage(key: String) -> String {
        "This row sets both \(key)= and a bare value. \(key)= is the one that runs."
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
