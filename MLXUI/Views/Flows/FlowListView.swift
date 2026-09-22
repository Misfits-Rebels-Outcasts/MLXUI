import SwiftUI
import AppKit

/// The flow detail view. R1 shipped it read-only; CFM-R2-8 wires Run: preflight → one
/// install sheet → the `FlowEvent` stream driving each row's dot `○ → ● → ✓`, error
/// sentences on `✗`, and a Cancel that keeps earned dots. Run is disabled with the reason
/// when `FlowRunner.canRun` says no, and blocked models disable it with the bridge's reason.
struct FlowListView: View {
    /// CFM-R12-1: where a flow comes from — the bundled gallery or the user's own shelf.
    enum Source: Equatable {
        case gallery
        case user
    }

    let flowID: String
    let source: Source
    /// CFM-R17-3: non-nil when this flow lives inside a workspace — its `.cat` paths resolve
    /// against the shared workspace directory, and it runs under a workspace-rooted scope.
    let workspaceRef: WorkspaceRef?

    /// CFM-R17-4: run the flow once as soon as it loads (the workspace index card's Build /
    /// Ask verbs).
    let autoRun: Bool
    @State private var didAutoRun = false
    /// WR-4 (`RSI/DelegateWorkspaceRunBacklog.md`, owner gate Q1, ruled (a) 2026-09-22): true
    /// exactly while this flow's *own* auto-run is the reason the install sheet is up — set
    /// right before the auto-run's `run(doc)` call raises it. Never set by a manual "Install
    /// Required Models" press, so `AutoRunResume.shouldRun` can never fire for one.
    /// Deliberately **not** `didAutoRun` reset — resetting that would re-arm the auto-run on
    /// Cancel and on every later, unrelated install too. Usually cleared the moment the sheet
    /// closes, but not always immediately on a successful install: finding B (owner review,
    /// 2026-09-22) — `session.canRun` can lag one `.onChange(of: installedModelIDs)` pass
    /// behind the install actually finishing, so this can stay armed for exactly one extra
    /// pass (`retryArmedAutoRunOnce()`) before giving up for good.
    @State private var autoRunAwaitingInstall = false

    /// A user flow has no bundled assets, so `prepare` is called with an empty list (it is
    /// already idempotent). A gallery flow's input assets come from the flattened bundle.
    init(flowID: String, source: Source = .gallery, workspace: WorkspaceRef? = nil,
         autoRun: Bool = false) {
        self.flowID = flowID
        self.source = source
        self.workspaceRef = workspace
        self.autoRun = autoRun
    }

    /// The workspace a flow's paths resolve against — the shared workspace directory for a
    /// workspace flow, `flows/` otherwise.
    private var flowWorkspace: FlowWorkspace { workspaceRef?.workspace ?? .shared }
    /// The id `flowWorkspace.directory(for:)` / `resolve` keys on — the workspace id for a
    /// workspace flow, the flow id otherwise.
    private var locationID: String { workspaceRef?.workspaceID ?? flowID }
    /// The run/edit scope: a workspace flow keeps its own identity but resolves in the shared
    /// directory (CFM-R17-1); a plain flow is `.plain(flowID)`.
    private func makeScope() -> FlowScope {
        workspaceRef?.scope(text: rawText) ?? .plain(flowID)
    }
    /// The flow's own `.cat` text, read at load — feeds the run seed for a workspace flow.
    @State private var rawText: String?

    @Environment(AppState.self) private var appState
    @State private var metadata: GalleryFlowMetadata?
    /// CFM-R12-1: the user-flow shelf entry (nil for a bundled flow).
    @State private var userEntry: UserFlowStore.Entry?
    /// WR-1: set once `loadWorkspaceFlow()` succeeds — mirrors `metadata`/`userEntry`'s
    /// "written on load, not at init" shape, so `display` reflects a completed load the same
    /// way for all three sources. (`workspaceRef` itself is available from init and would
    /// resolve early; this keeps `display`'s precondition uniform.)
    @State private var workspaceDisplayRef: WorkspaceRef?
    @State private var document: FlowDocument?
    @State private var loadError: String?
    /// A read-only `FlowEditorModel` over the loaded document — the inspector's frozen
    /// Properties tab is browse-only, but needs the same candidate/input/number resolution
    /// the editor uses. Never mutated (the flow list has no edit affordance).
    @State private var inspectModel: FlowEditorModel?
    /// The refusal reason this flow can't run, derived from `FlowRunner.canRun(doc)` — not
    /// blindly trusted from `_metadata.json` (B3). `nil` = runnable.
    @State private var notRunnableReason: String?
    /// MS-4 — the fix-it action paired with `notRunnableReason`, when the refusal has one
    /// (always `nil` today; Phase AFM/RM/WS are the first to produce one).
    @State private var notRunnableAction: SetupAction?
    /// FIX-2 — a non-`.model` row's setup advisory (e.g. a keyless `Web Search` row), or nil.
    /// Never blocks the run the way `notRunnableReason` does — it renders as a banner
    /// alongside a still-enabled Run (§2 of `RSI/DelegateFixItBacklog.md`).
    @State private var setupAdvisory: FlowPreflight.RowAdvisory?
    /// FIP-2 — a `Read *` row whose file isn't there yet, or nil. Same non-blocking shape and
    /// rendering as `setupAdvisory` (owner ruling: warn only, never block).
    @State private var inputAdvisory: FlowPreflight.RowAdvisory?
    /// The canonical serialized lines (CFM-R6-2) — the flow list *is* the file. Computed once
    /// in `load()`; `lineRanges` maps each row id to its lines' range in `serializedLines`.
    @State private var serializedLines: [String] = []
    @State private var lineRanges: [UUID: Range<Int>] = [:]
    /// QR12R2-1: a block's clause line (emitted after its children) — drawn after the header.
    @State private var clauseLineRanges: [UUID: Range<Int>] = [:]
    /// R7-5: block rows whose children are collapsed (start expanded).
    @State private var collapsedBlocks: Set<UUID> = []
    @State private var session = FlowRunSession()

    var body: some View {
        Group {
            if let error = loadError {
                ContentUnavailableView("Couldn't Load This Flow",
                                       systemImage: "exclamationmark.triangle",
                                       description: Text(error))
            } else if let display, let document {
                if let reason = notRunnableReason {
                    notRunnableView(display, doc: document, reason: reason, action: notRunnableAction)
                } else {
                    flowList(document, display: display)
                }
            } else {
                ProgressView("Loading flow…")
                    .onAppear(perform: load)
            }
        }
        .navigationTitle(display?.title ?? flowID)
        // `load()` runs only from the loading placeholder's `.onAppear`, which never
        // fires again when the editor is popped off the navigation stack. Returning
        // from an edit of *this* flow (a rename, or any row change) would otherwise
        // leave the page showing the pre-edit title, serialized rows and refusal.
        .onChange(of: appState.editingFlow) { old, new in
            guard new == nil, old?.flowID == flowID else { return }
            reload()
        }
        // WR-3 (`RSI/DelegateWorkspaceRunBacklog.md`): a re-opened view's refusal AND
        // preflight are both stale if an install finished while away — moved onto the outer
        // `Group` (was inside `flowList` alone) so it fires on the placeholder and on
        // `notRunnableView` too, not only on an already-runnable flow's row list (smoke-30:
        // "Install Required Models" reappearing after the install completed; and the owner's
        // report that a refusal never clears without navigating away and back).
        .onChange(of: appState.installedModelIDs) { _, _ in
            reassess()
            // WR-4 finding B: this is also the one grace pass a still-armed auto-run gets if
            // `session.canRun` hadn't caught up to a genuinely successful install yet.
            retryArmedAutoRunOnce()
        }
        .sheet(isPresented: $session.showInstallSheet) {
            if let result = session.preflight {
                installSheet(result)
                    .environment(appState)
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
            if let parked = session.parked, let document {
                FlowHumanPromptView(
                    parked: parked,
                    row: Self.row(parked.rowID, in: document),
                    doc: document,
                    session: session,
                    runner: FlowRunner(),
                    context: AppFlowExecutorFactory.cachingContext(scope: makeScope(), appState: appState, transforms: document.transforms))
            }
        }
    }

    /// A read-only view for a flow this version can't run (CFM-R4-4): an honest badge
    /// naming what it needs, the description, and the canonical serialized lines (the list
    /// is the file — R6-2/3, no raw-`.cat` disclosure in the gallery view). No Run button.
    private func notRunnableView(_ display: FlowDisplay, doc: FlowDocument, reason: String,
                                 action: SetupAction? = nil) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Label(display.title, systemImage: "flowchart")
                    .font(.title2.weight(.semibold))
                Text("\(doc.headerKeyword) \(doc.version)")
                    .font(.caption.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
                Spacer()
            }
            HStack(alignment: .top, spacing: 8) {
                Label(reason, systemImage: "exclamationmark.triangle.fill")
                    .font(.callout)
                    .foregroundStyle(.orange)
                    .frame(maxWidth: .infinity, alignment: .leading)
                // MS-4 — the owner's fifth ask: a button next to the refusal, when it's
                // fixable. Always absent today (no `SetupAction` is produced yet); Phase
                // AFM/KEY/RM/WS give it something real to open.
                if let action {
                    setupActionButton(action)
                        .buttonStyle(.bordered)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
            Text(display.description ?? "")
                .font(.callout)
                .foregroundStyle(.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                Text(serializedLines.joined(separator: "\n"))
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    /// MS-4 — the button label for a `SetupAction`. Plain and generic on purpose: nothing
    /// produces one of these yet, so this exists to compile and be fixture-tested, not to
    /// anticipate exact copy AFM/KEY/RM/WS will each want for their own case.
    private func setupActionButtonLabel(_ action: SetupAction) -> String {
        switch action {
        case .openSettings, .enableAppleIntelligence: return "Open Settings…"
        case .installModel: return "Install…"
        }
    }

    /// SET-2 (D3, `RSI/DelegateSettingsBacklog.md`) — this is the phase `SetupAction` was
    /// waiting for: `.openSettings(pane)` now lands on the pane it actually names, via
    /// `SettingsOpener`, instead of Settings' first tab regardless of what fixed the problem.
    /// `settingsPane(for:)` is the pure routing decision (FlowKit's G2 seam); a `nil` pane
    /// (only `.installModel`, never produced today — MS-4) falls back to a plain
    /// `SettingsLink`, same as every case did before this phase.
    @ViewBuilder
    private func setupActionButton(_ action: SetupAction) -> some View {
        if let pane = settingsPane(for: action) {
            SettingsOpener(pane: pane) { Text(setupActionButtonLabel(action)) }
        } else {
            SettingsLink { Text(setupActionButtonLabel(action)) }
        }
    }

    /// Shared rendering for `setupAdvisory` and `inputAdvisory` — same non-blocking shape,
    /// same fix-it button style when the advisory carries one (a missing-input advisory
    /// never does; there's no settings pane that creates a file).
    private func advisoryBanner(_ advisory: FlowPreflight.RowAdvisory) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Label(advisory.reason, systemImage: "exclamationmark.circle")
                .font(.callout)
                .foregroundStyle(.blue)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let action = advisory.action {
                setupActionButton(action)
                    .buttonStyle(.bordered)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.blue.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }

    // MARK: - Content

    private func flowList(_ doc: FlowDocument, display: FlowDisplay) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            header(doc, display: display)
            Divider()
            // FIX-2 — a non-blocking advisory (e.g. no Tavily/Brave key set): the flow stays
            // runnable and this is not `notRunnableView`'s red refusal, so it renders inline,
            // above the row list, with the same fix-it button style.
            if let advisory = setupAdvisory {
                advisoryBanner(advisory)
            }
            // FIP-2 — same shape, for a missing `Read *` input. Owner ruling: warn only, so
            // this never disables Run below.
            if let advisory = inputAdvisory {
                advisoryBanner(advisory)
            }
            if let sentence = session.errorSentence {
                HStack(alignment: .top, spacing: 8) {
                    Label(sentence, systemImage: "exclamationmark.triangle.fill")
                        .font(.callout)
                        .foregroundStyle(.red)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    copyButton(sentence)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.red.opacity(0.08), in: RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
            // DA-10-FIX-1: an `<each on_error=skip>` that dropped rows — surfaced up here so the
            // reason isn't only an inline caption under the (indented) block-child row.
            if let skip = session.skipSummary {
                HStack(alignment: .top, spacing: 8) {
                    Label(skip, systemImage: "exclamationmark.triangle")
                        .font(.callout)
                        .foregroundStyle(.orange)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    copyButton(skip)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.orange.opacity(0.09), in: RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            }
            HStack(spacing: 0) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        // CFM-R12-3: one entry per row — block children get their own status
                        // dot, selection, and error sentence; a collapsed block hides its
                        // children (its header dot aggregates them so a red can't hide).
                        ForEach(FlowRowFlatten.flatten(doc.rows, collapsed: collapsedBlocks)) { display in
                            let row = display.row
                            if FlowRowSummary.hasChainBreakBefore(row) {
                                Divider()
                                    .padding(.leading, 48)
                            }
                            if let range = lineRanges[row.id] {
                                let isBlock = row.blockKind != nil
                                let isCollapsed = isBlock && collapsedBlocks.contains(row.id)
                                rowView(row: row, range: range, isBlock: isBlock, isCollapsed: isCollapsed)
                                    .padding(.leading, CGFloat(display.depth) * 20)
                                    .background(session.selectedRowID == row.id ? Color.accentColor.opacity(0.12) : Color.clear)
                            }
                            if let sentence = session.errorSentence(for: row.id) {
                                // DA-10: a skipped row's sentence is a skip reason, not a
                                // failure — orange "Skipped —", never the red failure style.
                                let skipped = session.wasSkipped(row.id)
                                HStack(alignment: .top, spacing: 6) {
                                    Text(skipped ? "Skipped — \(sentence)" : sentence)
                                        .font(.caption)
                                        .foregroundStyle(skipped ? .orange : .red)
                                        .textSelection(.enabled)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                    copyButton(sentence)
                                }
                                .padding(.leading, 48 + CGFloat(display.depth) * 20)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                    .padding(12)
                }
                Divider()
                FlowInspectorPane(
                    output: session.selectedRowID.flatMap { session.outputs[$0] },
                    rowTitle: selectedRowTitle(doc),
                    substitutionNote: session.selectedRowID.flatMap { session.substitutionNotes[$0] },
                    statusNote: session.selectedRowID.flatMap { session.statusNote(for: $0) },
                    savedFile: savedFileURL(in: doc),
                    savedKind: savedFileKind(in: doc),
                    initialTab: .output
                ) {
                    // The Properties tab is browse-only here: bundled gallery flows are
                    // frozen (dimmed + a lock), a user's own flow reads at full opacity —
                    // either way the values are never mutable until opened in the editor.
                    if let inspectModel {
                        FlowRowInspectorView(model: inspectModel,
                                             rowID: session.selectedRowID ?? UUID(),
                                             catalog: appState.browserData?.domains.flatMap { $0.allModels } ?? [],
                                             totalRAMGB: appState.systemInfo.totalRAMGB,
                                             installedModelIDs: appState.installedModelIDs,
                                             claimableModelIDs: appState.claimableModelIDs,
                                             resolvePromptSupport: { appState.registry.bestModule(for: $0)?.sdk.promptSupport ?? .none },
                                             editable: false,
                                             isFrozen: source == .gallery)
                    }
                }
            }
            // CFM-R11-2 visibility: the flow's cache state + warm engines, so "is there a
            // cache?" is answerable at a glance instead of a guess.
            if cacheStatus != nil {
                Text(cacheStatus ?? "")
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .padding(.horizontal, 16)
                    .padding(.top, 2)
            }
            // CFM-R11-1: the run's peak-memory record — a real cold run reads this into the
            // journal to judge the preflight's largest-single-row RAM rule. Shown after any
            // run (even one with no GPU activity, so a 0 is visible, not silent).
            if !session.isRunning && !session.metrics.rowSamples.isEmpty {
                Text(session.metrics.summary())
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .trailing)
                    .padding(.horizontal, 16)
                    .padding(.bottom, 6)
            }
            // CFM-R12-8: the flow's outbox — staged effects are meant to be read by a person
            // before they go anywhere (nothing in the app can send one).
            outboxDisclosure
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        .confirmationDialog("Remove this flow?", isPresented: $flowPendingRemoval,
                            titleVisibility: .visible) {
            Button("Remove", role: .destructive) {
                appState.removeUserFlow(flowID: flowID)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("'\(display.title)' and its files will be deleted from your flows folder. This can't be undone.")
        }
    }

    /// Whether the user confirmed they want to delete this flow (user flows only).
    @State private var flowPendingRemoval = false

    private func selectedRowTitle(_ doc: FlowDocument) -> String {
        guard let id = session.selectedRowID,
              let row = doc.rows.first(where: { $0.id == id }) else { return "" }
        return FlowRowSummary.taskName(for: row)
    }

    /// CFM-R12-8: the flow's pending staged effects. Nothing sends — the whole point is that
    /// a person reads each entry before it goes anywhere (which in this version is nowhere).
    private var outboxDisclosure: some View {
        let entries = OutboxStore.entries(workspace: flowWorkspace, flowID: locationID)
        return DisclosureGroup {
            if entries.isEmpty {
                Text("No pending entries — a Stage Send / Stage Post row queues one here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(entries) { entry in
                    VStack(alignment: .leading, spacing: 2) {
                        HStack {
                            Text("\(entry.kind) → \(entry.destination)")
                                .font(.caption.monospaced().weight(.semibold))
                            Spacer()
                            Text(entry.id)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        Text(entry.text)
                            .font(.caption)
                            .textSelection(.enabled)
                            .lineLimit(3)
                        Text("staged \(entry.stagedAt)")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            }
        } label: {
            Label("Outbox (\(entries.count))", systemImage: "tray.full")
                .font(.subheadline)
        }
        .padding(.horizontal, 16)
        .padding(.bottom, 8)
    }

    /// The file a selected `Save *` row wrote, resolved against the flow's folder — lets the
    /// inspector play/view the saved result instead of just its status sentence.
    private func savedFileURL(in doc: FlowDocument) -> URL? {
        guard let id = session.selectedRowID,
              let row = doc.rows.first(where: { $0.id == id }) else { return nil }
        return FlowSavedFile.resolved(row: row, flowID: locationID, workspace: flowWorkspace)
    }

    /// The kind of the saved file (drives how the inspector presents it).
    private func savedFileKind(in doc: FlowDocument) -> Kind? {
        guard let id = session.selectedRowID,
              let row = doc.rows.first(where: { $0.id == id }) else { return nil }
        return FlowSavedFile.kind(forTask: row.task)
    }

    /// One row's `FlowSerializedRow` — extracted so the row builder stays type-checkable.
    private func rowView(row: Row, range: Range<Int>, isBlock: Bool, isCollapsed: Bool) -> some View {
        var shownLines: [String]
        if isCollapsed {
            shownLines = Array(serializedLines[range.lowerBound..<min(range.lowerBound + 1, range.upperBound)])
        } else {
            shownLines = Array(serializedLines[range])
        }
        // QR12R2-1: a block's clause line is part of its rendered lines (e.g. `-> 7`).
        if let clauseRange = clauseLineRanges[row.id] {
            shownLines.append(contentsOf: serializedLines[clauseRange])
        }
        return FlowSerializedRow(lines: shownLines,
                                 status: statusForDisplay(row, isCollapsed: isCollapsed),
                                 onSelect: { session.selectedRowID = row.id },
                                 isSemanticParallel: row.blockKind == .parallel,
                                 isCollapsible: isBlock,
                                 isCollapsed: isCollapsed,
                                 onToggleCollapse: {
                                     if collapsedBlocks.contains(row.id) {
                                         collapsedBlocks.remove(row.id)
                                     } else {
                                         collapsedBlocks.insert(row.id)
                                     }
                                 },
                                 cached: session.cacheHitRows.contains(row.id))
    }

    /// CFM-R12-3 item 3 / CFM-R12-FIX-3: a collapsed block's header dot aggregates its
    /// descendants recursively, so a red dot at any depth can never be hidden by collapsing.
    private func statusForDisplay(_ row: Row, isCollapsed: Bool) -> FlowStatus {
        let own = session.status(for: row.id)
        guard row.blockKind != nil, isCollapsed else { return own }
        return Self.aggregateDescendantStatus(of: row, session: session, fallback: own)
    }

    private static func aggregateDescendantStatus(of row: Row, session: FlowRunSession,
                                                              fallback: FlowStatus) -> FlowStatus {
        var worst = fallback
        for child in row.children {
            let childStatus = session.status(for: child.id)
            if childStatus == .failed { return .failed }
            if childStatus == .running { worst = .running }
            if childStatus == .succeeded, worst != .running { worst = .succeeded }
            if childStatus == .needsAttention, worst == .notRun { worst = .needsAttention }
            if !child.children.isEmpty {
                worst = aggregateDescendantStatus(of: child, session: session, fallback: worst)
                if worst == .failed { return .failed }
            }
        }
        return worst
    }

    /// The flow cache + engine-cache status line (R11-2 visibility): nil = nothing to say.
    /// A `✓ from cache` row marker plus this line answer "is there a cache?" at a glance.
    private var cacheStatus: String? {
        let store = FlowCacheStore.shared
        var parts: [String] = []
        let entryCount = store.entryCount
        if entryCount > 0 {
            let bytes = mb(store.totalBytes)
            parts.append("cache \(entryCount) output\(entryCount == 1 ? "" : "s") (\(bytes))")
        } else {
            parts.append("cache empty")
        }
        let engineCount = EngineCache.shared.count
        if engineCount > 0 {
            let bytes = mb(EngineCache.shared.totalCachedBytes)
            parts.append("\(engineCount) engine\(engineCount == 1 ? "" : "s") warm (\(bytes))")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    private func mb(_ bytes: Int64) -> String {
        let value = String(format: "%.1f", Double(bytes) / 1_048_576)
        return "\(value) MB"
    }

    /// CFM-R11-0: copy this bundled flow into the user's flow folder and open the editor on
    /// the copy. On failure the flow stays read-only with a plain sentence.
    private func duplicateAndEdit(_ display: FlowDisplay, _ doc: FlowDocument) {
        do {
            let target = try FlowEditRoute.duplicateAndEdit(
                flowID: flowID, title: display.title, document: doc,
                workspace: FlowWorkspace.shared,
                sourceDir: GalleryLoader.resourcesDirectory ?? Bundle.main.resourceURL ?? .init(fileURLWithPath: "/"))
            appState.editingFlow = target
        } catch {
            appState.openCatFlowError = "Couldn't copy '\(display.title)' into your flows folder — the flow stays read-only."
        }
    }

    /// A plain copy button for an error sentence (top banner and per-row). Copying the
    /// exact string is how a failure gets reported verbatim.
    private func copyButton(_ text: String) -> some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        } label: {
            Image(systemName: "doc.on.doc")
                .foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .help("Copy error text")
    }

    private func header(_ doc: FlowDocument, display: FlowDisplay) -> some View {
        HStack(spacing: 12) {
            Text(display.title)
                .font(.title2.weight(.semibold))
            Text("\(doc.headerKeyword) \(doc.version)")
                .font(.caption.monospaced())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
            // CFM-R12-9: a flow that declares `network` says so before it runs.
            if doc.flags.contains(.network) {
                Label("network", systemImage: "network")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary.opacity(0.5), in: Capsule())
                    .help("This flow fetches URLs over the network when it runs.")
            }
            Spacer()
            // CFM-R12-1: bundled flows are read-only — "Duplicate & Edit" copies the flow
            // into the user's flow folder. A user flow is already the user's: "Edit" opens
            // it in the editor in place.
            if source == .gallery {
                Button {
                    duplicateAndEdit(display, doc)
                } label: {
                    Label("Duplicate & Edit", systemImage: "square.and.pencil")
                }
                .help("Copy this flow into your flows folder and open it in the editor")
                .disabled(false) //comeback cbx
            } else {
                Button {
                    appState.editingFlow = FlowEditTarget(flowID: flowID, name: display.title,
                                                          document: doc,
                                                          savedText: CatSerializer.serialize(doc))
                } label: {
                    Label("Edit", systemImage: "square.and.pencil")
                }
                .help("Open this flow in the editor")
            }
            Button {
                try? flowWorkspace.prepare(
                    flowID: locationID,
                    sourceDir: source == .gallery ? GalleryLoader.resourcesDirectory : nil,
                    bundledAssets: source == .gallery ? GalleryLoader.bundledAssets(flowID: flowID) : [])
                flowWorkspace.revealInFinder(flowID: locationID)
            } label: {
                Label("Reveal in Finder", systemImage: "folder")
            }
            if appState.installingFlowIDs.contains(flowID) {
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
                    session.showInstallSheet = true
                } label: {
                    Label("Install Required Models", systemImage: "arrow.down.circle")
                }
            }
            if session.isRunning {
                Button {
                    session.cancel()
                } label: {
                    Label("Cancel", systemImage: "stop.fill")
                }
            } else {
                Button {
                    run(doc)
                } label: {
                    Label("Run", systemImage: "play.fill")
                }
                .buttonStyle(.borderedProminent)
                .disabled(!session.canRun)
                .help(session.runDisabledReason ?? "Run the flow (or re-run from the first gray row)")
            }
            // CFM-R10-Events: a trigger flow can be armed — the app watches the folder /
            // waits for the schedule / hooks the named flow, then fires with an occurrence.
            // FIX-1: a trigger flow carrying a door (§14.4) cannot arm (never unattended).
            if armSession.isTriggerFlow {
                if armSession.isArmed {
                    Button {
                        armSession.disarm()
                    } label: {
                        Label("Disarm", systemImage: "stop.circle")
                    }
                    .help(armSession.armedDescription ?? "Armed")
                } else {
                    Button {
                        armSession.arm(flowID: locationID, doc: doc,
                                       workspace: flowWorkspace,
                                       onFire: { occurrence in
                                           DispatchQueue.main.async { run(doc, occurrence: occurrence) }
                                       })
                    } label: {
                        Label("Arm", systemImage: "bell")
                    }
                    .disabled(!armSession.isArmable)
                    .help(armSession.armRefusal ?? armSession.armedDescription ?? "Arm the flow's trigger")
                }
                if let armed = armSession.armedDescription, armSession.isArmed {
                    Label(armed, systemImage: "bell.fill")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            // CACHE-Q: the ⋯ menu is shared with `FlowEditorView` (a working My Workflows flow
            // opens there, not here). FlowListView appends "Remove Flow…" for a user flow.
            FlowMaintenanceMenu(session: session, doc: doc, workspace: flowWorkspace,
                                flowID: locationID,
                                onNotice: { cacheClearNotice = $0 }) {
                // CFM-R12-1: a user's own flow can be removed from disk — the bundled gallery
                // flows never can. Disabled mid-run so a deleting folder never breaks a run.
                if source == .user {
                    Divider()
                    Button("Remove Flow…", systemImage: "trash", role: .destructive) {
                        flowPendingRemoval = true
                    }
                    .disabled(session.isRunning)
                }
            }
        }
        .padding(16)
    }

    /// The Clear Cache & Results / Undo Improvise result sentence. Set by `FlowMaintenanceMenu`
    /// via `onNotice`; write-only here today, as it always has been (surfacing it is a
    /// separate follow-on — CACHE-Q journal).
    @State private var cacheClearNotice: String?
    /// CFM-R10-Events: the arming session for trigger flows.
    @State private var armSession = FlowArmSession()

    // MARK: - Run

    private func run(_ doc: FlowDocument) {
        run(doc, occurrence: nil)
    }

    /// Run with an optional trigger occurrence (a fired `On File` / `On Schedule` / `On Flow`).
    private func run(_ doc: FlowDocument, occurrence: FlowInterpreter.Occurrence?) {
        // Preflight decides installs; the session gates Run until models exist.
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        let result = FlowPreflight.run(doc, catalog: catalog,
                                       installedModelIDs: appState.installedModelIDs,
                                       totalRAMGB: appState.systemInfo.totalRAMGB,
                                       claimableModelIDs: appState.claimableModelIDs)
        session.prepareInstall(result, doc: doc, scope: workspaceRef != nil ? makeScope() : nil)
        pendingOccurrence = occurrence
        guard !result.toDownload.isEmpty else {
            startRun(doc)
            return
        }
        // Defensive: the Run button is disabled while downloads are pending, but if run is
        // reached with models to download, surface the install sheet (Install, not Run).
        session.showInstallSheet = true
    }

    /// The occurrence an armed trigger fired, consumed by the next `startRun`.
    @State private var pendingOccurrence: FlowInterpreter.Occurrence?

    private func startRun(_ doc: FlowDocument) {
        // Re-run from here when some rows already have results; a fresh run otherwise.
        let resume = session.hasRunResults
        let context = AppFlowExecutorFactory.cachingContext(scope: makeScope(), appState: appState, transforms: doc.transforms)
        let occurrence = pendingOccurrence
        pendingOccurrence = nil
        session.start(doc: doc, runner: FlowRunner(), context: context, resume: resume,
                      occurrence: occurrence)
    }

    // MARK: - Install sheet (one prompt, not six)

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
                Divider()
                HStack {
                    Text("Total")
                        .fontWeight(.semibold)
                    Spacer()
                    Text(String(format: "%.1f GB", result.totalDownloadGB))
                        .fontWeight(.semibold)
                }
            }
            .padding(10)
            .background(.quaternary.opacity(0.3), in: RoundedRectangle(cornerRadius: 8))

            Text("Models download once and stay installed for future runs.")
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack {
                Spacer()
                Button("Cancel") {
                    session.showInstallSheet = false
                    // WR-4 finding A (owner review, 2026-09-22): route through the same
                    // decision the other two dismissals use, rather than clearing the flag
                    // imperatively here — `.cancelled` must be a real, reachable case, not a
                    // value only a test ever constructs.
                    finishInstallSheet(.cancelled)
                }
                .keyboardShortcut(.cancelAction)
                Button("Install") {
                    session.showInstallSheet = false
                    installRequiredModels(result)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    /// Install the flow's to-download models, then refresh preflight so the header flips to
    /// a ready Run button (and the install button disappears). Run stays on the user. The
    /// in-progress state is recorded on `AppState` so it survives navigating away (smoke-30).
    private func installRequiredModels(_ result: FlowPreflight.Result) {
        session.isInstalling = true
        appState.installingFlowIDs.insert(flowID)
        let models = Array(result.downloadSet)
        Task {
            await installSequentially(models)
        }
    }

    /// Drive `InstallManager.install` **sequentially** — one model at a time, awaiting each
    /// `.installed` marker before starting the next. Always reassesses when done (so the
    /// install sheet, `needsInstall` and any refusal reflect what actually landed, even after a
    /// partial install — M13), and surfaces a sentence when a download failed instead of
    /// silently reverting the button.
    private func installSequentially(_ models: [ModelEntry]) async {
        var succeeded = true
        for model in models {
            appState.installModel(model)
            let installed = await InstallPoller.awaitInstalled(model: model,
                                                               installManager: installManager)
            guard installed else { succeeded = false; break }
        }
        session.isInstalling = false
        appState.installingFlowIDs.remove(flowID)
        if !succeeded {
            session.errorSentence = "One of the required models failed to download — check your connection and try again."
        }
        reassess()
        // WR-4 (owner gate Q1, ruled (a) 2026-09-22): resume the auto-run, but only on the
        // exact path it asked for this install.
        finishInstallSheet(succeeded ? .installSucceeded : .installFailed)
    }

    /// WR-4 finding A (owner review, 2026-09-22): the one place all three ways an install
    /// sheet closes are resolved — Cancel, an install that succeeded, an install that failed.
    /// Before this, Cancel cleared `autoRunAwaitingInstall` imperatively instead of producing
    /// a `.cancelled` dismissal and running it through `AutoRunResume.shouldRun`, so `.cancelled`
    /// was reachable only from a test, never from the app itself.
    private func finishInstallSheet(_ dismissal: InstallSheetDismissal) {
        guard AutoRunResume.shouldRun(autoRunPending: autoRunAwaitingInstall, dismissal: dismissal) else {
            autoRunAwaitingInstall = false
            return
        }
        applyResumeAttempt(AutoRunResume.resumeAttempt(canRunNow: session.canRun, isFinalAttempt: false))
    }

    /// WR-4 finding B (owner review, 2026-09-22): `InstallPoller.awaitInstalled` resolves off
    /// the on-disk `.installed` marker; `appState.installedModelIDs` updates on a separate
    /// path (`installModel`'s completion callback). If the poller wins that race,
    /// `finishInstallSheet`'s `session.canRun` check reads a stale installed set even though
    /// the install genuinely succeeded, and `autoRunAwaitingInstall` stays armed rather than
    /// running immediately. This is the one grace pass it gets, from the next
    /// `.onChange(of: installedModelIDs)` — resolved here regardless of outcome, never a
    /// second time.
    private func retryArmedAutoRunOnce() {
        guard autoRunAwaitingInstall else { return }
        applyResumeAttempt(AutoRunResume.resumeAttempt(canRunNow: session.canRun, isFinalAttempt: true))
    }

    private func applyResumeAttempt(_ attempt: AutoRunResume.ResumeAttempt) {
        switch attempt {
        case .runNow:
            autoRunAwaitingInstall = false
            if let doc = document { startRun(doc) }
        case .stayArmedOnce:
            break
        case .giveUp:
            autoRunAwaitingInstall = false
        }
    }

    /// WR-3: re-derive both the refusal and the preflight from the current `document` — the
    /// installed-model set changed, so both the refusal (`FlowRunnability`, which reads the
    /// same `installed:` set) and the preflight's `toDownload` bucket are stale until this
    /// runs. Never re-reads the `.cat` from disk (that's `reload()`'s job, and it would
    /// discard run state) — `document` is already parsed and unchanged. The actual decision is
    /// `FlowReassessment.compute`, a pure function a test can call directly.
    private func reassess() {
        guard let doc = document else { return }
        apply(assess(doc), doc: doc)
    }

    /// WR-3 follow-up: the one call to `FlowReassessment.compute` every load path and
    /// `reassess()` share — was duplicated inline in `loadGallery`/`loadUserFlow`/
    /// `loadWorkspaceFlow` (two copies of one decision, this project's named recurring
    /// failure per R14-FIX) until this pass folded them back into the seam WR-3 already
    /// extracted the *logic* into but not yet the *call sites* out of.
    private func assess(_ doc: FlowDocument) -> FlowReassessment {
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        return FlowReassessment.compute(
            doc: doc, catalog: catalog, installed: appState.installedModelIDs,
            totalRAMGB: appState.systemInfo.totalRAMGB, claimableModelIDs: appState.claimableModelIDs,
            refusalScope: workspaceRef != nil ? makeScope() : nil, inputScope: makeScope())
    }

    /// Apply a `FlowReassessment` to `@State` — always calls `session.prepareInstall`, refused
    /// or not (owner review, 2026-09-22: never leave a stale preflight from before a refusal
    /// in place — see `FlowReassessment`'s own doc).
    private func apply(_ result: FlowReassessment, doc: FlowDocument) {
        notRunnableReason = result.notRunnableReason
        notRunnableAction = result.notRunnableAction
        setupAdvisory = result.setupAdvisory
        inputAdvisory = result.inputAdvisory
        session.prepareInstall(result.preflight, doc: doc, scope: workspaceRef != nil ? makeScope() : nil)
    }

    private var installManager: InstallManager {
        appState.installManager
    }

    /// Find a row by id (top-level or nested) for the human-prompt sheet.
    private static func row(_ id: UUID, in doc: FlowDocument) -> Row {
        func find(_ rows: [Row]) -> Row? {
            for row in rows {
                if row.id == id { return row }
                if let found = find(row.children) { return found }
            }
            return nil
        }
        return find(doc.rows) ?? Row(id: id, task: "Ask Human")
    }

    // MARK: - Loading

    /// CFM-R12-1: what the flow list renders about a flow, abstracted over its source
    /// (bundled gallery vs. the user's shelf) — never a faked `GalleryFlowMetadata`.
    private var display: FlowDisplay? {
        FlowDisplay.resolve(metadata: metadata, userEntry: userEntry, workspace: workspaceDisplayRef)
    }

    /// Re-read the flow from disk after the editor closes — its title, rows or
    /// runnability may have changed. Clears the transient error/refusal state a
    /// previous load left so a now-fixed flow isn't stuck on a stale sentence.
    private func reload() {
        loadError = nil
        notRunnableReason = nil
        notRunnableAction = nil
        setupAdvisory = nil
        inputAdvisory = nil
        load()
    }

    private func load() {
        if workspaceRef != nil {
            loadWorkspaceFlow()
        } else if source == .gallery {
            loadGallery()
        } else {
            loadUserFlow()
        }
    }

    /// CFM-R17-3: load a flow that lives inside a workspace. Its `.cat` sits beside its
    /// siblings in the shared directory; everything else (serialize, refusal, preflight)
    /// runs exactly as a user flow's does, only against the workspace-rooted scope.
    private func loadWorkspaceFlow() {
        guard let ref = workspaceRef else { return }
        let text: String
        do {
            text = try String(contentsOf: ref.fileURL, encoding: .utf8)
        } catch {
            loadError = "This workspace flow's file isn't there anymore — \(ref.flowFile)."
            return
        }
        rawText = text
        do {
            document = try CatParser.parse(text)
        } catch {
            loadError = (error as CustomStringConvertible).description
            return
        }
        guard let doc = document else { return }
        // WR-1: before the refusal early-return below — a refused workspace flow must
        // render `notRunnableView` with its title, not fall back to the spinner.
        workspaceDisplayRef = ref

        let serialized = CatSerializer.serializeLines(doc)
        serializedLines = serialized.lines
        lineRanges = serialized.lineRanges
        clauseLineRanges = serialized.clauseRanges

        let result = assess(doc)
        apply(result, doc: doc)
        guard result.notRunnableReason == nil else { return }
        finishLoad(doc)
    }

    private func loadGallery() {
        metadata = GalleryLoader.loadMetadata().first { $0.flowID == flowID }
        do {
            document = try GalleryLoader.loadDocument(flowID: flowID)
        } catch {
            loadError = (error as CustomStringConvertible).description
            return
        }
        guard let doc = document else { return }

        // R6-2: the flow list *is* the file — compute the canonical serialized lines once.
        let serialized = CatSerializer.serializeLines(doc)
        serializedLines = serialized.lines
        lineRanges = serialized.lineRanges
        clauseLineRanges = serialized.clauseRanges

        // CFM-R12-FIX-1: the refusal comes from the live gates (canRun + preflight), never
        // from a hand-written `_metadata.json` string that can go stale. `FlowRunnability`
        // covers language/doors/tools AND the model-preflight (a row whose model has no
        // bridge entry) + RAM.
        let result = assess(doc)
        apply(result, doc: doc)
        guard result.notRunnableReason == nil else { return }

        // Copy the bundled input assets into the flow's working folder before a run
        // can touch them — "Reveal in Finder" alone used to do this, so running a
        // fresh flow (e.g. 15-HouseStyle's `Read Text draft.md`) failed with
        // "couldn't read" until the user had clicked it. Idempotent: only missing
        // files are copied, so user edits to inputs survive.
        try? flowWorkspace.prepare(
            flowID: locationID,
            sourceDir: GalleryLoader.resourcesDirectory,
            bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        finishLoad(doc)
    }

    /// CFM-R12-1: load a user flow from its folder. A broken file shows its parse error
    /// (loadError), never a crash or a vanished badge.
    private func loadUserFlow() {
        userEntry = UserFlowStore.scan(workspace: FlowWorkspace.shared,
                                       bundledFlowIDs: Set(appState.galleryEntries.map(\.flowID)))
            .first { $0.flowID == flowID }
        guard let entry = userEntry else {
            loadError = "This flow's folder isn't in your flows directory anymore."
            return
        }
        if let issue = entry.parseIssue {
            loadError = issue
            return
        }
        do {
            document = try UserFlowStore.loadDocument(entry: entry)
        } catch {
            loadError = (error as CustomStringConvertible).description
            return
        }
        guard let doc = document else { return }

        let serialized = CatSerializer.serializeLines(doc)
        serializedLines = serialized.lines
        lineRanges = serialized.lineRanges
        clauseLineRanges = serialized.clauseRanges

        // CFM-R12-FIX-1: a user flow's refusal is the same live gates a bundled flow's is.
        let result = assess(doc)
        apply(result, doc: doc)
        guard result.notRunnableReason == nil else { return }
        // A user flow has no bundled assets; prepare with an empty list is a no-op.
        finishLoad(doc)
    }

    /// The load-only tail, once a flow is known not refused (WR-3 follow-up: the refusal +
    /// preflight themselves are `assess(_:)`/`apply(_:doc:)`'s job now, called by every
    /// `load*` function before this runs — this is only the work that doesn't overlap with
    /// `reassess()`, which must not rebuild the inspector or re-trigger auto-run on every
    /// install).
    private func finishLoad(_ doc: FlowDocument) {
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        // The inspector's frozen Properties tab reuses the editor's resolution machinery
        // (candidate models, input labels, display numbers) over a read-only model.
        inspectModel = FlowEditorModel(name: display?.title ?? flowID, flowID: locationID,
                                       document: doc, workspace: flowWorkspace,
                                       savedText: CatSerializer.serialize(doc))
        inspectModel?.modelCatalog = catalog
        inspectModel?.claimableModelIDs = appState.claimableModelIDs
        // CFM-R10-Events: establish the trigger kind so the Arm button shows (and the §14.4
        // refusal when the flow carries a door — CFM-R17-FIX-3: including one reached through
        // a workspace-sibling `uses:` flow, which needs the workspace + location to resolve).
        armSession.inspect(doc: doc, workspace: flowWorkspace, flowID: locationID)

        // CFM-R17-4: the workspace index card opened this flow to run it. `canRun` needs the
        // preflight (just set) to settle through `@Observable`, so hop a runloop.
        // CFM-R17-FIX-7: burn `didAutoRun` only once we actually act — run when we can, or
        // (when the sole blocker is missing models) let `run(doc)` raise the install sheet so
        // auto-run isn't a silent no-op. A hard refusal (door / capability / RAM) leaves the
        // disabled Run + `runDisabledReason` to explain it; there's nothing to open, so the
        // guard stays unspent.
        if autoRun, !didAutoRun {
            DispatchQueue.main.async {
                guard autoRun, !didAutoRun else { return }
                guard session.canRun || session.blockedOnlyOnDownloads else { return }
                didAutoRun = true
                // WR-4: mark that *this* install sheet, if `run(doc)` raises one, is the
                // auto-run's own — the only path `AutoRunResume.shouldRun` may fire for.
                if !session.canRun, session.blockedOnlyOnDownloads {
                    autoRunAwaitingInstall = true
                }
                run(doc)
            }
        }
    }
}

/// CFM-R12-1 — the display-side identity of a flow, abstracted over its source. Everything
/// the flow list renders about a flow, without faking gallery metadata for user flows.
nonisolated struct FlowDisplay {
    let title: String
    let description: String?

    /// CFM-R12-1, extended CFM-R17-3 (WR-1) — the same precedence `FlowListView.display` used
    /// to apply inline, now reachable from a test: gallery metadata first, the user shelf
    /// second, a workspace flow third (its title is the flow's own stem — never a faked
    /// `GalleryFlowMetadata`, CFM-R12-1). Owner gate Q3, **ruled (a), 2026-09-22**: "What WR-1
    /// shipped is what I want" — title `workspace.flowStem`, description `"in <workspaceID>"`,
    /// no change needed.
    static func resolve(metadata: GalleryFlowMetadata?, userEntry: UserFlowStore.Entry?,
                        workspace: WorkspaceRef?) -> FlowDisplay? {
        if let metadata {
            return FlowDisplay(title: metadata.title, description: metadata.description)
        }
        if let userEntry {
            return FlowDisplay(title: userEntry.title, description: nil)
        }
        if let workspace {
            return FlowDisplay(title: workspace.flowStem,
                               description: "in \(workspace.workspaceID)")
        }
        return nil
    }
}

/// WR-3 (`RSI/DelegateWorkspaceRunBacklog.md`) — the "refusal + preflight" tail shared by every
/// `load*` function in `FlowListView`, pulled out so `.onChange(of: installedModelIDs)` can
/// re-derive **both** (not just the preflight, which is all `refreshPreflight()` used to do)
/// from a test-reachable seam. `preflight` is **always** present, refused or not — owner
/// review, 2026-09-22: an earlier version returned `nil` on the refused path, on the theory
/// that a refused flow has no preflight worth keeping. But the caller (`FlowListView.apply`)
/// only called `session.prepareInstall` when `preflight != nil`, which left whatever
/// preflight was already in `session` (from before the refusal) in place — a stale download
/// set that could render the install sheet. Unreachable today only because `installed` alone
/// never flips a flow between refused and not (see below); reachable the moment any future
/// gate makes the refusal genuinely install-sensitive.
nonisolated struct FlowReassessment {
    let notRunnableReason: String?
    let notRunnableAction: SetupAction?
    let setupAdvisory: FlowPreflight.RowAdvisory?
    let inputAdvisory: FlowPreflight.RowAdvisory?
    let preflight: FlowPreflight.Result

    /// `refusalScope` matches each `load*` function's existing call: `nil` for a gallery/user
    /// flow, `makeScope()` for a workspace flow (so a `uses:` call isn't refused as an unknown
    /// task). `inputScope` is always concrete — `FlowInputAdvisory.advisory` has always taken
    /// `makeScope()` unconditionally (it already falls back to `.plain(flowID)`).
    ///
    /// Runs `FlowPreflight.run` exactly **once**. `FlowRunnability.refusal` computes its own
    /// preflight internally to derive `blockedReason`/`blockedAction` and throws it away — a
    /// caller that also needs the preflight object (every caller here does) would otherwise
    /// run it twice per call. The doors-check + `blockedReason`/`blockedAction` logic below is
    /// `FlowRunnability.refusal`'s own decision, inlined rather than duplicated blind: both
    /// read the identical `FlowRunner.canRun` → `FlowPreflight.blockedReason`/`blockedAction`
    /// gates `FlowRunnability.swift` defines, so there is one *authority* for what refuses a
    /// flow even though there are now two call sites expressing it — `FlowRunnability.swift`
    /// wasn't itself in this phase's five-file scope.
    static func compute(doc: FlowDocument, catalog: [ModelEntry], installed: Set<String>,
                        totalRAMGB: Double, claimableModelIDs: Set<String>,
                        refusalScope: FlowScope?, inputScope: FlowScope) -> FlowReassessment {
        let preflight = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: installed,
                                          totalRAMGB: totalRAMGB, claimableModelIDs: claimableModelIDs)
        let runnability = refusalScope.map { FlowRunner.canRun(doc, scope: $0) } ?? FlowRunner.canRun(doc)
        var notRunnableReason: String?
        var notRunnableAction: SetupAction?
        if case .notRunnable(let reason) = runnability {
            notRunnableReason = reason
        } else if let reason = FlowPreflight.blockedReason(preflight, totalRAMGB: totalRAMGB) {
            notRunnableReason = reason
            notRunnableAction = FlowPreflight.blockedAction(preflight)
        }
        let refused = notRunnableReason != nil
        return FlowReassessment(
            notRunnableReason: notRunnableReason, notRunnableAction: notRunnableAction,
            setupAdvisory: refused ? nil : FlowPreflight.setupAdvisory(preflight),
            inputAdvisory: refused ? nil : FlowInputAdvisory.advisory(for: doc, scope: inputScope),
            preflight: preflight)
    }
}

/// WR-4 (`RSI/DelegateWorkspaceRunBacklog.md`) — how the install sheet closed, when it was
/// raised by this flow's own auto-run. `.cancelled` is the sheet's Cancel button — reached
/// directly, or via Esc through that button's own `.keyboardShortcut(.cancelAction)`. A macOS
/// `.sheet` isn't click-outside-dismissible, so that was never a separate path (corrected
/// 2026-09-22 — an earlier version of this doc claimed otherwise; say what's true).
nonisolated enum InstallSheetDismissal: Equatable {
    case cancelled
    case installSucceeded
    case installFailed
}

/// WR-4, owner gate Q1 (**ruled (a), 2026-09-22**: auto-run resumes once, only on the path
/// that raised the sheet) — whether an auto-run that opened the install sheet should resume
/// once that sheet closes. Pure: no view, no session, no `didAutoRun`. `didAutoRun` burns once
/// by design (CFM-R17-FIX-7) so the auto-run never opens the sheet a second time; resetting it
/// to "try again" would also re-arm Cancel and every later, unrelated install, which is the
/// wrong answer (root cause 4's actual bug was the *lack* of any resume path, not that
/// `didAutoRun` needed clearing). This is the resume path instead, decided from exactly what
/// the backlog asks for: was *this* flow's auto-run the one waiting on an install, and what
/// happened to the sheet it raised.
///
/// **Finding A** (owner review, 2026-09-22): all three `InstallSheetDismissal` cases must be
/// constructed by production code and run through `shouldRun` from one call site
/// (`FlowListView.finishInstallSheet`) — not two paths through the decision and a third
/// enforced by an imperative flag write that only a test ever exercises as `.cancelled`.
///
/// **Finding B** (owner review, 2026-09-22): `shouldRun` alone doesn't decide *when* to run.
/// `InstallPoller.awaitInstalled` resolves off the on-disk `.installed` marker;
/// `appState.installedModelIDs` updates on a separate completion-callback path, so
/// `session.canRun` can briefly lag behind a genuinely successful install. `resumeAttempt`
/// covers that: run immediately if `canRunNow` already agrees, otherwise stay armed for
/// exactly one more `.onChange(of: installedModelIDs)` pass (`isFinalAttempt`) before giving
/// up for good — never longer than that one extra pass.
nonisolated enum AutoRunResume {
    static func shouldRun(autoRunPending: Bool, dismissal: InstallSheetDismissal) -> Bool {
        autoRunPending && dismissal == .installSucceeded
    }

    enum ResumeAttempt: Equatable {
        case runNow
        case stayArmedOnce
        case giveUp
    }

    static func resumeAttempt(canRunNow: Bool, isFinalAttempt: Bool) -> ResumeAttempt {
        if canRunNow { return .runNow }
        return isFinalAttempt ? .giveUp : .stayArmedOnce
    }
}
