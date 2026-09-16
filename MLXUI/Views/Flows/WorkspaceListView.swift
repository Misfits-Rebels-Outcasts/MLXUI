import SwiftUI
import AppKit

/// CFM-R17-3 — a workspace page: the flows in one `workspaces/<id>/` directory, the files
/// they share, and `Reveal in Finder` on the one directory they all resolve against. It is a
/// **container** around the existing flow surfaces — a flow opens in the same editor
/// (`FlowEditorView`) and runs through the same session, only with a `WorkspaceRef` so its
/// `.cat` paths land in the shared directory (CFM-R17-1).
struct WorkspaceListView: View {
    @Environment(AppState.self) private var appState
    /// KW-2-1-FIX: `workspace` used to be the `Workspace` value snapshotted at navigation
    /// time, so a flow added or renamed while this page stayed open never showed up here —
    /// `appState.selectedWorkspace` is never re-synced to a fresh scan except on removal, and
    /// `reloadWorkspaces()` only refreshed `appState.workspaceEntries`, which this view never
    /// read. `workspace` is now computed from that array by id every time it's read, so a
    /// mutation to `workspaceEntries` (from `addFlow()`, or the appear-time rescan after
    /// popping back from the editor) is picked up live, the same way `manifest(for:)` already
    /// reads the index off disk fresh every time rather than trusting a cached copy. `initial`
    /// is only the seed for the instant before the environment is available and for the rare
    /// case the id has vanished from the array entirely (e.g. removed from another window).
    private let initial: WorkspaceStore.Workspace
    private var workspace: WorkspaceStore.Workspace {
        appState.workspaceEntries.first { $0.workspaceID == initial.workspaceID } ?? initial
    }

    init(workspace: WorkspaceStore.Workspace) {
        self.initial = workspace
    }

    @State private var pendingRemoval = false
    /// KW-2-2: the flow a trash click is confirming — non-nil shows the confirmation dialog.
    @State private var flowPendingRemoval: WorkspaceStore.FlowFile?
    /// KW-2-2: the flow a `Rename…` click is prompting for a new name — non-nil shows the
    /// rename alert. `renameText` seeds from the flow's current stem each time it's set.
    @State private var flowPendingRename: WorkspaceStore.FlowFile?
    @State private var renameText = ""
    /// KW-2-FIX-4: surfaces a failed rename *or* delete — both call into `WorkspaceStore`
    /// functions with real, plain-voice refusals; neither should be swallowed by `try?`.
    @State private var flowActionError: String?

    private var columns: [GridItem] { [GridItem(.adaptive(minimum: 240), spacing: 14)] }

    /// CFM-R17-4: parse each flow once, then a Knowledge Base card per index worth rendering.
    /// CFM-R17-FIX-5/-9(c): an ambiguous side drops its own button and notes the collision; the
    /// unambiguous side keeps working. CFM-R17-FIX-11(c): a builder makes a card always
    /// actionable (Build); with no builder, only an index that already exists on disk earns a
    /// card (Ask, no Build) — a querier with nothing to read yet is a dead end, not a card.
    /// CFM-R17-FIX-11(e): a caller inherits its `uses:` callees' index use; a `.cat` that is
    /// itself the target of a sibling's `uses:` line is excluded from the candidate set (a
    /// library component, not a runnable entry point).
    ///
    /// CFM-R17-FIX-12(c): the candidate-set/graph construction that used to live in this
    /// property is now `WorkspaceKnowledge.classifyWorkspace` — a pure, directly-tested
    /// function. This is just plumbing: parse the flows, hand `UsesResolver`/`FlowWorkspace`
    /// to it as closures (the filesystem-touching halves it can't do itself), then apply the
    /// one filter that's still this view's own call — whether an index exists on disk.
    private var knowledgeCards: [WorkspaceKnowledge.IndexCard] {
        let ws = FlowWorkspace(root: ModelStore.shared.workspacesDirectory)
        let flowID = workspace.workspaceID
        let parsed = workspace.flows.compactMap { flow -> (file: String, doc: FlowDocument, url: URL)? in
            guard let doc = try? WorkspaceStore.loadDocument(flow: flow) else { return nil }
            return (flow.url.lastPathComponent, doc, flow.url)
        }
        let cards = WorkspaceKnowledge.classifyWorkspace(
            flows: parsed,
            resolveUses: { doc, selfFile in
                UsesResolver.resolve(doc, workspace: ws, flowID: flowID, selfFile: selfFile)
            },
            resolvePath: { rawPath in try? ws.resolve(rawPath, flowID: flowID) })
        return cards.filter {
            WorkspaceKnowledge.isCardWorthRendering($0, indexExists: manifest(for: $0.indexName) != nil)
        }
    }

    /// The `manifest.json` of an index in this workspace, read fresh from disk — never cached
    /// in app state, so a card that disagrees with the disk is impossible (CFM-R17-4).
    private func manifest(for indexName: String) -> IndexFormat.Manifest? {
        let url = workspace.url.appendingPathComponent(indexName).appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? IndexFormat.manifest(from: data)
    }

    /// The non-`.cat` entries in the directory the flows share — indexes, `docs/`, fixtures.
    private var sharedFiles: [URL] {
        let flowNames = Set(workspace.flows.map { $0.url.lastPathComponent })
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: workspace.url, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])) ?? []
        return contents
            .filter { !flowNames.contains($0.lastPathComponent) }
            .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                let cards = knowledgeCards
                if !cards.isEmpty { indexSection(cards) }
                if !workspace.flows.isEmpty { flowsSection }
                if !sharedFiles.isEmpty { sharedFilesSection }
            }
            .padding(20)
        }
        .navigationTitle(workspace.title)
        .onAppear { appState.reloadWorkspaces() }
        .confirmationDialog("Remove this workspace?", isPresented: $pendingRemoval,
                            titleVisibility: .visible) {
            Button("Remove", role: .destructive) {
                appState.removeWorkspace(workspaceID: workspace.workspaceID)
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(WorkspaceStore.deletionSummary(
                workspace, bundled: BundledWorkspaces.isBundled(workspace.workspaceID)))
        }
        // KW-2-2 (Q3): deleting the last flow is allowed, with a warning naming what stays.
        .confirmationDialog("Delete this flow?", isPresented: Binding(
            get: { flowPendingRemoval != nil },
            set: { if !$0 { flowPendingRemoval = nil } }
        ), titleVisibility: .visible) {
            Button("Delete", role: .destructive) { deleteFlow() }
            Button("Cancel", role: .cancel) { flowPendingRemoval = nil }
        } message: {
            Text(flowPendingRemoval.map {
                WorkspaceStore.flowDeletionSummary(fileName: $0.url.lastPathComponent,
                                                   isLastFlow: workspace.flows.count == 1,
                                                   workspace: workspace)
            } ?? "")
        }
        // KW-2-2 (Q3): Rename… as its own action rather than a side effect of the editor's
        // name field — renames the file in place, no need to open it.
        .alert("Rename Flow", isPresented: Binding(
            get: { flowPendingRename != nil },
            set: { if !$0 { flowPendingRename = nil } }
        )) {
            TextField("Name", text: $renameText)
            Button("Rename") { renameFlow() }
            Button("Cancel", role: .cancel) { flowPendingRename = nil }
        } message: {
            Text("Choose a new name for '\(flowPendingRename?.title ?? "")'.")
        }
        .alert("Couldn't Complete That", isPresented: Binding(
            get: { flowActionError != nil },
            set: { if !$0 { flowActionError = nil } }
        )) {
            Button("OK", role: .cancel) { flowActionError = nil }
        } message: {
            Text(flowActionError ?? "")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Label(workspace.title, systemImage: "folder.badge.gearshape")
                    .font(.largeTitle.weight(.bold))
                Spacer()
                Button { addFlow() } label: {
                    Label("New Flow", systemImage: "doc.badge.plus")
                }
                Button {
                    FlowWorkspace(root: ModelStore.shared.workspacesDirectory)
                        .revealInFinder(flowID: workspace.workspaceID)
                } label: {
                    Label("Reveal in Finder", systemImage: "folder")
                }
                Button(role: .destructive) { pendingRemoval = true } label: {
                    Label("Remove", systemImage: "trash")
                }
            }
            // KW-1-2 (Q2): a workspace can list with no `.cat` at all — whatever it holds
            // (an index, leftover files) is still reachable and removable, just not runnable.
            Text(workspace.flows.isEmpty
                 ? "No flows in this folder — whatever's left in it is listed below, and Remove still reaches it."
                 : "\(workspace.flows.count) flow\(workspace.flows.count == 1 ? "" : "s") sharing one folder — every relative path resolves against it.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - CFM-R17-4: the knowledge-base card

    private func indexSection(_ cards: [WorkspaceKnowledge.IndexCard]) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Knowledge Base")
                .font(.title3.weight(.semibold))
            ForEach(cards, id: \.indexName) { card in
                indexCard(card)
            }
        }
    }

    /// CFM-R17-FIX-9(c): a Build/Ask button appears only for a side with exactly one flow; an
    /// ambiguous side drops its button and the card carries a note naming the colliding flows.
    /// CFM-R17-FIX-11(b): the card names the flow each button runs — every row on this page
    /// renders as the literal `.cat` so the user can see what will happen before they click,
    /// and this card had stopped doing that the moment a manifest existed and the placeholder
    /// line (the card's only filename) disappeared. The caption below is unconditional on
    /// whether the index is built, so `runFlow`'s `autoRun: true` never fires unnamed.
    private func indexCard(_ card: WorkspaceKnowledge.IndexCard) -> some View {
        let m = manifest(for: card.indexName)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "books.vertical").foregroundStyle(.secondary)
                Text(card.indexName).font(.headline)
                Spacer()
            }
            if let caption = card.namesCaption {
                Text(caption)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            if let m {
                Text("\(m.embedder) · \(m.dims)-dim · \(m.count) chunk\(m.count == 1 ? "" : "s")")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else if let builder = card.buildFile {
                Text("not built yet — run \(builder) to create it")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text("not built yet")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            // CFM-R17-FIX-11(d): omit the row entirely when neither side has a button (both
            // ambiguous, or ambiguous + none) — `-11(c)` made that shape reachable, and an
            // empty `HStack { Spacer() }` was ~16pt of dead space where the buttons would be.
            if card.buildFile != nil || card.askFile != nil {
                HStack(spacing: 10) {
                    if let builder = card.buildFile {
                        Button {
                            runFlow(builder)
                        } label: {
                            Label(m == nil ? "Build" : "Rebuild", systemImage: "hammer")
                        }
                    }
                    if let querier = card.askFile {
                        Button {
                            runFlow(querier)
                        } label: {
                            Label("Ask", systemImage: "text.bubble")
                        }
                        .disabled(m == nil)
                    }
                    Spacer()
                }
            }
            if let note = card.ambiguityNote {
                HStack(alignment: .top, spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.orange)
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.accentColor.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(Color.accentColor.opacity(0.3), lineWidth: 1) }
    }

    /// Open the flow and run it — Build/Rebuild the builder, Ask the querier (which parks on
    /// its Human Input row and is drawn by the existing `FlowHumanPromptView`).
    private func runFlow(_ file: String) {
        let ref = WorkspaceRef(workspaceID: workspace.workspaceID, flowFile: file)
        appState.selectedFlow = FlowSelection(flowID: ref.workspaceID, workspace: ref, autoRun: true)
    }

    private var flowsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Flows")
                .font(.title3.weight(.semibold))
            LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                ForEach(workspace.flows) { flow in
                    flowBadge(flow)
                }
            }
        }
    }

    private var sharedFilesSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Shared Files")
                .font(.title3.weight(.semibold))
            VStack(alignment: .leading, spacing: 4) {
                ForEach(sharedFiles, id: \.self) { url in
                    Label(url.lastPathComponent,
                          systemImage: url.hasDirectoryPath ? "folder" : "doc")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    /// KW-2-1: add a second (or third, …) flow to this workspace — the same three-step
    /// template `createWorkspace` (`FlowGalleryView.swift:245`) uses one directory up: write
    /// the starter `.cat`, rescan, open. `WorkspaceStore.firstFreeFlowName` picks a name that
    /// can never collide with — and so can never overwrite — an existing sibling. Adding a
    /// flow can turn a Knowledge Base side ambiguous (two builders, say); that's an honest
    /// collision the card already words, not something to suppress here.
    private func addFlow() {
        let starter = "mlxflow 0.8\n1. Read Text   notes.txt\n2. Save Text   out.md\n"
        let filename = WorkspaceStore.firstFreeFlowName(stem: "Flow", in: workspace.url)
        let fileURL = workspace.url.appendingPathComponent(filename)
        do {
            try starter.write(to: fileURL, atomically: true, encoding: .utf8)
            let doc = try CatParser.parse(starter)
            appState.reloadWorkspaces()
            let ref = WorkspaceRef(workspaceID: workspace.workspaceID, flowFile: filename)
            appState.editingFlow = FlowEditTarget(flowID: ref.workspaceID, name: ref.flowStem,
                                                  document: doc, savedText: starter, workspace: ref)
        } catch {
            appState.workspaceImportError = "Couldn't create a new flow in this workspace."
        }
    }

    /// A click opens the flow in the editor (workspace-scoped). A file that no longer parses
    /// falls back to the read-only list so its parse sentence shows.
    private func open(_ flow: WorkspaceStore.FlowFile) {
        let ref = WorkspaceRef(workspaceID: workspace.workspaceID, flowFile: flow.url.lastPathComponent)
        if flow.parseIssue == nil, let doc = try? WorkspaceStore.loadDocument(flow: flow),
           let text = try? String(contentsOf: flow.url, encoding: .utf8) {
            appState.editingFlow = FlowEditTarget(flowID: ref.workspaceID, name: flow.title,
                                                  document: doc, savedText: text, workspace: ref)
        } else {
            appState.selectedFlow = FlowSelection(flowID: ref.workspaceID, workspace: ref)
        }
    }

    private func flowBadge(_ flow: WorkspaceStore.FlowFile) -> some View {
        Button { open(flow) } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "doc.plaintext").foregroundStyle(.secondary)
                    Spacer()
                    if flow.parseIssue != nil {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .accessibilityLabel("This flow doesn't parse")
                    }
                }
                Text(flow.title)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(flow.url.lastPathComponent)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 100, alignment: .leading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 14))
            .overlay { RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary, lineWidth: 1) }
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: 14))
        // KW-2-2: Rename… as a named action, not a side effect of the editor's name field.
        .contextMenu {
            Button("Rename…") {
                renameText = flow.title
                flowPendingRename = flow
            }
            Button("Delete", role: .destructive) { flowPendingRemoval = flow }
        }
        // KW-2-2: the trash overlay mirrors `workspaceBadge`'s (`FlowGalleryView.swift:228`).
        .overlay(alignment: .topTrailing) {
            Button { flowPendingRemoval = flow } label: {
                Image(systemName: "trash")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 22, height: 22)
                    .background(.quaternary.opacity(0.75), in: Circle())
                    .contentShape(Circle())
            }
            .buttonStyle(.plain)
            .help("Delete this flow")
            .padding(8)
        }
    }

    /// KW-2-2 (Q3): deletes the confirmed flow. The Knowledge Base card recomputes on the next
    /// render because `workspace` is a live lookup (KW-2-1-FIX) — a two-builder ambiguity that
    /// becomes single-builder gets its Build button back with no extra plumbing here.
    /// KW-2-FIX-4: `removeFlow` carries two real, plain-voice refusals — surface them instead
    /// of swallowing them with `try?`, the same way `renameFlow`'s failure already is.
    private func deleteFlow() {
        guard let flow = flowPendingRemoval else { return }
        flowPendingRemoval = nil
        do {
            try WorkspaceStore.removeFlow(file: flow.url, from: workspace.url)
        } catch {
            flowActionError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
            return
        }
        appState.reloadWorkspaces()
        let isTheDeletedFlow: (WorkspaceRef?) -> Bool = { ref in
            ref?.workspaceID == workspace.workspaceID && ref?.flowFile == flow.url.lastPathComponent
        }
        if isTheDeletedFlow(appState.editingFlow?.workspace) { appState.editingFlow = nil }
        if isTheDeletedFlow(appState.selectedFlow?.workspace) { appState.selectedFlow = nil }
    }

    /// KW-2-2 (Q3): renames the confirmed flow to `renameText`'s stem. A collision with a
    /// *different* sibling refuses (`WorkspaceStoreError.flowNameTaken`) rather than
    /// overwriting it — the same rule `KW-1-FIX-2` gives the editor's own Save.
    private func renameFlow() {
        guard let flow = flowPendingRename else { return }
        flowPendingRename = nil
        let stem = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stem.isEmpty else { return }
        do {
            _ = try WorkspaceStore.renameFlow(file: flow.url,
                                              toStem: FlowEditorModel.sanitizedFileName(stem),
                                              in: workspace.url)
            appState.reloadWorkspaces()
        } catch {
            flowActionError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
        }
    }
}
