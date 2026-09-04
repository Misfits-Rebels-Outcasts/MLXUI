import SwiftUI
import AppKit

/// CFM-R17-3 — a workspace page: the flows in one `workspaces/<id>/` directory, the files
/// they share, and `Reveal in Finder` on the one directory they all resolve against. It is a
/// **container** around the existing flow surfaces — a flow opens in the same editor
/// (`FlowEditorView`) and runs through the same session, only with a `WorkspaceRef` so its
/// `.cat` paths land in the shared directory (CFM-R17-1).
struct WorkspaceListView: View {
    @Environment(AppState.self) private var appState
    let workspace: WorkspaceStore.Workspace

    @State private var pendingRemoval = false

    private var columns: [GridItem] { [GridItem(.adaptive(minimum: 240), spacing: 14)] }

    /// CFM-R17-4: parse each flow once, then pair a builder with a querier of the same index.
    /// CFM-R17-FIX-5: a name with more than one builder or querier surfaces as a collision
    /// notice rather than a silent coin-flip.
    private var knowledge: WorkspaceKnowledge.Knowledge {
        let parsed = workspace.flows.compactMap { flow -> (file: String, doc: FlowDocument)? in
            guard let doc = try? WorkspaceStore.loadDocument(flow: flow) else { return nil }
            return (flow.url.lastPathComponent, doc)
        }
        return WorkspaceKnowledge.classify(flows: parsed)
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
                let kb = knowledge
                if !kb.isEmpty { indexSection(kb) }
                flowsSection
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
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Label(workspace.title, systemImage: "folder.badge.gearshape")
                    .font(.largeTitle.weight(.bold))
                Spacer()
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
            Text("\(workspace.flows.count) flow\(workspace.flows.count == 1 ? "" : "s") sharing one folder — every relative path resolves against it.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - CFM-R17-4: the knowledge-base card

    private func indexSection(_ kb: WorkspaceKnowledge.Knowledge) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Knowledge Base")
                .font(.title3.weight(.semibold))
            ForEach(kb.pairings, id: \.indexName) { pair in
                indexCard(pair)
            }
            ForEach(kb.collisions, id: \.indexName) { collision in
                collisionNotice(collision)
            }
        }
    }

    /// CFM-R17-FIX-5 — an index whose builders or queriers collide gets a notice, not a card
    /// wired to whichever flow happened to sort first.
    private func collisionNotice(_ collision: WorkspaceKnowledge.IndexCollision) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)
            Text(collision.message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(Color.orange.opacity(0.3), lineWidth: 1) }
    }

    private func indexCard(_ pair: WorkspaceKnowledge.IndexPairing) -> some View {
        let m = manifest(for: pair.indexName)
        return VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "books.vertical").foregroundStyle(.secondary)
                Text(pair.indexName).font(.headline)
                Spacer()
            }
            if let m {
                Text("\(m.embedder) · \(m.dims)-dim · \(m.count) chunk\(m.count == 1 ? "" : "s")")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            } else {
                Text("not built yet — run \(pair.builderFile) to create it")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            HStack(spacing: 10) {
                Button {
                    runFlow(pair.builderFile)
                } label: {
                    Label(m == nil ? "Build" : "Rebuild", systemImage: "hammer")
                }
                Button {
                    runFlow(pair.querierFile)
                } label: {
                    Label("Ask", systemImage: "text.bubble")
                }
                .disabled(m == nil)
                Spacer()
                Text("\(pair.builderFile) builds · \(pair.querierFile) queries")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
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
    }
}
