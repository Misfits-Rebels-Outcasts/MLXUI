import SwiftUI
import AppKit

/// The Automate → "AI Workflows" gallery: every bundled flow rendered as a badge in the
/// detail panel. Selecting a badge pushes the flow's detail (`FlowListView` — title, rows,
/// inspector) onto the detail stack, leaving the sidebar on "AI Workflows".
struct FlowGalleryView: View {
    @Environment(AppState.self) private var appState

    private let columns = [GridItem(.adaptive(minimum: 220), spacing: 14)]

    /// The My Workflows badge awaiting a Remove confirmation (nil = none).
    @State private var flowPendingRemoval: UserFlowStore.Entry?

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                // CFM-R12-1: the user's own saved flows. The New Flow badge lives here too —
                // it is a user action, not a bundled flow. With `hideMyWorkflows` on, the
                // whole section stays visible but is frozen — every button inside (New Flow,
                // Import, a flow's badge, its export/trash) is disabled.
                myWorkflowsSection
                    .disabled(AppState.hideMyWorkflows)
                    .opacity(AppState.hideMyWorkflows ? 0.5 : 1)
                basicGallerySection
                if !AppState.hideAdvanceGallery {
                    advanceGallerySection
                }
            }
            .padding(20)
        }
        .navigationTitle("AI Workflows")
        // A save or Duplicate & Edit adds a folder while the editor was open; refresh on
        // every appearance so the shelf is never stale.
        .onAppear {
            appState.reloadUserFlows()
            appState.refreshGalleryBlocked()
        }
        // The gallery is the NavigationStack root, so `onAppear` does **not** fire again
        // when the editor pops back to it after a rename-and-save — the badge would keep the
        // old title until the user left the section and returned. Refresh whenever the editor
        // closes instead (editingFlow goes non-nil → nil), which covers both direct edits and
        // Duplicate & Edit copies.
        .onChange(of: appState.editingFlow) { _, newValue in
            if newValue == nil {
                appState.reloadUserFlows()
            }
        }
        .confirmationDialog("Remove this flow?", isPresented: Binding(
            get: { flowPendingRemoval != nil },
            set: { if !$0 { flowPendingRemoval = nil } }
        ), titleVisibility: .visible) {
            Button("Remove", role: .destructive) {
                if let entry = flowPendingRemoval {
                    appState.removeUserFlow(flowID: entry.flowID)
                }
                flowPendingRemoval = nil
            }
            Button("Cancel", role: .cancel) { flowPendingRemoval = nil }
        } message: {
            Text(flowPendingRemoval.map { "'\($0.title)' and its files will be deleted from your flows folder. This can't be undone." } ?? "")
        }
    }

    /// The content-panel title: a visible heading above the badge grid (the navigation
    /// title alone reads as a window label, not a page heading).
    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("AI Workflows")
                .font(.largeTitle.weight(.bold))
            //Text("\(appState.galleryEntries.count) bundled flows — and anything you've saved under My Workflows. Pick one to see and run it, or start your own.")
            Text("Readable AI Workflows. Pick one to see, understand, and run it.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - CFM-R12-1: My Workflows

    private var myWorkflowsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Text("My Workflows")
                    .font(.title3.weight(.semibold))
                Spacer()
                // CFM — Import copies a picked flow folder (its .cat plus fixtures) into
                // the flows folder; Export copies a flow's whole folder out to the user.
                Button {
                    importFlow()
                } label: {
                    Label("Import", systemImage: "square.and.arrow.down")
                }
                .help("Copy a flow folder (.cat plus its audio, text, etc.) into My Workflows")
            }
            LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                newFlowBadge
                ForEach(appState.userFlowEntries) { entry in
                    userFlowBadge(entry)
                }
            }
            if appState.userFlowEntries.isEmpty {
                Text("Flows you save — or duplicate from the gallery — appear here. Start with New Flow, or Import one.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// The "Basic Gallery" shelf: the simple bundled flows — read-only like Advance
    /// Gallery (Duplicate & Edit / Reveal in Finder / Run), shown above it. Hidden
    /// when empty so a shelf with no basic flows renders no stray heading.
    private var basicGallerySection: some View {
        if appState.basicGalleryEntries.isEmpty {
            return AnyView(EmptyView())
        }
        return AnyView(
            VStack(alignment: .leading, spacing: 10) {
                Text("Basic Gallery")
                    .font(.title3.weight(.semibold))
                LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                    ForEach(Array(appState.basicGalleryEntries.enumerated()), id: \.element.id) { index, flow in
                        badge(for: flow, number: index + 1)
                    }
                }
            }
        )
    }

    /// The "Advance Gallery" shelf: every bundled flow that isn't basic. Numbered within its
    /// own shelf (starting at 1) just like Basic Gallery — the two shelves never share a
    /// number, so a "Transcribe Audio" badge in Basic Gallery reads 1 rather than 70.
    private var advanceGallerySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Advance Gallery")
                .font(.title3.weight(.semibold))
            LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                ForEach(Array(appState.advanceGalleryEntries.enumerated()), id: \.element.id) { index, flow in
                    badge(for: flow, number: index + 1)
                }
            }
        }
    }

    // MARK: - CFM: Import / Export

    /// CFM — Import: the user picks a flow folder (a `.cat`/`.catpipeline` plus its fixtures
    /// like audio and `.txt`); the whole folder is copied into the flows folder under a fresh
    /// id and the shelf refreshes. Failures surface as the app-level alert.
    private func importFlow() {
        let panel = NSOpenPanel()
        panel.title = "Import a Flow"
        panel.prompt = "Import"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            do {
                _ = try UserFlowStore.importFlow(from: url, workspace: FlowWorkspace.shared)
                appState.reloadUserFlows()
            } catch {
                appState.flowImportError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
            }
        }
    }

    /// CFM — Export: the user picks a destination folder; the flow's entire folder (its
    /// `.cat` plus fixtures) is copied there under the flow's title, then revealed in Finder.
    private func exportFlow(_ entry: UserFlowStore.Entry) {
        let panel = NSOpenPanel()
        panel.title = "Export '\(entry.title)'"
        panel.prompt = "Export"
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            do {
                let dest = try UserFlowStore.export(flowID: entry.flowID, name: entry.title,
                                                    to: url, workspace: FlowWorkspace.shared)
                NSWorkspace.shared.activateFileViewerSelecting([dest])
            } catch {
                appState.flowExportError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
            }
        }
    }

    /// A New Flow badge (CFM-R8) — assembles a flow from the step picker without touching
    /// text, saves it as `.cat`, and runs it.
    private var newFlowBadge: some View {
        Button {
            appState.editingFlow = FlowEditTarget(flowID: UUID().uuidString,
                                                  name: "Untitled Flow",
                                                  document: nil,
                                                  savedText: nil)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "plus")
                        .font(.title3)
                        .foregroundStyle(Color.accentColor)
                    Spacer()
                }
                Text("New Flow")
                    .font(.headline)
                Text("Build a flow from scratch — add steps, save as .cat, run it.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
            .background(Color.accentColor.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(Color.accentColor.opacity(0.4), style: StrokeStyle(lineWidth: 1, dash: [5]))
            }
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: 14))
    }

    /// CFM — a click on a My Workflows flow opens it **in the editor** (Edit mode, first row
    /// selected, inspector up on Properties). A broken file can't be edited — it falls back
    /// to the read-only list so the parse sentence shows.
    private func openUserFlow(_ entry: UserFlowStore.Entry) {
        guard entry.parseIssue == nil,
              let doc = try? UserFlowStore.loadDocument(entry: entry) else {
            appState.selectedFlow = FlowSelection(flowID: entry.flowID, isUserFlow: true)
            return
        }
        appState.editingFlow = FlowEditTarget(flowID: entry.flowID,
                                              name: entry.title,
                                              document: doc,
                                              savedText: CatSerializer.serialize(doc))
    }

    /// One user flow's badge: title + last-modified, a ⚠ when its file no longer parses.
    /// Opening it goes straight into the editor. The export/trash overlay (top-right) sits
    /// on top, so each captures its own click and never also opens the flow.
    private func userFlowBadge(_ entry: UserFlowStore.Entry) -> some View {
        Button {
            openUserFlow(entry)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "doc.plaintext")
                        .foregroundStyle(.secondary)
                    Spacer()
                    if entry.parseIssue != nil {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .accessibilityLabel("This flow doesn't parse")
                    }
                }
                Text(entry.title)
                    .font(.headline)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("Saved \(entry.modifiedAt.formatted(date: .abbreviated, time: .omitted))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(.quaternary, lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: 14))
        .overlay(alignment: .topTrailing) {
            HStack(spacing: 4) {
                Button {
                    exportFlow(entry)
                } label: {
                    Image(systemName: "square.and.arrow.up")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 22, height: 22)
                        .background(.quaternary.opacity(0.75), in: Circle())
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Export this flow to a folder you choose")
                Button {
                    flowPendingRemoval = entry
                } label: {
                    Image(systemName: "trash")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(width: 22, height: 22)
                        .background(.quaternary.opacity(0.75), in: Circle())
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Remove this flow")
            }
            .padding(8)
        }
    }

    /// One bundled gallery badge. `number` is the **shelf-local** number (the per-shelf
    /// ordinal 1…N, not the metadata's global `number`), so Basic and Advance each read
    /// starting at 1.
    private func badge(for flow: GalleryFlowMetadata, number: Int) -> some View {
        Button {
            appState.selectedFlow = FlowSelection(flowID: flow.flowID)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "flowchart")
                        .foregroundStyle(.secondary)
                    Spacer()
                    // CFM-R12-FIX-1: the badge's ⚠ comes from the live gates
                    // (`appState.galleryBlocked`), never a stale metadata string.
                    if appState.galleryBlocked.contains(flow.flowID) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .accessibilityLabel("Not runnable yet")
                    }
                }
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("\(number)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text(flow.title)
                        .font(.headline)
                }
                Text(flow.category)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(flow.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
            }
            .padding(14)
            .frame(maxWidth: .infinity, minHeight: 110, alignment: .leading)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 14))
            .overlay {
                RoundedRectangle(cornerRadius: 14)
                    .strokeBorder(.quaternary, lineWidth: 1)
            }
        }
        .buttonStyle(.plain)
        .contentShape(RoundedRectangle(cornerRadius: 14))
    }
}
