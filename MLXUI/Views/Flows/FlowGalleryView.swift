import SwiftUI

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
                // it is a user action, not a bundled flow.
                myWorkflowsSection
                gallerySection
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
            Text("\(appState.galleryEntries.count) bundled flows — and anything you've saved under My Workflows. Pick one to see and run it, or start your own.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - CFM-R12-1: My Workflows

    private var myWorkflowsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("My Workflows")
                .font(.title3.weight(.semibold))
            LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                newFlowBadge
                ForEach(appState.userFlowEntries) { entry in
                    userFlowBadge(entry)
                }
            }
            if appState.userFlowEntries.isEmpty {
                Text("Flows you save — or duplicate from the gallery — appear here. Start with New Flow.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var gallerySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Gallery Workflows")
                .font(.title3.weight(.semibold))
            LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                ForEach(appState.galleryEntries) { flow in
                    badge(for: flow)
                }
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

    /// One user flow's badge: title + last-modified, a ⚠ when its file no longer parses.
    /// Opening it pushes `FlowListView` in the user source. The trash overlay (top-right)
    /// removes the flow folder with a confirmation — it sits on top, so it captures its own
    /// click and never also opens the flow.
    private func userFlowBadge(_ entry: UserFlowStore.Entry) -> some View {
        Button {
            appState.selectedFlow = FlowSelection(flowID: entry.flowID, isUserFlow: true)
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
            .padding(8)
        }
    }

    private func badge(for flow: GalleryFlowMetadata) -> some View {
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
                    Text("\(flow.number)")
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
