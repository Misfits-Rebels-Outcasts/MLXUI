import SwiftUI

/// The Automate → "AI Workflows" gallery: every bundled flow rendered as a badge in the
/// detail panel. Selecting a badge pushes the flow's detail (`FlowListView` — title, rows,
/// inspector) onto the detail stack, leaving the sidebar on "AI Workflows".
struct FlowGalleryView: View {
    @Environment(AppState.self) private var appState

    private let columns = [GridItem(.adaptive(minimum: 220), spacing: 14)]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                    newFlowBadge
                    ForEach(appState.galleryEntries) { flow in
                        badge(for: flow)
                    }
                }
            }
            .padding(20)
        }
        .navigationTitle("AI Workflows")
    }

    /// The content-panel title: a visible heading above the badge grid (the navigation
    /// title alone reads as a window label, not a page heading).
    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("AI Workflows")
                .font(.largeTitle.weight(.bold))
            Text("\(appState.galleryEntries.count) bundled flows — pick one to see and run it, or start your own.")
                .font(.callout)
                .foregroundStyle(.secondary)
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

    private func badge(for flow: GalleryFlowMetadata) -> some View {
        Button {
            appState.selectedFlow = FlowSelection(flowID: flow.flowID)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Image(systemName: "flowchart")
                        .foregroundStyle(.secondary)
                    Spacer()
                    if !flow.isRunnable {
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
