import SwiftUI

/// The read-only flow detail view (CFM-R1-5): a friendly header, the flow's rows as a
/// numbered list with gray status dots, chain breaks as dividers, and a "Show raw `.cat`"
/// disclosure at the bottom. The Run button is present but disabled — its presence is what
/// makes the section legible as unfinished rather than broken.
struct FlowListView: View {
    let flowID: String

    @State private var metadata: GalleryFlowMetadata?
    @State private var document: FlowDocument?
    @State private var rawCat: String?
    @State private var loadError: String?

    var body: some View {
        Group {
            if let error = loadError {
                ContentUnavailableView("Couldn't Load This Flow",
                                       systemImage: "exclamationmark.triangle",
                                       description: Text(error))
            } else if let document, let metadata {
                flowList(document, metadata)
            } else {
                ProgressView("Loading flow…")
                    .onAppear(perform: load)
            }
        }
        .navigationTitle(metadata?.title ?? flowID)
    }

    // MARK: - Content

    private func flowList(_ doc: FlowDocument, _ meta: GalleryFlowMetadata) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            header(meta, doc)
            Divider()
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(doc.rows.enumerated()), id: \.element.id) { index, row in
                        if FlowRowSummary.hasChainBreakBefore(row) {
                            Divider()
                                .padding(.leading, 48)
                        }
                        FlowRowView(row: row,
                                    number: index + 1,
                                    referenceScope: doc.rows,
                                    status: .notRun)
                    }
                }
                .padding(12)
            }
            rawCatDisclosure
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }

    private func header(_ meta: GalleryFlowMetadata, _ doc: FlowDocument) -> some View {
        HStack(spacing: 12) {
            Text(meta.title)
                .font(.title2.weight(.semibold))
            Text("catflow \(doc.version)")
                .font(.caption.monospaced())
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
            Spacer()
            Button {
                FlowWorkspace.shared.revealInFinder(flowID: flowID)
                try? FlowWorkspace.shared.prepare(
                    flowID: flowID,
                    sourceDir: GalleryLoader.resourcesDirectory,
                    bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
                FlowWorkspace.shared.revealInFinder(flowID: flowID)
            } label: {
                Label("Reveal in Finder", systemImage: "folder")
            }
            Button(action: {}) {
                Label("Run", systemImage: "play.fill")
            }
            .disabled(true)
            .help("Running arrives in the next update.")
        }
        .padding(16)
    }

    private var rawCatDisclosure: some View {
        DisclosureGroup("Show raw `.cat`") {
            if let rawCat {
                ScrollView(.horizontal) {
                    Text(rawCat)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                }
                .frame(maxWidth: .infinity)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
            }
        }
        .font(.callout)
        .padding(.horizontal, 16)
        .padding(.bottom, 16)
    }

    // MARK: - Loading

    private func load() {
        do {
            metadata = GalleryLoader.loadMetadata().first { $0.flowID == flowID }
            document = try GalleryLoader.loadDocument(flowID: flowID)
            rawCat = try? GalleryLoader.rawCatText(flowID: flowID)
        } catch {
            loadError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
        }
    }
}
