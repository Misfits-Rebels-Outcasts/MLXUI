import SwiftUI

/// The flow detail view. R1 shipped it read-only; CFM-R2-8 wires Run: preflight → one
/// install sheet → the `FlowEvent` stream driving each row's dot `○ → ● → ✓`, error
/// sentences on `✗`, and a Cancel that keeps earned dots. Run is disabled with the reason
/// when `FlowRunner.canRun` says no, and blocked models disable it with the bridge's reason.
struct FlowListView: View {
    let flowID: String

    @Environment(AppState.self) private var appState
    @State private var metadata: GalleryFlowMetadata?
    @State private var document: FlowDocument?
    @State private var rawCat: String?
    @State private var loadError: String?
    @State private var session = FlowRunSession()

    var body: some View {
        Group {
            if let error = loadError {
                ContentUnavailableView("Couldn't Load This Flow",
                                       systemImage: "exclamationmark.triangle",
                                       description: Text(error))
            } else if let metadata, metadata.notRunnableReason != nil, let rawCat {
                notRunnableView(metadata, rawCat)
            } else if let document, let metadata {
                flowList(document, metadata)
            } else {
                ProgressView("Loading flow…")
                    .onAppear(perform: load)
            }
        }
        .navigationTitle(metadata?.title ?? flowID)
        .sheet(isPresented: $session.showInstallSheet) {
            if let result = session.preflight {
                installSheet(result)
                    .environment(appState)
            }
        }
    }

    /// A read-only view for a flow this version can't run (CFM-R4-4): an honest badge
    /// naming what it needs, the description, and the raw `.cat` text. No Run button.
    private func notRunnableView(_ meta: GalleryFlowMetadata, _ raw: String) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Label(meta.title, systemImage: "flowchart")
                    .font(.title2.weight(.semibold))
                Text("catflow 0.8")
                    .font(.caption.monospaced())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
                Spacer()
            }
            Label(meta.notRunnableReason ?? "", systemImage: "exclamationmark.triangle.fill")
                .font(.callout)
                .foregroundStyle(.orange)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.orange.opacity(0.10), in: RoundedRectangle(cornerRadius: 8))
            Text(meta.description)
                .font(.callout)
                .foregroundStyle(.secondary)
            DisclosureGroup("Show raw `.cat`") {
                ScrollView(.horizontal) {
                    Text(raw)
                        .font(.system(.caption, design: .monospaced))
                        .textSelection(.enabled)
                        .padding(8)
                }
                .frame(maxWidth: .infinity)
                .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 6))
            }
            .font(.callout)
            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    // MARK: - Content

    private func flowList(_ doc: FlowDocument, _ meta: GalleryFlowMetadata) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            header(meta, doc)
            Divider()
            HStack(spacing: 0) {
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
                                        status: session.status(for: row.id))
                                .contentShape(Rectangle())
                                .onTapGesture { session.selectedRowID = row.id }
                                .background(session.selectedRowID == row.id ? Color.accentColor.opacity(0.12) : Color.clear)
                            if let sentence = session.errorSentence(for: row.id) {
                                Text(sentence)
                                    .font(.caption)
                                    .foregroundStyle(.red)
                                    .padding(.leading, 48)
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
                    substitutionNote: session.selectedRowID.flatMap { session.substitutionNotes[$0] }
                )
            }
            rawCatDisclosure
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
    }

    private func selectedRowTitle(_ doc: FlowDocument) -> String {
        guard let id = session.selectedRowID,
              let row = doc.rows.first(where: { $0.id == id }) else { return "" }
        return FlowRowSummary.taskName(for: row)
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
                try? FlowWorkspace.shared.prepare(
                    flowID: flowID,
                    sourceDir: GalleryLoader.resourcesDirectory,
                    bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
                FlowWorkspace.shared.revealInFinder(flowID: flowID)
            } label: {
                Label("Reveal in Finder", systemImage: "folder")
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
            Menu {
                Button("Clear Run", systemImage: "arrow.counterclockwise") {
                    confirmClearRun(doc)
                }
                .disabled(session.hasRunResults)
                Button("Clear Cache", systemImage: "trash") {
                    confirmClearCache()
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
            .menuStyle(.borderlessButton)
            .help("Clear run results or the flow cache")
        }
        .padding(16)
        .confirmationDialog("Clear run results?", isPresented: $showClearRunConfirm) {
            Button("Clear", role: .destructive) { session.clearRun(doc: doc) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Every row's dot resets to gray and the inspector outputs are dropped. The cache is kept.")
        }
        .confirmationDialog("Clear the flow cache?", isPresented: $showClearCacheConfirm) {
            Button("Clear Cache", role: .destructive) {
                let cleared = session.clearCache()
                if cleared == nil {
                    cacheClearNotice = "The cache was already empty."
                } else {
                    cacheClearNotice = "Cleared \(cleared ?? 0) cached row\(cleared == 1 ? "" : "s")."
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Stored row outputs are deleted. The next run recomputes everything.")
        }
    }

    @State private var showClearRunConfirm = false
    @State private var showClearCacheConfirm = false
    @State private var cacheClearNotice: String?

    private func confirmClearRun(_ doc: FlowDocument) { showClearRunConfirm = true }
    private func confirmClearCache() { showClearCacheConfirm = true }

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

    // MARK: - Run

    private func run(_ doc: FlowDocument) {
        // Preflight decides installs; the session gates Run until models exist.
        let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
        let result = FlowPreflight.run(doc, catalog: catalog,
                                       installedModelIDs: appState.installedModelIDs,
                                       totalRAMGB: appState.systemInfo.totalRAMGB)
        session.prepareInstall(result, doc: doc)
        guard !result.toDownload.isEmpty else {
            startRun(doc)
            return
        }
        // The install sheet's confirm callback starts the run.
    }

    private func startRun(_ doc: FlowDocument) {
        // Re-run from here when some rows already have results; a fresh run otherwise.
        let resume = session.hasRunResults
        let context = AppFlowExecutorFactory.cachingContext(flowID: flowID, appState: appState)
        session.start(doc: doc, runner: FlowRunner(), context: context, resume: resume)
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
                Button("Cancel") { session.showInstallSheet = false }
                    .keyboardShortcut(.cancelAction)
                Button("Install & Run") {
                    session.showInstallSheet = false
                    installAndRun(result)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    private func installAndRun(_ result: FlowPreflight.Result) {
        let models = Array(result.downloadSet)
        Task {
            await installSequentially(models)
        }
    }

    /// Drive `InstallManager.install` **sequentially** — one model at a time, awaiting each
    /// `.installed` marker before starting the next, then Run once all are installed.
    private func installSequentially(_ models: [ModelEntry]) async {
        guard let doc = document else { return }
        for model in models {
            appState.installModel(model)
            let installed = await InstallPoller.awaitInstalled(modelID: model.id,
                                                               installManager: installManager)
            guard installed else {
                // A failed install leaves the flow ready to retry; don't run half-installed.
                return
            }
        }
        startRun(doc)
    }

    private var installManager: InstallManager {
        appState.installManager
    }

    // MARK: - Loading

    private func load() {
        do {
            metadata = GalleryLoader.loadMetadata().first { $0.flowID == flowID }
            rawCat = try? GalleryLoader.rawCatText(flowID: flowID)
            // Not-runnable flows (CFM-R4-4) have no parse document — metadata + raw .cat is
            // enough for their read-only view.
            if let meta = metadata, meta.notRunnableReason != nil {
                return
            }
            document = try GalleryLoader.loadDocument(flowID: flowID)
            if let doc = document {
                let catalog = appState.browserData?.domains.flatMap { $0.allModels } ?? []
                session.prepareInstall(FlowPreflight.run(doc, catalog: catalog,
                                                          installedModelIDs: appState.installedModelIDs,
                                                          totalRAMGB: appState.systemInfo.totalRAMGB),
                                       doc: doc)
            }
        } catch {
            loadError = (error as? CustomStringConvertible)?.description ?? error.localizedDescription
        }
    }
}
