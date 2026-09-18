import SwiftUI

/// CACHE-Q — the flow-detail ⋯ maintenance menu: **Clear Cache & Results** (and **Undo
/// Improvise** when the flow carries an Improvise row).
///
/// Extracted from `FlowListView.header` so `FlowEditorView` — the view a *working* My
/// Workflows flow opens in — carries the same control. Before this it lived only in
/// `FlowListView`, which serves the read-only gallery shelves and the parse-failure
/// fallback; a working user flow (the one you actually edit) had no way to force a cold
/// re-run of an *unchanged* flow. DA-5's determinism check and DA-10's `on_error=skip`
/// test both need one.
///
/// The host owns only the `extraItems` it appends (FlowListView's "Remove Flow…") and the
/// `onNotice` sink for the result sentence. `clearAll` is exactly `FlowListView`'s old
/// sequence: `clearCache()` then `clearRun(doc:)` — warm engines are a performance cache,
/// not results, so they are kept.
struct FlowMaintenanceMenu<ExtraItems: View>: View {
    let session: FlowRunSession
    let doc: FlowDocument
    let workspace: FlowWorkspace
    let flowID: String
    /// Called with the human-readable result of a clear / undo. `FlowListView` routes it to
    /// its existing `cacheClearNotice` state.
    var onNotice: (String) -> Void = { _ in }
    @ViewBuilder var extraItems: () -> ExtraItems

    @State private var showClearConfirm = false

    /// The same check `FlowListView` used inline: results on screen, or a non-empty store.
    private var hasAnythingToClear: Bool {
        session.hasRunResults || FlowCacheStore.shared.entryCount > 0
    }

    var body: some View {
        Menu {
            Button("Clear Cache & Results", systemImage: "trash") {
                showClearConfirm = true
            }
            .disabled(!hasAnythingToClear)
            // CFM-R10-FIX-6: an Improvise row's workdir can be restored to its pre-run
            // snapshot (Direct build only — the App Store tier refuses Improvise).
            if session.hasImprovise(doc) {
                Button("Undo Improvise", systemImage: "arrow.uturn.backward") {
                    do {
                        onNotice(try session.undoImprovise(doc: doc, workspace: workspace,
                                                           flowID: flowID))
                    } catch {
                        onNotice((error as CustomStringConvertible).description)
                    }
                }
            }
            extraItems()
        } label: {
            Image(systemName: "ellipsis.circle")
        }
        .menuStyle(.borderlessButton)
        .help("Clear cached results and reset the run display")
        .confirmationDialog("Clear cached results?", isPresented: $showClearConfirm) {
            Button("Clear Cache & Results", role: .destructive) { clearAll() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("The saved row outputs are deleted and every dot resets to gray. The next Run recomputes everything from scratch — this can take minutes.")
        }
    }

    /// The single destructive reset: drop the saved outputs *and* the run display, so the
    /// next Run is a genuinely fresh compute.
    private func clearAll() {
        let cleared = session.clearCache()
        session.clearRun(doc: doc)
        onNotice(FlowMaintenance.clearNotice(clearedCount: cleared))
    }
}

/// The non-View helpers behind `FlowMaintenanceMenu`, split out so they are testable without
/// importing SwiftUI.
nonisolated enum FlowMaintenance {
    /// The result sentence for a Clear Cache & Results action. A `nil` count means the store
    /// was already empty.
    static func clearNotice(clearedCount: Int?) -> String {
        if let clearedCount, clearedCount > 0 {
            return "Cleared \(clearedCount) cached output\(clearedCount == 1 ? "" : "s"); dots reset."
        }
        return "Dots reset. The cache was already empty."
    }
}

extension FlowMaintenanceMenu where ExtraItems == EmptyView {
    /// No trailing items (the flow editor — a user flow is removed from its own list, not
    /// this menu).
    init(session: FlowRunSession, doc: FlowDocument, workspace: FlowWorkspace, flowID: String,
         onNotice: @escaping (String) -> Void = { _ in }) {
        self.init(session: session, doc: doc, workspace: workspace, flowID: flowID,
                  onNotice: onNotice, extraItems: { EmptyView() })
    }
}
