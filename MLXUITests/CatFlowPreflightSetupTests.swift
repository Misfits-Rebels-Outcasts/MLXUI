import Testing
import Foundation
@testable import MLXUI

/// Phase FIX, FIX-2 (`RSI/DelegateFixItBacklog.md`) — the pre-run setup pass now asks every
/// row, not only `.model` ones, whether it needs setup — routed to a **non-blocking**
/// advisory (`FlowPreflight.Result.rowAdvisories` / `setupAdvisory(_:)`), never into
/// `blocked`/`needsSetup`. §2 of the backlog is the trap this file exists to catch: those two
/// buckets are model-row-specific, and `FlowRunner.rowClassRefusal` deliberately keeps a
/// `.needsSetup` net row selectable and runnable (WS-3).
struct CatFlowPreflightSetupTests {

    /// A fixture Apple Foundation Models readiness, injected via `checkerOverride` — the same
    /// seam `CatFlowAppleFoundationTests` uses, redeclared here (that one's is `private` to its
    /// own file) so this file's "unchanged model-row refusal" test doesn't need a real device.
    private struct FixedAFMChecker: AppleFoundationAvailabilityChecking {
        let readiness: Readiness
    }

    private func resetAFMOverride() {
        AppleFoundationAvailability.checkerOverride = nil
    }

    private func withCleanCredentials<T>(_ body: () throws -> T) rethrows -> T {
        let tavilyAccount = KeychainHelper.providerAccount("tavily")
        let braveAccount = KeychainHelper.providerAccount("brave")
        let originalTavily = KeychainHelper.get(account: tavilyAccount)
        let originalBrave = KeychainHelper.get(account: braveAccount)
        KeychainHelper.delete(account: tavilyAccount)
        KeychainHelper.delete(account: braveAccount)
        defer {
            if let originalTavily { KeychainHelper.save(originalTavily, account: tavilyAccount) }
            if let originalBrave { KeychainHelper.save(originalBrave, account: braveAccount) }
        }
        return try body()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    // MARK: - A keyless `Web Search` row: advisory, not a block

    @Test func keylessWebSearchFlowYieldsAnAdvisoryAndStaysRunnable() throws {
        try withCleanCredentials {
            let doc = FlowDocument(version: "0.8", rows: [
                Row(task: "Read Text", settings: "memo.txt"),
                Row(task: "Web Search", settings: "\"weather in Tokyo\""),
                Row(task: "Save Text", settings: "out.md"),
            ])
            let result = FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 16)
            let advisory = try #require(FlowPreflight.setupAdvisory(result))
            #expect(advisory.task == "Web Search")
            #expect(advisory.action == .openSettings(.providers))
            // Never a block: `blockedReason`/`isBlocked` read `blocked`/`needsSetup` only —
            // both stay empty for a net row's advisory.
            #expect(FlowPreflight.blockedReason(result, totalRAMGB: 16) == nil)
            #expect(FlowPreflight.blockedAction(result) == nil)
            #expect(!result.isBlocked)
            // And `FlowRunner.canRun` — the other half of §2's trap — must still say runnable
            // (WS-3, tested independently in `CatFlowNotRunnableTests`/`CatFlowWebSearchTests`;
            // re-asserted here so FIX-2 can never regress it silently).
            #expect(FlowRunner.canRun(doc) == .runnable)
        }
    }

    @Test func webSearchFlowWithAKeySetHasNoAdvisory() throws {
        try withCleanCredentials {
            KeychainHelper.save("throwaway-\(UUID().uuidString)", account: KeychainHelper.providerAccount("tavily"))
            defer { KeychainHelper.delete(account: KeychainHelper.providerAccount("tavily")) }
            let doc = FlowDocument(version: "0.8", rows: [
                Row(task: "Web Search", settings: "\"weather in Tokyo\""),
            ])
            let result = FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 16)
            #expect(result.rowAdvisories.isEmpty)
            #expect(FlowPreflight.setupAdvisory(result) == nil)
        }
    }

    // MARK: - A `.model` row's blocking refusal is unchanged

    @Test func appleIntelligenceOffModelRowStillProducesTheBlockingRefusalUnchanged() throws {
        resetAFMOverride()
        defer { resetAFMOverride() }
        AppleFoundationAvailability.checkerOverride = FixedAFMChecker(
            readiness: .needsSetup(reason: "Turn on Apple Intelligence in System Settings",
                                   action: .enableAppleIntelligence))
        let doc = FlowDocument(version: "0.8", rows: [
            Row(task: "Summarize", model: "apple-foundation @ system", settings: "\"x\""),
        ])
        let result = FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 16)
        #expect(result.needsSetup.count == 1)
        #expect(result.blocked.isEmpty)
        #expect(!result.isBlocked)
        #expect(FlowPreflight.blockedReason(result, totalRAMGB: 16) == "Turn on Apple Intelligence in System Settings")
        #expect(FlowPreflight.blockedAction(result) == .enableAppleIntelligence)
        // FIX-2 touches only non-`.model` rows — a `.model` row never contributes to the new
        // advisory bucket, unchanged from before this phase.
        #expect(result.rowAdvisories.isEmpty)
    }

    // MARK: - Regression sweep: the new advisory pass never moves a gallery flow's verdict

    private func flatten(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + flatten($0.children) }
    }

    /// The pre-FIX-2 view of a document: only its `.model` rows, flattened out of any blocks
    /// (block nesting doesn't matter to `FlowPreflight.run`, which flattens internally too).
    private func modelRowsOnly(_ doc: FlowDocument) -> FlowDocument {
        let modelRows = flatten(doc.rows).filter { row in
            guard let task = row.task, let desc = TaskCatalog.get(task) else { return false }
            return desc.taskClass == .model
        }
        return FlowDocument(version: doc.version, headerKeyword: doc.headerKeyword, rows: modelRows,
                            models: doc.models, modelsOrder: doc.modelsOrder)
    }

    /// "The preflight verdict of every flow under `Resources/Gallery/` is identical to
    /// main's" (backlog, FIX-2's test list): every gallery flow's non-`.model` rows are the
    /// only thing FIX-2 added a code path for, and that path writes only to `rowAdvisories` —
    /// never `needs`. So the verdict computed from the *full* document must equal the verdict
    /// computed from the document with every non-`.model` row stripped, for every shipping
    /// flow. A future change that feeds `rowAdvisories` into `blocked`/`needsSetup` (§2's
    /// trap) would silently un-runnable a flow and fail this test.
    @Test func newAdvisoryPassNeverMovesAnyGalleryFlowsBlockedVerdict() throws {
        let catalog = try loadCatalog()
        let claimable = Set(catalog.map(\.id))
        var checked = 0
        for metadata in GalleryLoader.loadMetadata() {
            guard let doc = try? GalleryLoader.loadDocument(flowID: metadata.flowID) else { continue }
            checked += 1
            let full = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16,
                                         claimableModelIDs: claimable)
            let modelOnly = FlowPreflight.run(modelRowsOnly(doc), catalog: catalog, installedModelIDs: [],
                                              totalRAMGB: 16, claimableModelIDs: claimable)
            #expect(FlowPreflight.blockedReason(full, totalRAMGB: 16) == FlowPreflight.blockedReason(modelOnly, totalRAMGB: 16),
                    "flow \(metadata.flowID)'s blocked reason must not move")
            #expect(FlowPreflight.blockedAction(full) == FlowPreflight.blockedAction(modelOnly),
                    "flow \(metadata.flowID)'s blocked action must not move")
            #expect(full.isBlocked == modelOnly.isBlocked,
                    "flow \(metadata.flowID)'s isBlocked must not move")
        }
        #expect(checked > 0, "the gallery sweep must actually load flows, or it proves nothing")
    }
}
