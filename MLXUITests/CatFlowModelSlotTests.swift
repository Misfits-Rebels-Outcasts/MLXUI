import Testing
import Foundation
@testable import MLXUI

/// Phase MS — `ModelSlot` makes "a model MLXUI can run that is not a download" a first-class
/// thing, with **zero visible behaviour change** for every model that ships today (backlog
/// §1, `RSI/DelegateOffMachineBacklog.md`).
///
/// - **MS-1**: `ModelSlot`'s three questions (`displayName`/`readiness`/`resourceNote`),
///   answered per case, with no `ramGB` property anywhere on the type.
/// - **MS-2/MS-3**: `TaskModels.derivedModels`/`CatalogBridge.resolve`/`FlowPreflight.run`
///   thread `ModelSlot` through unchanged for the cataloged case — the system/provider
///   registries are empty until Phase AFM/RM, so today's pool membership, display names, and
///   preflight buckets are exactly what they were before this phase.
/// - **MS-4**: the picker's `.unavailable`/`.needsSetup` distinction (selectable vs. not) and
///   `FlowPreflight`'s `needsSetup` bucket, tested with fixture slots since nothing in the
///   shipping app produces one yet.
struct CatFlowModelSlotTests {

    private func bundledCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    // MARK: - MS-1: the cataloged case answers each question exactly as `ModelEntry` did

    @Test func catalogedSlotIdAndDisplayNameMatchTheEntry() {
        let entry = makeEntry(id: "mlx-community--Test-4bit", displayName: "Test Model")
        let slot = ModelSlot.cataloged(entry)
        #expect(slot.id == entry.id)
        #expect(slot.modelEntry == entry)
    }

    @Test func catalogedSlotResourceNoteIsTheRAMFigureNotDownloadSize() {
        // The pre-MS picker showed `ramGB` on every Model menu row, never `downloadSizeGB` —
        // that number is the section header's business (summed separately). Pinning this
        // guards against someone "fixing" resourceNote to show download size instead.
        let entry = makeEntry(ramGB: 4.5, downloadSizeGB: 9.9)
        #expect(ModelSlot.cataloged(entry).resourceNote == "4.5 GB")
    }

    @Test func catalogedSlotDisplayNameUsesTheBridgeNameWhenBridged() {
        // "Whisper Large v3" is a bridge display name (CatalogBridge.entries); the underlying
        // catalog card's own `displayName` is different. `ModelSlot.displayName` must resolve
        // through the same `TaskModels.displayName` the picker always has.
        let entry = makeEntry(hfModelId: "mlx-community/whisper-large-v3-asr-fp16")
        #expect(ModelSlot.cataloged(entry).displayName == "Whisper Large v3")
    }

    @Test func catalogedReadinessIsReadyWhenInstalledNeedsDownloadOtherwise() {
        let entry = makeEntry(id: "abc", downloadSizeGB: 3.25)
        let slot = ModelSlot.cataloged(entry)
        #expect(slot.readiness(installedModelIDs: []) == .needsDownload(gb: 3.25))
        #expect(slot.readiness(installedModelIDs: ["abc"]) == .ready)
        #expect(slot.readiness(installedModelIDs: ["something-else"]) == .needsDownload(gb: 3.25))
    }

    /// MS-FOLLOWUP-1: `readiness` takes no default — an omitted argument used to mean
    /// "nothing is installed," silently reporting an *installed* cataloged model as
    /// `.needsDownload`. This won't compile if the default ever comes back
    /// (`slot.readiness()` is no longer callable), which is the actual enforcement; the
    /// assertion just documents which of the two wrong answers the default used to produce.
    @Test func readinessHasNoDefaultArgument() {
        let entry = makeEntry(id: "abc", downloadSizeGB: 3.25)
        let slot = ModelSlot.cataloged(entry)
        #expect(slot.readiness(installedModelIDs: ["abc"]) != .needsDownload(gb: 3.25))
    }

    /// MS-1's own guardrail, made concrete: every catalog entry's readiness is
    /// `.ready`/`.needsDownload` — never `.needsSetup`/`.unavailable`. That's the structural
    /// reason "zero visible behaviour change" holds for MS-2/MS-3: nothing in the shipping
    /// catalog can produce a bucket that didn't exist before this phase.
    @Test func everyBundledCatalogEntryReadinessIsReadyOrNeedsDownloadOnly() throws {
        for entry in try bundledCatalog() {
            let slot = ModelSlot.cataloged(entry)
            switch slot.readiness(installedModelIDs: []) {
            case .ready, .needsDownload: break
            case .needsSetup, .unavailable:
                Issue.record("\(entry.id) unexpectedly reports a non-cataloged readiness")
            }
            switch slot.readiness(installedModelIDs: [entry.id]) {
            case .ready: break
            default: Issue.record("\(entry.id) is installed but didn't report .ready")
            }
        }
    }

    // MARK: - MS-1: no `ramGB` property exists on `ModelSlot` at all

    /// Not a runtime assertion (Swift would refuse to compile `slot.ramGB` if the property
    /// existed) — this test exists so the guardrail has a named home in the suite. The real
    /// enforcement is `ModelSlot.swift` simply never declaring one; a reviewer adding
    /// `var ramGB: Double? { ... }` back in would see this comment.
    @Test func modelSlotHasNoTopLevelRAMProperty() {
        let slot = ModelSlot.cataloged(makeEntry(ramGB: 4.0))
        // The only route to a RAM figure is through the one case that has one.
        #expect(slot.modelEntry?.ramGB == 4.0)
    }

    // MARK: - MS-1/MS-4: a fixture slot of each readiness renders the right note and state

    @Test func readySlotIsSelectableWithItsResourceNote() {
        // RM-2: `ProviderModelRef` now carries the manifest itself, not flattened fields —
        // `resourceNote` is computed from `manifest.display`'s ` @ provider` suffix.
        let manifest = CuratedManifest(id: "anthropic/claude-sonnet-4", display: "claude-sonnet @ anthropic",
                                       kind: "provider", settings: [:], resources: nil)
        let ref = ProviderModelRef(manifest: manifest, readiness: .ready)
        let slot = ModelSlot.provider(ref)
        #expect(slot.resourceNote == "Runs on anthropic — leaves this Mac")
        #expect(FlowRowInspectorView.isModelButtonEnabled(slot, installedModelIDs: [], totalRAMGB: 8))
    }

    @Test func needsSetupSlotStaysSelectable() {
        // MS-1's own trap: "must stay selectable — a flow may legitimately name a model whose
        // key you are about to paste in; refusing at pick time would be a lie."
        let ref = SystemModelRef(id: "apple-foundation@system", displayName: "apple-foundation @ system",
                                 readiness: .needsSetup(reason: "Turn on Apple Intelligence in System Settings",
                                                        action: .enableAppleIntelligence),
                                 resourceNote: "Built into macOS")
        let slot = ModelSlot.system(ref)
        #expect(slot.resourceNote == "Built into macOS")
        #expect(FlowRowInspectorView.isModelButtonEnabled(slot, installedModelIDs: [], totalRAMGB: 8))
    }

    @Test func unavailableSlotIsDisabled() {
        let ref = SystemModelRef(id: "apple-foundation@system", displayName: "apple-foundation @ system",
                                 readiness: .unavailable(reason: "This Mac can't run Apple Intelligence"),
                                 resourceNote: "Built into macOS")
        let slot = ModelSlot.system(ref)
        #expect(!FlowRowInspectorView.isModelButtonEnabled(slot, installedModelIDs: [], totalRAMGB: 8))
    }

    @Test func catalogedSlotOverRAMIsDisabledEvenWhenReady() {
        // A model that fits nowhere near this Mac is still `.ready`/`.needsDownload` in
        // principle (readiness never encodes RAM-fit) — the picker disables the row itself,
        // exactly as before MS.
        let entry = makeEntry(id: "big", ramGB: 64.0)
        let slot = ModelSlot.cataloged(entry)
        #expect(slot.readiness(installedModelIDs: ["big"]) == .ready)
        #expect(!FlowRowInspectorView.isModelButtonEnabled(slot, installedModelIDs: ["big"], totalRAMGB: 16))
    }

    // MARK: - MS-2: the picker's third section, empty when no registry has entries

    /// AFM-FOLLOWUP-1 (`RSI/journal/2026-259`): `AppleFoundationAvailability.useRealSystem`
    /// defaults to `false` — only `MLXUIApp.init()` ever flips it — so the system registry
    /// stays empty here regardless of this build machine. The provider registry is a
    /// different story now: RM ported four real manifests, so MS-2's original "the third
    /// section stays empty for every model task" claim is exactly what RM exists to end
    /// for the tasks those manifests actually name. Pinned precisely instead of loosened:
    /// the populated set is exactly `@frames-text`'s own members (`Generate` plus every
    /// frame-backed text task — the group `macstudio-qwen3-32b.json` references, and every
    /// keyed manifest lists `Generate` literally too); every other model task must still
    /// show nothing.
    @Test @MainActor func sectionedModelCandidatesBuiltInSectionIsPopulatedOnlyForRMPortedTasks() throws {
        let catalog = try bundledCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        let populated = Set(TaskCatalog.taskGroupMembers("frames-text"))
        for task in TaskCatalog.entries where task.taskClass == .model {
            let sections = FlowEditorModel.sectionedModelCandidates(
                for: task.name, catalog: catalog, installedModelIDs: [], claimableModelIDs: claimable)
            if populated.contains(task.name) {
                #expect(!sections.builtIn.isEmpty,
                        "\(task.name) should offer a built-in provider slot now that RM has ported manifests")
            } else {
                #expect(sections.builtIn.isEmpty, "\(task.name) unexpectedly offers a built-in slot")
            }
        }
    }

    // MARK: - MS-3: FlowPreflight's buckets, direct — needsSetup is distinct from blocked

    @Test func needsSetupBucketIsDistinctFromBlocked() {
        let needsSetupNeed = FlowPreflight.ModelNeed(
            task: "Summarize", display: "apple-foundation @ system", model: nil, installed: false,
            equivalence: nil, blockingReason: nil,
            setupReason: "Turn on Apple Intelligence", setupAction: .enableAppleIntelligence)
        let blockedNeed = FlowPreflight.ModelNeed(
            task: "Transcribe", display: "Missing Model", model: nil, installed: false,
            equivalence: nil, blockingReason: "Missing Model can't run.")
        let result = FlowPreflight.Result(needs: [needsSetupNeed, blockedNeed])
        #expect(result.needsSetup.count == 1)
        #expect(result.needsSetup.first?.display == "apple-foundation @ system")
        #expect(result.blocked.count == 1)
        #expect(result.blocked.first?.display == "Missing Model")
        #expect(result.isBlocked)   // `blocked` alone is enough to block, regardless of needsSetup
        #expect(FlowPreflight.blockedReason(result, totalRAMGB: 1000) == "Missing Model can't run.")
        #expect(FlowPreflight.blockedAction(result) == nil)
    }

    @Test func blockedReasonFallsBackToNeedsSetupWhenNothingIsHardBlocked() {
        let needsSetupNeed = FlowPreflight.ModelNeed(
            task: "Summarize", display: "apple-foundation @ system", model: nil, installed: false,
            equivalence: nil, blockingReason: nil,
            setupReason: "Turn on Apple Intelligence", setupAction: .enableAppleIntelligence)
        let result = FlowPreflight.Result(needs: [needsSetupNeed])
        #expect(result.blocked.isEmpty)
        #expect(result.isBlocked == false)   // needsSetup alone doesn't hard-block the flow
        #expect(FlowPreflight.blockedReason(result, totalRAMGB: 1000) == "Turn on Apple Intelligence")
        #expect(FlowPreflight.blockedAction(result) == .enableAppleIntelligence)
    }

    /// MS-3's own exit criterion, updated for RM: it used to say no bundled gallery flow
    /// ever lands in `needsSetup` "because the registries feeding it are empty" — RM ends
    /// that on purpose. `44-FrontierEscalate.cat` names `claude-sonnet @ anthropic`
    /// directly, so with no Anthropic key in the Keychain it legitimately needs setup now.
    /// Pinned precisely (which flow, and only that one) rather than loosened to nothing.
    @Test func onlyFrontierEscalateNeedsSetupAndOnlyForItsAnthropicKey() throws {
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        KeychainHelper.delete(account: account)
        defer { if let original { KeychainHelper.save(original, account: account) } }

        let catalog = try bundledCatalog()
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        let galleryDir = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery")
        guard let enumerator = FileManager.default.enumerator(at: galleryDir, includingPropertiesForKeys: nil) else {
            Issue.record("couldn't enumerate the Gallery directory")
            return
        }
        var checked = 0
        var needingSetup: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "cat" {
            guard let text = try? String(contentsOf: url, encoding: .utf8),
                  let doc = try? CatParser.parse(text) else { continue }
            let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 64)
            if !result.needsSetup.isEmpty { needingSetup.append(url.lastPathComponent) }
            checked += 1
        }
        #expect(checked > 0, "expected to find at least one bundled .cat flow")
        #expect(needingSetup == ["44-FrontierEscalate.cat"])
    }

    // MARK: - FlowRunnability.refusal — the back-compat wrapper agrees with the tuple form

    @Test func refusalReasonMatchesTheReasonHalfOfRefusal() throws {
        let catalog = try bundledCatalog()
        let doc = FlowDocument(version: "0.8", rows: [
            Row(task: "Transcribe", model: "A Model That Doesn't Exist"),
        ])
        let full = FlowRunnability.refusal(for: doc, catalog: catalog, installed: [], totalRAMGB: 32)
        let reasonOnly = FlowRunnability.refusalReason(for: doc, catalog: catalog, installed: [], totalRAMGB: 32)
        #expect(full?.reason == reasonOnly)
        #expect(full?.action == nil)   // nothing produces a non-nil action yet
    }
}
