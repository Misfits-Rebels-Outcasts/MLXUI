import Testing
import Foundation
@testable import MLXUI

/// MC5-T-chat: what the chat `generate_music` tool does with ACE-Step 1.5 and MusicGen, as `CapabilityResolver.status` decides it today (owner-approved F6 rule: walk the pool in preference
/// order, keep what FITS this Mac by the same `ramGB <= RAM` rule Browse dims by, take the first INSTALLED one, else offer the first fitting download). Pool order is ACE-Step first (MC5-D).
/// These pin today's behaviour; the cases that need an owner ruling (a 16 GB Mac, an installed model over RAM, licence wording) are deliberately not here yet.
struct ChatGenerateMusicCapabilityTests {

    private static let aceID = "ACE-Step--Ace-Step1.5"
    private static let musicGenID = "jasonvassallo--mlx-musicgen-small"

    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    @MainActor private func claimable(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    private func status(installed: Set<String>, ram: Double) async throws -> ModelToolStatus {
        let cat = try catalog()
        let claim = await claimable(cat)
        let spec = try #require(ModelToolSpec.spec(named: "generate_music"))
        return CapabilityResolver.status(for: spec, catalog: cat, installedModelIDs: installed, claimableModelIDs: claim, totalRAMGB: ram)
    }

    @Test func musicGenOnlyInstalledIsReadyWithNoOffer() async throws {
        guard case .ready(let model) = try await status(installed: [Self.musicGenID], ram: 19) else { Issue.record("not ready"); return }
        #expect(model.id == Self.musicGenID)
    }

    @Test func aceStepOnlyInstalledIsReady() async throws {
        guard case .ready(let model) = try await status(installed: [Self.aceID], ram: 19) else { Issue.record("not ready"); return }
        #expect(model.id == Self.aceID)
    }

    @Test func bothInstalledPicksAceStep() async throws {
        guard case .ready(let model) = try await status(installed: [Self.aceID, Self.musicGenID], ram: 19) else { Issue.record("not ready"); return }
        #expect(model.id == Self.aceID)
    }

    @Test func nothingInstalledOn19GBOffersAceStepWithItsCatalogSize() async throws {
        guard case .needsDownload(let model, let gb) = try await status(installed: [], ram: 19) else { Issue.record("not needsDownload"); return }
        #expect(model.id == Self.aceID)
        #expect(gb == model.downloadSizeGB && gb == 6.33)
    }

    @Test func nothingInstalledOn8GBOffersMusicGenBecauseAceStepDoesNotFit() async throws {
        let cat = try catalog()
        let ace = try #require(cat.first { $0.id == Self.aceID })
        #expect(ace.ramGB > 8, "ACE-Step is dimmed on an 8 GB Mac by the same rule")
        guard case .needsDownload(let model, let gb) = try await status(installed: [], ram: 8) else { Issue.record("not needsDownload"); return }
        #expect(model.id == Self.musicGenID)
        #expect(gb == model.downloadSizeGB && gb == 2.5)
    }

    // MARK: Owner rulings, 2026-11-21 (A, B, C)

    /// A: one fit rule, Browse's dim test (`ramGB > Mac RAM`). ACE-Step's 10.9 GB fits a 16 GB Mac, so it is offered there. (Whether a measured 16 GB run behaves is a backlog follow-up: the
    /// booking evicts the other models and proceeds over the 9.6 GiB ledger rather than refusing, MC5-T-book.)
    @Test func nothingInstalledOn16GBOffersAceStepBecauseItFitsByTheDimRule() async throws {
        let cat = try catalog()
        let ace = try #require(cat.first { $0.id == Self.aceID })
        #expect(ace.ramGB <= 16)
        guard case .needsDownload(let model, _) = try await status(installed: [], ram: 16) else { Issue.record("not needsDownload"); return }
        #expect(model.id == Self.aceID)
    }

    /// B: an installed model over this Mac's RAM is not a candidate (it could not run: `StageFactory` refuses `ramGB > RAM`), so on an 8 GB Mac with ACE-Step installed the chat offers MusicGen and
    /// never routes to ACE-Step.
    @Test func anInstalledAceStepOn8GBIsNeverRoutedToAndMusicGenIsOffered() async throws {
        guard case .needsDownload(let model, _) = try await status(installed: [Self.aceID], ram: 8) else { Issue.record("not needsDownload"); return }
        #expect(model.id == Self.musicGenID)
    }

    // MARK: C: the install card names model, size and licence, all from the catalog entry

    private func option(_ id: String) throws -> ModelInstallOption {
        let e = try #require(try catalog().first { $0.id == id })
        return ModelInstallOption(id: e.id, displayName: TaskModels.displayName(for: e), downloadGB: e.downloadSizeGB, license: e.license)
    }

    @MainActor @Test func theOfferLinesNameModelSizeAndLicence() throws {
        #expect(ModelInstallCard.offerText(try option(Self.aceID)) == "ACE-Step 1.5 · 6.3 GB · MIT")
        #expect(ModelInstallCard.offerText(try option(Self.musicGenID)) == "MusicGen · 2.5 GB · non-commercial")
        #expect(ModelInstallCard.offerText(ModelInstallOption(id: "x", displayName: "X", downloadGB: 0.6)) == "X · 0.6 GB", "no licence, no licence part")
    }

    @MainActor @Test func theCardsDetailLineCarriesTheLicenceAndTheOldFormIsUnchangedWithoutOne() throws {
        let ace = ModelInstallCard(conversationID: nil, toolName: "generate_music", target: try option(Self.aceID), alternatives: [], phase: .offering)
        #expect(ace.headline == "Making sound needs ACE-Step 1.5")
        #expect(ace.detail == "6.3 GB download · MIT · runs on this Mac only")
        let mg = ModelInstallCard(conversationID: nil, toolName: "generate_music", target: try option(Self.musicGenID), alternatives: [], phase: .offering)
        #expect(mg.detail == "2.5 GB download · non-commercial · runs on this Mac only")
        let plain = ModelInstallCard(conversationID: nil, toolName: "generate_music", target: ModelInstallOption(id: "x", displayName: "X", downloadGB: 0.6), alternatives: [], phase: .offering)
        #expect(plain.detail == "0.6 GB download · runs on this Mac only")
    }

    /// The router's own suffix is unchanged: size only, never a model name or a licence.
    @Test func theRoutersSuffixIsUnchanged() async throws {
        let status = try await status(installed: [], ram: 19)
        #expect(status.routerSuffix == "[needs a one-time 6.3 GB download]")
    }
}
