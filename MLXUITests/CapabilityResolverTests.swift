import Testing
import Foundation
@testable import MLXUI

/// MT1-2 — `CapabilityResolver` is pure; these run on the real catalog plus small fixtures.
struct CapabilityResolverTests {

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

    private func entry(_ hf: String, in catalog: [ModelEntry]) throws -> ModelEntry {
        try #require(catalog.first { $0.hfModelId == hf }, "\(hf) not in catalog")
    }

    private func spec(_ name: String) throws -> ModelToolSpec { try #require(ModelToolSpec.spec(named: name)) }

    private func status(_ tool: String, installed: Set<String> = [], ram: Double = 19) async throws -> ModelToolStatus {
        let cat = try catalog()
        let claim = await claimable(cat)
        return CapabilityResolver.status(for: try spec(tool), catalog: cat, installedModelIDs: installed,
                                         claimableModelIDs: claim, totalRAMGB: ram)
    }

    @Test func installedAndFittingIsReady() async throws {
        let cat = try catalog()
        let kokoro = try entry("mlx-community/Kokoro-82M-bf16", in: cat)
        guard case .ready(let model) = try await status("speak", installed: [kokoro.id]) else { Issue.record("not ready"); return }
        #expect(model.id == kokoro.id)
    }

    @Test func nothingInstalledIsNeedsDownloadWithSize() async throws {
        guard case .needsDownload(let model, let gb) = try await status("ocr_image") else { Issue.record("not needsDownload"); return }
        #expect(gb == model.downloadSizeGB)
        #expect(gb > 0)
        #expect(model.displayName == "GLM-OCR" || TaskModels.displayName(for: model) == "GLM-OCR")   // seeded first choice
    }

    @Test func onlyParakeetInstalledResolvesToParakeetNotWhisper() async throws {
        let cat = try catalog()
        let parakeet = try entry("mlx-community/parakeet-tdt-0.6b-v3", in: cat)
        guard case .ready(let model) = try await status("transcribe_audio", installed: [parakeet.id]) else { Issue.record("not ready"); return }
        #expect(model.id == parakeet.id)
    }

    @Test func poolOrderPicksTheFirstInstalled() async throws {
        let cat = try catalog()
        let tiny = try entry("mlx-community/whisper-tiny-asr-fp16", in: cat)
        let parakeet = try entry("mlx-community/parakeet-tdt-0.6b-v3", in: cat)
        guard case .ready(let both) = try await status("transcribe_audio", installed: [tiny.id, parakeet.id]) else { Issue.record("x"); return }
        #expect(both.id == tiny.id)   // Whisper Tiny precedes Parakeet in the pool
        guard case .needsDownload(let none, _) = try await status("transcribe_audio") else { Issue.record("y"); return }
        #expect(none.id == tiny.id)   // no install: the pool's best
        #expect(TaskModels.defaultModel(forTask: "Transcribe", catalog: cat, claimableModelIDs: await claimable(cat)) == "Whisper Tiny")  // unchanged
    }

    @Test func tooBigIsCantRunWithAReason() async throws {
        let s = try await status("generate_image", ram: 2)
        guard case .cantRun(let reason) = s else { Issue.record("expected cantRun, got \(s)"); return }
        #expect(reason.contains("2.0 GB"))
        #expect(s.routerSuffix == nil)
    }

    @Test func aFittingAlternativeBeatsATooBigFirstChoice() async throws {
        var cat = try catalog()
        let claim = await claimable(cat)
        let klein = try entry("mlx-community/FLUX.2-klein-4B-bf16", in: cat)
        // Fixture: the preferred model claims 30 GB (same entry, bigger ramGB), so it does not fit a 19 GB Mac.
        var json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(klein)) as? [String: Any])
        json["ramGB"] = 30.0
        let huge = try JSONDecoder().decode(ModelEntry.self, from: JSONSerialization.data(withJSONObject: json))
        cat = cat.map { $0.id == klein.id ? huge : $0 }
        #expect(CapabilityResolver.orderedCandidates(task: "Generate Image", kinds: nil, catalog: cat, claimableModelIDs: claim).first?.id == klein.id)
        guard case .needsDownload(let model, _) = CapabilityResolver.status(for: try spec("generate_image"), catalog: cat, installedModelIDs: [],
                                                                             claimableModelIDs: claim, totalRAMGB: 19) else { Issue.record("x"); return }
        #expect(model.id != klein.id)
        #expect(model.ramGB <= 19)
        // And when nothing fits at all it is cantRun.
        guard case .cantRun = CapabilityResolver.status(for: try spec("generate_image"), catalog: cat, installedModelIDs: [],
                                                        claimableModelIDs: claim, totalRAMGB: 4) else { Issue.record("y"); return }
    }

    @Test func generateImageOnA19GBMacCarriesTheDownloadSize() async throws {
        guard case .needsDownload(let model, let gb) = try await status("generate_image", ram: 19) else { Issue.record("x"); return }
        #expect(model.hfModelId == "mlx-community/FLUX.2-klein-4B-bf16")
        #expect(gb > 15 && gb < 17)
        #expect(try await status("generate_image", ram: 19).routerSuffix == "[needs a one-time 16 GB download]")
    }

    @Test func translateUsesOnlyTheDedicatedModel() async throws {
        let cat = try catalog()
        let claim = await claimable(cat)
        guard case .needsDownload(let model, _) = try await status("translate") else { Issue.record("x"); return }
        #expect(model.runnerKind == .translate)
        // An installed general LLM never makes translate ready.
        let llm = try #require(cat.first { $0.runnerKind == .llm && claim.contains($0.id) })
        guard case .needsDownload = CapabilityResolver.status(for: try spec("translate"), catalog: cat, installedModelIDs: [llm.id],
                                                              claimableModelIDs: claim, totalRAMGB: 19) else { Issue.record("y"); return }
        // Without a .translate model in the catalog: .none (no candidate in this build).
        let without = cat.filter { $0.runnerKind != .translate }
        #expect(CapabilityResolver.status(for: try spec("translate"), catalog: without, installedModelIDs: [],
                                          claimableModelIDs: claim, totalRAMGB: 19) == .none)
    }

    @Test func anInstalledButUnclaimableModelIsNotReady() async throws {
        let cat = try catalog()
        let parakeet = try entry("mlx-community/parakeet-tdt-0.6b-v3", in: cat)
        let claim = await claimable(cat).subtracting([parakeet.id])
        let s = CapabilityResolver.status(for: try spec("transcribe_audio"), catalog: cat, installedModelIDs: [parakeet.id],
                                          claimableModelIDs: claim, totalRAMGB: 19)
        if case .ready = s { Issue.record("an unclaimable model must not be ready") }
    }

    @Test func taskLessSpecsAreNone() async throws {
        for name in ["list_capabilities", "read_asset"] {
            #expect(try await status(name) == .none, "\(name)")
        }
    }

    @Test func everyModelToolHasACandidateOnTheRealCatalog() async throws {
        for s in ModelToolSpec.all {
            guard s.task != nil else { continue }   // task-less rows are .none by design
            let st = try await status(s.name)
            if case .none = st { Issue.record("\(s.name) has no candidate") }
        }
    }

    @Test func routerSuffixNamesNoModel() async throws {
        let cat = try catalog()
        let names = cat.flatMap { [$0.id, $0.hfModelId, $0.displayName, TaskModels.displayName(for: $0)] }.map { $0.lowercased() }
        for s in ModelToolSpec.all {
            guard let suffix = try await status(s.name).routerSuffix else { continue }
            #expect(suffix == "[ready]" || suffix.hasPrefix("[needs a one-time "))
            for n in names where !n.isEmpty { #expect(!suffix.lowercased().contains(n), "\(s.name): \(suffix)") }
        }
        let ready = try entry("mlx-community/Kokoro-82M-bf16", in: cat)
        #expect(ModelToolStatus.ready(ready).routerSuffix == "[ready]")
        #expect(ModelToolStatus.needsDownload(ready, gb: 0.16).routerSuffix == "[needs a one-time 0.2 GB download]")
        #expect(ModelToolStatus.none.routerSuffix == nil)
    }
}
