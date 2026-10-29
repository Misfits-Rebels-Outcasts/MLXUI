import Foundation
import Testing
@testable import MLXUI

/// RIFE-3d: SDK claim rules, the stage's honest refusal, and the registry. Needs no weights.
@MainActor
struct RIFEModuleTests {

    private func rife() throws -> ModelEntry {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == "mlx-community/RIFE-4.25" })
    }

    @Test func sdkClaimsOnlyRIFEInterpolateModels() throws {
        let e = try rife()
        #expect(RIFESDK().claim(e) == .exact)
        #expect(RIFEUI().claim(e) == .exact)
        let all = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")))
            .domains.flatMap { $0.allModels }
        for other in all where other.hfModelId != e.hfModelId {
            #expect(RIFESDK().claim(other) == .no, "\(other.id) must not be claimed by RIFE")
        }
    }

    @Test func seedVR2StillOwnsUpscaleAndRIFEDoesNotTakeIt() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let all = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")))
            .domains.flatMap { $0.allModels }
        let seed = try #require(all.first { $0.runnerKind == .upscale })
        #expect(registry.bestModule(for: seed)?.descriptor.id == "seedvr2")
    }

    @Test func makeStageReturnsATextToTextRIFEStageWithDefaultOptions() throws {
        let e = try rife()
        let stage = try RIFESDK().makeStage(for: e, config: .default)
        let rife = try #require(stage as? RIFEStage)
        #expect(stage.id == "rife.mlx-community--RIFE-4.25")
        #expect(rife.accepts == .text && rife.produces == .text)
        #expect(rife.options == RIFEVideoOptions(times: 2, scale: 1.0))
    }

    @Test func stageRefusesTheWrongKindAndAnUnreadableFile() async throws {
        let stage = RIFEStage(modelDirectory: FileManager.default.temporaryDirectory.appendingPathComponent("no-such-rife"))
        await #expect(throws: StageError.self) { _ = try await stage.run(.audio(AudioBuffer(samples: [0], sampleRate: 16000)), progress: { _ in }) }
        await #expect(throws: RIFEVideoError.self) { _ = try await stage.run(.text("/no/such/video.mp4"), progress: { _ in }) }
    }

    @Test func savedFileNameCarriesTheMultiplier() {
        #expect(RIFERunModel.savedFileName(for: URL(fileURLWithPath: "/a/holiday.mov"), times: 4) == "holiday-4x.mp4")
        #expect(RIFERunModel.savedFileName(for: nil, times: 2) == "video-2x.mp4")
    }

    @Test func descriptorRecordsTheLicenceAndKind() {
        #expect(RIFEModule.descriptor.packageLicense == "MIT")
        #expect(RIFEModule.descriptor.modelTypes == [.interpolate])
        #expect(RIFEModule.descriptor.maintainers == ["hzwer"])
    }

    /// RIFE-5c: the ledger books the measured 6.4 GB working set (Case E), not the 2.0 GB it booked as a provisional catalog figure.
    @Test func theLedgerBooksTheMeasuredWorkingSet() {
        let row = ModelLoadProfile.entry(for: .interpolate)
        #expect(row.measured)
        #expect(row.workingSetBytes == 6_400_000_000)
        #expect(ModelLoadProfile.workingSetBytes(for: .interpolate) == 6_400_000_000)
        #expect(ModelLoadProfile.charge(kind: .interpolate, directory: FileManager.default.temporaryDirectory.appendingPathComponent("no-such-\(UUID().uuidString)"), catalogFallback: 6_400_000_000) == 6_400_000_000)
    }
}
