import Testing
import Foundation
@testable import MLXUI

/// MC5-3i: the SDK's claim, its by-name refusals, the stage contract, and the module registration.
struct AceStepModuleTests {

    private func entries() throws -> (ace: ModelEntry, musicgen: ModelEntry, others: [ModelEntry]) {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let ace = try #require(all.first { $0.family == "ace-step" })
        let mg = try #require(all.first { $0.hfModelId == "jasonvassallo/mlx-musicgen-small" })
        return (ace, mg, all.filter { $0.id != ace.id })
    }

    @Test func aceStepClaimsItsFamilyAndMusicGenDoesNot() throws {
        let e = try entries()
        #expect(AceStepSDK().claim(e.ace) == .exact && AceStepUI().claim(e.ace) == .exact)
        #expect(MusicGenSDK().claim(e.ace) == .no)
        #expect(MusicGenSDK().claim(e.musicgen) == .exact)
        #expect(AceStepSDK().claim(e.musicgen) == .no && AceStepUI().claim(e.musicgen) == .no, "ACE-Step never captures MusicGen")
        #expect(e.others.allSatisfy { AceStepSDK().claim($0) == .no }, "and nothing else in the catalog")
    }

    @MainActor @Test func theModuleRegistersAndResolvesTheEntry() throws {
        let e = try entries()
        #expect(installedModules.contains { $0.descriptor.id == "acestep" })
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        #expect(registry.bestModule(for: e.ace)?.descriptor.id == "acestep")
        #expect(registry.bestModule(for: e.musicgen)?.descriptor.id == "musicgen")
        #expect(AceStepModule.descriptor.modelTypes == [.music])
    }

    @Test func settingsThatCannotBeHonouredAreRefusedByName() throws {
        let e = try entries()
        let sdk = AceStepSDK()
        for (setting, config) in [("steps", StageConfig(steps: 21)), ("width", StageConfig(width: 512)), ("height", StageConfig(height: 512)),
                                  ("quantize", StageConfig(quantize: 4)), ("scale", StageConfig(scale: 2))] {
            do { _ = try sdk.makeStage(for: e.ace, config: config); Issue.record("\(setting) should be refused") }
            catch StageError.unsupportedSetting(let name) { #expect(name == setting) }
        }
        let stage = try sdk.makeStage(for: e.ace, config: StageConfig(seed: 5))
        #expect(stage.accepts == .text && stage.produces == .audio)
    }

    @Test func theStageTurnsAPromptIntoAMonoMixdownAndReportsProgress() async throws {
        final class Box: @unchecked Sendable { var request: AceStepRequest?; var progress: [Double] = [] }
        let box = Box()
        let stage = AceStepStage(id: "t", name: "t", settings: AceStepRunSettings(duration: 12, seed: 9)) { request, onPhase in
            box.request = request
            onPhase(.denoise(step: 4, of: 8))
            return AceStepAudio(samples: [0.2, 0.4, -0.2, -0.4])
        }
        let out = try await stage.run(.text("lofi beats")) { box.progress.append($0) }
        guard case .audio(let buffer) = out else { Issue.record("expected audio"); return }
        #expect(buffer.sampleRate == 48_000 && buffer.samples.count == 2)
        #expect(abs(buffer.samples[0] - 0.3) < 1e-6 && abs(buffer.samples[1] + 0.3) < 1e-6)
        #expect(box.request == AceStepRequest(caption: "lofi beats", duration: 12, seed: 9))
        #expect(box.progress.first == 0 && box.progress.last == 1 && box.progress.contains(AceStepPhase.denoise(step: 4, of: 8).fraction))
        await #expect(throws: StageError.self) { _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 1))) { _ in } }
    }

    @Test func aFlowRowDefaultsToThirtySeconds() {
        #expect(AceStepRunSettings().duration == 30)
    }
}
