import Testing
import Foundation
@testable import MLXUI

/// LY-4 (`RSI/DelegateLayaBacklog.md`) — `Modules/Laya/`: SDK, stage, registration.
struct LayaModuleTests {
    private func layaEntry() -> ModelEntry {
        makeEntry(
            id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
            modelType: .decision, source: .mlx, ramGB: 1.3, downloadSizeGB: 0.85,
            hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
    }

    @Test func layaHasNoSupportGap() {
        #expect(ModelSupport.unsupportedReason(for: layaEntry()) == nil)
    }

    @Test func layaRoutesToDecisionKind() {
        #expect(layaEntry().runnerKind == .decision)
    }

    @MainActor
    @Test func layaResolvesToLayaSDK() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: layaEntry()))
        #expect(resolved.descriptor.id == "laya")
        #expect(resolved.sdk.claim(layaEntry()) == .exact)
    }

    @Test func laySDKClaimsOnlyMLXDecisionEntries() {
        #expect(LayaSDK().claim(layaEntry()) == .exact)
        #expect(LayaSDK().claim(makeEntry(modelType: .llm, source: .mlx)) == .no)
        #expect(LayaSDK().claim(makeEntry(modelType: .rerank, source: .mlx)) == .no)
        #expect(LayaSDK().claim(makeEntry(modelType: .decision, source: .research)) == .no)
    }

    @Test func layaMakeStageProducesTextToTextStage() throws {
        let stage = try LayaSDK().makeStage(for: layaEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }

    // MARK: - LayaStage: request/response plumbing, testable without weights

    @Test func stageRunsTheInjectedAskClosureAndReturnsItsText() async throws {
        var received: String?
        let stage = LayaStage(id: "test", name: "Test") { requestJSON in
            received = requestJSON
            return #"{"type":"choice","choice":"billing"}"#
        }
        let result = try await stage.run(.text(#"{"type":"choice","state":"x"}"#), progress: { _ in })
        #expect(received == #"{"type":"choice","state":"x"}"#)
        guard case .text(let text) = result else {
            Issue.record("expected .text, got \(result)")
            return
        }
        #expect(text.contains("billing"))
    }

    @Test func stageRejectsNonTextInput() async {
        let stage = LayaStage(id: "test", name: "Test") { _ in "{}" }
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 16000)), progress: { _ in })
        }
    }

    // MARK: - LayaUI

    @Test func layaUIClaimMirrorsSDK() {
        #expect(LayaUI().claim(layaEntry()) == .exact)
        #expect(LayaUI().claim(makeEntry(modelType: .llm, source: .mlx)) == .no)
    }
}
