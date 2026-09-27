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

    /// CL-1: a `.decision`/`.mlx` entry from a different family, to prove the gate-B
    /// `family == "Laya"` discriminator keeps Laya from claiming it.
    private func clmEntry() -> ModelEntry {
        makeEntry(
            id: "RealityCat--CLM-v0.1-8B-MLX-8bit", family: "CLM", displayName: "CLM 8B",
            modelType: .decision, source: .mlx, ramGB: 12.2, downloadSizeGB: 8.12,
            hfRepo: "RealityCat", hfModelId: "RealityCat/CLM-v0.1-8B-MLX-8bit")
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

    /// CL-1 gate B: a `.decision`/`.mlx` entry from another family (CLM 8B) is refused by
    /// both `LayaSDK` and `LayaUI`, and doesn't resolve to any module — Laya keeps claiming
    /// only its own family.
    @MainActor
    @Test func layaDoesNotClaimCLM() throws {
        #expect(LayaSDK().claim(clmEntry()) == .no)
        #expect(LayaUI().claim(clmEntry()) == .no)

        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        #expect(registry.bestModule(for: clmEntry()) == nil)
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
