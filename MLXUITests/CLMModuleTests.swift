import Testing
import Foundation
@testable import MLXUI

/// CL-4b — `Modules/CLM/`: SDK, UI, and registration claim resolution (no real model needed;
/// engine parity is `CLMEncoderTests`/`CLMEngineTests`).
struct CLMModuleTests {
    private func clmEntry() -> ModelEntry {
        makeEntry(
            id: "RealityCat--CLM-v0.1-8B-MLX-8bit", family: "CLM", displayName: "CLM 8B",
            modelType: .decision, source: .mlx, ramGB: 12.2, downloadSizeGB: 8.12,
            hfRepo: "RealityCat", hfModelId: "RealityCat/CLM-v0.1-8B-MLX-8bit")
    }

    private func layaEntry() -> ModelEntry {
        makeEntry(
            id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
            modelType: .decision, source: .mlx, ramGB: 1.3, downloadSizeGB: 0.85,
            hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
    }

    @Test func clmSDKClaimsOnlyMLXDecisionCLMEntries() {
        #expect(CLMSDK().claim(clmEntry()) == .exact)
        #expect(CLMSDK().claim(layaEntry()) == .no)
        #expect(CLMSDK().claim(makeEntry(modelType: .llm, source: .mlx)) == .no)
        #expect(CLMSDK().claim(makeEntry(modelType: .decision, source: .research)) == .no)
        #expect(CLMSDK().claim(makeEntry(family: "CLM", modelType: .decision, source: .research)) == .no)
    }

    @Test func clmUIClaimMirrorsSDK() {
        #expect(CLMUI().claim(clmEntry()) == .exact)
        #expect(CLMUI().claim(layaEntry()) == .no)
    }

    @MainActor
    @Test func clmResolvesToCLMSDK() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        let resolved = try #require(registry.bestModule(for: clmEntry()))
        #expect(resolved.descriptor.id == "clm")
        #expect(resolved.sdk.claim(clmEntry()) == .exact)
    }

    /// CL-4b registers `CLMModule` alongside `LayaModule` — confirms neither steps on the
    /// other's entries in a shared registry.
    @MainActor
    @Test func layaAndCLMBothResolveInTheSameRegistry() throws {
        let registry = ModelRegistry()
        for module in installedModules {
            module.register(into: registry)
        }
        #expect(registry.bestModule(for: layaEntry())?.descriptor.id == "laya")
        #expect(registry.bestModule(for: clmEntry())?.descriptor.id == "clm")
    }

    /// CL-5: the real Run screen exists now, so CL-1's gap flag is removed.
    @MainActor
    @Test func clmNoLongerHasSupportGap() {
        #expect(ModelSupport.unsupportedReason(for: clmEntry()) == nil)
    }

    /// CL-5 gate G ("a duplicate Run screen"): `CLMUI.makeRunView` returns the real
    /// `CLMRunView`, not the generic `UnsupportedModelView`.
    @MainActor
    @Test func clmUIMakeRunViewReturnsCLMRunView() throws {
        let stage = try CLMSDK().makeStage(for: clmEntry(), config: StageConfig())
        let view = CLMUI().makeRunView(for: clmEntry(), stage: stage)
        #expect("\(view)".contains("CLMRunView"))
    }
}
