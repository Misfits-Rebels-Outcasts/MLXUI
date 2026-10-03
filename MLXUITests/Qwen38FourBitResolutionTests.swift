import Foundation
import Testing
@testable import MLXUI

/// Q38-5 — the official `mlx-community/Qwen3.8-27B-4bit` catalog entry (Phase B of
/// `RSI/DelegateQwen38Backlog.md`), ruled by the owner 2026-10-03 at Q38-A: **C1** (`ramGB` 26.0
/// rather than the 1.5× formula's 24.12, so a 24 GB Mac is blocked deliberately) and **D1**
/// (`AppState.hideQwen3827B = true` until smoke row Q38-S5 passes on a 32 GB+ Mac).
///
/// Modelled on `Qwen38TwentySevenBResolutionTests` (the 3-bit card, Q38-2). The architecture is
/// SUPPORTED, so this entry carries no `ModelSupport` gap and routes through `ChatModule`; the
/// **difference that matters** is that it must stay invisible in Browse while its 3-bit sibling
/// stays visible.
@MainActor
struct Qwen38FourBitResolutionTests {

    private static let hfModelId = "mlx-community/Qwen3.8-27B-4bit"

    /// The shipped catalog, decoded the way `AppState.loadBrowserData()` does — so these tests
    /// fail if the entry is removed from `browser.json`, not just if the decoder breaks.
    private func bundledCatalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)   // .../MLXUITests/Qwen38FourBit…swift
            .deletingLastPathComponent()                 // MLXUITests
            .deletingLastPathComponent()                 // PipelineStudio-q38
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func bundledEntry() throws -> ModelEntry {
        let all = try bundledCatalog().domains.flatMap { $0.allModels }
        return try #require(
            all.first { $0.hfModelId == Self.hfModelId },
            "the shipped browser.json has no \(Self.hfModelId) entry")
    }

    private func makeEntry() -> ModelEntry {
        MLXUITests.makeEntry(
            id: "mlx-community--Qwen3.8-27B-4bit",
            family: "Qwen3.8",
            displayName: "Qwen3.8-27B",
            modelType: .llm,
            source: .mlx,
            ramGB: 26.0,
            downloadSizeGB: 16.08,
            hfRepo: "mlx-community",
            hfModelId: Self.hfModelId
        )
    }

    /// The entry is in the shipped catalog with the shape Q38-5's contract and ruling C1 ask for.
    @Test func qwen38FourBitEntryIsInTheBundledCatalog() throws {
        let entry = try bundledEntry()
        #expect(entry.id == "mlx-community--Qwen3.8-27B-4bit")
        #expect(entry.modelType == .llm)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-4bit")
        #expect(entry.family == "Qwen3.8")
        #expect(entry.displayName == "Qwen3.8-27B")
        #expect(entry.hfRepo == "mlx-community")
        #expect(entry.hfModelId == Self.hfModelId)
        #expect(entry.contextWindow == 262144)
        #expect(entry.license == "apache-2.0")
        #expect(entry.architecture == "Qwen3_5ForConditionalGeneration")
        #expect(entry.downloadSizeGB == 16.08)
        #expect(entry.speedTokensPerSec == nil)
        #expect(entry.speedEstimated == true)
        #expect(entry.variants?.count == 1)
        #expect(entry.variants?.first?.quantization == "4-bit")
        #expect(entry.variants?.first?.format == "mlx-4bit")
        #expect(entry.variants?.first?.recommended == true)
        #expect(entry.description?.contains("10c35caafbb80f7dc6a7a432cdd11af10a6d4818") == true)
        #expect(entry.description?.contains("32 GB") == true)
        #expect(entry.description?.contains("text only") == true)
    }

    /// C1: `ramGB` is the **ruled** 26.0, deliberately *not* the 1.5× formula (≈24.12). This is
    /// the one catalog number in this stream that is an owner decision rather than a measurement,
    /// so it is pinned by its own test — a future "consistency" edit to `ramGB * 1.5` fails here.
    @Test func qwen38FourBitRAMIsTheRuled26NotTheFormula() throws {
        let entry = try bundledEntry()
        #expect(entry.ramGB == 26.0)
        #expect(entry.ramGB != entry.downloadSizeGB * 1.5)   // 24.12 — C2, not taken
        // A 32 GB Mac (≈32.00) clears it; the owner's 19.33 GB Mac does not, by 6.67.
        #expect(entry.ramGB < 32.0)
        #expect(entry.variants?.first?.ramGB == 26.0)
    }

    /// `qwen3_5` is supported, so no gap and no `UnsupportedModelView`.
    @Test func qwen38FourBitHasNoSupportGap() throws {
        #expect(ModelSupport.unsupportedReason(for: try bundledEntry()) == nil)
        #expect(ModelSupport.unsupportedReason(for: makeEntry()) == nil)
    }

    /// `.llm` routes to the LLM run path, and `ChatSDK` claims it exactly.
    @Test func qwen38FourBitResolvesToChatModule() throws {
        #expect(try bundledEntry().runnerKind == .llm)
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: try bundledEntry()))
        #expect(resolved.descriptor.id == "chat")
        #expect(resolved.sdk.claim(try bundledEntry()) == .exact)
        let stage = try ChatSDK().makeStage(for: makeEntry(), config: .default)
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
    }

    /// D1's committed value. If the owner flips this after Q38-S5, this test is the place the
    /// flip is recorded (and `hideRemovesOnlyTheFourBit` below is written so it stays correct
    /// either way — it asserts the *sibling* is never hidden, which is the part that must hold).
    @Test func hideFlagIsCommittedTrue() {
        #expect(AppState.hideQwen3827B == true)
    }

    /// The two cards are separate `id`s, so the flag can hide exactly one of them. The 3-bit card
    /// is the one the whole of Phase A was smoked on (Q38-S1…S4) and must keep showing.
    @Test func hideRemovesOnlyTheFourBitFromBrowse() throws {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        state.selectedSection = .browse("chat-text")

        let visible = state.filteredModels
        let ids = Set(visible.map(\.id))
        #expect(ids.contains("lukaskremla--Qwen3.8-27B-3bit-MLX"),
                "the 3-bit card must never be hidden by hideQwen3827B")
        #expect(!ids.contains("mlx-community--Qwen3.8-27B-4bit"),
                "hideQwen3827B is committed true (D1), so the 4-bit card must not appear in Browse")

        // The catalog itself still holds it — the flag filters the *view*, it does not delete the
        // entry, so a flow that names the 4-bit card can still resolve it (Q38-6's check).
        let all = try #require(state.browserData).domains.flatMap { $0.allModels }
        #expect(all.contains { $0.hfModelId == Self.hfModelId })
    }
}
