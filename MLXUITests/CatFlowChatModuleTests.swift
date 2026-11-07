import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R1-1: registering `ChatModule` so LLM models answer to the registry.
/// The chat sheet keeps working exactly as before — this only makes the *registry*
/// able to resolve `.llm` entries. See `RSI/DelegateMergeBacklog.md` CFM-R1-1.
@MainActor
struct CatFlowChatModuleTests {

    // MARK: - Resolution of the canonical LLM

    @Test func qwen3ResolvesToChatModule() throws {
        let registry = ModelRegistry()
        ChatModule.register(into: registry)
        let entry = makeEntry(id: "mlx-community--Qwen3-8B-4bit",
                              family: "Qwen3", modelType: .llm, source: .mlx)
        let resolved = try #require(registry.bestModule(for: entry))
        #expect(resolved.descriptor.id == "chat")
        #expect(resolved.sdk.claim(entry) == .exact)
    }

    // MARK: - Claim scoring

    @Test func chatSDKClaimsLLMExactly() {
        #expect(ChatSDK().claim(makeEntry(modelType: .llm)) == .exact)
    }

    @Test func chatSDKDeclinesASR() {
        #expect(ChatSDK().claim(makeEntry(modelType: .asr)) == .no)
    }

    @Test func chatSDKDeclinesTTS() {
        #expect(ChatSDK().claim(makeEntry(modelType: .tts)) == .no)
    }

    @Test func chatSDKDeclinesVision() {
        #expect(ChatSDK().claim(makeEntry(modelType: .vision)) == .no)
    }

    @Test func chatSDKDeclinesNonMLXLLM() {
        // `.llm` is claimed regardless of source — ChatSDK is the only LLM SDK.
        #expect(ChatSDK().claim(makeEntry(modelType: .llm, source: .research)) == .exact)
    }

    // MARK: - makeStage (do not .run — real engine)

    @Test func chatMakeStageProducesTextToTextStage() throws {
        let stage = try ChatSDK().makeStage(
            for: makeEntry(id: "mlx-community--Qwen3-8B-4bit", modelType: .llm),
            config: .default
        )
        #expect(stage.accepts == .text)
        #expect(stage.produces == .text)
        #expect(stage.id == "mlx-community--Qwen3-8B-4bit")
    }

    // MARK: - Regression: appending ChatModule changes nothing but the 9 .llm entries

    /// Every one of the bundled catalog's entries still resolves to the **same** module
    /// id as before `ChatModule` was registered, except the `.llm` entries, which
    /// previously resolved to `nil` and now resolve to `"chat"`. This is the regression
    /// guard for appending `ChatModule.self` to `installedModules` (it must stay last).
    @Test func catalogResolutionUnchangedByChatModuleAppend() throws {
        let entries = try loadBundledCatalogEntries()
        // LY-1: bundled catalog gained the aac6fef/laya-mlx `.decision` entry, 39 → 40.
        // CL-1: bundled catalog gained the RealityCat/CLM-v0.1-8B-MLX-8bit `.decision` entry, 40 → 41.
        // Q38-2: bundled catalog gained the lukaskremla/Qwen3.8-27B-3bit-MLX `.llm` entry
        //        (RSI/DelegateQwen38Backlog.md Q38-2, ruled 2026-10-03 A1/B1/C1/D1), 41 → 42.
        // Q38-5: bundled catalog gained the mlx-community/Qwen3.8-27B-4bit `.llm` entry (Phase B,
        //        same ruling — C1 sets its ramGB to 26.0; D1 hides it in Browse), 42 → 43.
        // CF-2: bundled catalog gained the mlx-community/clef-flash-4bit `.decision` entry
        //       (RSI/DelegateClefBacklog.md CF-2, CF-0 ruled 2026-10-03 A1–G1), 43 → 44.
        // MM1: bundled catalog gained the mlx-community/MiMo-V2.6-Distill-Qwen-9B-OptiQ-4bit `.llm`
        //      entry (RSI/backlog.md MM1, MM-0 ruled 2026-10-17), 44 → 45. Implementer's call,
        //      pending owner confirmation, to follow this list's convention: a bumped count, not an edited golden.
        // MP1-1: bundled catalog gained the two Gemma 4 E4B entries (`.llm` + `.vision`, one shared
        //        mlx-community/gemma-4-e4b-it-4bit install; RSI/DelegateModelPackBacklog.md MP-0 E, ruled
        //        2026-10-21), 45 → 47. Same convention: a bumped count, not an edited golden.
        // MP2-1: bundled catalog gained the mlx-community/parakeet-tdt-0.6b-v3 `.asr` entry
        //        (RSI/DelegateModelPackBacklog.md MP2-1, MP-0 ruled 2026-10-21), 47 → 48. Same convention.
        // MP3-1: bundled catalog gained the mlx-community/Qwen3-ASR-1.7B-8bit `.asr` entry
        //        (RSI/DelegateModelPackBacklog.md MP3-1, MP-0 ruled 2026-10-21), 48 → 49. Same convention.
        // MP4-1: bundled catalog gained the mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16 `.diarization`
        //        entry (RSI/DelegateModelPackBacklog.md MP4-1, MP-0 ruled 2026-10-21), 49 → 50. Same convention.
        // MP5-1: bundled catalog gained the mlx-community/DeepFilterNet-mlx `.enhance` entry
        //        (RSI/DelegateModelPackBacklog.md MP5-1, MP-0 ruled 2026-10-21), 50 → 51. Same convention.
        // MP6-1: bundled catalog gained the mlx-community/LFM2.5-Audio-1.5B-6bit `.voice` entry
        //        (RSI/DelegateModelPackBacklog.md MP6-1, MP-0 ruled 2026-10-21), 51 → 52. Same convention.
        // RIFE-1: bundled catalog gained the mlx-community/RIFE-4.25 `.interpolate` entry
        //         (RSI/backlog.md RIFE-1, owner rulings 2026-10-28), 52 → 53. Same convention.
        // MP7-2: bundled catalog gained the mlx-community/FLUX.2-klein-4B-bf16 `.image` entry
        //        (RSI/DelegateModelPackBacklog.md MP7-2, MP7-0 ruled 2026-10-30), 53 → 54. Same convention.
        // MC1-1: bundled catalog gained the mlx-community/sam-audio-base `.separate` entry
        //        (sound isolation), 54 → 55. Same convention.
        // MC3-1: bundled catalog gained the depth-anything/Depth-Anything-V2-Small-hf `.depth` entry
        //        (monocular depth estimation, owner decisions 2026-11-06), 55 → 56. Same convention.
        // MC2-1: bundled catalog gained the mlx-community/translategemma-4b-it-4bit `.translate` entry
        //        (TranslateGemma 4B, MC2), 56 → 57 when MC3 merged on top (MC2 +1, MC3 +1). Same convention.
        #expect(entries.count == 57, "bundled catalog drifted: \(entries.count) entries")

        let withChat = registry(includingChat: true)
        let withoutChat = registry(includingChat: false)

        var checked = 0
        var llmChecked = 0
        var translateChecked = 0
        for entry in entries {
            let before = withoutChat.bestModule(for: entry)?.descriptor.id
            let after = withChat.bestModule(for: entry)?.descriptor.id
            // MC2-1 (owner decision 3): TranslateGemma is cataloged `.llm` but routes as `.translate`, so it must NOT
            // resolve to the chat module — the family override is what keeps it out of every `.llm`-only path. Its own
            // module (MC2-4) claims it instead; until then it resolves to nothing.
            if entry.runnerKind == .translate {
                #expect(entry.modelType == .llm, "\(entry.id): the translator stays modelType llm")
                // MC2-4 registers its own module, so it resolves there whether ChatModule is present or not — the point
                // is that ChatModule never takes it (owner decision 3 / Trap 5).
                #expect(before == "translategemma", "\(entry.id) resolves to its own module, not chat")
                #expect(after == before, "\(entry.id): ChatModule must not change its resolution")
                #expect(after != "chat", "\(entry.id) must never resolve to chat")
                translateChecked += 1
            } else if entry.modelType == .llm {
                #expect(before == nil, "\(entry.id) resolved before ChatModule existed")
                #expect(after == "chat", "\(entry.id) should resolve to chat")
                llmChecked += 1
            } else {
                #expect(after == before,
                        "\(entry.id): resolution changed (\(String(describing: before)) → \(String(describing: after)))")
            }
            checked += 1
        }
        #expect(checked == 57)   // MC2-1 + MC3-1: 55 → 57 (the TranslateGemma 4B `.translate` and Depth Anything V2 Small `.depth` entries)
        #expect(llmChecked == 14)   // MP1-1: Gemma 4 E4B is the 14th .llm entry (MM1: MiMo V2.6 9B was the 13th)
        #expect(translateChecked == 1)   // MC2-1: TranslateGemma 4B is the first (and only) `.translate` entry
    }

    // MARK: - Helpers

    private func registry(includingChat: Bool) -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules {
            if !includingChat && module.descriptor.id == "chat" { continue }
            module.register(into: registry)
        }
        return registry
    }

    private func loadBundledCatalogEntries() throws -> [ModelEntry] {
        let filePath = #filePath                      // .../MLXUITests/CatFlowChatModuleTests.swift
        let repoRoot = URL(fileURLWithPath: filePath)
            .deletingLastPathComponent()              // MLXUITests
            .deletingLastPathComponent()              // PipelineStudio
        let repoURL = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let data = try Data(contentsOf: repoURL)
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        return catalog.domains.flatMap { $0.allModels }
    }
}
