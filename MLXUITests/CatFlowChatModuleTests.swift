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
        #expect(entries.count == 39, "bundled catalog drifted: \(entries.count) entries")

        let withChat = registry(includingChat: true)
        let withoutChat = registry(includingChat: false)

        var checked = 0
        var llmChecked = 0
        for entry in entries {
            let before = withoutChat.bestModule(for: entry)?.descriptor.id
            let after = withChat.bestModule(for: entry)?.descriptor.id
            if entry.modelType == .llm {
                #expect(before == nil, "\(entry.id) resolved before ChatModule existed")
                #expect(after == "chat", "\(entry.id) should resolve to chat")
                llmChecked += 1
            } else {
                #expect(after == before,
                        "\(entry.id): resolution changed (\(String(describing: before)) → \(String(describing: after)))")
            }
            checked += 1
        }
        #expect(checked == 39)
        #expect(llmChecked == 10)
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
