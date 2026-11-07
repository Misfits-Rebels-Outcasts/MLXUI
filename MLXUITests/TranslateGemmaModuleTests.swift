import Foundation
import Jinja
import Testing
@testable import MLXUI

/// MC2-4 — the TranslateGemma module with no weights: claim/routing, the structured request shape, the pure prompt
/// renderer (against the pinned template fixture), detection, the refusal paths, the stage seam and the SDK's
/// config plumbing. The real-weights load/translate/determinism/Stop checks are in `TranslateGemmaRealLoadTests.swift`
/// (gated; skipped unless the weights are pointed at).
@MainActor
struct TranslateGemmaModuleTests {

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func entry() -> ModelEntry {
        makeEntry(id: TranslateGemmaCatalogTests.cardID, family: "translategemma", modelType: .llm,
                  source: .mlx, ramGB: 3.33, downloadSizeGB: 2.22,
                  hfRepo: "mlx-community", hfModelId: TranslateGemmaCatalogTests.hfID)
    }

    private func registry() -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return registry
    }

    private func goldens() throws -> [[String: Any]] {
        let data = try Data(contentsOf: root.appendingPathComponent("Fixtures/TranslateGemma/prompt_goldens.json"))
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return try #require(object["cases"] as? [[String: Any]])
    }

    private func templateText() throws -> String {
        try String(contentsOf: root.appendingPathComponent("Fixtures/TranslateGemma/chat_template.jinja"),
                   encoding: .utf8)
    }

    // MARK: - Claim and routing (Trap 5)

    @Test func resolvesToTheTranslateModule() throws {
        let model = entry()
        let resolved = try #require(registry().bestModule(for: model))
        #expect(resolved.descriptor.id == "translategemma")
        #expect(resolved.sdk.claim(model) == .exact)
        #expect(resolved.ui.claim(model) == .exact)
        #expect(ChatSDK().claim(model) == .no)
    }

    @Test func claimsOnlyTheTranslateGemmaFamilyMLXTranslate() {
        let sdk = TranslateGemmaSDK()
        #expect(sdk.claim(makeEntry(family: "translategemma", modelType: .llm, source: .mlx)) == .exact)
        // A different family with the `.translate` kind is not ours.
        #expect(sdk.claim(makeEntry(family: "gemma", modelType: .llm, source: .mlx)) == .no)
        // A non-MLX source never is.
        #expect(sdk.claim(makeEntry(family: "translategemma", modelType: .llm, source: .coreml)) == .no)
        // And the family override is what makes this family route as `.translate` whatever its modelType says.
        #expect(makeEntry(family: "translategemma", modelType: .llm).runnerKind == .translate)
        #expect(makeEntry(family: "translategemma", modelType: .vision).runnerKind == .translate)
    }

    @Test func descriptorNamesTheModuleAndItsBacking() {
        let d = TranslateGemmaModule.descriptor
        #expect(d.id == "translategemma")
        #expect(d.modalities == ["text"])
        #expect(d.modelTypes == [.llm])
        #expect(d.backingPackage == "ml-explore/mlx-swift-lm")
        #expect(d.packageLicense == "MIT")
        #expect(d.notes.contains("2,048"))
        #expect(d.notes.contains("text only"))
    }

    // MARK: - The request shape (Trap 2)

    @Test func theMessageIsTheTemplatesOneItemContentList() throws {
        let message = try TranslateGemmaPrompt.message(sourceCode: "en", targetCode: "ms",
                                                       text: "The weather is lovely today.")
        #expect(message["role"] as? String == "user")
        let content = try #require(message["content"] as? [[String: any Sendable]])
        #expect(content.count == 1)
        let item = try #require(content.first)
        #expect(item["type"] as? String == "text")
        #expect(item["source_lang_code"] as? String == "en")
        #expect(item["target_lang_code"] as? String == "ms")
        #expect(item["text"] as? String == "The weather is lovely today.")
        // The underscore spelling is normalised before it reaches the template (which would also normalise it).
        let underscored = try TranslateGemmaPrompt.message(sourceCode: "en_GB", targetCode: "ms", text: "Hi")
        let underscoredContent = try #require(underscored["content"] as? [[String: any Sendable]])
        #expect(underscoredContent.first?["source_lang_code"] as? String == "en-GB")
    }

    @Test func thePureRendererMatchesThePinnedTemplate() throws {
        let template = try Template(try templateText(), with: .init(lstripBlocks: true, trimBlocks: true))
        for testCase in try goldens() {
            let name = try #require(testCase["name"] as? String)
            let expected = try #require(testCase["expected"] as? String)
            let context = try #require(testCase["context"] as? [String: Any])
            let content = try #require((context["messages"] as? [[String: Any]])?.first?["content"] as? [[String: Any]])
            let source = try #require(content.first?["source_lang_code"] as? String)
            let target = try #require(content.first?["target_lang_code"] as? String)
            let text = try #require(content.first?["text"] as? String)
            let rendered = try TranslateGemmaPrompt.render(sourceCode: source, targetCode: target, text: text)
            #expect(rendered == expected, "\(name): the pure renderer and the pinned template disagree")
            let jinja = try template.render(context.mapValues { try Value(any: $0 as Any? ?? ()) })
            #expect(rendered == jinja)
        }
    }

    @Test func theRenderedPromptHasOneBOSAndNoUndefinedLanguage() throws {
        let rendered = try TranslateGemmaPrompt.render(sourceCode: "zh-Hans", targetCode: "en", text: "今天天气很好。")
        #expect(rendered.hasPrefix("<bos>"))
        #expect(rendered.components(separatedBy: "<bos>").count == 2)
        // The template's own name for zh-Hans is "Chinese" (it has one name for all Chinese codes); our UI splits it.
        #expect(rendered.contains("Chinese (zh-Hans)"))
        #expect(!rendered.contains("<start_of_turn>user\nYou are a professional <"))
    }

    @Test func theRendererValidatesBothCodesBeforeRendering() {
        #expect(throws: TranslateGemmaError.unknownLanguage("Klingon")) {
            _ = try TranslateGemmaPrompt.render(sourceCode: "Klingon", targetCode: "en", text: "Hi")
        }
        #expect(throws: TranslateGemmaError.unknownLanguage("tlh")) {
            _ = try TranslateGemmaPrompt.render(sourceCode: "en", targetCode: "tlh", text: "Hi")
        }
        #expect(throws: TranslateGemmaError.emptyText) {
            _ = try TranslateGemmaPrompt.render(sourceCode: "en", targetCode: "ms", text: "   \n ")
        }
    }

    // MARK: - Refusal paths before any weight loads

    @Test func engineRefusesAnEmptyText() {
        #expect(throws: TranslateGemmaError.emptyText) {
            try TranslateGemmaEngine.validate(.init(text: "  \n", sourceCode: "en", targetCode: "ms"))
        }
    }

    @Test func engineRefusesAnUnknownLanguage() {
        #expect(throws: TranslateGemmaError.unknownLanguage("Klingon")) {
            try TranslateGemmaEngine.validate(.init(text: "Hi", sourceCode: "Klingon", targetCode: "ms"))
        }
        #expect(throws: TranslateGemmaError.unknownLanguage("tlh")) {
            try TranslateGemmaEngine.validate(.init(text: "Hi", sourceCode: "en", targetCode: "tlh"))
        }
    }

    @Test func theTwoThousandTokenLimitIsEnforcedOnTheRenderedPrompt() throws {
        try TranslateGemmaEngine.requireWithinLimit(promptTokens: 2048)
        #expect(throws: TranslateGemmaError.promptTooLong(tokens: 2049)) {
            try TranslateGemmaEngine.requireWithinLimit(promptTokens: 2049)
        }
        let sentence = TranslateGemmaError.promptTooLong(tokens: 2049).description
        #expect(sentence.contains("2049"))
        #expect(sentence.contains("2,048"))
        #expect(sentence.hasSuffix("Shorten it and try again."))
    }

    // MARK: - Detection (decision 5)

    @Test func detectionFindsTheClearCases() {
        #expect(TranslateGemmaLanguageDetector.detect("The weather is lovely today. We walked along the river.")?.code == "en")
        // Malay and Indonesian share a script; a short sentence can read as either, and both are template languages.
        // The screen shows what was detected and the person can override it, so either answer is correct here.
        #expect(["ms", "id"].contains(TranslateGemmaLanguageDetector.detect("Cuaca hari ini sangat baik. Saya suka membaca buku.")?.code))
        #expect(TranslateGemmaLanguageDetector.detect("今天天气很好。我们沿着河边散步。")?.code == "zh-Hans")
        #expect(TranslateGemmaLanguageDetector.detect("今日は天気がとてもいいです。")?.code == "ja")
    }

    @Test func detectionNeverGuesses() {
        #expect(TranslateGemmaLanguageDetector.detect("") == nil)
        #expect(TranslateGemmaLanguageDetector.detect("   \n") == nil)
        // A confidence no hypothesis can reach: the caller must ask the person to pick (decision 5).
        #expect(TranslateGemmaLanguageDetector.detect("Hello", threshold: 1.1) == nil)
    }

    // MARK: - The stage seam

    @Test func theStagePassesTextThroughTheInjectedTranslator() async throws {
        let stage = TranslateGemmaStage(id: "t", name: "TranslateGemma", sourceCode: "en", targetCode: "ms") { text in
            TranslateGemmaEngine.Outcome(text: "Terjemahan: \(text)", promptTokens: 12, generatedTokens: 4,
                                         cancelled: false)
        }
        let result = try await stage.run(.text("Hello")) { _ in }
        guard case .text(let translated) = result else {
            Issue.record("the stage must produce text, got \(result.kind)")
            return
        }
        #expect(translated == "Terjemahan: Hello")
        #expect(stage.accepts == .text && stage.produces == .text)
    }

    @Test func aCancelledRunIsNotATranslation() async {
        let stage = TranslateGemmaStage(id: "t", name: "TranslateGemma", sourceCode: "en", targetCode: "ms") { _ in
            .cancelled
        }
        await #expect(throws: CancellationError.self) {
            _ = try await stage.run(.text("Hello")) { _ in }
        }
    }

    @Test func theStageRefusesANonTextInput() async {
        let stage = TranslateGemmaStage(id: "t", name: "TranslateGemma", sourceCode: "en", targetCode: "ms") { _ in
            .cancelled
        }
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [0], sampleRate: 16_000))) { _ in }
        }
    }

    // MARK: - The SDK's config plumbing

    @Test func theSDKCarriesBothCodesFromTheConfig() throws {
        let stage = try TranslateGemmaSDK().makeStage(
            for: entry(), config: StageConfig(sourceLanguage: "ms", targetLanguage: "zh-Hans"))
        #expect((stage as? TranslateGemmaStage)?.sourceCode == "ms")
        #expect((stage as? TranslateGemmaStage)?.targetCode == "zh-Hans")
    }

    @Test func weightsOnlyDropsTheTranslateLanguagesButKeepsQuantize() {
        var config = StageConfig(sourceLanguage: "ms", targetLanguage: "zh-Hans")
        config.trimSilence = true
        config.quantize = 4
        let weights = config.weightsOnly
        #expect(weights.sourceLanguage == nil)
        #expect(weights.targetLanguage == nil)
        #expect(weights.trimSilence == nil)
        #expect(weights.quantize == 4)
    }

    @Test func theSDKsProbeStageRefusesWithoutLanguages() async throws {
        // The Run sheet builds this probe stage before languages are picked (MC2-5's screen replaces it per run); it
        // must exist, and a run through it must say so rather than translate from a guess.
        let stage = try TranslateGemmaSDK().makeStage(for: entry(), config: .default)
        await #expect(throws: TranslateGemmaError.missingLanguages) {
            _ = try await stage.run(.text("Hello")) { _ in }
        }
    }
}
