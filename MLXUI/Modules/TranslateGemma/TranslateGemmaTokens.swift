import Foundation
import MLXLMCommon
import Tokenizers

/// MC2-5 — counts the **rendered** prompt with the model's own tokenizer, without loading a weight. The Translate
/// screen shows this count live against the card's 2K limit (owner decision 8) and refuses before any work; the engine
/// counts the same ids from its prepared `LMInput`, so the screen and the run cannot disagree.
///
/// It reads the installed `tokenizer.json` + `chat_template.jinja` through swift-transformers' `AutoTokenizer` (the same
/// loader the engine's tokenizer comes from), once per model directory.
actor TranslateGemmaTokenCounter {
    static let shared = TranslateGemmaTokenCounter()

    private var cached: (directory: URL, tokenizer: any Tokenizers.Tokenizer)?

    /// The rendered request's token count, exactly as `applyChatTemplate` encodes it (the engine's `LMInput`).
    func tokenCount(for request: TranslateGemmaEngine.Request, modelDirectory: URL) async throws -> Int {
        let tokenizer = try await tokenizer(for: modelDirectory)
        let messages = try TranslateGemmaPrompt.messages(sourceCode: request.sourceCode,
                                                        targetCode: request.targetCode,
                                                        text: request.text)
        return try tokenizer.applyChatTemplate(messages: messages).count
    }

    /// For a test to check the cache without a second load.
    func hasLoadedTokenizer(for directory: URL) -> Bool {
        cached?.directory == directory
    }

    private func tokenizer(for directory: URL) async throws -> any Tokenizers.Tokenizer {
        if let cached, cached.directory == directory { return cached.tokenizer }
        let loaded = try await AutoTokenizer.from(modelFolder: directory)
        cached = (directory, loaded)
        return loaded
    }
}
