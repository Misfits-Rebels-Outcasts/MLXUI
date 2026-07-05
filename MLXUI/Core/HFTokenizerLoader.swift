import Foundation
import MLXLMCommon
import Tokenizers

/// The real tokenizer loader: loads the model's actual BPE tokenizer + Jinja chat template
/// via swift-transformers' `AutoTokenizer`, bridged to `MLXLMCommon.Tokenizer`. Replaces the
/// old hand-rolled `SimpleTokenizer` stub (whitespace-split, no chat template), which broke
/// tokenization for every model and, for VLMs, never emitted the `<|image_pad|>` vision
/// markers → "placeholder tokens ≠ frames" (journal/2026-37). Bridge mirrors mlx-swift-lm's
/// `#adaptHuggingFaceTokenizer`.
struct HFTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        let upstream = try await AutoTokenizer.from(modelFolder: directory)
        return HFTokenizerBridge(upstream)
    }
}

/// Adapts a swift-transformers `Tokenizers.Tokenizer` to `MLXLMCommon.Tokenizer`.
private struct HFTokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer

    init(_ upstream: any Tokenizers.Tokenizer) {
        self.upstream = upstream
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        upstream.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    // swift-transformers uses `decode(tokens:)` rather than `decode(tokenIds:)`.
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        upstream.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
    }

    func convertTokenToId(_ token: String) -> Int? { upstream.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { upstream.convertIdToToken(id) }

    var bosToken: String? { upstream.bosToken }
    var eosToken: String? { upstream.eosToken }
    var unknownToken: String? { upstream.unknownToken }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        do {
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
