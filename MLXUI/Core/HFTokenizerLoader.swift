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
        // Strict Qwen3.5-style templates raise on ChatSession's tool-only restart render;
        // pass a sanitized copy instead of the checkpoint's template when needed.
        let override = ChatTemplateSanitizer.sanitizedTemplate(inModelDirectory: directory)
        return HFTokenizerBridge(upstream, chatTemplateOverride: override)
    }
}

/// Adapts a swift-transformers `Tokenizers.Tokenizer` to `MLXLMCommon.Tokenizer`.
private struct HFTokenizerBridge: MLXLMCommon.Tokenizer, @unchecked Sendable {
    private let upstream: any Tokenizers.Tokenizer
    /// Sanitized chat template (`ChatTemplateSanitizer`); nil → use the model's own.
    private let chatTemplateOverride: String?

    init(_ upstream: any Tokenizers.Tokenizer, chatTemplateOverride: String? = nil) {
        self.upstream = upstream
        self.chatTemplateOverride = chatTemplateOverride
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
            if let chatTemplateOverride {
                return try upstream.applyChatTemplate(
                    messages: messages,
                    chatTemplate: .literal(chatTemplateOverride),
                    // Match the defaults of the no-template overload below.
                    addGenerationPrompt: true,
                    truncation: false,
                    maxLength: nil,
                    tools: tools,
                    additionalContext: additionalContext)
            }
            return try upstream.applyChatTemplate(
                messages: messages, tools: tools, additionalContext: additionalContext)
        } catch Tokenizers.TokenizerError.missingChatTemplate {
            throw MLXLMCommon.TokenizerError.missingChatTemplate
        }
    }
}
