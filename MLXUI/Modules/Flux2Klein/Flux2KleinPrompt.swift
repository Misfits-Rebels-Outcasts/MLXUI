import Foundation
import MLXLMCommon

// FLUX.2 Klein prompt template and tokenization (MP7-3b).
// The template, the truncate-then-right-pad order and the attention mask follow
// VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `KleinEmbeddingExtractor` (`formatQwen3ChatTemplate`,
// `getKleinTokenIds`); every id is checked against diffusers 0.37.1's `Flux2KleinPipeline._get_qwen3_prompt_embeds` front half
// (MLXUITests/Fixtures/Flux2Klein/g3b_tokens.json, g3b_tokens_edge.json).
//
// "[IMG]": the reference (flux-2-swift-mlx @ 315c27e) strips the literal "[IMG]" from the prompt. Owner ruling 2026-10-31: pass it
// through as plain text like diffusers and catflow-mlx's runtime. Later slices must not port the strip back.
//
// Shape of the result: always `maxLength` (512) ids, right-padded with `<|endoftext|>` (151643), and a mask of 1 for real tokens and 0
// for padding. The DiT sees all 512 text rows (the text encoder masks the pad rows' attention, the transformer does not), so the pad
// rows are model input and the mask belongs with the ids.
//
// Truncation: a prompt longer than 512 tokens is cut at the END (diffusers `truncation=True`, reference `prefix(maxLength)`), which
// drops the closing `<|im_end|>` and the assistant/`<think>` tail along with the last words. The cut is never silent: `cutTokenCount`
// says how many tokens went.

nonisolated enum Flux2KleinPromptError: Error, Equatable, LocalizedError {
    /// The tokenizer's `<|endoftext|>` is not 151643, i.e. this is not the Qwen3 vocabulary the checkpoint was trained with.
    case padTokenMismatch(found: Int?)

    var errorDescription: String? {
        switch self {
        case let .padTokenMismatch(found):
            return "FLUX.2 Klein expects the Qwen3 tokenizer (<|endoftext|> = 151643); this one has \(found.map(String.init) ?? "no <|endoftext|> token")."
        }
    }
}

nonisolated struct Flux2KleinEncodedPrompt: Equatable, Sendable {
    /// `maxLength` ids: the real tokens, then padding.
    let ids: [Int32]
    /// 1 for a real token, 0 for padding; same length as `ids`.
    let attentionMask: [Int32]
    /// Number of real tokens kept (the count of 1s in the mask).
    let realLength: Int
    /// Tokens removed from the end because the prompt (with its template) was longer than `maxLength`; 0 when nothing was cut.
    let cutTokenCount: Int

    var wasTruncated: Bool { cutTokenCount > 0 }
}

nonisolated enum Flux2KleinPrompt {
    static let maxLength = 512
    /// `<|endoftext|>`, the checkpoint tokenizer's `pad_token` (tokenizer_config.json).
    static let padTokenID = 151_643

    /// The prompt as the model was trained to see it: the Qwen3 chat template with no system message and `enable_thinking=False`
    /// (an empty think block after the assistant tag). The prompt is inserted verbatim (a literal "[IMG]" is not stripped).
    static func template(_ prompt: String) -> String {
        "<|im_start|>user\n" + prompt + "<|im_end|>\n" + "<|im_start|>assistant\n" + "<think>\n\n</think>\n\n"
    }

    /// Template, tokenize with `encode` (the real Qwen3 BPE: text -> ids, special tokens parsed from text), cut at `maxLength`, right-pad.
    static func encode(_ prompt: String, using encode: (String) -> [Int]) -> Flux2KleinEncodedPrompt {
        let all = encode(template(prompt))
        let kept = Array(all.prefix(maxLength))
        let pad = maxLength - kept.count
        return Flux2KleinEncodedPrompt(
            ids: kept.map { Int32($0) } + Array(repeating: Int32(padTokenID), count: pad),
            attentionMask: Array(repeating: 1, count: kept.count) + Array(repeating: 0, count: pad),
            realLength: kept.count,
            cutTokenCount: all.count - kept.count)
    }

    /// The same with an `MLXLMCommon.Tokenizer` (what `HFTokenizerLoader` returns); refuses a vocabulary whose pad token is not 151643.
    static func encode(_ prompt: String, tokenizer: any MLXLMCommon.Tokenizer) throws -> Flux2KleinEncodedPrompt {
        let pad = tokenizer.convertTokenToId("<|endoftext|>")
        guard pad == padTokenID else { throw Flux2KleinPromptError.padTokenMismatch(found: pad) }
        return encode(prompt) { tokenizer.encode(text: $0, addSpecialTokens: true) }
    }
}
