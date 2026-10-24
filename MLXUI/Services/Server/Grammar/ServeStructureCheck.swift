import Foundation
import MLXGuidedGeneration
import MLXLMCommon

/// WW4-1-2 — every way a structured-output request is refused *before* anything runs (pure: no model, no server wiring, no
/// dialect parsing; WW4-1-3 calls these from `ServePipeline.prepare` and WW4-1-4 supplies each dialect's `param`).
///
/// The three refusals the backlog names, in the order `validate` applies them:
/// 1. **`invalid_json_schema`** — the schema is unusable (`ServeSchemaCheck`: malformed, a keyword xgrammar would silently
///    drop, a construct whose compile takes too long) or xgrammar itself rejected it (`GrammarError.invalidJSONSchema`).
/// 2. **`unsupported_combination`** — `tools` that will be honoured together with a JSON format. A tool call is `<tool_call>…`
///    text, which the grammar forbids. `.none` — which is what `{"type": "text"}` decodes to — is a no-op: never this
///    error, never constrained (owner ruling AO, 2026-10-24).
/// 3. **`max_tokens_too_small_for_schema`** — the smallest valid reply is longer than the reply budget, so a cut is certain.
nonisolated enum ServeStructureCheck {
    static let invalidSchemaCode = "invalid_json_schema"
    static let unsupportedCombinationCode = "unsupported_combination"
    static let maxTokensTooSmallCode = "max_tokens_too_small_for_schema"
    /// What a failure that isn't the client's schema is reported as (never xgrammar's own words — S2's rule that a backend
    /// failure doesn't leak its internals).
    static let unavailableCode = "structured_output_unavailable"

    /// The first sentences of a refusal list; the rest are counted, not printed.
    static let sentencesShown = 6

    // MARK: - 1. The schema

    /// `nil` when the schema may be compiled. `param` is the dialect's name for the field (`response_format.json_schema.schema`,
    /// `text.format.schema`), prefixed to every JSON pointer so the client can find the construct.
    static func schemaRefusal(_ schema: String, param: String) -> ServeError? {
        refusal(for: ServeSchemaCheck.check(schema), param: param)
    }

    static func refusal(for outcome: ServeSchemaCheck.Outcome, param: String) -> ServeError? {
        switch outcome {
        case .ok:
            return nil
        case .malformed(let reason):
            return ServeError(status: 400, message: "\(param) is not a usable JSON Schema: \(reason).", code: invalidSchemaCode)
        case .refused(let findings):
            let shown = findings.prefix(sentencesShown).map { $0.sentence(param: param) }
            let more = findings.count > sentencesShown ? " (and \(findings.count - sentencesShown) more)" : ""
            return ServeError(status: 400,
                              message: "This schema can't be used for structured output: " + shown.joined(separator: " ") + more,
                              code: invalidSchemaCode)
        }
    }

    /// A `GrammarError` raised while compiling, as the error the client sees. Only `invalidJSONSchema` is the client's
    /// doing (the shim reports every compile-path exception as one); the rest are this build's, so they are a 500 with a plain
    /// sentence and none of xgrammar's text.
    static func error(from grammarError: GrammarError, param: String) -> ServeError {
        switch grammarError {
        case .invalidJSONSchema(let message):
            let text = message.trimmingCharacters(in: .whitespacesAndNewlines)
            return ServeError(status: 400,
                              message: "\(param) was rejected: \(text.isEmpty ? "it is not a valid JSON Schema" : text)",
                              code: invalidSchemaCode)
        default:
            return ServeError(status: 500, message: "Structured output couldn't be set up for this model.",
                              type: "server_error", code: unavailableCode)
        }
    }

    // MARK: - 2. tools + a JSON format

    /// `honouredTools`: the request carries tools the model's family will actually be shown (`backend.effectiveTools` is not
    /// empty) — a family that ignores `tools`, or `tool_choice: "none"`, is unaffected.
    static func combinationRefusal(structure: ServeStructure, honouredTools: Bool) -> ServeError? {
        guard structure.isConstrained, honouredTools else { return nil }
        return ServeError(status: 400,
                          message: "tools can't be combined with a JSON response format on this server: a tool call is not JSON "
                              + "the format allows. Send either tools or response_format {\"type\": \"json_object\" | "
                              + "\"json_schema\"}, or use tool_choice \"none\".",
                          code: unsupportedCombinationCode)
    }

    // MARK: - 3. The reply budget

    /// The token count of the smallest valid reply to `schema`, or `nil` when it can't be known
    /// (`CompletionReserve.estimate` falls back to a default for a schema it can't synthesise; that fallback is *not* a
    /// reserve, so it is turned into `nil` here). The tokenizer's special tokens (a BOS) that `encode` adds are subtracted.
    static func minimumReplyTokens(schema: String, tokenizer: any MLXLMCommon.Tokenizer) -> Int? {
        let unknown = -1
        let estimate = CompletionReserve.estimate(schemaJSON: schema, tokenizer: tokenizer, defaultReserve: unknown)
        guard estimate != unknown else { return nil }
        let special = tokenizer.encode(text: "", addSpecialTokens: true).count
        return max(1, estimate - special)
    }

    static func replyBudgetRefusal(effectiveMaxTokens: Int, minimumTokens: Int?) -> ServeError? {
        guard let minimumTokens, effectiveMaxTokens < minimumTokens else { return nil }
        return ServeError(status: 400,
                          message: "The smallest reply this schema allows is \(minimumTokens) tokens, but max_tokens allows "
                              + "\(effectiveMaxTokens) (after the context window and memory limits). Raise max_tokens or shrink the schema.",
                          code: maxTokensTooSmallCode)
    }

    // MARK: - All three

    /// The first refusal, in the order above, or `nil`. `minimumTokens` is a closure so a request that fails earlier never
    /// pays for a tokenizer walk; `json_object`'s smallest reply (`{}`) is never worth refusing for.
    static func validate(structure: ServeStructure, param: String, honouredTools: Bool, effectiveMaxTokens: Int,
                         minimumTokens: () -> Int?) -> ServeError? {
        switch structure {
        case .none:
            return nil
        case .jsonObject:
            return combinationRefusal(structure: structure, honouredTools: honouredTools)
        case .jsonSchema(_, let schema, _):
            if let refusal = schemaRefusal(schema, param: param) { return refusal }
            if let refusal = combinationRefusal(structure: structure, honouredTools: honouredTools) { return refusal }
            return replyBudgetRefusal(effectiveMaxTokens: effectiveMaxTokens, minimumTokens: minimumTokens())
        }
    }
}
