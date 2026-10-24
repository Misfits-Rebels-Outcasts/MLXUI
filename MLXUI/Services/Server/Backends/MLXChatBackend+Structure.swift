import Foundation
import MLXGuidedGeneration
import MLXLMCommon

// WW4-1-3 — the structured-output half of `MLXChatBackend` that runs BEFORE a request takes a queue place or a model lease
// (ruling AS): the allow-list, the schema checks, the model's vocabulary and the grammar compile. Nothing here loads weights.

/// `any Tokenizer` crossing into the compile thread's build closure. The tokenizer is only read there, by one job at a time.
private struct TokenizerBox: @unchecked Sendable { let value: any MLXLMCommon.Tokenizer }

extension MLXChatBackend {
    /// The `param` named in the 400s until each dialect supplies its own (WW4-1-4).
    static let structureParam = "response_format"

    func prepareStructure(_ request: ServeRequest) async throws -> ServeGrammarGrant? {
        let structure = request.structure
        guard structure.isConstrained else { return nil }

        // Ruling AX: only families verified on real weights.
        guard ServeStructuredOutputFamilies.allows(modelType: checkpointModelType) else {
            throw ServeStructureCheck.unsupportedModelRefusal(model: request.model)
        }
        // The pure refusals first (schema, tools + format) — they cost no tokenizer.
        if let refusal = ServeStructureCheck.validate(structure: structure, param: Self.structureParam,
                                                      honouredTools: !effectiveTools(request).isEmpty,
                                                      effectiveMaxTokens: request.effectiveMaxTokens, minimumTokens: { nil }) {
            throw refusal
        }
        guard let tokenizer = await ServePromptCounter.shared.tokenizer(for: directory) else {
            throw ServeStructureCheck.unavailable
        }
        var closingReserve: Int?
        if case .jsonSchema(_, let schema, _) = structure {
            let minimum = ServeStructureCheck.minimumReplyTokens(schema: schema, tokenizer: tokenizer)
            if let refusal = ServeStructureCheck.replyBudgetRefusal(effectiveMaxTokens: request.effectiveMaxTokens, minimumTokens: minimum) {
                throw refusal
            }
            if closingBiasEnabled { closingReserve = minimum }
        }
        // Ruling AT: the stop token is read from the model's own files, with the loader's precedence.
        guard let eos = ServeGrammarEOS.pick(directory: directory, tokenizer: tokenizer),
              let format = structure.grammarFormat else { throw ServeStructureCheck.unavailable }

        let directory = directory
        let box = TokenizerBox(value: tokenizer)
        do {
            let vocabulary = try await grammarCache.vocabulary(key: ServeGrammarVocabulary.key(directory: directory, eosTokenID: eos)) {
                try ServeGrammarVocabulary(directory: directory, tokenizer: box.value, eosTokenID: eos)
            }
            let lease = try await grammarCache.lease(vocabulary: vocabulary, format: format)
            return ServeGrammarGrant(cache: grammarCache, lease: lease, vocabulary: vocabulary, format: format, closingReserve: closingReserve)
        } catch let error as GrammarError {
            throw ServeStructureCheck.error(from: error, param: Self.structureParam)
        }
    }
}
