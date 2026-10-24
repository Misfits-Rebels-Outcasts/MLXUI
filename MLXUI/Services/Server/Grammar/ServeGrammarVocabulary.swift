import Foundation
import MLXGuidedGeneration
import MLXLMCommon

// WW4-1-1 — the per-model half of grammar-constrained decoding (design note `RSI/notes/claude-code/wwdc26/ww4-design.md`,
// owner ruling AN). Nothing here is wired into the server yet (WW4-1-3 does that); it is the pieces, tested on their own.

/// Which stop token a grammar ends on.
///
/// `GrammarTokenizer` registers **one** stop token, and once the grammar is complete its mask allows only that one. The
/// generation loop stops on any id in `configuration.eosTokenIds ∪ tokenizer.eosTokenId ∪ extraEOSTokens`, so the pick must
/// come from that set (a token outside it would be forced by the mask and then not stop the loop), and it should be the
/// token the model actually ends a turn with, not the base `<eos>`: Gemma 4's `eos_token_id` is `[1, 106, 50]` where `1` is
/// `<eos>` and `106` is `<turn|>`.
nonisolated enum ServeGrammarEOS {
    /// The turn-ender spellings of the families the server serves, most specific first. Matched on the token's string, never
    /// on a family name or an id, so a new family whose turn-ender is one of these needs no table entry.
    static let turnEnders = ["<turn|>", "<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]

    /// The loop's stop set: the same three sources as mlx-swift-lm's `buildStopTokenIds`.
    static func stopIDs(configuration: ModelConfiguration, tokenizer: any MLXLMCommon.Tokenizer) -> Set<Int> {
        var ids = configuration.eosTokenIds
        if let eos = tokenizer.eosTokenId { ids.insert(eos) }
        for token in configuration.extraEOSTokens {
            if let id = tokenizer.convertTokenToId(token) { ids.insert(id) }
        }
        return ids
    }

    /// The id to register: the first turn-ender (in `turnEnders` order) found in the stop set; else the tokenizer's own EOS
    /// when it is in the set; else the smallest id in the set. `nil` only when the set is empty (the caller refuses).
    static func pick(stopIDs: Set<Int>, tokenizerEOS: Int?, tokenString: (Int) -> String?) -> Int? {
        let named = stopIDs.compactMap { id in tokenString(id).map { (id, $0) } }
        for ender in turnEnders {
            if let match = named.filter({ $0.1 == ender }).map(\.0).min() { return match }
        }
        if let tokenizerEOS, stopIDs.contains(tokenizerEOS) { return tokenizerEOS }
        return stopIDs.min()
    }

    static func pick(configuration: ModelConfiguration, tokenizer: any MLXLMCommon.Tokenizer) -> Int? {
        pick(stopIDs: stopIDs(configuration: configuration, tokenizer: tokenizer),
             tokenizerEOS: tokenizer.eosTokenId, tokenString: { tokenizer.convertIdToToken($0) })
    }
}

/// What a model contributes to grammar-constrained decoding: xgrammar's tokenizer tables (read-only after construction, so
/// shared by every request), the stop token the grammar ends on, and the whitespace tokens the processor penalises once a
/// model starts looping on them. Built once per model by `ServeGrammarCache`.
nonisolated struct ServeGrammarVocabulary: @unchecked Sendable {
    /// `<directory path>#eos=<id>` — the cache key, and the first half of a constraint's key.
    let key: String
    let tokenizer: GrammarTokenizer
    let eosTokenID: Int
    let whitespaceTokenIDs: Set<Int>
    /// `-200` at each whitespace token, `0` elsewhere, one entry per vocabulary token — plain floats, so no `MLXArray` is
    /// built on the thread that builds this (the iterator runs on two threads; see `ServeGrammarProcessor`).
    let whitespaceBias: [Float]
    /// WW4-1-3 (soft-zone closing bias): the library's `ClosingTokenBias` tiers as plain floats — `+200` at the stop token, `+100`
    /// at `"`, `}`, `]` and the single digits, `0` elsewhere. Added to the logits in the last few tokens of a reply so the model
    /// closes the document it is in; the grammar mask still decides what is allowed (the stop token only where the document is whole).
    let closingBias: [Float]

    var vocabularyCount: Int { tokenizer.vocabSize }

    /// From strings already in hand (tests, and the production initialiser below).
    init(key: String, vocab: [String], vocabType: VocabType, eosTokenID: Int, whitespaceTokenIDs: Set<Int>) throws {
        self.key = key
        self.tokenizer = try GrammarTokenizer(vocab: vocab, vocabType: vocabType, eosTokenId: Int32(eosTokenID))
        self.eosTokenID = eosTokenID
        self.whitespaceTokenIDs = whitespaceTokenIDs
        var bias = [Float](repeating: 0, count: vocab.count)
        for id in whitespaceTokenIDs where id >= 0 && id < bias.count { bias[id] = ServeGrammarProcessor.whitespacePenalty }
        self.whitespaceBias = bias
        var closing = [Float](repeating: 0, count: vocab.count)
        for (id, token) in vocab.enumerated() where Self.closingTokens.contains(token) { closing[id] = 100 }
        if eosTokenID >= 0, eosTokenID < closing.count { closing[eosTokenID] = 200 }
        self.closingBias = closing
    }

    private static let closingTokens: Set<String> = ["\"", "}", "]", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    /// From a loaded tokenizer. Synchronous and CPU-heavy (a walk of the whole vocabulary, then xgrammar's tables): the cache
    /// runs it off the main actor.
    init(directory: URL, tokenizer: any MLXLMCommon.Tokenizer, eosTokenID: Int) throws {
        let extracted = TokenizerVocabExtractor.extractForGrammar(from: tokenizer)
        // `WhitespaceTokenBias.compute` also builds an `MLXArray`, which is dropped unread; only the id set is used.
        let whitespace = WhitespaceTokenBias.compute(tokenizer: tokenizer).tokenIDs
        try self.init(key: Self.key(directory: directory, eosTokenID: eosTokenID), vocab: extracted.vocab,
                      vocabType: extracted.vocabType, eosTokenID: eosTokenID, whitespaceTokenIDs: whitespace)
    }

    static func key(directory: URL, eosTokenID: Int) -> String {
        "\(directory.standardizedFileURL.path)#eos=\(eosTokenID)"
    }
}
