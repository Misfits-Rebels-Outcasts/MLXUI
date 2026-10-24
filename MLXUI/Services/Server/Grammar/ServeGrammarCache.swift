import CryptoKit
import Foundation
import MLXGuidedGeneration
import MLXLMCommon

/// What the reply must look like (the two `response_format` kinds, neutral of dialect).
nonisolated enum ServeGrammarFormat: Sendable, Equatable {
    /// `{"type": "json_object"}` — any JSON **object**.
    case jsonObject
    /// `{"type": "json_schema", …}` — the schema text, exactly as the client sent it (already validated by WW4-1-2).
    case jsonSchema(String)

    /// The schema xgrammar compiles. `json_object` states `additionalProperties: true` explicitly rather than relying on a
    /// default: the shim compiles with `strict_mode = true`, which forbids undeclared properties *when the schema declares
    /// some*. (WW4-1-1 measured on the real xgrammar that a bare `{"type":"object"}` also accepts `{"a":1}`; the design note's
    /// worry that it allowed only `{}` was wrong, and the explicit form is kept because it doesn't depend on that default.)
    /// A root that is not an object is refused either way — `jsonObjectRefusesWhatIsNotAnObject`.
    var schemaText: String {
        switch self {
        case .jsonObject: return #"{"type":"object","additionalProperties":true}"#
        case .jsonSchema(let text): return text
        }
    }
}

/// A constraint on loan to one request. Return it with `ServeGrammarCache.release` when the generation ends, however it ends.
nonisolated struct ServeGrammarLease: @unchecked Sendable {
    let constraint: GrammarConstraint
    fileprivate let key: String
}

/// The two caches grammar-constrained decoding needs (design note, "Per-request cost").
///
/// - **Vocabularies**, one per model (`ServeGrammarVocabulary`): xgrammar copies the whole vocabulary into its tables, which
///   is the per-model cost, and the tables are read-only afterwards.
/// - **Compiled constraints**, per (vocabulary, schema): the shim makes a new `GrammarCompiler` for every `GrammarConstraint`,
///   so xgrammar's own compile cache is thrown away each time, and the package's README warns a cold compile blocks for
///   hundreds of milliseconds. A constraint is therefore **leased**, not shared: a request takes an idle one (or compiles a
///   new one when none is idle), and on return the cache rolls it back to its initial state and keeps it for the next request.
///
/// **Deviation from the design note, found by this task's tests (WW4-1-1):** the note planned `root.clone()` per request. In
/// the vendored xgrammar (v0.1.30) `xg_matcher_fork` is a stub that always fails ("Fork() not available in xgrammar v0.1.30";
/// `Fork()` arrived in v0.1.34), so a matcher can't be copied. `rollback(n)` is wired and the matcher is created with unlimited
/// rollback, so a used constraint is reset by rolling back every token it accepted. The server runs one generation per model,
/// so in practice a request finds an idle constraint and pays no compile.
///
/// RAM only, and nothing in it is request content: a schema is the client's, the key is its hash, and it is never logged.
/// An `actor` (like `ServePromptCounter`) so the compile never runs on the main actor and two requests can't race the
/// dictionaries.
actor ServeGrammarCache {
    static let shared = ServeGrammarCache()

    private var vocabularies: [String: ServeGrammarVocabulary] = [:]
    private var vocabularyRecency: [String] = []     // most recent first
    private var idle: [String: [GrammarConstraint]] = [:]
    private var idleRecency: [String] = []
    private let vocabularyCapacity: Int
    private let keyCapacity: Int
    private let idlePerKey: Int

    /// Counts for tests and for the journal's measurement.
    private(set) var vocabularyBuilds = 0
    private(set) var compiles = 0
    private(set) var resets = 0

    init(vocabularyCapacity: Int = 4, keyCapacity: Int = 16, idlePerKey: Int = 2) {
        self.vocabularyCapacity = vocabularyCapacity
        self.keyCapacity = keyCapacity
        self.idlePerKey = idlePerKey
    }

    var cachedVocabularies: Int { vocabularies.count }
    /// Distinct (vocabulary, schema) keys holding at least one idle constraint.
    var cachedKeys: Int { idle.count }
    var idleConstraints: Int { idle.values.reduce(0) { $0 + $1.count } }

    /// The model's vocabulary, built on first use. `build` runs inside the actor, so a second request for the same model waits
    /// for the first instead of building it twice.
    func vocabulary(key: String, build: () throws -> ServeGrammarVocabulary) throws -> ServeGrammarVocabulary {
        if let cached = vocabularies[key] {
            Self.touch(key, in: &vocabularyRecency)
            return cached
        }
        let built = try build()
        vocabularyBuilds += 1
        if vocabularies.count >= vocabularyCapacity, let oldest = vocabularyRecency.last {
            vocabularies[oldest] = nil
            vocabularyRecency.removeLast()
            // Constraints compiled against the evicted vocabulary can't be used with a new one (their key starts with it).
            let stale = idle.keys.filter { $0.hasPrefix(oldest + "|") }
            for key in stale { idle[key] = nil }
            idleRecency.removeAll { stale.contains($0) }
        }
        vocabularies[key] = built
        Self.touch(key, in: &vocabularyRecency)
        return built
    }

    /// A constraint at its initial state for this request: an idle one, else a fresh compile.
    /// - Throws: `GrammarError` — `.invalidJSONSchema` for a schema xgrammar rejects (WW4-1-2 turns it into a 400).
    func lease(vocabulary: ServeGrammarVocabulary, format: ServeGrammarFormat) throws -> ServeGrammarLease {
        let key = Self.key(vocabulary: vocabulary, format: format)
        if var stack = idle[key], let constraint = stack.popLast() {
            idle[key] = stack.isEmpty ? nil : stack
            if stack.isEmpty { idleRecency.removeAll { $0 == key } } else { Self.touch(key, in: &idleRecency) }
            return ServeGrammarLease(constraint: constraint, key: key)
        }
        let constraint = try Self.compile(vocabulary: vocabulary, format: format)
        compiles += 1
        return ServeGrammarLease(constraint: constraint, key: key)
    }

    /// Give a leased constraint back. `committedTokens` is how many tokens the matcher accepted (the stop token included); they
    /// are rolled back so the next lease starts from the beginning. A constraint that failed, or whose rollback fails, is
    /// dropped rather than reused.
    func release(_ lease: ServeGrammarLease, committedTokens: Int, reusable: Bool = true) {
        guard reusable else { return }
        do {
            if committedTokens > 0 { try lease.constraint.rollback(Int32(committedTokens)) }
        } catch { return }
        resets += 1
        var stack = idle[lease.key] ?? []
        guard stack.count < idlePerKey else { return }
        if idle[lease.key] == nil, idle.count >= keyCapacity, let oldest = idleRecency.last {
            idle[oldest] = nil
            idleRecency.removeLast()
        }
        stack.append(lease.constraint)
        idle[lease.key] = stack
        Self.touch(lease.key, in: &idleRecency)
    }

    /// A new constraint. Fast-forward is off (design note): it saves no model passes and re-encodes the forced bytes, which can
    /// disagree with the grammar's own tokenisation; no host tokenizer is needed either way.
    nonisolated static func compile(vocabulary: ServeGrammarVocabulary, format: ServeGrammarFormat) throws -> GrammarConstraint {
        try GrammarConstraint(tokenizer: vocabulary.tokenizer, jsonSchema: format.schemaText, fastForward: false)
    }

    static func key(vocabulary: ServeGrammarVocabulary, format: ServeGrammarFormat) -> String {
        let digest = SHA256.hash(data: Data(format.schemaText.utf8))
        return vocabulary.key + "|" + digest.map { String(format: "%02x", $0) }.joined()
    }

    private static func touch(_ key: String, in recency: inout [String]) {
        recency.removeAll { $0 == key }
        recency.insert(key, at: 0)
    }
}
