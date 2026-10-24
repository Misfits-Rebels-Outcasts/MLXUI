import Foundation
import MLXGuidedGeneration
import MLXLMCommon

/// One request's hold on a leased grammar constraint (WW4-1-3, ruling AS).
///
/// `prepareStructure` compiles (or reuses) the constraint **before** the request takes a queue place or a model lease and
/// returns this grant; whoever ends the request calls `finish(clean:)`. It is idempotent — the first call releases, later
/// ones do nothing — so the backend (which knows whether generation ended cleanly) and the pipeline (which owns every other
/// exit: a refused reservation, a cancelled wait, a thrown error) can both call it, and a path nobody remembered is caught
/// by `deinit`, which releases as *not* reusable.
///
/// Reusable only after a clean finish: the generation loop has ended on its own, so nothing else is touching the matcher
/// when it is rolled back. After an error, a cancellation or a client `stop` the loop may still be running; the constraint
/// is dropped, never reset.
nonisolated final class ServeGrammarGrant: @unchecked Sendable {
    let vocabulary: ServeGrammarVocabulary
    let format: ServeGrammarFormat
    /// The smallest valid reply's token count, when known (a schema, not `json_object`); nil = no closing bias.
    let closingReserve: Int?
    private let cache: ServeGrammarCache
    private let lease: ServeGrammarLease
    private let lock = NSLock()
    private var processor: ServeGrammarProcessor?
    private var finished = false

    init(cache: ServeGrammarCache, lease: ServeGrammarLease, vocabulary: ServeGrammarVocabulary, format: ServeGrammarFormat,
         closingReserve: Int? = nil) {
        self.closingReserve = closingReserve
        self.cache = cache
        self.lease = lease
        self.vocabulary = vocabulary
        self.format = format
    }

    /// The logit processor for the one generation this grant serves (the same instance on every call). `maxTokens` (the reply
    /// limit) arms the soft-zone closing bias when the grant has a reserve.
    func makeProcessor(maxTokens: Int? = nil) -> ServeGrammarProcessor {
        lock.lock(); defer { lock.unlock() }
        if let processor { return processor }
        let made = ServeGrammarProcessor(constraint: lease.constraint, vocabulary: vocabulary, format: format)
        if let maxTokens, let closingReserve { made.enableClosingBias(maxTokens: maxTokens, reserve: closingReserve) }
        processor = made
        return made
    }

    var isFinished: Bool { lock.lock(); defer { lock.unlock() }; return finished }

    /// Give the constraint back. `clean`: the generation loop ended by itself. A processor that recorded a matcher failure is
    /// never reused whatever the caller says. Returns `true` for the call that actually released.
    @discardableResult
    func finish(clean: Bool) async -> Bool {
        lock.lock()
        if finished { lock.unlock(); return false }
        finished = true
        let used = processor
        lock.unlock()
        let reusable = clean && used?.failure == nil
        await cache.release(lease, committedTokens: used?.committedTokens ?? 0, reusable: reusable)
        return true
    }

    deinit {
        lock.lock()
        let already = finished
        lock.unlock()
        guard !already else { return }
        let cache = cache, lease = lease
        Task { await cache.release(lease, committedTokens: 0, reusable: false) }
    }
}

/// The families verified for structured output on real weights (rulings AT, AX). Anything else answers a JSON format with
/// 400 `structured_output_unsupported`. A family is added here only when WW4-1-3's owner-run real-model check has passed
/// for it, and the journal says so.
nonisolated enum ServeStructuredOutputFamilies {
    /// `model_type`s. `gemma4`: verified 2026-10-24 on gemma-4-e4b-it-4bit (journal 2026-573: stop token, 30/30 parses, cache equality, leases). `qwen3` is ruling AX's second
    /// family but is NOT here: Qwen3-8B-4bit was not installed when the check was due.
    static let verified: Set<String> = ["gemma4"]

    /// Tests lift the gate for a candidate family (`MLXUI_TEST_STRUCT_CANDIDATE=1` in the real-model suite) — that is how a
    /// family earns its place.
    nonisolated(unsafe) static var overrideForTesting: Set<String>?

    static func allows(modelType: String) -> Bool { (overrideForTesting ?? verified).contains(modelType) }
}

nonisolated extension ServeGrammarEOS {
    /// The stop set the model's loader will build, computed from the same files and with the same precedence as mlx-swift-lm's
    /// factory (`config.json` `eos_token_id`, overridden by `generation_config.json`'s), plus the tokenizer's own EOS — so the
    /// pick can be made **before** the weights are loaded. `extraEOSTokens` only exist on a loaded configuration; the
    /// generation re-checks the pick against that one (`ServeGrammarEOS.pickIsStopToken`) and fails closed if they disagree.
    static func stopIDs(directory: URL, tokenizer: any MLXLMCommon.Tokenizer) -> Set<Int> {
        var ids: Set<Int> = []
        if let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
           let base = try? JSONDecoder.json5().decode(BaseConfiguration.self, from: data) {
            ids = base.effectiveEOSTokenIds
        }
        if let data = try? Data(contentsOf: directory.appendingPathComponent("generation_config.json")),
           let generation = try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: data),
           let override = generation.eosTokenIds?.values {
            ids = Set(override)
        }
        if let eos = tokenizer.eosTokenId { ids.insert(eos) }
        return ids
    }

    /// The pick for a model directory, or `nil` when it has no stop token at all.
    static func pick(directory: URL, tokenizer: any MLXLMCommon.Tokenizer) -> Int? {
        pick(stopIDs: stopIDs(directory: directory, tokenizer: tokenizer), tokenizerEOS: tokenizer.eosTokenId,
             tokenString: { tokenizer.convertIdToToken($0) })
    }

    /// The generation loop stops on `configuration.eosTokenIds ∪ tokenizer.eosTokenId ∪ extraEOSTokens`. The grammar forces
    /// its stop token once the document is complete, so that token must be in the loaded model's own set.
    static func pickIsStopToken(_ pick: Int, configuration: ModelConfiguration, tokenizer: any MLXLMCommon.Tokenizer) -> Bool {
        stopIDs(configuration: configuration, tokenizer: tokenizer).contains(pick)
    }
}
