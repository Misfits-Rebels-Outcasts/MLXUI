import Foundation
import MLX
import MLXGuidedGeneration
import MLXLMCommon

/// The grammar as a `LogitProcessor` (owner ruling AN, design note "(a)"): the mask for the matcher's current state is added
/// to the logits, and the sampled token is committed to the matcher. Handed to `generateTokens` through
/// `GenerationComponents.logitProcessorFactory` (WW4-1-3), so temperature, top_p, seed, the prompt cache and raw token ids
/// all stay with the ordinary generation.
///
/// **Three rules the protocol forces** (design note, "Does `computeMask()` fit…"):
/// 1. `didSample` is a synchronisation point: it reads the sampled token (`MLXErrorGuard.item`) to advance the matcher, so
///    the iterator's one-token-ahead pipelining is gone for a structured request.
/// 2. **The mask lives on the CPU between `didSample` and `process`.** The iterator's first token runs on the caller's thread
///    and the rest on `GenerationWorker`'s; under mlx-swift 0.32 an array built on one thread can't be evaluated on another
///    (the Kokoro B10 crash class). So only `[Int32]` words are held, and the `MLXArray` is built *inside* `process`.
/// 3. The library's own mask builder is `internal`, so the unpacking is here (`additiveMask`).
///
/// A class because the matcher is shared mutable state and the backend reads the outcome afterwards (`isTerminated`,
/// `failure`, `committedTokens` — which the cache needs to reset the constraint). `copy()` is required (the protocol traps on a
/// class that doesn't implement it); the vendored xgrammar can't fork a matcher (see `ServeGrammarCache`), so a copy compiles
/// a fresh constraint and replays the committed tokens. Only the speculative iterators call it, and the server uses none.
/// Not a `final` class so that `copy()` can return `Self`.
nonisolated class ServeGrammarProcessor: LogitProcessor, @unchecked Sendable {
    /// Added to a whitespace-only token's logit once a run of three has been sampled (the loop's `WhitespaceTokenBias`
    /// magnitude).
    static let whitespacePenalty: Float = -200

    private let constraint: GrammarConstraint
    private let vocabulary: ServeGrammarVocabulary
    private let format: ServeGrammarFormat
    private let lock = NSLock()

    /// The mask for the next step: `nil` until computed, recomputed after each commit.
    private var pending: MaskResult?
    private var terminated = false
    /// After a failure the mask admits only the stop token, so the generation ends instead of producing text the grammar
    /// didn't allow.
    private var failed = false
    private var failureMessage: String?
    private var whitespace: WhitespaceRunTracker
    /// Every token the matcher accepted, in order: `copy()` replays it, and its count is what a rollback undoes.
    private var history: [Int32] = []

    init(constraint: GrammarConstraint, vocabulary: ServeGrammarVocabulary, format: ServeGrammarFormat) {
        self.constraint = constraint
        self.vocabulary = vocabulary
        self.format = format
        self.whitespace = WhitespaceRunTracker(whitespaceTokenIDs: vocabulary.whitespaceTokenIDs)
    }

    required init(copying other: ServeGrammarProcessor) {
        other.lock.lock()
        defer { other.lock.unlock() }
        vocabulary = other.vocabulary
        format = other.format
        whitespace = other.whitespace
        history = other.history
        terminated = other.terminated
        failed = other.failed
        failureMessage = other.failureMessage
        do {
            let fresh = try ServeGrammarCache.compile(vocabulary: other.vocabulary, format: other.format)
            for token in other.history { _ = try fresh.commitToken(token) }
            constraint = fresh
        } catch {
            // Unreachable in practice (the same grammar accepted these tokens once); if it happens the copy must not run
            // unconstrained, so it only admits the stop token.
            constraint = other.constraint
            failed = true
            failureMessage = "the grammar could not be copied"
        }
    }

    // MARK: - Outcome (read by the backend after the stream ends)

    /// The grammar accepted the stop token: the reply is a complete instance.
    var isTerminated: Bool { lock.lock(); defer { lock.unlock() }; return terminated }
    /// Something went wrong inside the grammar (a commit the matcher refused, an MLX read that failed). Metadata only.
    var failure: String? { lock.lock(); defer { lock.unlock() }; return failureMessage }
    /// Tokens the matcher accepted (the stop token included) — what `ServeGrammarCache.release` rolls back.
    var committedTokens: Int { lock.lock(); defer { lock.unlock() }; return history.count }

    // MARK: - LogitProcessor

    func prompt(_ prompt: MLXArray) {}      // the grammar constrains the reply, not the prompt (so the prompt cache needs no bypass)

    func process(logits: MLXArray) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        let width = logits.dim(-1)
        var out = logits
        if whitespace.isActive, !terminated, !failed {
            out = out + Self.padded(vocabulary.whitespaceBias, to: width, with: 0).asType(logits.dtype)
        }
        if let mask = maskLocked(width: width) { out = out + mask.asType(logits.dtype) }
        return out
    }

    func didSample(token: MLXArray) {
        lock.lock()
        defer { lock.unlock() }
        guard !terminated, !failed else { return }
        do {
            let id = Int(try MLXErrorGuard.item(token, as: Int32.self))
            _ = whitespace.record(tokenID: id)
            let result = try constraint.commitToken(Int32(id))
            history.append(Int32(id))
            if result.isTerminated {
                terminated = true
                pending = nil
            } else {
                pending = try constraint.computeMask()
            }
        } catch {
            failed = true
            failureMessage = String(describing: type(of: error)) + ": " + ((error as? LocalizedError)?.errorDescription ?? "\(error)")
        }
    }

    func copy() -> Self { Self(copying: self) }

    // MARK: - The mask

    private func maskLocked(width: Int) -> MLXArray? {
        if terminated || failed {
            return Self.onlyAllowed(vocabulary.eosTokenID, width: width)
        }
        if pending == nil {
            do { pending = try constraint.computeMask() } catch {
                failed = true
                failureMessage = String(describing: type(of: error)) + ": " + ((error as? LocalizedError)?.errorDescription ?? "\(error)")
                return Self.onlyAllowed(vocabulary.eosTokenID, width: width)
            }
        }
        guard let mask = pending, mask.needsApply else { return nil }
        return Self.additiveMask(words: mask.mask, vocabularyCount: vocabulary.vocabularyCount, width: width)
    }

    /// xgrammar's packed bitmask (LSB-first: bit `i` of word `w` is token `w·32 + i`) as an additive mask over `width` logits:
    /// `0` where the grammar allows the token, `-inf` where it doesn't. Tokens past the tokenizer's vocabulary (a model's
    /// padded embedding rows) are `-inf`, as in the library's own loop. Unpacked on the device, not with a per-bit Swift loop
    /// over ~150–260 k entries.
    static func additiveMask(words: [Int32], vocabularyCount: Int, width: Int) -> MLXArray {
        let usable = min(vocabularyCount, width, words.count * 32)
        guard usable > 0 else { return MLXArray([Float](repeating: -Float.infinity, count: width)) }
        let packed = MLXArray(words).expandedDimensions(axis: 1)                     // [W, 1]
        let shifts = MLXArray((0 ..< 32).map { Int32($0) })                          // [32]
        let bits = ((packed >> shifts) & 1).reshaped(-1)                             // [W·32]; an arithmetic shift is fine, `& 1` follows
        let allowed = bits[0 ..< usable] .== 1
        let head = which(allowed, Float(0), -Float.infinity)
        return padded(head, to: width, with: -Float.infinity)
    }

    /// `-inf` everywhere except `id`.
    static func onlyAllowed(_ id: Int, width: Int) -> MLXArray {
        var values = [Float](repeating: -Float.infinity, count: width)
        if id >= 0, id < width { values[id] = 0 }
        return MLXArray(values)
    }

    static func padded(_ values: [Float], to width: Int, with fill: Float) -> MLXArray {
        let clipped = Array(values.prefix(width))
        return MLXArray(clipped + [Float](repeating: fill, count: max(0, width - clipped.count)))
    }

    static func padded(_ head: MLXArray, to width: Int, with fill: Float) -> MLXArray {
        let have = head.dim(0)
        if have == width { return head }
        if have > width { return head[0 ..< width] }
        return concatenated([head, MLXArray([Float](repeating: fill, count: width - have))])
    }
}
