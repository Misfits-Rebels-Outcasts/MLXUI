import Foundation
import MLX
import MLXLMCommon

// PC-2 — where to checkpoint (pure) and how to prefill to a checkpoint (MLX).

/// The probe method (PC-0 item 4, ruled E1): render the conversation **plus a probe assistant turn
/// and a probe user turn** and take the longest common token prefix with the real prompt — that is
/// where the next turn's prompt will still agree (*T*). Render `[system…, probe user]` for the
/// shared system prefix (*S*), which is what separate sessions of one harness have in common.
nonisolated enum PromptCheckpoints {
    // Text that cannot appear in a real conversation by accident; only its first tokens matter.
    static let probeAssistant = "\u{2063}PCPROBE-assistant-\u{2063}"
    static let probeUser = "\u{2063}PCPROBE-user-\u{2063}"

    /// The probe conversations, or `nil` where a boundary doesn't exist.
    static func probeMessages(for messages: [ServeMessage]) -> (turn: [ServeMessage], system: [ServeMessage]?) {
        let turn = messages + [ServeMessage(role: .assistant, text: probeAssistant),
                               ServeMessage(role: .user, text: probeUser)]
        let leading = Array(messages.prefix { $0.role == .system })
        let system = leading.isEmpty ? nil : leading + [ServeMessage(role: .user, text: probeUser)]
        return (turn, system)
    }

    /// The positions to checkpoint, ascending: the system boundary *S* (when it is before *T*) and
    /// the turn boundary *T*, each clamped to `n − 1` (a tail token must remain, T6) and dropped
    /// when below `minimumLength` (a snapshot the store would never serve).
    static func positions(actual: [Int], turnProbe: [Int]?, systemProbe: [Int]?, minimumLength: Int) -> [Int] {
        let n = actual.count
        func common(_ probe: [Int]?) -> Int? {
            guard let probe else { return nil }
            var i = 0
            while i < n, i < probe.count, actual[i] == probe[i] { i += 1 }
            return min(i, n - 1)
        }
        var out: [Int] = []
        if let s = common(systemProbe), s >= minimumLength { out.append(s) }
        if let t = common(turnProbe), t >= minimumLength, !out.contains(t) { out.append(t) }
        return out.sorted()
    }
}

/// The result of preparing a request's KV cache.
nonisolated struct PromptPlan<Cache>: @unchecked Sendable {
    /// `nil` = no cache work happened: generate from the full input exactly as without the cache.
    let cache: Cache?
    /// The tokens still to process (`tokens[prefilled...]`); the whole prompt when `cache == nil`.
    let tailTokens: [Int]
    var outcome: PromptCacheOutcome
}

typealias PreparedPrompt = PromptPlan<[KVCache]>

nonisolated enum PromptCheckpointing {
    /// The whole algorithm, with the model's three operations passed in so it can be pinned without
    /// a model (PC-3 carry-over 2): what is **reused + prefilled + handed to `generateTokens` as the
    /// tail** is the prompt exactly — no gap, no overlap, and the tail is never empty.
    ///
    /// - `restore`: turns a hit into a cache holding exactly `hit.reusedLength` tokens (copy, then
    ///   trim), or `nil` if that can't be done — a miss, never a wrong cache.
    /// - `prefill(from, to, cache)`: feeds `tokens[from..<to]` and evaluates; throws on cancellation.
    /// - `snapshot(cache)`: an evaluated copy, its bytes, and whether it can be trimmed.
    /// - `render`: renders a probe conversation **the same way the real prompt was rendered** (T5).
    static func plan<Cache, Snapshot: Sendable>(
        store: PromptCacheStore<Snapshot>,
        key: PromptCacheModelKey,
        messages: [ServeMessage],
        tokens: [Int],
        render: ([ServeMessage]) async throws -> [Int],
        newCache: () -> Cache,
        isTrimmable: (Cache) -> Bool,
        restore: (PromptCacheHit<Snapshot>) -> Cache?,
        prefill: (_ from: Int, _ to: Int, _ cache: Cache) throws -> Void,
        snapshot: (Cache) -> (snapshot: Snapshot, bytes: Int64, trimmable: Bool)
    ) async throws -> PromptPlan<Cache> {
        var outcome = PromptCacheOutcome(promptTokens: tokens.count)
        let n = tokens.count
        guard n >= 2 else { return PromptPlan(cache: nil, tailTokens: tokens, outcome: outcome) }

        // 1. lookup → a copy made on the generating side, trimmed if needed.
        var working: Cache?
        var position = 0
        var mark = ContinuousClock.now
        func lap() -> Double {
            let now = ContinuousClock.now
            defer { mark = now }
            let d = now - mark
            return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
        }
        let hit = await store.lookup(key: key, prompt: tokens)
        outcome.timings.lookup = lap()
        if let hit, let restored = restore(hit) {
            working = restored
            position = hit.reusedLength
            outcome.reusedTokens = hit.reusedLength
            outcome.hitStoredLength = hit.storedLength
        }
        outcome.timings.restore = lap()

        // 2. where to checkpoint (probe renders; a failed probe just means no checkpoint there)
        let probes = PromptCheckpoints.probeMessages(for: messages)
        let turnProbe = try? await render(probes.turn)
        let systemProbe: [Int]? = if let system = probes.system { try? await render(system) } else { nil }
        outcome.timings.probes = lap()
        let wanted = PromptCheckpoints.positions(actual: tokens, turnProbe: turnProbe, systemProbe: systemProbe,
                                                 minimumLength: store.minimumReuse).filter { $0 > position }

        // 3. prefill segment by segment; store each snapshot only after its segment is evaluated.
        if !wanted.isEmpty, working == nil { working = newCache() }
        for target in wanted {
            guard let cache = working else { break }
            _ = lap()
            try prefill(position, target, cache)
            position = target
            outcome.timings.prefill += lap()
            // PC-3b: a trimmable cache stores only the **last** checkpoint (T). T's snapshot covers
            // the system boundary S by trimming (`lookup` matches a trimmable snapshot by common
            // prefix), so a second snapshot of the same prefix is a near-duplicate that costs
            // memory and a copy. Non-trimmable caches (hybrids, sliding windows past their window)
            // can't be trimmed back, so they keep both.
            if target != wanted.last, isTrimmable(cache) { continue }
            let taken = snapshot(cache)
            let stored = await store.insert(key: key, tokens: Array(tokens[0..<target]), snapshot: taken.snapshot,
                                            bytes: taken.bytes, trimmable: taken.trimmable)
            if stored { outcome.storedAt.append(target) }
            outcome.timings.snapshot += lap()
        }

        guard let cache = working, position > 0 else {
            return PromptPlan(cache: nil, tailTokens: tokens, outcome: outcome)
        }
        return PromptPlan(cache: cache, tailTokens: Array(tokens[position...]), outcome: outcome)
    }

    /// The real model's version: copy and trim a hit, prefill through the model, snapshot as a
    /// `KVSnapshot`. Cancellation throws before the unfinished segment is stored.
    static func prepare(
        store: PromptCacheStore<KVSnapshot>,
        key: PromptCacheModelKey,
        messages: [ServeMessage],
        tokens: [Int],
        context: ModelContext,
        parameters: GenerateParameters,
        render: ([ServeMessage]) async throws -> [Int]
    ) async throws -> PreparedPrompt {
        try await plan(
            store: store, key: key, messages: messages, tokens: tokens, render: render,
            newCache: { context.model.newCache(parameters: parameters) },
            isTrimmable: { canTrimPromptCache($0) },
            restore: { hit in
                let copy = hit.makeCopy()           // made here, inside the caller's `perform` (PC-1 carry-over 3)
                let trimmed = hit.trimBy == 0 ? 0 : trimPromptCache(copy.caches, numTokens: hit.trimBy)
                guard trimmed == hit.trimBy else { return nil }
                eval(copy.caches)                   // PC-3b: evaluate here so "copy+eval" is measurable, not hidden in the first step
                return copy.caches
            },
            prefill: { from, to, cache in
                try prefill(tokens, from: from, to: to, cache: cache, context: context,
                            step: parameters.prefillStepSize)
            },
            snapshot: { cache in
                let snapshot = KVSnapshot(caches: cache.map { $0.copy() })     // evaluated by prefill (T3)
                return (snapshot, snapshot.bytes, canTrimPromptCache(snapshot.caches))
            })
    }

    /// Mirrors `LLMModel.prepare` (F7): chunks through the model, `eval(cache)` after each, with a
    /// cancellation check between chunks.
    static func prefill(_ tokens: [Int], from: Int, to: Int, cache: [KVCache], context: ModelContext,
                        step: Int) throws {
        var i = from
        while i < to {
            try Task.checkCancellation()
            let j = min(i + max(step, 1), to)
            let chunk = MLXArray(Array(tokens[i..<j]))[.newAxis, 0...]
            _ = context.model(LMInput.Text(tokens: chunk), cache: cache, state: nil)
            eval(cache)
            i = j
        }
    }
}
