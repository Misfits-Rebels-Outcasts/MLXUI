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
nonisolated struct PreparedPrompt: @unchecked Sendable {
    /// `nil` = no cache work happened: generate from the full input exactly as without the cache.
    let cache: [KVCache]?
    /// The tokens still to process (`tokens[prefilled...]`); the whole prompt when `cache == nil`.
    let tailTokens: [Int]
    var outcome: PromptCacheOutcome
}

nonisolated enum PromptCheckpointing {
    /// Look the prompt up, copy and trim the hit (inside the caller's `perform`), prefill to the
    /// checkpoints, insert a snapshot after each finished segment, and hand back the cache plus the
    /// tail. Cancellation throws before the unfinished segment is stored.
    ///
    /// - Parameter render: renders a conversation the **same way the real prompt was rendered**
    ///   (T5) and returns its token ids; a throw means "no checkpoint from that probe".
    static func prepare(
        store: PromptCacheStore<KVSnapshot>,
        key: PromptCacheModelKey,
        messages: [ServeMessage],
        tokens: [Int],
        context: ModelContext,
        parameters: GenerateParameters,
        render: ([ServeMessage]) async throws -> [Int]
    ) async throws -> PreparedPrompt {
        var outcome = PromptCacheOutcome(promptTokens: tokens.count)
        let n = tokens.count
        guard n >= 2 else { return PreparedPrompt(cache: nil, tailTokens: tokens, outcome: outcome) }

        // 1. lookup → a copy made here, on the generating side (carry-over 3), trimmed if needed.
        var working: [KVCache]?
        var position = 0
        if let hit = await store.lookup(key: key, prompt: tokens) {
            let snapshot = hit.makeCopy()
            let trimmed = hit.trimBy == 0 ? 0 : trimPromptCache(snapshot.caches, numTokens: hit.trimBy)
            if trimmed == hit.trimBy {
                working = snapshot.caches
                position = hit.reusedLength
                outcome.reusedTokens = hit.reusedLength
                outcome.hitStoredLength = hit.storedLength
            }   // else: a trim that didn't take is a miss, never a wrong cache
        }

        // 2. where to checkpoint (probe renders; a failed probe just means no checkpoint there)
        let probes = PromptCheckpoints.probeMessages(for: messages)
        let turnProbe = try? await render(probes.turn)
        let systemProbe: [Int]? = if let system = probes.system { try? await render(system) } else { nil }
        let minimum = await store.minimumReuse
        let wanted = PromptCheckpoints.positions(actual: tokens, turnProbe: turnProbe, systemProbe: systemProbe,
                                                 minimumLength: minimum).filter { $0 > position }

        // 3. prefill segment by segment; store each snapshot only after its segment is evaluated.
        if !wanted.isEmpty, working == nil { working = context.model.newCache(parameters: parameters) }
        for target in wanted {
            guard let cache = working else { break }
            try prefill(tokens, from: position, to: target, cache: cache, context: context,
                        step: parameters.prefillStepSize)
            position = target
            let snapshot = KVSnapshot(caches: cache.map { $0.copy() })     // evaluated above (T3)
            let stored = await store.insert(key: key, tokens: Array(tokens[0..<target]), snapshot: snapshot,
                                            bytes: snapshot.bytes, trimmable: canTrimPromptCache(snapshot.caches))
            if stored { outcome.storedAt.append(target) }
        }

        guard let cache = working, position > 0 else {
            return PreparedPrompt(cache: nil, tailTokens: tokens, outcome: outcome)
        }
        return PreparedPrompt(cache: cache, tailTokens: Array(tokens[position...]), outcome: outcome)
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
