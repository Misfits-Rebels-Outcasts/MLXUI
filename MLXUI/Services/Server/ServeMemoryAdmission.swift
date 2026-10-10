import Foundation

// S3-5a — memory admission before prefill (R47). INV-1 (journal 2026-439) showed one request with a
// ~35-40k-token prompt takes Qwen3-8B to ~13 GB on an 18 GiB Mac and then swaps for minutes (and, with the
// Data volume full, was killed by the kernel). "No request may be able to kill the app": the server
// predicts what a request needs and refuses — in the dialect's own error shape — what the machine cannot give.

/// What the admission rule needs to know about one model on this Mac. Built by the MLX backend; `nil`
/// (Apple Foundation Models, test doubles, a config that doesn't say enough) means "exempt".
nonisolated struct ServeMemoryProfile: Sendable, Equatable {
    /// The weights' bytes on disk (`*.safetensors`).
    var weightsBytes: Int64
    /// State that does not grow with the prompt (a hybrid model's linear-attention layers).
    var fixedBytes: Int64
    /// Bytes one more token costs: the KV bytes per token × `ServeMemoryAdmission.overheadFactor`.
    var bytesPerToken: Double
    /// What this model can occupy: the shared `MemoryBudget`'s capacity (see `ServeMemoryAdmission.room`).
    var roomBytes: Int64

    /// `need(prompt, reply) = weights + fixed + (prompt + reply) × bytesPerToken`.
    func needBytes(promptTokens: Int, replyTokens: Int) -> Int64 {
        weightsBytes + fixedBytes + Int64((Double(max(0, promptTokens) + max(0, replyTokens)) * bytesPerToken).rounded(.up))
    }

    /// The most tokens (prompt + reply) the room holds beyond the weights.
    var tokenCapacity: Int {
        let free = roomBytes - weightsBytes - fixedBytes
        return free > 0 && bytesPerToken > 0 ? Int((Double(free) / bytesPerToken).rounded(.down)) : 0
    }

    /// The largest prompt `ServeMemoryAdmission.decide` admits: the one that still leaves `minimumReplyTokens`.
    var largestAdmittedPromptTokens: Int {
        max(0, tokenCapacity - ServeMemoryAdmission.minimumReplyTokens)
    }
}

nonisolated enum ServeMemoryAdmission {

    // MARK: - Constants (calibrated; journal 2026-440)

    /// The multiplier from the raw KV bytes per token to what a token really costs during prefill.
    ///
    /// **Calibrated against Qwen3-8B on an 18 GiB Mac, but not by dividing a footprint by tokens**: peak
    /// footprint is not linear in tokens (journals `2026-440`, `2026-442`). What *is* measured is where a
    /// request stops being usable. With MLX's buffer pool capped (S3-5c) and doomed snapshots skipped: 20k
    /// tokens finish in 80 s, 25k in 105 s, 30k in 136 s with the system's swap unchanged, and 39.5k in 396 s
    /// with swap growing. **1.75** (≈ 258 KB/token) admits prompts up to ~29.4k tokens for Qwen3-8B on this Mac
    /// (weights ~4.6 GB, room 11.6 GiB): every admitted run under ~2.3 minutes and out of swap. 2.0 (S3-5a's
    /// value, set before the pool was capped) admitted ≤ ~25.6k. Owner ruling R50 (2026-10-10) accepted 1.75.
    static let overheadFactor = 1.75

    /// The smallest reply the server will admit a prompt for (R47): a prompt that leaves less than this is refused.
    static let minimumReplyTokens = 1024

    /// Bytes per token used **only** when a prompt cannot be counted. Measured on captured Claude Code
    /// bodies: 4.1-4.4 UTF-8 bytes per token (167,823 B ↔ 39,492 tokens; 187 KB ↔ 45,804), English prose is
    /// similar; code, CJK and base64-like text run lower (~2-3). 3 is therefore conservative for the
    /// typical body (it over-counts tokens by ~1.4×) and still under-counts the worst cases a little.
    static let conservativeBytesPerToken = 3

    /// The dtype width of the KV cache this server uses (fp16; nothing here enables a quantized cache).
    static let kvElementBytes = 2

    // MARK: - The estimator

    /// What the checkpoint's `config.json` says the KV cache costs: bytes **per token** (attention layers
    /// that keep a full KV: `2 × layers × kv_heads × head_dim × dtype`) and a **fixed** part (a hybrid's
    /// linear-attention state). `nil` when the config doesn't declare enough.
    ///
    /// This is the only KV parser: it began as a copy of S2-4's `AgentReadiness.kvBytes`, which R51 removed along
    /// with the Agent-ready badge.
    static func kvCost(configData: Data?) -> (perToken: Int, fixed: Int)? {
        guard let data = configData,
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        let text = (root["text_config"] as? [String: Any]) ?? root          // wrapper checkpoints
        func number(_ key: String) -> Int? {
            ((text[key] as? NSNumber) ?? (root[key] as? NSNumber)).map(\.intValue)
        }
        guard let layers = number("num_hidden_layers"), layers > 0 else { return nil }
        guard let headDim = number("head_dim")
                ?? number("hidden_size").flatMap({ hidden in number("num_attention_heads").map { hidden / $0 } }),
              headDim > 0 else { return nil }
        guard let kvHeads = number("num_key_value_heads") ?? number("num_attention_heads"), kvHeads > 0 else {
            return nil
        }
        // Only a **linear-attention** layer holds a fixed state instead of a growing cache; a sliding layer
        // still holds K and V (counted in full: an upper bound), and an unknown type counts as attention.
        let types = (text["layer_types"] as? [String]) ?? (root["layer_types"] as? [String])
        let linearLayers = min(types.map { $0.filter { $0 == "linear_attention" }.count } ?? 0, layers)
        let attentionLayers = layers - linearLayers
        return (attentionLayers * kvHeads * headDim * 2 * kvElementBytes, linearLayers * linearStateBytes(number: number))
    }

    /// The fixed state one linear-attention layer holds (the shapes `mlx-swift-lm` allocates; see
    /// S2-4's original comments for the sources).
    private static func linearStateBytes(number: (String) -> Int?) -> Int {
        let valueHeads = number("linear_num_value_heads") ?? 0
        let keyHeads = number("linear_num_key_heads") ?? 0
        let keyHeadDim = number("linear_key_head_dim") ?? 0
        let valueHeadDim = number("linear_value_head_dim") ?? 0
        guard valueHeads > 0, keyHeadDim > 0, valueHeadDim > 0 else { return 0 }
        let recurrent = valueHeads * valueHeadDim * keyHeadDim * kvElementBytes
        let convDim = 2 * keyHeads * keyHeadDim + valueHeads * valueHeadDim
        let conv = max(0, (number("linear_conv_kernel_dim") ?? 0) - 1) * convDim * kvElementBytes
        return recurrent + conv
    }

    /// The profile for one model, or `nil` (exempt) when the weights or the config can't be read.
    static func profile(configData: Data?, weightsBytes: Int64, roomBytes: Int64) -> ServeMemoryProfile? {
        guard weightsBytes > 0, let kv = kvCost(configData: configData), kv.perToken > 0 else { return nil }
        return ServeMemoryProfile(weightsBytes: weightsBytes, fixedBytes: Int64(kv.fixed),
                                  bytesPerToken: Double(kv.perToken) * overheadFactor, roomBytes: roomBytes)
    }

    /// **Room.** What this model can occupy is the shared `MemoryBudget`'s whole capacity
    /// (`max(4 GB, RAM × 0.6)`, the figure the pool and `EngineCache` already reserve against). Why the whole
    /// capacity and not "capacity − used": a generation runs alone (global cap 1), so every other resident
    /// (idle weights in the pool or `EngineCache`, prompt-cache snapshots) is evictable for it: `reserve`
    /// evicts snapshots first, then LRU weights, and snapshots are lowest priority. The ledger's only
    /// *un*evictable entries are loads in flight, which `reserve` waits for rather than overshoots, and
    /// the ledger does not say whose they are, so subtracting them would charge a model's own first load
    /// against itself. The ledger also does not see KV or prefill buffers at all, which is why this module
    /// has its own `need`.
    static func room(budget: MemoryBudget = .shared) -> Int64 { budget.capacityBytes }

    // MARK: - The rule

    enum Decision: Equatable, Sendable {
        /// Admitted; the reply may be at most this many tokens (always ≥ `minimumReplyTokens`).
        case admit(maxReplyTokens: Int)
        case refuse(needBytes: Int64, roomBytes: Int64)
    }

    /// 1. `need(prompt, minimumReply) > room` → refuse. 2. Otherwise the reply is capped at
    /// `floor((room − weights − fixed) / bytesPerToken) − prompt` (only ever a ceiling for the caller to `min`).
    static func decide(profile: ServeMemoryProfile, promptTokens: Int) -> Decision {
        let minimum = profile.needBytes(promptTokens: promptTokens, replyTokens: minimumReplyTokens)
        guard minimum <= profile.roomBytes else { return .refuse(needBytes: minimum, roomBytes: profile.roomBytes) }
        return .admit(maxReplyTokens: max(minimumReplyTokens, profile.tokenCapacity - promptTokens))
    }

    /// **The usable context (R53)** — what the Connect sheet tells a client app it may send:
    /// `min(the model's context_length, the largest prompt memory admission accepts on this Mac)`, so the client
    /// compacts before the server refuses. A model whose memory isn't the limit (small weights, a big room), or
    /// one with no profile (Apple Foundation Models, an unreadable config), gets its own `context_length`;
    /// `nil` only when neither fact is known. A plain number, not a verdict.
    static func usableTokens(profile: ServeMemoryProfile?, contextLength: Int?) -> Int? {
        switch (profile?.largestAdmittedPromptTokens, contextLength) {
        case let (memory?, window?): return min(memory, window)
        case let (memory?, nil): return memory
        case let (nil, window): return window
        }
    }

    /// A prompt's token count when the render could not count it: UTF-8 bytes ÷ 3, rounded up. For admission only.
    static func estimatedTokens(utf8Bytes: Int) -> Int {
        (max(0, utf8Bytes) + conservativeBytesPerToken - 1) / conservativeBytesPerToken
    }

    /// The `400` — `ServeError` is dialect-neutral; each dialect's tracker renders it in its own shape.
    static func refusal(needBytes: Int64, roomBytes: Int64) -> ServeError {
        ServeError(status: 400,
                   message: "This prompt needs about \(gigabytes(needBytes)) GB of memory, and this Mac can give the model "
                       + "about \(gigabytes(roomBytes)) GB. Use a shorter prompt (for example, fewer tools) or a smaller model.",
                   code: "context_too_large_for_memory")
    }

    private static func gigabytes(_ bytes: Int64) -> String { String(format: "%.1f", Double(bytes) / 1_073_741_824) }
}
