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
}

nonisolated enum ServeMemoryAdmission {

    // MARK: - Constants (calibrated; journal 2026-440)

    /// The multiplier from the raw KV bytes per token to what a token really costs during prefill.
    ///
    /// **Calibrated against Qwen3-8B on an 18 GiB Mac, but not by dividing a footprint by tokens**: the
    /// measured peak footprint is *not linear* in tokens: it rises at ~1 MB/token and then saturates at the
    /// working-set ceiling (≈ 13.8 GB) from about 10k tokens on (2k → +2.0 GB, 10k → +9.0 GB, 20k → +9.0 GB,
    /// 30k → +9.0 GB, INV-1's 39.5k → +8.5 GB), because MLX keeps freed buffers. What *is* measured is where
    /// the request stops being usable: 20k tokens finished in 91 s, 30k in 519 s, 39.5k not in 300 s. The
    /// factor puts the admit boundary between 20k and 30k on this Mac. 2.0 × 144 KiB ≈ 295 KB/token (INV-1's
    /// plateau figure, 215 KB/token, would admit the 519 s case). Implementer's call, pending owner
    /// confirmation.
    static let overheadFactor = 2.0

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
    /// Mirrors `AgentReadiness.kvBytes` (S2-4), which holds the owner's uncommitted edits and was not
    /// touched; `ServeMemoryAdmissionTests` pins the two to the same number for the same config.
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
    /// `AgentReadiness.kvBytes` for the sources).
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

    /// Criterion (d) of Agent-ready, asked of the same estimator (R47): does a 32,768-token context fit?
    /// **Not wired** — doing so edits `AgentReadiness.swift`, which holds the owner's uncommitted changes.
    static func fitsAgentContext(_ profile: ServeMemoryProfile, contextTokens: Int = 32_768) -> Bool {
        profile.needBytes(promptTokens: contextTokens, replyTokens: 0) <= profile.roomBytes
    }

    private static func gigabytes(_ bytes: Int64) -> String { String(format: "%.1f", Double(bytes) / 1_073_741_824) }
}
