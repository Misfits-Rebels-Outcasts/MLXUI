import Foundation

/// **Is this model Agent-ready?** (S2-4.) One pure function over facts, in one place, beside
/// `ToolCallFormatTable` — the badge, the Connect sheet's line and the tests all ask this, so they
/// cannot disagree.
///
/// A model is Agent-ready when **all** of these hold:
///
/// | | Criterion | Where it comes from |
/// |---|---|---|
/// | a | its family's template renders tools | `ToolCallFormatTable.acceptsTools` (S2-1) |
/// | b | its context window is ≥ 32k | `ModelContextWindow` (S2-6) |
/// | c | a **recorded** SV-6 pass | `verified` — ships empty; a later commit adds rows |
/// | d | 32k fits this Mac's memory | `kvBytes` + the catalog's `ramGB` ≤ the shared budget |
///
/// (d) is the planner's recommendation, **owner decision pending**; (c) is deliberately a table no
/// code path can infer from — "it would probably pass" is not a pass.
nonisolated enum AgentReadiness {

    /// What "agent-ready" means by context length: 32k (S2-4's criterion b, and the window the
    /// memory estimate is computed for).
    static let requiredContextTokens = 32_768

    /// The four states, in the order the criteria are checked.
    enum State: Equatable, Sendable {
        /// a + b + c + d.
        case agentReady
        /// a + b pass, but no SV-6 row yet. The honest state for a model nobody has verified.
        case eligibleNotVerified
        /// a + b + c pass, (d) fails: it works, but a 32k context may not fit this Mac.
        case mayNotFit
        /// a or b fails. `reason` is plain English and is what the UI shows.
        case notAgentReady(reason: String)
    }

    /// A **recorded** SV-6 pass: an HF repo id, the date it passed, and the journal that says so.
    struct Verification: Equatable, Sendable {
        var hfModelID: String
        var date: String
        var journal: String
    }

    /// The bundled SV-6 table. **Ships empty** (2026-10-07): no SV-6 has been run yet, and a pass is
    /// never inferred from a criterion — only the owner running the harness makes a row.
    static let verified: [String: Verification] = [:]

    // MARK: - The criteria

    /// Criteria a…d over facts only. `verification` is "is there an SV-6 row for this id" — the
    /// caller owns the table lookup, so this stays pure.
    static func state(modelType: String, contextLength: Int?, catalogRAMGB: Double,
                      configData: Data?, budgetBytes: Int64, verification: Verification? = nil) -> State {
        // (a) without tool support there is no agent, whatever else is true.
        guard ToolCallFormatTable.acceptsTools(forModelType: modelType) else {
            return .notAgentReady(reason: "This model's template has no tool support.")
        }
        // (b) the window, from the config when the caller didn't already read it.
        guard let limit = contextLength ?? ModelContextWindow.limit(configData: configData),
              limit >= requiredContextTokens else {
            let declared = (contextLength ?? ModelContextWindow.limit(configData: configData))
                .map { "\($0) tokens" } ?? "unknown"
            return .notAgentReady(reason: "Its context window is \(declared); agent use needs "
                                   + "\(requiredContextTokens).")
        }
        // (c) a recorded pass, never an inference.
        guard verification != nil else { return .eligibleNotVerified }
        // (d) the planner's recommendation: does 32k fit this Mac? Fail **open** — an estimate we
        // can't make is not evidence that the model won't fit.
        guard budgetBytes > 0, let kv = kvBytes(configData: configData), kv > 0 else { return .agentReady }
        let kvBytes = Int64(kv)
        let modelBytes = Int64((catalogRAMGB * 1_073_741_824).rounded())
        return modelBytes + kvBytes <= budgetBytes ? .agentReady : .mayNotFit
    }

    /// Criterion (d): the KV cache a **32k** context needs, from the checkpoint's own config.
    ///
    /// * Attention layers: `layers × kv_heads × head_dim × 2 (K and V) × 2 bytes × tokens`.
    /// * Hybrid `qwen3_5`: the **linear-attention** layers are subtracted from that — each holds a
    ///   fixed state instead, whose two parts are the shapes
    ///   `mlx-swift-lm` actually allocates — the recurrent state `[B, v_heads, v_head_dim, k_head_dim]`
    ///   (`GatedDelta.swift:290`) and the convolution state
    ///   `[B, conv_kernel − 1, 2 × k_heads × k_head_dim + v_heads × v_head_dim]` (`Qwen35.swift:245`).
    ///
    /// `nil` when the config doesn't declare enough to estimate.
    ///
    /// Known conservatisms (implementer's call, pending owner confirmation): 2 bytes per element
    /// assumes an fp16 cache, which is what this server uses — nothing here enables a quantized KV
    /// cache; sliding-window layers (`gemma2`, `gemma3_text`, `gpt_oss`) are counted at the full 32k
    /// rather than at their window, so the estimate is an **upper bound** for those families — and a
    /// `layer_types` value we don't recognise is treated as a plain attention layer, never as zero;
    /// expert weights are already inside the catalog's `ramGB`.
    static func kvBytes(configData: Data?, contextTokens: Int = requiredContextTokens) -> Int? {
        guard let data = configData,
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return nil }
        // The wrapper checkpoints keep the text model's dimensions under `text_config`.
        let text = (root["text_config"] as? [String: Any]) ?? root
        func number(_ key: String) -> Int? {
            ((text[key] as? NSNumber) ?? (root[key] as? NSNumber)).map(\.intValue)
        }

        guard let layers = number("num_hidden_layers"), layers > 0 else { return nil }
        // `head_dim` when the checkpoint states it, else the usual hidden/heads ratio.
        guard let headDim = number("head_dim")
                ?? number("hidden_size").flatMap({ hidden in number("num_attention_heads").map { hidden / $0 } }),
              headDim > 0 else { return nil }
        guard let kvHeads = number("num_key_value_heads") ?? number("num_attention_heads"), kvHeads > 0 else {
            return nil
        }

        // Only a **linear-attention** layer holds a fixed state instead of a growing cache. A
        // sliding-attention layer (gpt-oss's `sliding_attention`) still holds K and V, and is counted
        // at the full `contextTokens` — an upper bound, since its window is shorter.
        let layerTypes = (text["layer_types"] as? [String]) ?? (root["layer_types"] as? [String])
        let linearLayers = min(layerTypes.map { types in types.filter { $0 == "linear_attention" }.count } ?? 0,
                               layers)
        let attentionLayers = layers - linearLayers

        let attention = attentionLayers * kvHeads * headDim * 2 * 2 * contextTokens
        let linear = linearLayers * linearStateBytes(number: number)
        return attention + linear
    }

    /// The fixed state one linear-attention layer holds, in bytes (fp16).
    private static func linearStateBytes(number: (String) -> Int?) -> Int {
        let valueHeads = number("linear_num_value_heads") ?? 0
        let keyHeads = number("linear_num_key_heads") ?? 0
        let keyHeadDim = number("linear_key_head_dim") ?? 0
        let valueHeadDim = number("linear_value_head_dim") ?? 0
        guard valueHeads > 0, keyHeadDim > 0, valueHeadDim > 0 else { return 0 }
        let recurrent = valueHeads * valueHeadDim * keyHeadDim * 2
        // `convDim = 2 × key channels + value channels` and the cache keeps `kernel − 1` rows.
        let convDim = 2 * keyHeads * keyHeadDim + valueHeads * valueHeadDim
        let conv = max(0, (number("linear_conv_kernel_dim") ?? 0) - 1) * convDim * 2
        return recurrent + conv
    }

    // MARK: - What the user sees

    /// The badge's short label. Copy is implementer's call, pending owner confirmation.
    static func badge(for state: State) -> String {
        switch state {
        case .agentReady: "Agent-ready"
        case .eligibleNotVerified: "Eligible"
        case .mayNotFit: "May not fit"
        case .notAgentReady: "Not agent-ready"
        }
    }

    /// The badge's tooltip / the Connect sheet's one line — **one short line**, and for
    /// `notAgentReady` it *is* the reason, so the user is told why in plain English.
    static func line(for state: State) -> String {
        switch state {
        case .agentReady:
            "Agent-ready: tools work, the context is 32k or more, and it fits this Mac's memory."
        case .eligibleNotVerified:
            "Tools and context are fine; not yet verified with a tool-calling harness (SV-6)."
        case .mayNotFit:
            "Tools and context are fine, but a 32k context may not fit this Mac's memory."
        case .notAgentReady(let reason):
            reason
        }
    }

    /// Rule 14/R27: the badge and the sheet's line are Local Server surfaces, so they exist only
    /// when the server does. Same three facts as the Serve toggle (S1-5's
    /// `ServeToggleVisibility.shouldShow`), asked here so the badge can be tested on its own.
    static func isVisible(gateAvailable: Bool, runnerKind: RunnerKind, isInstalled: Bool) -> Bool {
        gateAvailable && runnerKind == .llm && isInstalled
    }
}
