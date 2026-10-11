import Foundation

/// S3.1b — trade a generation's unused **reply room** for a prompt-cache snapshot, without ever reserving more.
///
/// S3-8B reserves `weights + (prompt + reply) × KV-per-token` for a generation, and Claude Code always asks for a
/// 32,000-token reply (~1.8 GB on Qwen3.5-9B) it almost never uses. Above ~60k tokens that unused room is what leaves a
/// ~2 GB snapshot with nowhere to go (journal `2026-453`, Finding 1). When the snapshot store would refuse a snapshot
/// *only* for lack of room, the request gives back exactly the missing bytes — and the same number of reply tokens:
///
/// 1. **Reserved bytes never go up.** The request's reply reservation is only lowered (`MemoryBudget.shrink`), by
///    `ceil(shortfall ÷ bytesPerToken) × bytesPerToken ≥ shortfall`; snapshot + reservations then fit the capacity.
/// 2. **`maxTokens` only goes down**, by that many tokens; a reply that reaches the lowered limit ends with the
///    dialect's normal length stop (the backend compares against the lowered limit).
/// 3. **The floor:** the trade happens only if the reply limit stays ≥ `minimumReplyTokens` (8,192); otherwise the
///    snapshot is skipped exactly as before.
/// 4. Weights still win (snapshots stay first to be evicted — PC rule 8), and S3-5a admission, `overheadFactor`, the
///    buffer-cache cap and S3-8B's evict-before-prefill are untouched: this only acts after them.
///
/// Decided at the snapshot point, inside prefill, so the clamp is known before generation starts.
nonisolated final class ReplyTrade: @unchecked Sendable {
    static let minimumReplyTokens = 8_192

    private let lock = NSLock()
    private var limit: Int
    private var traded = 0
    private let bytesPerToken: Double
    private let shrink: @Sendable (Int64) -> Bool

    /// - Parameters:
    ///   - replyLimit: the request's effective `maxTokens` as reserved.
    ///   - bytesPerToken: `ServeMemoryProfile.bytesPerToken` (KV bytes × `overheadFactor`, S3-5a's conversion).
    ///   - shrink: lowers the request's reply reservation by that many bytes; `false` = it could not.
    init(replyLimit: Int, bytesPerToken: Double, shrink: @escaping @Sendable (Int64) -> Bool) {
        self.limit = replyLimit
        self.bytesPerToken = bytesPerToken
        self.shrink = shrink
    }

    /// The reply limit now (after any trades).
    var replyLimit: Int { lock.lock(); defer { lock.unlock() }; return limit }
    /// Reply tokens given up so far.
    var tradedTokens: Int { lock.lock(); defer { lock.unlock() }; return traded }

    /// Give back at least `shortfallBytes`. `true` = done (the reservation and the limit are both lowered);
    /// `false` = nothing changed.
    func trade(shortfallBytes: Int64) -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard shortfallBytes > 0, bytesPerToken > 0 else { return false }
        let tokens = Int((Double(shortfallBytes) / bytesPerToken).rounded(.up))
        guard tokens > 0, limit - tokens >= Self.minimumReplyTokens else { return false }
        let bytes = Int64((Double(tokens) * bytesPerToken).rounded(.up))
        guard shrink(bytes) else { return false }
        limit -= tokens
        traded += tokens
        return true
    }
}
