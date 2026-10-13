import Foundation

// SC-3 — the serving numbers (gate E, ruled 2026-10-07), aggregated per model over the in-memory request log.
// A pure function of the entries: nothing is stored, and there is no text to leak (Serve rule 13).

nonisolated struct ModelServeStats: Sendable, Equatable {
    var requestCount: Int
    /// Requests that ended `503` (busy, waiting for memory, too large…).
    var unavailableCount: Int
    /// Median time to first token over the requests that produced one; nil when none did.
    var p50TTFTSeconds: Double?
    /// Median decode rate over the requests that have one; nil when none do.
    var p50DecodeTokensPerSecond: Double?
    /// Prompt tokens reused from the prompt cache ÷ prompt tokens, over the requests the cache applied to (0…1);
    /// nil when it applied to none.
    var promptTokensReusedShare: Double?
    var lastRequestTime: Date
}

nonisolated enum ServeStats {
    /// Per model id. A line with no model (a refusal before one was named) belongs to no model and is skipped.
    static func summary(entries: [RequestLogEntry]) -> [String: ModelServeStats] {
        var grouped: [String: [RequestLogEntry]] = [:]
        for entry in entries {
            guard let model = entry.model else { continue }
            grouped[model, default: []].append(entry)
        }
        return grouped.mapValues { lines in
            let reusing = lines.filter { $0.promptTokensReused != nil && $0.promptTokens != nil }
            let promptTotal = reusing.reduce(0) { $0 + ($1.promptTokens ?? 0) }
            let reusedTotal = reusing.reduce(0) { $0 + ($1.promptTokensReused ?? 0) }
            return ModelServeStats(
                requestCount: lines.count,
                unavailableCount: lines.filter { $0.status == 503 }.count,
                p50TTFTSeconds: median(lines.compactMap(\.ttftSeconds)),
                p50DecodeTokensPerSecond: median(lines.compactMap(\.decodeTokensPerSecond)),
                promptTokensReusedShare: promptTotal > 0 ? Double(reusedTotal) / Double(promptTotal) : nil,
                lastRequestTime: lines.map(\.time).max() ?? .distantPast)
        }
    }

    /// The middle value; the mean of the two middle ones for an even count; nil for none.
    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
}
