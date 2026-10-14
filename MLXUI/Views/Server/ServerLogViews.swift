import SwiftUI

// SC-4b — moved from Settings → Local Server (it now lives on the Local Server screen). Unchanged otherwise.

/// The last 200 requests, newest first — time, status, model, tokens, duration, caller, and
/// "tools ignored". **Metadata only**: the log has no field for content (rule 13).
struct RequestLogList: View {
    let log: RequestLog

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            let entries = Array(log.entries.reversed())
            Group {
                if entries.isEmpty {
                    Text("No requests yet.").foregroundStyle(.secondary)
                } else {
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 4) {
                            ForEach(Array(entries.enumerated()), id: \.offset) { _, entry in
                                Text(Self.line(entry))
                                    .font(.system(.caption, design: .monospaced))
                                    .foregroundStyle(entry.status >= 400 ? Color.red : Color.primary)
                                    .textSelection(.enabled)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(height: 160)
                }
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    static func line(_ entry: RequestLogEntry) -> String {
        var parts = [entry.time.formatted(date: .omitted, time: .standard), String(entry.status),
                     entry.model ?? "—"]
        if let prompt = entry.promptTokens, let completion = entry.completionTokens {
            parts.append("\(prompt)→\(completion) tok")
        }
        parts.append(String(format: "%.1fs", entry.durationSeconds))
        if let agent = entry.userAgent { parts.append(agent) }
        if entry.toolsIgnored { parts.append("tools ignored") }
        // S2-3: the flag only. The thinking text has nowhere to go — the entry has no field for it.
        if entry.reasoningOn { parts.append("thinking") }
        // S2-2: how many tool calls the model made — a count only, never a name, an argument or a
        // result (rule 13). This line is the request log's only consumer.
        if entry.toolCalls > 0 { parts.append(entry.toolCalls == 1 ? "1 tool call" : "\(entry.toolCalls) tool calls") }
        // PC-3: how much of the prompt came from the cache — a count — and, when the cache path
        // failed and the request ran without it, a marker, so a broken cache can't hide as "no hit".
        if entry.cachedTokens > 0 { parts.append("reused \(entry.cachedTokens)") }
        if entry.cacheFellBack { parts.append("cache fell back") }
        return parts.joined(separator: "  ")
    }
}

/// PC-3: "Prompt cache: N snapshots, X GB". Refreshes while it is on screen; counts and bytes only.
struct PromptCacheLine: View {
    @State private var snapshots = 0
    @State private var bytes: Int64 = 0
    @State private var fallbacks = 0

    var body: some View {
        Text(PromptCacheSummary.line(snapshots: snapshots, bytes: bytes, fallbacks: fallbacks))
            .font(.callout)
            .foregroundStyle(fallbacks > 0 ? Color.orange : Color.secondary)
            .task {
                while !Task.isCancelled {
                    snapshots = await PromptCacheShared.store.count
                    bytes = await PromptCacheShared.store.totalBytes
                    fallbacks = PromptCacheCounters.shared.snapshot.fallbacks
                    try? await Task.sleep(for: .seconds(2))
                }
            }
    }
}
