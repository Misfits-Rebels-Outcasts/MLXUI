import Foundation

// SC-4b — the Local Server screen's view model (gates A, E; F20 "Serve just serves"). A pure value built from
// `LocalServerController`: the view renders it and does nothing else, so every row, state line, memory figure and stat
// is testable without a window. Plain facts only — no ratings, badges or gating (R51/R56).

/// Bytes the budget and pool ledgers report right now (not estimates), before they are split for display.
nonisolated struct MemoryLedgerSample: Sendable, Equatable {
    /// Everything booked in `MemoryBudget` (weights, engines, prompt-cache snapshots).
    var used: Int64
    /// Everything resident in `ModelContainerPool` (served and not served).
    var poolWeights: Int64
    /// Prompt-cache snapshots (`PromptCacheStore.totalBytes`).
    var snapshots: Int64
}

/// The memory bar: four slices against the budget's capacity.
nonisolated struct ServeMemoryReading: Sendable, Equatable {
    var capacity: Int64
    /// Served models that are in memory and kept ready (pinned).
    var servedKeptReady: Int64
    /// Every other model resident in the pool (chat, a Run window, a served model that is not kept ready).
    var otherModels: Int64
    /// What the budget holds beyond the pool's weights and the snapshots: flow and Run-UI engines.
    var engines: Int64
    var snapshots: Int64

    var used: Int64 { servedKeptReady + otherModels + engines + snapshots }
}

enum ServeDisk {
    /// Under this much free space on the Data volume, serving can end in a `low-swap` kill (R47, journal 2026-439).
    static let lowSpaceBytes: Int64 = 15 * 1_073_741_824
    static let lowSpaceWarning = "Low disk space — macOS may stop MLXUI while it's serving. Free up space."

    /// Free space where the app's data and swap live (the Data volume holds the home folder).
    static func freeBytes() -> Int64? {
        let url = URL(fileURLWithPath: NSHomeDirectory())
        guard let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]),
              let free = values.volumeAvailableCapacityForImportantUsage else { return nil }
        return free
    }

    static func warning(freeBytes: Int64?) -> String? {
        guard let freeBytes, freeBytes < lowSpaceBytes else { return nil }
        return lowSpaceWarning
    }

    static func factText(freeBytes: Int64?) -> String {
        guard let freeBytes else { return "Free disk space: —" }
        return "Free disk space: \(ByteCountFormatter.string(fromByteCount: freeBytes, countStyle: .file))"
    }
}

struct ServerScreenModel: Equatable {
    struct Row: Identifiable, Equatable {
        let id: String
        let name: String
        let isAppleFoundation: Bool
        /// The catalog's RAM figure — only to keep the Serve checkbox off a model that can't fit this Mac.
        let ramGB: Double
        let sizeText: String?
        let contextText: String?
        let isServed: Bool
        /// The §UI states line; `nil` when the model is not served.
        let stateLine: String?
    }

    struct StatsRow: Identifiable, Equatable {
        let id: String
        let name: String
        let requests: Int
        let unavailable: Int
        let ttft: String
        let decode: String
        let reused: String
        let lastRequest: Date
    }

    enum RunState: Equatable { case running, stopped, failed }

    var runState: RunState
    var statusSentence: String
    /// "127.0.0.1:1212" while running.
    var address: String?
    /// The running switch ("Let other apps on this Mac use my models").
    var switchOn: Bool
    var rows: [Row]
    var stats: [StatsRow]
    var memory: ServeMemoryReading?
    var diskFact: String
    var lowDiskWarning: String?

    // MARK: Derivations (pure)

    /// §UI states. A pushed-out or not-yet-loaded model is still **served**; only the list says whether it is.
    static func stateLine(served: Bool, serverOn: Bool, state: ModelLoadState) -> String? {
        guard served else { return nil }
        guard serverOn else { return "Server is off" }
        switch state {
        case .ready: return "Ready"
        case .loading: return "Loading…"
        case .notLoaded(let reason):
            let base = "Not loaded · loads on next request"
            guard let reason, !reason.isEmpty else { return base }
            return base + " · " + reason
        case .failed: return "Couldn't load · tries again on the next request"      // the message can carry paths
        }
    }

    static func ttftText(_ seconds: Double?) -> String { seconds.map { String(format: "%.1f s", $0) } ?? "—" }
    static func decodeText(_ tokensPerSecond: Double?) -> String { tokensPerSecond.map { String(format: "%.1f tok/s", $0) } ?? "—" }
    /// `nil` = the prompt cache applied to no request, so there is no figure: "—". A miss is a real "0%".
    static func reusedText(_ share: Double?) -> String { share.map { String(format: "%.0f%%", $0 * 100) } ?? "—" }

    static func contextText(_ tokens: Int?) -> String? {
        tokens.map { "\($0.formatted()) token context" }
    }

    static func sizeText(downloadGB: Double) -> String { String(format: "%.1f GB", downloadGB) }
}

extension LocalServerController {
    /// The Server screen's model. `memory` / `freeDiskBytes` / `contextLength` are what the view fetched
    /// (async ledgers, disk, `config.json`); everything else is read here, from the controller.
    func screenModel(memory: ServeMemoryReading?, freeDiskBytes: Int64?,
                     contextLength: (String) -> Int?) -> ServerScreenModel {
        var rows: [ServerScreenModel.Row] = []
        var seen: Set<String> = []
        // Servable = installed MLX chat models (+ Apple Foundation Models below). Nothing else is ever listed.
        for entry in installedIndex().entries where entry.kind == .llm && seen.insert(entry.hfModelId).inserted {
            let served = isServing(entry.hfModelId)
            rows.append(.init(
                id: entry.hfModelId,
                name: entry.hfModelId.split(separator: "/").last.map(String.init) ?? entry.hfModelId,
                isAppleFoundation: false, ramGB: entry.ramGB,
                sizeText: ServerScreenModel.sizeText(downloadGB: entry.downloadSizeGB),
                contextText: ServerScreenModel.contextText(contextLength(entry.hfModelId)),
                isServed: served,
                stateLine: ServerScreenModel.stateLine(served: served, serverOn: enabled,
                                                       state: loadState(for: entry.hfModelId))))
        }
        rows.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
        if let readiness = appleFoundation() {
            let id = ServedModels.appleFoundationID
            let served = isServing(id)
            var line = ServerScreenModel.stateLine(served: served, serverOn: enabled, state: loadState(for: id))
            if served, readiness != .ready { line = ReadinessSentence.text(for: readiness) }
            rows.append(.init(id: id, name: "Apple Foundation Models", isAppleFoundation: true, ramGB: 0,
                              sizeText: nil, contextText: nil, isServed: served, stateLine: line))
        }

        let stats = servingStats
        let statsRows: [ServerScreenModel.StatsRow] = rows.compactMap { row in
            guard let s = stats[row.id] else { return nil }
            return .init(id: row.id, name: row.name, requests: s.requestCount, unavailable: s.unavailableCount,
                         ttft: ServerScreenModel.ttftText(s.p50TTFTSeconds),
                         decode: ServerScreenModel.decodeText(s.p50DecodeTokensPerSecond),
                         reused: ServerScreenModel.reusedText(s.promptTokensReusedShare),
                         lastRequest: s.lastRequestTime)
        }

        let runState: ServerScreenModel.RunState
        var address: String?
        switch server.status {
        case .running(let port): runState = .running; address = "127.0.0.1:\(port)"
        case .failed: runState = .failed
        case .starting, .stopped: runState = .stopped
        }
        return ServerScreenModel(
            runState: runState, statusSentence: statusSentence, address: address, switchOn: enabled,
            rows: rows, stats: statsRows, memory: memory,
            diskFact: ServeDisk.factText(freeBytes: freeDiskBytes),
            lowDiskWarning: ServeDisk.warning(freeBytes: freeDiskBytes))
    }

    /// The memory bar from the ledgers: kept-ready served weights come from the pool's own per-model bookings; the
    /// rest of the pool is "other models"; what the budget holds beyond pool + snapshots is the flow / Run engines.
    func memoryReading() async -> ServeMemoryReading {
        let sample = await ledger()
        var kept: Int64 = 0
        if let pool = preloadPool {
            for id in reachableIDs where id != ServedModels.appleFoundationID && loadState(for: id) == .ready {
                kept += await pool.residentBytes(directory: ModelStore.shared.directory(forHFModelID: id)) ?? 0
            }
        }
        kept = min(kept, sample.poolWeights)
        return ServeMemoryReading(
            capacity: budgetBytes, servedKeptReady: kept,
            otherModels: sample.poolWeights - kept,
            engines: max(0, sample.used - sample.poolWeights - sample.snapshots),
            snapshots: sample.snapshots)
    }
}

extension LocalServerPolicy {
    /// SC-4b: the sidebar can't stay on the Local Server screen once serving is switched off — land on Home.
    static func resolvedSidebarItem(_ item: SidebarItem, serving enabled: Bool) -> SidebarItem {
        if case .server = item, !enabled { return .home }
        return item
    }
}
