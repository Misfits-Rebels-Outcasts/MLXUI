import Foundation

/// The "before you press Run" half of CFM-R2-7: walk a flow, resolve every model row
/// through `CatalogBridge`, and bucket the models into already-installed / to-download /
/// no-candidate. Also does the RAM preflight — checked against the **largest single row's**
/// `ramGB`, never the sum (rows run sequentially, one engine released before the next loads).
nonisolated struct FlowPreflight {

    /// One model row's resolution.
    struct ModelNeed: Sendable, Equatable {
        /// The row display name (e.g. "Transcribe").
        let task: String
        /// The `.cat` display name (e.g. "Whisper Large v3").
        let display: String
        /// The installable catalog entry, when one exists.
        let model: ModelEntry?
        /// Already installed on disk?
        let installed: Bool
        /// The bridge equivalence (`.same`/`.requantized`/`.substitute`/`.sameFamily`).
        let equivalence: Equivalence?
        /// Why this model can't run, when `model == nil`.
        let blockingReason: String?
    }

    /// The three buckets + the flow-wide verdict.
    struct Result: Sendable, Equatable {
        var needs: [ModelNeed] = []
        var installed: [ModelNeed] { needs.filter { $0.installed } }
        var toDownload: [ModelNeed] { needs.filter { !$0.installed && $0.model != nil } }
        var blocked: [ModelNeed] { needs.filter { $0.model == nil } }

        /// Models that must be installed before Run, deduplicated by catalog id.
        var downloadSet: [ModelEntry] {
            var seen = Set<String>()
            var out: [ModelEntry] = []
            for need in toDownload {
                guard let model = need.model, seen.insert(model.id).inserted else { continue }
                out.append(model)
            }
            return out
        }

        /// Total download size in GB across the to-download set.
        var totalDownloadGB: Double {
            downloadSet.reduce(0) { $0 + $1.downloadSizeGB }
        }

        /// The largest single row's RAM footprint among all model rows that will run.
        var largestRowRAMGB: Double {
            needs.compactMap { $0.model?.ramGB }.max() ?? 0
        }

        /// True when every model row resolved to an installable model.
        var isBlocked: Bool { !blocked.isEmpty }
    }

    /// Run the preflight. `catalog` is the flat `browser.json` entries; `installedModelIDs`
    /// the set of catalog ids already on disk; `totalRAMGB` this machine's RAM.
    static func run(
        _ doc: FlowDocument,
        catalog: [ModelEntry],
        installedModelIDs: Set<String>,
        totalRAMGB: Double
    ) -> Result {
        var result = Result()
        // Flatten rows (blocks aren't in the linear subset, but be safe).
        let rows = allRows(doc.rows)
        for row in rows {
            guard let desc = TaskCatalog.get(row.task ?? ""), desc.taskClass == .model else {
                continue   // instant tools don't need a model
            }
            guard let display = row.model else {
                // SPEC-Q214 (DA-9): a model-optional task (`Text to Table`) with **no** model
                // named is legitimate — its deterministic fast path needs none, and the executor
                // branches ahead of `resolveModel`. Don't block the flow. This is the third site
                // to learn the rule after the editor warning and the executor (both DA-6).
                if TaskModels.isModelOptional(row.task ?? "") { continue }
                result.needs.append(ModelNeed(task: row.task ?? "?", display: "an unnamed model",
                                              model: nil, installed: false, equivalence: nil,
                                              blockingReason: "Row \(row.task ?? "?") has no model named."))
                continue
            }
            switch CatalogBridge.resolve(display, catalog: catalog) {
            case .runnable(let model, let equivalence, _):
                result.needs.append(ModelNeed(task: row.task ?? "?", display: display,
                                              model: model, installed: installedModelIDs.contains(model.id),
                                              equivalence: equivalence, blockingReason: nil))
            case .notRunnable(let displayName, let reason):
                result.needs.append(ModelNeed(task: row.task ?? "?", display: displayName,
                                              model: nil, installed: false, equivalence: nil,
                                              blockingReason: reason))
            }
        }
        return result
    }

    /// Whether a run is blocked on RAM: the largest single row's `ramGB` must fit.
    static func fitsRAM(_ result: Result, totalRAMGB: Double) -> Bool {
        result.largestRowRAMGB <= totalRAMGB
    }

    /// The plain sentence the UI shows when the flow can't run (models missing or RAM).
    static func blockedReason(_ result: Result, totalRAMGB: Double) -> String? {
        if let first = result.blocked.first {
            return first.blockingReason ?? "A model in this flow can't run yet."
        }
        if !fitsRAM(result, totalRAMGB: totalRAMGB) {
            return String(format: "This flow needs %.1f GB of RAM at once, but this Mac has %.1f GB.",
                          result.largestRowRAMGB, totalRAMGB)
        }
        return nil
    }

    // MARK: - Helpers

    private static func allRows(_ rows: [Row]) -> [Row] {
        var out: [Row] = []
        for row in rows {
            out.append(row)
            out.append(contentsOf: allRows(row.children))
        }
        return out
    }
}
