import Foundation

/// MT1-2 — what the app can do for a Model Tool right now (`RSI/plan-chat-model-tools-2026-11.md` §3.5).
/// The `ModelEntry` inside is for the app layer (install card, chip, row text); a router only ever gets `routerSuffix`.
nonisolated enum ModelToolStatus: Sendable, Equatable {
    /// An installed, claimable, fitting model will serve the tool.
    case ready(ModelEntry)
    /// The best fitting model is not installed; `gb` is its download size.
    case needsDownload(ModelEntry, gb: Double)
    /// Candidates exist but none fits this Mac. Shown dimmed with the reason (rule 7); never offered.
    case cantRun(reason: String)
    /// No candidate in this build, or the spec runs no model (`list_capabilities`, `read_asset`).
    case none

    /// The plan §3.5 text appended to a tool description. `nil` = the tool is not offered with a status line
    /// (`cantRun`, `none`; `list_capabilities` reports `cantRun` itself). Never contains a model id or name.
    var routerSuffix: String? {
        switch self {
        case .ready: return "[ready]"
        case .needsDownload(_, let gb): return "[needs a one-time \(Self.format(gb)) GB download]"
        case .cantRun, .none: return nil
        }
    }

    private static func format(_ gb: Double) -> String {
        gb < 10 ? String(format: "%.1f", max(gb, 0.1)) : String(format: "%.0f", gb)
    }
}

/// Pure and synchronous: the same tables flows use (`TaskModels`, `FlowPreflight`), no I/O.
nonisolated enum CapabilityResolver {

    /// The status of `spec`.
    ///
    /// Rule (owner-approved F6 fix): walk the task's derived pool in preference order and take the **first installed, fitting**
    /// model; only when none is installed take the first fitting download. `TaskModels.defaultModel` is deliberately not used
    /// (it ignores what is installed). The derived pool already excludes models the registry does not claim and those
    /// `ModelSupport` blocks, and only cataloged models are candidates (a system or provider slot is not a Model Tool).
    /// A spec with no task is `.none`: it is not a capability that needs a model.
    static func status(for spec: ModelToolSpec, catalog: [ModelEntry], installedModelIDs: Set<String>,
                       claimableModelIDs: Set<String>, totalRAMGB: Double) -> ModelToolStatus {
        guard let task = spec.task else { return .none }
        let candidates = orderedCandidates(task: task, kinds: spec.kinds, catalog: catalog, claimableModelIDs: claimableModelIDs)
        guard !candidates.isEmpty else { return .none }
        let fitting = candidates.filter { fits($0, totalRAMGB: totalRAMGB) }
        guard !fitting.isEmpty else {
            let smallest = candidates.map(\.ramGB).min() ?? 0
            return .cantRun(reason: String(format: "Needs about %.1f GB of memory, and this Mac has %.1f GB.", smallest, totalRAMGB))
        }
        if let installed = fitting.first(where: { installedModelIDs.contains($0.id) }) { return .ready(installed) }
        guard let download = fitting.first else { return .none }
        return .needsDownload(download, gb: download.downloadSizeGB)
    }

    /// The task's cataloged candidates in preference order: `TaskModels.models(forTask:)` names first (the order
    /// `defaultModel` uses), then the rest of the derived pool in its own order; narrowed by the spec's `kinds`.
    static func orderedCandidates(task: String, kinds: Set<RunnerKind>?, catalog: [ModelEntry],
                                  claimableModelIDs: Set<String>) -> [ModelEntry] {
        let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimableModelIDs)
            .compactMap(\.modelEntry)
            .filter { kinds == nil || kinds?.contains($0.runnerKind) == true }
        var ordered: [ModelEntry] = []
        var seen = Set<String>()
        for display in TaskModels.models(forTask: task) {
            for entry in derived where TaskModels.displayName(for: entry) == display && seen.insert(entry.id).inserted {
                ordered.append(entry)
            }
        }
        for entry in derived where seen.insert(entry.id).inserted { ordered.append(entry) }
        return ordered
    }

    /// The rule `FlowPreflight` uses (`fitsRAM`: the row's `ramGB` must be <= this Mac's RAM), reused through a one-row result.
    static func fits(_ model: ModelEntry, totalRAMGB: Double) -> Bool {
        let need = FlowPreflight.ModelNeed(task: "", display: model.displayName, model: model, installed: false,
                                           equivalence: nil, blockingReason: nil)
        return FlowPreflight.fitsRAM(FlowPreflight.Result(needs: [need]), totalRAMGB: totalRAMGB)
    }
}
