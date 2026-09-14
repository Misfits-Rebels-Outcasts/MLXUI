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
        /// The installable catalog entry, when the resolved slot is `.cataloged` — `slot
        /// .modelEntry`, kept as its own field so existing readers (`RealExecutor`'s cache
        /// key, `FlowRunSession.substitutionNotes`) don't need to unwrap a slot themselves.
        let model: ModelEntry?
        /// Already installed on disk?
        let installed: Bool
        /// The bridge equivalence (`.same`/`.requantized`/`.substitute`/`.sameFamily`).
        let equivalence: Equivalence?
        /// Why this model can't run at all, when `model == nil` and it isn't merely waiting
        /// on setup (see `setupReason` below).
        let blockingReason: String?
        /// MS-3 — non-nil when the resolved slot's `readiness` is `.needsSetup`: a fixable
        /// gap (a key not yet pasted in, a toggle not yet flipped), not a hard block. Always
        /// `nil` today — the system/provider registries are empty until Phase AFM/RM.
        let setupReason: String?
        /// MS-3 — the fix-it action attached to `blockingReason`/`setupReason`, when one
        /// exists (MS-4 renders it as a trailing button). Always `nil` today.
        let setupAction: SetupAction?

        init(task: String, display: String, model: ModelEntry?, installed: Bool,
             equivalence: Equivalence?, blockingReason: String?,
             setupReason: String? = nil, setupAction: SetupAction? = nil) {
            self.task = task
            self.display = display
            self.model = model
            self.installed = installed
            self.equivalence = equivalence
            self.blockingReason = blockingReason
            self.setupReason = setupReason
            self.setupAction = setupAction
        }
    }

    /// FIX-2 — a non-`.model` row (net/instant/agent/staged) whose `TaskAvailability` verdict
    /// is `.needsSetup` (e.g. a `Web Search` row with no Tavily/Brave key yet). Advisory only:
    /// it never feeds `blocked`/`needsSetup` below — those buckets' semantics are
    /// model-row-specific (§2 of `RSI/DelegateFixItBacklog.md`), and `FlowRunner
    /// .rowClassRefusal` deliberately keeps a `.needsSetup` net row selectable and runnable.
    struct RowAdvisory: Sendable, Equatable {
        let task: String
        let reason: String
        let action: SetupAction?
    }

    /// The buckets + the flow-wide verdict.
    struct Result: Sendable, Equatable {
        var needs: [ModelNeed] = []
        /// FIX-2 — every non-`.model` row's setup advisory, in row order. Never blocks a run
        /// (`isBlocked` below reads only `blocked`, unchanged) — `setupAdvisory(_:)` is the
        /// banner's own read of this.
        var rowAdvisories: [RowAdvisory] = []
        var installed: [ModelNeed] { needs.filter { $0.installed } }
        var toDownload: [ModelNeed] { needs.filter { !$0.installed && $0.model != nil } }
        /// MS-3 — rows whose slot needs a setup step (a key, a toggle) rather than a
        /// download. Empty until a `.needsSetup` slot exists (AFM's `.system` is the first).
        var needsSetup: [ModelNeed] { needs.filter { $0.setupReason != nil } }
        /// AFM-1 caught this one: `model == nil` used to mean "blocked" by construction,
        /// because only `.cataloged` could ever be `.ready`/`.needsDownload`, and both of
        /// those always carried a `model`. A `.ready` `.system`/`.provider` slot breaks that
        /// — it's genuinely fine (`installed == true`) with no `ModelEntry` at all — so
        /// `blocked` must also exclude anything already `installed`, not just anything with
        /// a `setupReason`.
        var blocked: [ModelNeed] { needs.filter { !$0.installed && $0.model == nil && $0.setupReason == nil } }

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

        /// True when every model row resolved to an installable (or already-setup) model.
        var isBlocked: Bool { !blocked.isEmpty }
    }

    /// Run the preflight. `catalog` is the flat `browser.json` entries; `installedModelIDs`
    /// the set of catalog ids already on disk; `totalRAMGB` this machine's RAM.
    ///
    /// MS-3: reads each resolution's `ModelSlot.readiness`, not a hand-rolled
    /// `installed`/`ramGB` check — `.ready`/`.needsDownload` land in `installed`/`toDownload`
    /// exactly as before (the only slot kind today, `.cataloged`, never answers anything
    /// else), `.needsSetup` is the new `needsSetup` bucket, and `.unavailable` blocks the run
    /// with its own reason, same as an unresolvable display name always has.
    static func run(
        _ doc: FlowDocument,
        catalog: [ModelEntry],
        installedModelIDs: Set<String>,
        totalRAMGB: Double,
        claimableModelIDs: Set<String>,
        isAppStore: Bool = CapabilityGate.isAppStoreBuild
    ) -> Result {
        var result = Result()
        // Flatten rows (blocks aren't in the linear subset, but be safe).
        let rows = allRows(doc.rows)
        for row in rows {
            guard let desc = TaskCatalog.get(row.task ?? "") else { continue }
            guard desc.taskClass == .model else {
                // FIX-2: every non-`.model` row is asked too, not just skipped — but only as
                // an *advisory*. Feeding a `.needsSetup` net/instant row into `blocked` here
                // would make every keyless `Web Search` flow un-runnable, silently reversing
                // WS-3's own tested decision (`FlowRunner.rowClassRefusal` keeps it selectable).
                if case .needsSetup(let reason, let action) = TaskAvailability.state(
                    for: desc, isAppStore: isAppStore, catalog: catalog, claimableModelIDs: claimableModelIDs
                ) {
                    result.rowAdvisories.append(RowAdvisory(task: desc.name, reason: reason, action: action))
                }
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
            case .runnable(let slot, let equivalence, _):
                switch slot.readiness(installedModelIDs: installedModelIDs) {
                // `.ready`/`.needsDownload` already encode installedness — re-deriving it
                // from `installedModelIDs.contains(slot.modelEntry?.id ?? "")` here was
                // AFM's own trap in miniature: correct for `.cataloged` (where the two can
                // never disagree) and silently wrong for `.system`/`.provider`, whose
                // `modelEntry` is always nil, so the old line always read `installed: false`
                // even when readiness was `.ready` (caught building Phase AFM's preflight
                // test, `journal 2026-258`).
                case .ready:
                    result.needs.append(ModelNeed(
                        task: row.task ?? "?", display: display, model: slot.modelEntry,
                        installed: true, equivalence: equivalence, blockingReason: nil))
                case .needsDownload:
                    result.needs.append(ModelNeed(
                        task: row.task ?? "?", display: display, model: slot.modelEntry,
                        installed: false, equivalence: equivalence, blockingReason: nil))
                case .needsSetup(let reason, let action):
                    result.needs.append(ModelNeed(
                        task: row.task ?? "?", display: display, model: slot.modelEntry,
                        installed: false, equivalence: equivalence, blockingReason: nil,
                        setupReason: reason, setupAction: action))
                case .unavailable(let reason):
                    result.needs.append(ModelNeed(
                        task: row.task ?? "?", display: display, model: nil,
                        installed: false, equivalence: nil, blockingReason: reason))
                }
            case .notRunnable(let displayName, let reason, let action):
                result.needs.append(ModelNeed(task: row.task ?? "?", display: displayName,
                                              model: nil, installed: false, equivalence: nil,
                                              blockingReason: reason, setupAction: action))
            }
        }
        return result
    }

    /// Whether a run is blocked on RAM: the largest single row's `ramGB` must fit.
    static func fitsRAM(_ result: Result, totalRAMGB: Double) -> Bool {
        result.largestRowRAMGB <= totalRAMGB
    }

    /// The plain sentence the UI shows when the flow can't run (models missing, needing
    /// setup, or RAM).
    static func blockedReason(_ result: Result, totalRAMGB: Double) -> String? {
        if let first = result.blocked.first {
            return first.blockingReason ?? "A model in this flow can't run yet."
        }
        if let first = result.needsSetup.first {
            return first.setupReason ?? "A model in this flow needs setup before it can run."
        }
        if !fitsRAM(result, totalRAMGB: totalRAMGB) {
            return String(format: "This flow needs %.1f GB of RAM at once, but this Mac has %.1f GB.",
                          result.largestRowRAMGB, totalRAMGB)
        }
        return nil
    }

    /// MS-4 — the fix-it action paired with `blockedReason`, when one exists (always `nil`
    /// today; Phase AFM/RM/WS are the first to produce one).
    static func blockedAction(_ result: Result) -> SetupAction? {
        if let first = result.blocked.first { return first.setupAction }
        if let first = result.needsSetup.first { return first.setupAction }
        return nil
    }

    /// FIX-2 — the non-blocking advisory banner's content, or nil: the first row advisory
    /// (row order), never consulted by `blockedReason`/`blockedAction` above, which keep
    /// their model-row-only semantics exactly (§2 of `RSI/DelegateFixItBacklog.md`).
    static func setupAdvisory(_ result: Result) -> RowAdvisory? {
        result.rowAdvisories.first
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
