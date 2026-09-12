import Foundation

/// CFM-R12-4 — the one availability truth: a task is offerable only when it can actually run
/// in this build. Generalizes R11-0b's "seedable ⟺ runnable, one list" rule to the whole
/// catalog, so the step picker never offers a task the runtime will die on three rows in.
///
/// `supportedInstantTools` is the **single** list — it must match `RealExecutor.runInstant`'s
/// `case`s exactly. The cross-check test (`CatFlowTaskAvailabilityTests`) enforces that they
/// cannot disagree by driving the executor for every catalog task; porting a tool to
/// `RealExecutor` is the only edit needed to make it available everywhere.
nonisolated enum TaskAvailability {

    /// One task's verdict.
    enum State: Equatable {
        case available
        /// An instant tool with no implementation yet (the picker's "needs newer support").
        case needsNewerSupport
        /// `net` / `staged` / App-Store `agent` — this channel refuses, with the reason.
        case refusedByChannel(reason: String)
        /// WS-3 — the task genuinely runs in this build, but a fixable step (a key) stands
        /// between here and running it. Distinct from `.refusedByChannel` (which names a
        /// dead end — nothing the user does in this app changes it) and from `.available`
        /// (which would be a lie: the row can't actually run yet). Mirrors `Readiness
        /// .needsSetup` — same shape, same reason this codebase already keeps the two
        /// separate for model rows (MS-3).
        case needsSetup(reason: String, action: SetupAction?)
    }

    /// The instant tools with a real `case` in `RealExecutor.runInstant` (46 today — the
    /// catalog has zero unported instant tools since R13-3 ported Join Video).
    static let supportedInstantTools: Set<String> = [
        "Read Audio", "Read Text", "Read Image", "Read Images", "Read Files", "Read PDF",
        "Read Index", "Read Context", "Read Video",
        "Save Audio", "Save Text", "Save Image", "Save Images", "Save Video", "Save Context",
        "Store Index", "Retrieve", "Keyword Search",
        "Count Context", "Calculate", "Compare", "Range", "Chart",
        "Resize", "Crop", "Convert", "Watermark", "Overlay Text", "Contact Sheet",
        "Extract Frame", "Extract Audio", "Trim", "Mux", "Detect Edges", "Detect Pose",
        "Join Video",
        "Split", "Filter", "Dedupe", "Sort", "Extract", "Count", "Join Text", "Template",
        "Read CSV", "Read JSON", "Query Table", "Set Field", "Table to Text", "Append Row", "Merge Record",
        "Store Query", "Store Read", "Store Write", "Diff",
    ]

    /// CFM-R12-9 (approved scope): the networked tools with a real `URLSession`
    /// implementation. `Web Search` stayed unported ("no provider to name in App
    /// Review") until **backlog `RSI/DelegateOffMachineBacklog.md` §0 ruling 2
    /// superseded that, 2026-09-11** — Tavily and Brave are exactly the providers that
    /// ruling said didn't exist. Phase WS (journal `2026-09-12`) ports it; its
    /// availability is still special-cased below (`.needsSetup` when no key is set),
    /// not a bare membership check like the other four.
    static let supportedNetTools: Set<String> = ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File", "Web Search"]

    /// A task's verdict in this build.
    /// - Parameters:
    ///   - catalog: the flat `browser.json` entries. Only the `.model` branch consults it, but
    ///     it is a **required** argument (CFM-R14-FIX-3): the derived pool is the availability
    ///     authority, and a silently-empty catalog would report every model task as unavailable
    ///     — a wrong answer that looks correct. `FlowRunner` passes explicit empties because it
    ///     only ever asks about `.instant`/`.net` tasks, which never read the catalog.
    ///   - claimableModelIDs: the catalog ids the registry can claim (CFM-R14-2).
    static func state(for task: TaskDescriptor,
                      isAppStore: Bool = CapabilityGate.isAppStoreBuild,
                      catalog: [ModelEntry],
                      claimableModelIDs: Set<String>) -> State {
        switch task.taskClass {
        case .instant:
            return supportedInstantTools.contains(task.name) ? .available : .needsNewerSupport
        case .model:
            // CFM-R14-2 + CFM-R14-FIX-2: a model task is available exactly when its **derived**
            // pool (the registry's own claim answer + corrected runnerKind + the executor
            // genuinely serving the task) is non-empty. Segment was pending the owner's
            // CFM-R13-6 ruling; it was **ruled option (1) 2026-08-27** and the executor now
            // serves `engines.diffusion.segment` (CFM-R15-1), so `SAM3` derives and Segment is
            // available. The latent tasks still have no stage accepting what they hand it, and
            // `Rerank`/`Upscale`/`Estimate Depth` have no catalog model at all. Their pools are
            // empty for these different reasons; the picker's single "needs newer support"
            // marker is the honest surface for all of them.
            return TaskModels.derivedModels(for: task.name, catalog: catalog,
                                            claimableModelIDs: claimableModelIDs).isEmpty
                ? .needsNewerSupport : .available
        case .human, .trigger:
            return .available
        case .agent:
            return isAppStore
                ? .refusedByChannel(reason: "runs only in the direct build")
                : .available
        case .net:
            // WS-3: `Web Search` is structurally ported (in `supportedNetTools`, and
            // `RealExecutor` genuinely dispatches it) but needs a Tavily or Brave key to
            // actually run — `.needsSetup`, never `.available` with nothing behind it and
            // never `.refusedByChannel` (which would say this app can never run it, a lie).
            if task.name == "Web Search" {
                if WebSearchProvider.resolve(explicit: nil) != nil { return .available }
                return .needsSetup(reason: "Add a Tavily or Brave key in Settings",
                                   action: .openSettings(.providers))
            }
            return supportedNetTools.contains(task.name)
                ? .available
                : .refusedByChannel(reason: "no provider for this network tool is ported")
        case .staged:
            // CFM-R12-8: Stage Send / Stage Post queue a visible outbox entry (never send).
            return .available
        }
    }

    /// Whether a task (by name) can run in this build. `catalog` + `claimableModelIDs` are
    /// required (CFM-R14-FIX-3): the derived pool is the availability authority, and a model
    /// task asked about without them would get a silently-wrong answer.
    static func isAvailable(_ taskName: String,
                            isAppStore: Bool = CapabilityGate.isAppStoreBuild,
                            catalog: [ModelEntry],
                            claimableModelIDs: Set<String>) -> Bool {
        guard let desc = TaskCatalog.get(taskName) else { return false }
        if case .available = state(for: desc, isAppStore: isAppStore,
                                   catalog: catalog, claimableModelIDs: claimableModelIDs) { return true }
        return false
    }

    /// The one-line marker the picker shows for an unavailable task (or nil when available).
    /// `catalog` + `claimableModelIDs` are required (CFM-R14-FIX-3), same as `state`.
    static func marker(for task: TaskDescriptor,
                       isAppStore: Bool = CapabilityGate.isAppStoreBuild,
                       catalog: [ModelEntry],
                       claimableModelIDs: Set<String>) -> String? {
        switch state(for: task, isAppStore: isAppStore,
                     catalog: catalog, claimableModelIDs: claimableModelIDs) {
        case .available: return nil
        case .needsNewerSupport: return "needs newer support"
        case .refusedByChannel(let reason): return "refused — \(reason)"
        case .needsSetup(let reason, _): return reason
        }
    }
}
