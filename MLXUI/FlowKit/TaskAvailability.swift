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

    /// CFM-R12-9 (approved scope): the networked tools with a real `URLSession` GET
    /// implementation. `Web Search` stays unported — no provider to name in App Review.
    static let supportedNetTools: Set<String> = ["Web Fetch", "HTTP Get", "Fetch Feed", "Download File"]

    /// A task's verdict in this build.
    /// - Parameters:
    ///   - catalog: the flat `browser.json` entries. Only the `.model` branch needs it —
    ///     instant/net/staged/agent verdicts are catalog-free, so `FlowRunner` (which never
    ///     asks about model tasks) passes nothing.
    ///   - claimableModelIDs: the catalog ids the registry can claim (CFM-R14-2).
    static func state(for task: TaskDescriptor,
                      isAppStore: Bool = CapabilityGate.isAppStoreBuild,
                      catalog: [ModelEntry] = [],
                      claimableModelIDs: Set<String> = []) -> State {
        switch task.taskClass {
        case .instant:
            return supportedInstantTools.contains(task.name) ? .available : .needsNewerSupport
        case .model:
            // CFM-R14-2: a model task is available exactly when its **derived** pool (the
            // registry's own claim answer + corrected runnerKind) is non-empty. Segment,
            // Upscale, Generate Image, … stay marked — their derived pool is empty because no
            // claimable catalog model serves them (or, for the hidden flags, is still served).
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
            return supportedNetTools.contains(task.name)
                ? .available
                : .refusedByChannel(reason: "no provider for this network tool is ported")
        case .staged:
            // CFM-R12-8: Stage Send / Stage Post queue a visible outbox entry (never send).
            return .available
        }
    }

    /// Whether a task (by name) can run in this build. Model tasks need `catalog` +
    /// `claimableModelIDs` (the derived pool is the availability authority, CFM-R14-2).
    static func isAvailable(_ taskName: String,
                            isAppStore: Bool = CapabilityGate.isAppStoreBuild,
                            catalog: [ModelEntry] = [],
                            claimableModelIDs: Set<String> = []) -> Bool {
        guard let desc = TaskCatalog.get(taskName) else { return false }
        if case .available = state(for: desc, isAppStore: isAppStore,
                                   catalog: catalog, claimableModelIDs: claimableModelIDs) { return true }
        return false
    }

    /// The one-line marker the picker shows for an unavailable task (or nil when available).
    /// Model tasks need `catalog` + `claimableModelIDs` (the derived pool is the authority).
    static func marker(for task: TaskDescriptor,
                       isAppStore: Bool = CapabilityGate.isAppStoreBuild,
                       catalog: [ModelEntry] = [],
                       claimableModelIDs: Set<String> = []) -> String? {
        switch state(for: task, isAppStore: isAppStore,
                     catalog: catalog, claimableModelIDs: claimableModelIDs) {
        case .available: return nil
        case .needsNewerSupport: return "needs newer support"
        case .refusedByChannel(let reason): return "refused — \(reason)"
        }
    }
}
