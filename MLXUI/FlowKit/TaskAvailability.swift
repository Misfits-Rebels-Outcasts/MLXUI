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

    /// The instant tools with a real `case` in `RealExecutor.runInstant` (28 today).
    static let supportedInstantTools: Set<String> = [
        "Read Audio", "Read Text", "Read Image", "Read Images", "Read Files", "Read PDF",
        "Save Audio", "Save Text", "Save Image", "Save Images", "Save Video",
        "Split", "Filter", "Dedupe", "Sort", "Extract", "Count", "Join Text", "Template",
        "Read CSV", "Read JSON", "Query Table", "Set Field", "Table to Text",
        "Store Query", "Store Read", "Store Write", "Diff",
    ]

    /// A task's verdict in this build.
    static func state(for task: TaskDescriptor,
                      isAppStore: Bool = CapabilityGate.isAppStoreBuild) -> State {
        switch task.taskClass {
        case .instant:
            return supportedInstantTools.contains(task.name) ? .available : .needsNewerSupport
        case .model, .human, .trigger:
            return .available
        case .agent:
            return isAppStore
                ? .refusedByChannel(reason: "runs only in the direct build")
                : .available
        case .net:
            return .refusedByChannel(reason: "no networked tool is ported in this version")
        case .staged:
            return .refusedByChannel(reason: "this version can't complete staged rows")
        }
    }

    /// Whether a task (by name) can run in this build.
    static func isAvailable(_ taskName: String,
                            isAppStore: Bool = CapabilityGate.isAppStoreBuild) -> Bool {
        guard let desc = TaskCatalog.get(taskName) else { return false }
        if case .available = state(for: desc, isAppStore: isAppStore) { return true }
        return false
    }

    /// The one-line marker the picker shows for an unavailable task (or nil when available).
    static func marker(for task: TaskDescriptor,
                       isAppStore: Bool = CapabilityGate.isAppStoreBuild) -> String? {
        switch state(for: task, isAppStore: isAppStore) {
        case .available: return nil
        case .needsNewerSupport: return "needs newer support"
        case .refusedByChannel(let reason): return "refused — \(reason)"
        }
    }
}
