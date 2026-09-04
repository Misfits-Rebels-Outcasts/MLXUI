import Foundation

/// Whether a flow can run under `FlowRunner.linear` (CFM-R2-6). A `FlowDocument` using any
/// feature outside the linear subset is **unrunnable** — the UI disables Run with the
/// reason sentence (standing rule 5: refuse rather than approximate).
nonisolated enum Runnability: Equatable, Sendable {
    case runnable
    case notRunnable(reason: String)
}

/// The typed event stream a run emits — the Swift analogue of catflow's event stream,
/// **never printed** (the Python's "events, not prints" isolation rule). `FlowRunner.linear`
/// produces these; R2-8's UI consumes them to drive the dots.
nonisolated enum FlowEvent: Sendable {
    case queued(rowID: UUID)
    case started(rowID: UUID)
    case progress(rowID: UUID, Double)
    case finished(rowID: UUID, Asset)
    case failed(rowID: UUID, FlowError)
    case cacheHit(rowID: UUID)
    case flagRaised(rowID: UUID, message: String)
    /// CFM-R10-Human: a `wait=forever` human row parked the run. `execPath` is the
    /// activation key a resumed answer must use (`"3@1"`).
    case parked(rowID: UUID, prompt: String, policy: String, execPath: String)
}

/// Executes one row's transform given its gathered inputs — the seam `FlowRunner` runs
/// against, mirroring `catflow-mlx/src/catflow/engines/mock.py::MockExecutor.execute` (and,
/// later, `engines/real.py`'s real dispatch). The linear runner never decides *how* a row
/// runs, only *what feeds it* and in what order events fire.
nonisolated protocol FlowExecutor: Sendable {
    /// Run `row`, whose gathered inputs are `inputs`. `path` is the row's dotted path
    /// (e.g. `"1"`, `"2"` — the Python's fingerprint identity). `transcript` is a
    /// transcript-keeping decider's (`Think`'s) accumulated "So far" entries, `context` the
    /// `· ctx` journal snapshot this activation read, `usedFlowContent` the ambient `uses:`
    /// flow's exact text — the three ingredients the cache key folds in (FIX-12), mirroring
    /// `interpreter.py:1220-1221`. Throws a `FlowError`.
    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?) async throws -> Asset

    /// Whether the last `execute` served from cache. Default `false` for plain executors;
    /// `CachingExecutor` overrides (a reference type so the mutation persists across the
    /// protocol existential). The runner reads this to emit `.cacheHit`.
    var lastCacheHit: Bool { get }

    /// The tag a decider row fired on its last `execute` (the Python's `last_tag`), or nil
    /// for a non-decider row. The interpreter routes a `DecideClause` on it. Default `nil`;
    /// `MockExecutor` sets it for decider rows.
    var lastTag: String? { get }

    /// A human row's timeout default (the Python's `last_timeout_flag`): `(code, message)`
    /// for F002, or nil. `MockExecutor` sets it for Ask Human / Human Input with a `timeout=`.
    var lastTimeoutFlag: (code: String, message: String)? { get }

    /// A staged row's queued outbox effect (the Python's `last_staged`), or nil.
    var lastStaged: (id: String, kind: String, summary: String)? { get }
}

extension FlowExecutor {
    /// The 3-arg convenience form — a row with no transcript/context/uses ambient. Provided
    /// as an extension so the interpreter always calls the full seam while existing call
    /// sites keep compiling.
    func execute(path: String, row: Row, inputs: [Asset]) async throws -> Asset {
        try await execute(path: path, row: row, inputs: inputs,
                          transcript: nil, context: nil, usedFlowContent: nil)
    }
    var lastCacheHit: Bool { false }
    var lastTag: String? { nil }
    var lastTimeoutFlag: (code: String, message: String)? { nil }
    var lastStaged: (id: String, kind: String, summary: String)? { nil }
}

/// The execution engine (CFM-R7-1): the full `FlowInterpreter` port drives every run,
/// mapped to the UUID-keyed `FlowEvent` stream the UI consumes. The interpreter handles
/// straight-line rows, blocks, clauses, budgets, deciders, composites, `uses:`, triggers,
/// and human rows; `canRun` refuses only the features no Swift engine exists for.
///
/// Stateless: per-run state lives in an explicit `RunContext` value (the documented
/// divergence from the Python's thread-locals), never in this type. A **task** is the unit
/// of execution (each run creates one); cancellation is Swift `Task` cancellation.
nonisolated struct FlowRunner {

    /// Everything the linear runner needs per run, passed explicitly rather than stored on
    /// the runner — the `RunContext` divergence (SPEC-Q): `last_frame`/`last_tag`/
    /// `last_duration_ms` thread-locals become this value, passed alongside.
    struct RunContext: Sendable {
        /// CFM-R17-1: identity vs. location, carried together. A plain flow's `scope` has
        /// `identity == locationID` and a `flows/`-rooted workspace.
        let scope: FlowScope
        let blobDirectory: URL
        /// The executor that turns rows into assets (mock in CI, real in the app).
        let executor: any FlowExecutor
        /// True when running off the main actor's registry (real dispatch); the linear
        /// runner itself never touches the registry — the executor does.
        var realism: String = "mock"

        /// Identity — the run seed source, `On Flow` matching, run records.
        var flowID: String { scope.identity }
        /// Location — the root a `.cat`'s paths resolve against.
        var workspace: FlowWorkspace { scope.workspace }

        init(scope: FlowScope, blobDirectory: URL, executor: any FlowExecutor) {
            self.scope = scope
            self.blobDirectory = blobDirectory
            self.executor = executor
        }

        /// Back-compat: a plain flow whose identity and location are the same id.
        init(flowID: String, workspace: FlowWorkspace, blobDirectory: URL,
             executor: any FlowExecutor) {
            self.init(scope: FlowScope(identity: flowID, workspace: workspace, locationID: flowID),
                      blobDirectory: blobDirectory, executor: executor)
        }
    }

    /// Whether `doc` is runnable by the full interpreter (CFM-R7-4): the interpreter handles
    /// blocks, clauses, budgets, deciders, composites, `uses:`, triggers, and human rows, so
    /// those are no longer refused. Still refused are the features no engine exists for, and
    /// the doors the App Store tier refuses outright (CFM-R10-Direct): in the **App Store**
    /// build, `code`/`improvise` header flags, `transforms:`, and `.agent` rows (Improvise)
    /// are refused; in the **Direct** build they run fenced (`sandbox-exec`).
    /// CFM-R17-1 convenience: resolve the E117 `uses:` chain against a `FlowScope` — the
    /// used `.cat` sits at `workspace/<locationID>/`, so location, not identity, is the key.
    /// CFM-R17-5: also resolves the `uses:` graph so a row that calls a used flow is
    /// recognised (rather than refused as an unknown task).
    static func canRun(_ doc: FlowDocument, scope: FlowScope) -> Runnability {
        let graph = doc.uses.isEmpty ? [:] : UsesResolver.resolve(
            doc, workspace: scope.workspace, flowID: scope.locationID, selfFile: scope.selfFile)
        return canRun(doc, flowID: scope.locationID, workspace: scope.workspace, usesGraph: graph)
    }

    static func canRun(_ doc: FlowDocument, flowID: String? = nil,
                       workspace: FlowWorkspace? = nil,
                       usesGraph: [String: FlowInterpreter.UsedFlow] = [:]) -> Runnability {
        if CapabilityGate.isAppStoreBuild {
            // E117 (CFM-R10-FIX-3): the App Store refusal must see capabilities inherited
            // through `uses:` — a used flow declaring `code`/`improvise` while the header
            // doesn't is a hole straight through the channel guarantee. `flowID`/`workspace`
            // resolve the chain; without them only the declared flags are checked.
            let flags = CapabilityGate.effectiveFlags(doc, workspace: workspace, flowID: flowID)
            if let refusal = CapabilityGate.appStoreRefusal(flags: flags) {
                return .notRunnable(reason: refusal)
            }
        }
        // CFM-R10-Direct: `transforms:` runs only in the Direct build, behind the fence.
        if !doc.transforms.isEmpty, CapabilityGate.isAppStoreBuild {
            return .notRunnable(reason:
                "This flow declares `transforms:`, which the App Store build refuses — distribute it directly instead.")
        }
        var indexByID: [UUID: Int] = [:]
        for (i, row) in doc.rows.enumerated() { indexByID[row.id] = i + 1 }

        // Check every row including nested block children — `canRun` must see the Web Search
        // inside a block, or 53-MorningBriefing would wrongly pass (B3, the same class of
        // gap the store-door metadata used to mask).
        for (n, row) in Self.enumeratedRows(doc.rows).enumerated() {
            let position = n + 1

            // FIX-2: a block row has no task by construction (`blockKind != nil`) and the
            // interpreter runs it — only a row with *neither* task nor block kind is refused.
            if row.task == nil {
                if row.blockKind != nil { continue }
                return .notRunnable(reason:
                    "Row \(position) is neither a task row nor a block, which can't be run.")
            }
            // A row naming a `transforms:` entry is a transform call (no catalog entry).
            if doc.transforms[row.task!] != nil {
                if CapabilityGate.isAppStoreBuild {
                    return .notRunnable(reason:
                        "Row \(position) calls transform '\(row.task!)', which the App Store build refuses — distribute it directly instead.")
                }
                continue
            }
            // CFM-R17-5: a row naming a `uses:` entry. The `scope` overload resolved the
            // sibling `.cat` into `usesGraph`; the interpreter expands it like a composite.
            // The entry must have resolved, and every row inside the used flow must be one
            // this version knows (recursively, through its own nested `uses:`).
            if doc.uses[row.task!] != nil {
                guard let used = usesGraph[row.task!] else {
                    return .notRunnable(reason:
                        "Row \(position) uses `\(row.task!)`, but that flow couldn't be resolved — check its `uses:` line points at a `.cat` beside it in this workspace.")
                }
                // CFM-R17-FIX-2: the same per-row class verdicts below, applied recursively
                // to the used flow's rows — an undeclared `Improvise`, an unported instant/net
                // task or a transform call inside a used flow must be refused up front, not
                // reached mid-run.
                if let bad = Self.usedFlowRefusal(used) {
                    return .notRunnable(reason: "Row \(position) uses `\(row.task!)`, which names \(bad).")
                }
                continue
            }
            switch Self.rowClassRefusal(row.task!) {
            case .unknownTask:
                return .notRunnable(reason:
                    "Row \(position) uses '\(row.task!)', which isn't a task this version of Flows knows.")
            case .appStoreDoor(let cls):
                // CFM-R10-Direct: `Improvise` runs only in the Direct build, behind the fence.
                return .notRunnable(reason:
                    "Row \(position) is an \(cls) row, which the App Store build refuses — distribute it directly instead.")
            case .unported:
                // CFM-R12-4 / R12-9: refuse an unported instant/net tool *before* the install
                // prompt, naming the task. Both verdicts are catalog-free (CFM-R14-FIX-3).
                return .notRunnable(reason:
                    "Row \(position) uses '\(row.task!)', which this version of Flows doesn't run yet.")
            case nil:
                break   // model / human / trigger / staged, and available instant/net — in scope
            }
        }
        return .runnable
    }

    /// Why this tier refuses `task` as a bare row — independent of position or the flow it
    /// sits in, so `canRun`'s top-level loop and `usedFlowRefusal` share one rule set
    /// (CFM-R17-FIX-2: two loops with different rules is what let a capability through `uses:`).
    /// `nil` = the row's class is in scope.
    enum RowClassRefusal: Equatable {
        case unknownTask
        case appStoreDoor(String)   // the task class name, e.g. "agent"
        case unported               // an instant/net task with no runner yet
    }

    static func rowClassRefusal(_ task: String) -> RowClassRefusal? {
        guard let desc = TaskCatalog.get(task) else { return .unknownTask }
        switch desc.taskClass {
        case .agent where CapabilityGate.isAppStoreBuild:
            return .appStoreDoor(desc.taskClass.rawValue)
        case .instant, .net:
            return TaskAvailability.isAvailable(task, catalog: [], claimableModelIDs: [])
                ? nil : .unported
        default:
            return nil   // model / human / trigger / staged, and `.agent` in the Direct build
        }
    }

    /// Run `doc` with the linear engine, starting at `startIndex` (0 = the whole flow; a
    /// re-run from the first gray row passes the index of the first `✓`-to-`○` boundary).
    /// `resumeOutputs` seeds rows **before** `startIndex` (their already-earned outputs), so
    /// downstream refs still resolve. The caller drains the returned stream; each row emits
    /// `.started` → (`.progress`…) → (`.cacheHit` when served from cache) → `.finished` or
    /// `.failed`, and the run halts on the first failure (no later `.started`s). A cancelled
    /// `Task` stops the stream.
    func run(_ doc: FlowDocument, context: RunContext, startIndex: Int = 0,
             resumeOutputs: [UUID: Asset] = [:],
             answers: [String: FlowInterpreter.HumanAnswer] = [:],
             occurrence: FlowInterpreter.Occurrence? = nil) -> AsyncStream<FlowEvent> {
        // The full interpreter (CFM-R7-1) drives every run now — `startIndex`/`resumeOutputs`
        // are retained for the session's resume bookkeeping, but the interpreter always runs
        // the whole flow from the top (unchanged rows replay as cache hits, which is fast).
        AsyncStream { continuation in
            let task = Task {
                let pathToID = Self.buildPathMap(doc.rows, prefix: "")
                // CFM-R10-FIX-1: no UI path can route around the refusal — the runner itself
                // consults `canRun` and refuses before starting. CFM-R17-1: pass the run's
                // scope so the App Store E117 door sees capabilities inherited through a
                // workspace-sibling `uses:` flow, not just the header's declared flags.
                if case .notRunnable(let reason) = Self.canRun(doc, scope: context.scope) {
                    if let row = pathToID["1"] {
                        continuation.yield(.failed(rowID: row,
                                                   FlowError.stageFailure(row: "1", message: reason)))
                    }
                    continuation.finish()
                    return
                }
                // CFM-R17-5: resolve the flow's `uses:` section into the graph the interpreter
                // expands. Empty for every plain flow (`doc.uses` is empty), so the conformance
                // traces are byte-for-byte unchanged.
                let usesGraph: [String: FlowInterpreter.UsedFlow] = doc.uses.isEmpty ? [:]
                    : UsesResolver.resolve(doc, workspace: context.scope.workspace,
                                           flowID: context.scope.locationID,
                                           selfFile: context.scope.selfFile)
                do {
                    // `onEvent` streams each event to the UI the moment it lands — the dots
                    // advance row-by-row while the flow is still running, instead of the whole
                    // event list arriving when the run completes (the pre-live-stream behavior).
                    let _ = try await FlowInterpreter.run(doc, executor: context.executor,
                                                          definitions: doc.definitions,
                                                          presets: doc.presets,
                                                          usesGraph: usesGraph,
                                                          occurrence: occurrence,
                                                          answers: answers,
                                                          // The GUI parks `timeout=` human rows (wait + fallback); the
                                                          // reference runtime proceeds to the default unattended.
                                                          parkOnTimeout: true) { event in
                        if let flowEvent = Self.map(event, pathToID: pathToID) {
                            continuation.yield(flowEvent)
                        }
                    }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch let error as FlowError {
                    // FIX-5: the interpreter's non-row failures (e.g. `on_budget=fail` →
                    // `budgetExceeded`) must not silently stop the run — surface a `.failed`
                    // on the row it names, then finish.
                    let row = Self.rowID(for: Self.errorRow(error), in: pathToID)
                        ?? pathToID["1"]
                    if let row {
                        continuation.yield(.failed(rowID: row, error))
                    }
                    continuation.finish()
                } catch {
                    // CFM-R8-FIX-4: a catch that discards an error without surfacing
                    // anything is how defects reach a user as "nothing happened". Surface a
                    // `.failed` with a generic sentence instead of finishing silently.
                    let row = pathToID["1"]
                    if let row {
                        continuation.yield(.failed(rowID: row,
                                                   FlowError.stageFailure(row: "1",
                                                                          message: "This run stopped unexpectedly — \(String(describing: error))")))
                    }
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Interpreter → FlowEvent mapping

    /// Every row in the document, top-level first then each block's children (depth-first)
    /// — `canRun` must see a task nested inside a block.
    private static func enumeratedRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + enumeratedRows($0.children) }
    }

    /// CFM-R17-5 / CFM-R17-FIX-2: the first row inside a resolved used flow (recursively,
    /// through its own nested `uses:` and block children) that this tier refuses — an unknown
    /// task, an App-Store-refused door (`.agent`), or an unported instant/net task — as a
    /// short phrase for the caller's refusal, or `nil` when every row runs. Applies the
    /// **same** `rowClassRefusal` rule as `canRun`'s top-level loop, so a capability can't
    /// reach a run through `uses:` that a plain flow would be refused for.
    private static func usedFlowRefusal(_ used: FlowInterpreter.UsedFlow) -> String? {
        for row in enumeratedRows(used.rows) {
            guard let task = row.task else {
                if row.blockKind != nil { continue }
                return "a row that is neither a task nor a block"
            }
            if let nested = used.nested[task] {
                if let bad = usedFlowRefusal(nested) { return bad }
                continue
            }
            if used.definitions[task] != nil { continue }
            switch rowClassRefusal(task) {
            case .unknownTask:
                return "`\(task)`, which isn't a task this version of Flows knows"
            case .appStoreDoor(let cls):
                return "`\(task)`, an \(cls) row the App Store build refuses — distribute it directly instead"
            case .unported:
                return "`\(task)`, which this version of Flows doesn't run yet"
            case nil:
                continue
            }
        }
        return nil
    }

    /// Every row's dotted exec path (the interpreter's path identity) → its row id, so the
    /// path-based `FlowInterpreter.PathEvent` stream can be turned into the UUID-keyed
    /// `FlowEvent` stream the UI consumes.
    private static func buildPathMap(_ rows: [Row], prefix: String) -> [String: UUID] {
        var map: [String: UUID] = [:]
        for (i, row) in rows.enumerated() {
            let path = "\(prefix)\(i + 1)"
            map[path] = row.id
            if !row.children.isEmpty {
                map.merge(buildPathMap(row.children, prefix: "\(path).")) { a, _ in a }
            }
        }
        return map
    }

    /// The interpreter error's row path (for `.failed`), from whichever `FlowError` carries
    /// one; empty for the row-less cases (the caller falls back to row 1).
    private static func errorRow(_ error: FlowError) -> String {
        switch error {
        case .budgetExceeded(let row, _): return row
        case .badInputCardinality(let row, _, _): return row
        case .unsupportedKind(let row, _): return row
        case .missingInlineValue(let row, _): return row
        case .fileReadFailed(let row, _): return row
        case .writeFailed(let row, _): return row
        case .frameInputOutOfRange: return ""
        case .unknownTask(let row): return row
        case .unsupportedReference(let row): return row
        case .referenceNotFound(let row): return row
        case .stageFailure(let row, _): return row
        case .modelNotRunnable(let row, _, _): return row
        case .unsupportedTask(let row, _): return row
        case .invalidSettings(let row, _, _): return row
        }
    }

    /// `"2@3"` → the base path `"2"` (an activation suffix refers to the same row id).
    private static func rowID(for path: String, in map: [String: UUID]) -> UUID? {
        let base = path.split(separator: "@").first.map(String.init) ?? path
        return map[base]
    }

    /// One interpreter event → the UUID `FlowEvent` the session consumes (nil = no UI
    /// equivalent: per-item/journal/transcript/budget events don't map).
    private static func map(_ event: FlowInterpreter.PathEvent,
                            pathToID: [String: UUID]) -> FlowEvent? {
        switch event.kind {
        case .rowStarted:
            return rowID(for: event.path, in: pathToID).map { .started(rowID: $0) }
        case .rowCompleted:
            return rowID(for: event.path, in: pathToID).map { .finished(rowID: $0, event.output ?? Asset(items: [])) }
        case .cacheHit:
            return rowID(for: event.path, in: pathToID).map { .cacheHit(rowID: $0) }
        case .rowFailed:
            return rowID(for: event.path, in: pathToID).map {
                .failed(rowID: $0, FlowError.stageFailure(row: event.path, message: event.error ?? "row failed"))
            }
        case .flagRaised:
            return rowID(for: event.path, in: pathToID).map { .flagRaised(rowID: $0, message: event.message ?? "") }
        case .runParked:
            return rowID(for: event.path, in: pathToID).map {
                .parked(rowID: $0, prompt: event.parkPrompt ?? "", policy: event.parkPolicy ?? "",
                        execPath: event.path)
            }
        case .runCompleted, .eachItemStarted, .eachItemCompleted, .budgetForced,
             .transcriptAppended, .journalAppended, .runResumed, .effectStaged:
            return nil
        }
    }

}
