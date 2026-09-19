import Foundation

/// The full interpreter — a faithful port of `catflow-mlx/src/catflow/core/interpreter.py`
/// `run_v04_flat`/`_run_scope` (CFM-R7-1). Walks `Row.clause` edges with a FIFO of threads,
/// gathers inputs (explicit refs, `(input:N)`, edge-arrival override, auto-chain with the
/// shape gate), fan-out for `<each>`/`<parallel>`, and budget forcing.
///
/// It emits the **path-based** event stream the Python yields (the conformance traces' ground
/// truth): `row_started` / `row_completed` / `each_item_started` / `each_item_completed` /
/// `run_completed`, with the same dotted exec paths (`"2.1"`). `FlowRunner.linear` is kept
/// until this port passes the conformance traces, then retired (per CFM-R7-1). `<parallel>`
/// is *semantic* — chains run sequentially in declared order (CFM-R7-2), never concurrently.
nonisolated enum FlowInterpreter {

    /// One emitted event, in the Python's wire shape.
    struct PathEvent: Sendable {
        enum Kind: String, Sendable {
            case rowStarted = "row_started"
            case rowCompleted = "row_completed"
            case eachItemStarted = "each_item_started"
            case eachItemCompleted = "each_item_completed"
            case runCompleted = "run_completed"
            case budgetForced = "budget_forced"
            case flagRaised = "flag_raised"
            case transcriptAppended = "transcript_appended"
            case journalAppended = "journal_appended"
            case runParked = "run_parked"
            case rowFailed = "row_failed"
            case rowSkipped = "row_skipped"
            case cacheHit = "cache_hit"
            case runResumed = "run_resumed"
            case effectStaged = "effect_staged"
        }
        let kind: Kind
        let path: String
        let output: Asset?
        let index: Int?
        let total: Int?
        let durationMS: Int
        let firedTag: String?
        let edge: String?
        let code: String?
        let message: String?
        let error: String?
        let staged: (id: String, kind: String, summary: String)?
        let transcript: (tool: String, input: String, observation: String)?
        let entry: (label: String, content: String)?
        let entryIndex: Int?
        let parkPrompt: String?
        let parkPolicy: String?
        let context: [(label: String, content: String)]?

        static func rowStarted(_ path: String) -> PathEvent {
            PathEvent(kind: .rowStarted, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func rowCompleted(_ path: String, _ output: Asset, firedTag: String?,
                                 context: [(label: String, content: String)]?) -> PathEvent {
            PathEvent(kind: .rowCompleted, path: path, output: output, index: nil, total: nil,
                      durationMS: 0, firedTag: firedTag, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: context)
        }
        static func eachStarted(_ path: String, _ index: Int, _ total: Int) -> PathEvent {
            PathEvent(kind: .eachItemStarted, path: path, output: nil, index: index, total: total,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func eachCompleted(_ path: String, _ index: Int, _ total: Int) -> PathEvent {
            PathEvent(kind: .eachItemCompleted, path: path, output: nil, index: index, total: total,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func budgetForced(_ path: String, _ edge: String) -> PathEvent {
            PathEvent(kind: .budgetForced, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: edge, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func transcriptAppended(_ path: String, _ tool: String, _ input: String,
                                       _ observation: String) -> PathEvent {
            PathEvent(kind: .transcriptAppended, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil,
                      transcript: (tool, input, observation), entry: nil, entryIndex: nil,
                      parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func journalAppended(_ path: String, _ index: Int, _ label: String, _ content: String) -> PathEvent {
            PathEvent(kind: .journalAppended, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: (label, content), entryIndex: index, parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func rowFailed(_ path: String, _ error: String) -> PathEvent {
            PathEvent(kind: .rowFailed, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil,
                      error: error, staged: nil, transcript: nil, entry: nil, entryIndex: nil,
                      parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        /// SPEC-Q216 (DA-10): a row an enclosing `<each on_error=skip>` chose to drop — the
        /// block continues, so the row is neither `rowCompleted` nor `rowFailed`. The reason
        /// travels in `error`. Terminal for its path: a skipped row must never be left in
        /// `rowStarted`.
        static func rowSkipped(_ path: String, _ reason: String) -> PathEvent {
            PathEvent(kind: .rowSkipped, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil,
                      error: reason, staged: nil, transcript: nil, entry: nil, entryIndex: nil,
                      parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func cacheHit(_ path: String) -> PathEvent {
            PathEvent(kind: .cacheHit, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil,
                      error: nil, staged: nil, transcript: nil, entry: nil, entryIndex: nil,
                      parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func runResumed(_ path: String) -> PathEvent {
            PathEvent(kind: .runResumed, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil,
                      error: nil, staged: nil, transcript: nil, entry: nil, entryIndex: nil,
                      parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func effectStaged(_ path: String, _ id: String, _ kind: String, _ summary: String) -> PathEvent {
            PathEvent(kind: .effectStaged, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil,
                      error: nil, staged: (id, kind, summary), transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: nil)
        }
        static func flagRaised(_ path: String, _ code: String, _ message: String) -> PathEvent {
            PathEvent(kind: .flagRaised, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: code, message: message, error: nil, staged: nil,
                      transcript: nil, entry: nil, entryIndex: nil, parkPrompt: nil,
                      parkPolicy: nil, context: nil)
        }
        static func runParked(_ path: String, _ prompt: String, _ policy: String) -> PathEvent {
            PathEvent(kind: .runParked, path: path, output: nil, index: nil, total: nil,
                      durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                      entry: nil, entryIndex: nil, parkPrompt: prompt, parkPolicy: policy, context: nil)
        }
    }

    // MARK: - Engine state

    /// One activation of a row — what `(N)`/`(N@k)` refs resolve to, and what a budget's
    /// forced edge carries forward.
    private struct Activation {
        let path: String
        let k: Int
        let input: [Asset]
        let output: Asset
        let firedTag: String?
    }

    private struct CallFrame {
        let returnTo: Int
        let owner: Int?
        let tool: String?
        let toolInput: String?
    }

    private struct Thread {
        var position: Int
        var callStack: [CallFrame]
        var lastOutput: Asset?
        var edgeArrival: Bool
        var edgeSource: Int?
    }

    private struct NotReadyYet: Error {}
    private struct ItemFailed: Error { let reason: String }
    private struct StopRun: Error {}
    private struct ParkRun: Error { let path: String; let prompt: String; let policy: String }

    /// One `· ctx+`/`Read Context` journal entry (`JournalEntry`).
    struct JournalEntry: Sendable, Equatable {
        let label: String
        let content: String
    }

    /// `Occurrence` — one synthetic trigger firing (Spec §11.1); exactly one field set.
    struct Occurrence: Sendable {
        var path: URL?
        var tick: String?
        var payload: String?
    }

    /// `HumanAnswer` — one resolved live answer for a human row's activation (P3-MC-02):
    /// `tag` for Ask Human (must name a declared edge), `text` for Human Input.
    struct HumanAnswer: Sendable {
        var tag: String?
        var text: String?
        /// True when this answer is the timeout fallback — nobody answered by the deadline.
        /// Ask Human resolves to its `default=` tag, Human Input passes its input through
        /// ("unchanged"), both with an F002 disclosure.
        var isTimeoutDefault: Bool = false
    }

    /// A resolved `uses:` entry (P5-NC-02, Spec §6.7) — the used flow's body plus its own
    /// scope (definitions/presets), which take over once a call expands into its body.
    /// `nested` (P5-NC-02) is the used flow's *own* further `uses:` imports — a used flow's
    /// calls must resolve against its imports, never the caller's. `sourceText` (P5-NC-03) is
    /// the used flow's exact resolved text — the cache-key's `used_flow_content` ingredient
    /// for every row that ends up inside it.
    struct UsedFlow: Sendable {
        var params: [ParamDecl]
        var rows: [Row]
        var definitions: [String: CompositeDef]
        var presets: [String: PresetDecl]
        var nested: [String: UsedFlow] = [:]
        var sourceText: String? = nil
        /// CFM-R17-FIX-10: the **names** of any `transforms:` the used flow declares — not the
        /// bodies. `RealExecutor.transforms` only ever carries the *caller's*, so a used flow's
        /// own transform can't execute; `FlowRunner.canRun` refuses a row that calls one, with
        /// a sentence that says *that* rather than "unknown task". Carrying names (not scripts)
        /// keeps the Direct fence exactly where it was.
        var transformNames: Set<String> = []
    }

    /// One completed tool-call round trip for a transcript-keeping decider (`Think`,
    /// P2-M13-01) — the Python's `TranscriptEntry`.
    struct TranscriptEntry: Sendable, Equatable {
        let tool: String
        let input: String
        let observation: String
    }

    /// `_NAME_PLACEHOLDER_TASKS` — the Save-family rows eligible for `{name}` substitution.
    static let namePlaceholderTasks: Set<String> = [
        "Save Text", "Save Audio", "Save Image", "Save Video", "Save Context",
    ]

    static let triggerTasks: Set<String> = ["On File", "On Schedule", "On Flow"]

    // MARK: - Public entry

    static let defaultMaxSteps = 1000

    /// A small accumulator that also pushes each event to a listener as it lands — the live
    /// half of `FlowRunner.run`'s stream (the GUI's dots advance row-by-row instead of all
    /// arriving when the run finishes). Every `events.append(...)` site is unchanged; a nil
    /// listener makes it a plain array, so the conformance traces stay byte-for-byte.
    private struct EventSink {
        var events: [PathEvent] = []
        var onEvent: ((PathEvent) -> Void)?
        mutating func append(_ event: PathEvent) {
            events.append(event)
            onEvent?(event)
        }
        mutating func append(contentsOf more: [PathEvent]) {
            for event in more { append(event) }
        }
    }

    /// Run `doc` to completion, returning the full path-event stream (the Python's
    /// `run_v04_flat`). `definitions` is the flow's own `definitions:` section — a row whose
    /// task names one is expanded into an ordinary `<list>` block (CFM-R7-1, `_expand_composite_call`).
    /// Throws on a row failure (the Python yields `RowFailed` and stops).
    ///
    /// **`parkOnTimeout`** — a deliberate, GUI-requested divergence from the Python reference:
    /// the Python resolves a `timeout=` human row to its default immediately (F002, unattended).
    /// The app instead **parks** it (like `wait=forever`) so the user can answer before the
    /// deadline; past it, the session falls back to the default. Default `false` keeps the
    /// conformance trace byte-for-byte; `FlowRunner.run` passes `true` for the GUI.
    ///
    /// **`onEvent`** — called synchronously as each event lands. `FlowRunner.run` maps it to a
    /// `FlowEvent` and yields it live; the return value is still the full array for the callers
    /// that assert on the completed trace.
    static func run(_ doc: FlowDocument, executor: any FlowExecutor,
                    maxSteps: Int = defaultMaxSteps,
                    definitions: [String: CompositeDef] = [:],
                    presets: [String: PresetDecl] = [:],
                    usesGraph: [String: UsedFlow] = [:],
                    occurrence: Occurrence? = nil,
                    answers: [String: HumanAnswer] = [:],
                    parkOnTimeout: Bool = false,
                    onEvent: ((PathEvent) -> Void)? = nil) async throws -> [PathEvent] {
        var events = EventSink(onEvent: onEvent)
        var journal: [JournalEntry] = []
        do {
            _ = try await runScope(
                rows: doc.rows, pathPrefix: "", blockInputs: [], executor: executor,
                events: &events, journal: &journal,
                onError: "fail", start: 1, end: nil, eachIndex: nil, eachItem: nil,
                definitions: definitions, presets: presets, usesGraph: usesGraph,
                usedContent: nil,
                triggerName: triggerName(rows: doc.rows, occurrence: occurrence),
                occurrence: occurrence, answers: answers, maxSteps: maxSteps,
                parkOnTimeout: parkOnTimeout)
            events.append(PathEvent(kind: .runCompleted, path: "", output: nil, index: nil, total: nil,
                                    durationMS: 0, firedTag: nil, edge: nil, code: nil, message: nil, error: nil, staged: nil, transcript: nil,
                                    entry: nil, entryIndex: nil, parkPrompt: nil, parkPolicy: nil, context: nil))
        } catch let park as ParkRun {
            events.append(PathEvent.runParked(park.path, park.prompt, park.policy))
        } catch is StopRun {
            // A row failure was emitted as `.rowFailed`; the run stops there (no run_completed).
        }
        return events.events
    }

    // MARK: - _run_scope

    /// One scope (the top-level flow, or a block body) to completion, returning its final
    /// traveling asset. `events` accumulates the Python's yield order.
    private static func runScope(
        rows: [Row], pathPrefix: String, blockInputs: [Asset],
        executor: any FlowExecutor,
        events: inout EventSink, journal: inout [JournalEntry],
        onError: String, start: Int, end: Int?, eachIndex: Int?, eachItem: String?,
        definitions: [String: CompositeDef], presets: [String: PresetDecl],
        usesGraph: [String: UsedFlow], usedContent: String?,
        triggerName: String, occurrence: Occurrence?, answers: [String: HumanAnswer],
        maxSteps: Int, parkOnTimeout: Bool
    ) async throws -> Asset? {
        // `visits`/`activations` are **per scope** (the Python's `_run_scope` creates them
        // locally): a block's children get fresh ones, so a child at position 1 doesn't
        // inherit the parent's `visits[1]` and acquire a spurious `@2` activation suffix.
        // `transcripts` is per scope too (`interpreter.py:929`) — a transcript-keeping
        // decider's activations accumulate within the scope that contains them, never across
        // `<each>` items or `<parallel>` chains.
        var visits: [Int: Int] = [:]
        var activations: [Int: [Activation]] = [:]
        var transcripts: [Int: [TranscriptEntry]] = [:]
        let blockInput = blockInputs.first
        let n = rows.count
        if n == 0 { return blockInput ?? Asset(items: []) }
        let stop = end ?? n
        let scope = Dictionary(rows.enumerated().map { ($1.id, $0 + 1) }, uniquingKeysWith: { a, _ in a })

        var queue: [Thread] = [Thread(position: start, callStack: [], lastOutput: blockInput,
                                      edgeArrival: false, edgeSource: nil)]
        var finalOutput: Asset? = blockInput
        var steps = 0

        while !queue.isEmpty {
            steps += 1
            if steps > maxSteps {
                throw FlowError.stageFailure(row: pathPrefix.isEmpty ? "run" : pathPrefix,
                                             message: "exceeded \(maxSteps) steps")
            }
            let thread = queue.removeFirst()
            let position = thread.position
            let callStack = thread.callStack

            // A composite call (a row whose task names a `definitions:` entry) expands into
            // an ordinary `<list>` block — params bound, body substituted (R7-1). A `uses:`
            // entry expands the same way, but the child scope swaps to the used flow's own
            // definitions/presets *and* its own nested `uses:` imports (P5-NC-02), and every
            // row inside it is keyed against the used file's text (P5-NC-03).
            let row: Row
            var childDefinitions = definitions
            var childPresets = presets
            var childUsesGraph = usesGraph
            var childUsedContent = usedContent
            if rows[position - 1].blockKind == nil {
                if let comp = definitions[rows[position - 1].task ?? ""] {
                    row = expandCompositeCall(row: rows[position - 1], comp: comp)
                } else if let used = usesGraph[rows[position - 1].task ?? ""] {
                    let comp = CompositeDef(name: rows[position - 1].task ?? "",
                                            params: used.params, rows: used.rows)
                    row = expandCompositeCall(row: rows[position - 1], comp: comp)
                    childDefinitions = used.definitions
                    childPresets = used.presets
                    childUsesGraph = used.nested
                    childUsedContent = used.sourceText
                } else {
                    row = rows[position - 1]
                }
            } else {
                row = rows[position - 1]
            }
            let path = "\(pathPrefix)\(position)"

            // Budget forcing: a row at its visit cap routes to its on_budget edge, carrying the
            // prior activation's output (the Python's `_forced_target`).
            if let forced = try forcedTarget(row, position: position, path: path,
                                             activations: activations, visits: visits) {
                let (edgeTag, target, forcedOutput) = forced
                events.append(PathEvent.budgetForced(path, edgeTag))
                let (nextPosition, _, _) = advanceTarget(target, callStack: callStack, callerPosition: nil,
                                                         tool: nil, toolInput: nil)
                // SPEC-Q72: a forced edge whose *target* is a call is routing (via_edge), the
                // same as any other call edge — never a plain fallthrough.
                let viaEdge = isCallTarget(target)
                if let nextPosition {
                    queue.append(Thread(position: nextPosition, callStack: callStack,
                                        lastOutput: forcedOutput, edgeArrival: viaEdge,
                                        edgeSource: viaEdge ? position : nil))
                } else {
                    finalOutput = forcedOutput
                }
                continue
            }

            if row.blockKind != nil {
                // FIX-10: a ref that hasn't resolved yet defers the whole thread to the back
                // of the FIFO (the Python's `_NotReadyYet` re-queue), so a later row's
                // completion can satisfy it this pass.
                let gathered: ([Asset], (Int, String)?)
                do {
                    gathered = try gatherInputs(row, scope: scope, activations: activations,
                                                lastOutput: thread.lastOutput, blockInputs: blockInputs,
                                                edgeArrival: thread.edgeArrival, edgeSource: thread.edgeSource)
                } catch is NotReadyYet {
                    if !queue.isEmpty {
                        queue.append(thread)
                        continue
                    }
                    throw NotReadyYet()
                }
                let (bundle, override) = gathered
                let childBundle = bundle.isEmpty ? [Asset(items: [])] : bundle
                let childPrefix = "\(path)."
                let k = (visits[position] ?? 0) + 1
                visits[position] = k
                let execPath = k == 1 ? path : "\(path)@\(k)"
                events.append(PathEvent.rowStarted(execPath))
                // FIX-9: a block row's position-1 edge-arrival override is disclosed as F007
                // exactly like an ordinary row's (the Python yields `_flag_r17` right after
                // the block's RowStarted).
                if let override {
                    events.append(PathEvent.flagRaised(execPath, "F007",
                        "This activation's first input arrived from row \(override.0)'s edge, overriding the `(\(override.1))` reference for this pass."))
                }

                let output: Asset
                if row.blockKind == .each {
                    output = try await runEach(execPath, children: row.children, pathPrefix: childPrefix,
                                               bundle: childBundle, executor: executor,
                                               events: &events, journal: &journal,
                                               onError: eachOnError(row), eachIndex: eachIndex,
                                               eachItem: eachItem, definitions: childDefinitions,
                                               presets: childPresets, usesGraph: childUsesGraph,
                                               usedContent: childUsedContent,
                                               triggerName: triggerName, occurrence: occurrence,
                                               answers: answers, maxSteps: maxSteps,
                                               parkOnTimeout: parkOnTimeout)
                } else if row.blockKind == .parallel {
                    output = try await runParallel(children: row.children, pathPrefix: childPrefix,
                                                   bundle: childBundle, executor: executor,
                                                   events: &events, journal: &journal,
                                                   onError: onError, eachIndex: eachIndex, eachItem: eachItem,
                                                   definitions: childDefinitions, presets: childPresets,
                                                   usesGraph: childUsesGraph, usedContent: childUsedContent,
                                                   triggerName: triggerName,
                                                   occurrence: occurrence, answers: answers,
                                                   maxSteps: maxSteps, parkOnTimeout: parkOnTimeout)
                } else {
                    output = try await runScope(rows: row.children, pathPrefix: childPrefix,
                                                blockInputs: childBundle, executor: executor,
                                                events: &events, journal: &journal,
                                                onError: onError, start: 1, end: nil,
                                                eachIndex: eachIndex, eachItem: eachItem,
                                                definitions: childDefinitions, presets: childPresets,
                                                usesGraph: childUsesGraph, usedContent: childUsedContent,
                                                triggerName: triggerName,
                                                occurrence: occurrence, answers: answers,
                                                maxSteps: maxSteps, parkOnTimeout: parkOnTimeout)
                            ?? Asset(items: [])
                }
                activations[position, default: []].append(
                    Activation(path: path, k: k, input: bundle, output: output, firedTag: nil))
                // A block row's own `· ctx+` appends, labeled by its name (or kind when the
                // author left it unnamed — the auto-name, P3-MB-05).
                if rowCtxMarkers(row).appends {
                    journal.append(JournalEntry(label: row.blockName ?? row.blockKind?.rawValue ?? "?",
                                                content: assetTextTruncated(output)))
                    events.append(PathEvent.journalAppended(execPath, journal.count - 1,
                                                            journal[journal.count - 1].label,
                                                            journal[journal.count - 1].content))
                }
                events.append(PathEvent.rowCompleted(execPath, output, firedTag: nil, context: nil))

                if case .fork(let targets) = row.clause {
                    for target in targets {
                        switch target {
                        case .done: finalOutput = output
                        default:
                            queue.append(Thread(position: targetNumber(target), callStack: callStack,
                                                lastOutput: output, edgeArrival: true, edgeSource: position))
                        }
                    }
                } else {
                    let (nextPosition, popped, viaEdge, newStack) = try advance(row, firedTag: nil, position: position,
                                                                                stop: stop, callStack: callStack, output: output)
                    if let popped {
                        events.append(contentsOf: recordTranscript(&transcripts, popped, output: output,
                                                                    pathPrefix: pathPrefix))
                    }
                    if let nextPosition {
                        queue.append(Thread(position: nextPosition, callStack: newStack,
                                            lastOutput: output, edgeArrival: viaEdge,
                                            edgeSource: viaEdge ? position : nil))
                    } else {
                        finalOutput = output
                    }
                }
                continue
            }

            // Ordinary row.
            // FIX-10: a ref that hasn't resolved yet defers the whole thread to the back of
            // the FIFO, so a later row's completion can satisfy it this pass.
            let gathered: ([Asset], (Int, String)?)
            do {
                gathered = try gatherInputs(row, scope: scope, activations: activations,
                                            lastOutput: thread.lastOutput, blockInputs: blockInputs,
                                            edgeArrival: thread.edgeArrival, edgeSource: thread.edgeSource)
            } catch is NotReadyYet {
                if !queue.isEmpty {
                    queue.append(thread)
                    continue
                }
                throw NotReadyYet()
            }
            var inputs = gathered.0
            let override = gathered.1
            // Save Context / Count Context read the run's live journal, not a ref (P3-MB-05).
            if row.task == "Save Context" || row.task == "Count Context" {
                inputs = [Asset(items: [Item(kind: .context, value: entriesToJSON(journal),
                                             path: nil, sourceText: nil)])]
            }
            // A trigger row's real input is the occurrence that fired it (P3-MD-01).
            if Self.triggerTasks.contains(row.task ?? "") {
                if let occurrence {
                    inputs = [occurrenceAsset(task: row.task ?? "", occurrence: occurrence)]
                } else {
                    inputs = []
                }
            }
            let k = (visits[position] ?? 0) + 1
            visits[position] = k
            let execPath = k == 1 ? path : "\(path)@\(k)"
            events.append(PathEvent.rowStarted(execPath))
            // SPEC-Q72/R17: an edge-arrival override of position 1 is disclosed as F007.
            if let override {
                events.append(PathEvent.flagRaised(execPath, "F007",
                    "This activation's first input arrived from row \(override.0)'s edge, overriding the `(\(override.1))` reference for this pass."))
            }

            // Settings substitution into a dispatch-only copy of the row: `{name}` (R19,
            // Save-family), `{index}`/`{item}` (an enclosing `<each>`, P6-LC-04).
            var execRow = row
            if let settings = execRow.settings, settings.contains("{name}"),
               Self.namePlaceholderTasks.contains(execRow.task ?? "") {
                execRow = Row(id: execRow.id, task: execRow.task, blockKind: execRow.blockKind,
                              blockName: execRow.blockName, model: execRow.model,
                              settings: settings.replacingOccurrences(of: "{name}", with: triggerName),
                              refs: execRow.refs, chainBreak: execRow.chainBreak,
                              children: execRow.children, clause: execRow.clause, tags: execRow.tags,
                              visitsLeq: execRow.visitsLeq, onBudget: execRow.onBudget,
                              declaredSignature: execRow.declaredSignature, comment: execRow.comment,
                              leadingComments: execRow.leadingComments)
            }
            if let settings = execRow.settings {
                var newSettings = settings
                if let eachIndex, execRow.task == "Template", settings.contains("{index}") {
                    newSettings = newSettings.replacingOccurrences(of: "{index}", with: String(eachIndex))
                }
                if let eachItem, settings.contains("{item}") {
                    newSettings = newSettings.replacingOccurrences(of: "{item}", with: eachItem)
                }
                if newSettings != settings {
                    execRow = Row(id: execRow.id, task: execRow.task, blockKind: execRow.blockKind,
                                  blockName: execRow.blockName, model: execRow.model, settings: newSettings,
                                  refs: execRow.refs, chainBreak: execRow.chainBreak,
                                  children: execRow.children, clause: execRow.clause, tags: execRow.tags,
                                  visitsLeq: execRow.visitsLeq, onBudget: execRow.onBudget,
                                  declaredSignature: execRow.declaredSignature, comment: execRow.comment,
                                  leadingComments: execRow.leadingComments)
                }
            }
            // Q199 (P6-PRESET-02): a row naming `preset=<name>` against the pipeline's
            // `presets:` block expands before dispatch (the resolved values are what the
            // cache key and engine call see). Deferred/unknown presets leave the settings
            // with the mechanism token stripped.
            if !presets.isEmpty, let settings = execRow.settings, settings.contains("preset=") {
                let resolution = resolvePresetSettings(settings, presets: presets)
                if resolution.unknown == nil && resolution.settings != settings {
                    execRow = Row(id: execRow.id, task: execRow.task, blockKind: execRow.blockKind,
                                  blockName: execRow.blockName, model: execRow.model,
                                  settings: resolution.settings,
                                  refs: execRow.refs, chainBreak: execRow.chainBreak,
                                  children: execRow.children, clause: execRow.clause, tags: execRow.tags,
                                  visitsLeq: execRow.visitsLeq, onBudget: execRow.onBudget,
                                  declaredSignature: execRow.declaredSignature, comment: execRow.comment,
                                  leadingComments: execRow.leadingComments)
                }
            }

            // P3-MB-05 (Spec §9.1/§9.3): the `context` a `· ctx` row reads is the journal
            // snapshot as of *before* this activation — captured ahead of dispatch, exactly
            // like the Python's `context_arg` (so the executor and the row's completion both
            // see the same pre-append snapshot).
            let markers = rowCtxMarkers(execRow)
            let contextArg = markers.reads ? journal.map { ($0.label, $0.content) } : nil

            // A human row parks when it waits forever **or** declares a timeout: it resolves a
            // live answer (P3-MC-02) — or, past the deadline, its declared fallback — instead
            // of running through the executor. A row with neither resolves to its default
            // immediately (the executor's non-parked F002 path).
            var output: Asset
            var firedTag: String?
            if (execRow.task == "Ask Human" || execRow.task == "Human Input")
                && (waitsForever(execRow) || (hasTimeout(execRow) && parkOnTimeout)) {
                guard let answer = answers[execPath] else {
                    throw ParkRun(path: execPath, prompt: rowPromptText(execRow), policy: parkPolicy(execRow))
                }
                // A timeout default (nobody answered by the deadline) is disclosed as F002,
                // the same sentence the non-parked executor path emits.
                if answer.isTimeoutDefault, let f002 = timeoutDefaultSentence(execRow) {
                    events.append(PathEvent.flagRaised(execPath, "F002", f002))
                }
                // Resolved without ever touching an executor: Ask Human is a passthrough with
                // the person's tag; Human Input's reply replaces the content. Emit RunResumed.
                let resolved = resolveHumanAnswer(row: execRow, inputs: inputs, answer: answer)
                output = resolved.0
                firedTag = resolved.1
                events.append(PathEvent.runResumed(execPath))
            } else {
                do {
                    // FIX-12: the seam passes the row's accumulated transcript, its `· ctx`
                    // journal snapshot, and the ambient `uses:` content — the same three
                    // `interpreter.py:1220-1221` folds into the cache key.
                    output = try await executor.execute(path: execPath, row: execRow, inputs: inputs,
                                                        transcript: transcripts[position],
                                                        context: contextArg,
                                                        usedFlowContent: childUsedContent)
                } catch {
                    if onError == "skip" {
                        // SPEC-Q216 (DA-10): emit a terminal event for this row before
                        // unwinding to `_run_each`'s per-item catch — the UI must be able to
                        // leave `rowStarted` and show *why* the item was dropped, not a dot
                        // stuck on ●. Reference parity: the Python raises `_ItemFailed` with
                        // no per-row event; logged as SPEC-Q216 for it to mirror.
                        let reason = String(describing: error)
                        events.append(PathEvent.rowSkipped(execPath, reason))
                        throw ItemFailed(reason: reason)
                    }
                    events.append(PathEvent.rowFailed(execPath, String(describing: error)))
                    throw StopRun()
                }
                if executor.lastCacheHit {
                    events.append(PathEvent.cacheHit(execPath))
                }
                firedTag = executor.lastTag
            }
            activations[position, default: []].append(
                Activation(path: path, k: k, input: inputs, output: output, firedTag: firedTag))
            // FIX-11: `Read Context` loads its output back into the run's *live* journal —
            // extended, never replaced (Spec §9.3, the Python's `journal.extend`).
            if row.task == "Read Context", let first = output.items.first, let value = first.value {
                journal.append(contentsOf: entriesFromJSON(value))
            }
            // A `· ctx+` row appends its own output to the run's live journal (P3-MB-05) —
            // emitted before the row's own completion, matching the Python's order.
            if markers.appends {
                journal.append(JournalEntry(label: execRow.task ?? "?", content: assetTextTruncated(output)))
                events.append(PathEvent.journalAppended(execPath, journal.count - 1,
                                                        journal[journal.count - 1].label,
                                                        journal[journal.count - 1].content))
            }
            events.append(PathEvent.rowCompleted(execPath, output, firedTag: firedTag, context: contextArg))
            // A non-forever human row's timeout default is disclosed as F002; a staged row
            // announces its queued outbox effect — both after the row completes (Python order).
            if let timeout = executor.lastTimeoutFlag {
                events.append(PathEvent.flagRaised(execPath, timeout.code, timeout.message))
            }
            // RM-2: a provider decider's F010 disclosure — "this row's tag is parsed, not
            // guaranteed" — after the row completes, same ordering as F002 above.
            if let providerFlag = executor.lastProviderDeciderFlag {
                events.append(PathEvent.flagRaised(execPath, providerFlag.code, providerFlag.message))
            }
            if let staged = executor.lastStaged {
                events.append(PathEvent.effectStaged(execPath, staged.id, staged.kind, staged.summary))
            }

            if case .fork(let targets) = row.clause {
                for target in targets {
                    switch target {
                    case .done: finalOutput = output
                    default:
                        queue.append(Thread(position: targetNumber(target), callStack: callStack,
                                            lastOutput: output, edgeArrival: true, edgeSource: position))
                    }
                }
            } else {
                let (nextPosition, popped, viaEdge, newStack) = try advance(row, firedTag: firedTag, position: position,
                                                                            stop: stop, callStack: callStack, output: output)
                if let popped {
                    events.append(contentsOf: recordTranscript(&transcripts, popped, output: output,
                                                                pathPrefix: pathPrefix))
                }
                if let nextPosition {
                    queue.append(Thread(position: nextPosition, callStack: newStack,
                                        lastOutput: output, edgeArrival: viaEdge,
                                        edgeSource: viaEdge ? position : nil))
                } else {
                    finalOutput = output
                }
            }
        }
        return finalOutput ?? Asset(items: [])
    }

    // MARK: - Blocks

    /// `_run_each` — one fresh child scope per item, in order. The first bundle element is the
    /// iterated source; the rest are broadcast to every item as `(input:2)`, `(input:3)`, …
    private static func runEach(_ path: String, children: [Row], pathPrefix: String, bundle: [Asset],
                                executor: any FlowExecutor,
                                events: inout EventSink, journal: inout [JournalEntry], onError: String,
                                eachIndex: Int?, eachItem: String?,
                                definitions: [String: CompositeDef], presets: [String: PresetDecl],
                                usesGraph: [String: UsedFlow], usedContent: String?,
                                triggerName: String,
                                occurrence: Occurrence?, answers: [String: HumanAnswer],
                                maxSteps: Int, parkOnTimeout: Bool) async throws -> Asset {
        let source = bundle.first ?? Asset(items: [])
        let broadcast = Array(bundle.dropFirst())
        let total = source.items.count
        var resultItems: [Item] = []
        for (index, item) in source.items.enumerated() {
            events.append(PathEvent.eachStarted(path, index, total))
            let itemInputs = [Asset(items: [item])] + broadcast
            do {
                let output = try await runScope(rows: children, pathPrefix: pathPrefix,
                                                blockInputs: itemInputs, executor: executor,
                                                events: &events, journal: &journal,
                                                onError: onError, start: 1, end: nil,
                                                eachIndex: index + 1, eachItem: itemValue(item),
                                                definitions: definitions, presets: presets,
                                                usesGraph: usesGraph, usedContent: usedContent,
                                                triggerName: triggerName,
                                                occurrence: occurrence, answers: answers,
                                                maxSteps: maxSteps, parkOnTimeout: parkOnTimeout) ?? Asset(items: [])
                resultItems.append(contentsOf: output.items)
            } catch let failed as ItemFailed {
                // FIX-6 (R6/`on_error=skip`): a skipped item flags F003 first — the Python's
                // "Item N of M failed (…) and was skipped. K delivered." — before the
                // item-complete bracket, so the drop is never silent.
                events.append(PathEvent.flagRaised(path, "F003",
                    "Item \(index + 1) of \(total) failed (\(failed.reason)) and was skipped. \(total - 1) delivered."))
                events.append(PathEvent.eachCompleted(path, index, total))
                continue
            }
            events.append(PathEvent.eachCompleted(path, index, total))
        }
        return Asset(items: resultItems)
    }

    /// `_parallel_chains` — maximal runs of consecutive children, split at chain breaks.
    private static func parallelChains(_ children: [Row]) -> [[Int]] {
        var chains: [[Int]] = []
        var current: [Int] = []
        for (i, child) in children.enumerated() {
            let position = i + 1
            if child.chainBreak && !current.isEmpty {
                chains.append(current)
                current = []
            }
            current.append(position)
        }
        if !current.isEmpty { chains.append(current) }
        return chains
    }

    /// `_run_parallel` — every chain gets the same block input and runs **sequentially** in
    /// declared chain order (CFM-R7-2: semantic parallel). Output is the chains' tails
    /// concatenated item-wise.
    ///
    /// FIX-7 (SPEC-Q104/§9.2): each chain runs against its **own private copy** of the
    /// journal (snapshotted at block entry), and its `journal_appended` entries commit onto
    /// the real shared journal only after the chain completes — in chain order, with
    /// `entry_index` rewritten against the real journal. A chain's `· ctx` row never sees a
    /// sibling's `ctx+` entry (the Python's `context=[]` for a lone `· ctx` chain).
    private static func runParallel(children: [Row], pathPrefix: String, bundle: [Asset],
                                    executor: any FlowExecutor,
                                    events: inout EventSink, journal: inout [JournalEntry], onError: String,
                                    eachIndex: Int?, eachItem: String?,
                                    definitions: [String: CompositeDef], presets: [String: PresetDecl],
                                    usesGraph: [String: UsedFlow], usedContent: String?,
                                    triggerName: String,
                                    occurrence: Occurrence?, answers: [String: HumanAnswer],
                                    maxSteps: Int, parkOnTimeout: Bool) async throws -> Asset {
        let chains = parallelChains(children)
        let journalSnapshot = journal
        var tails: [Asset] = []
        for chain in chains {
            // A chain's events buffer without a listener — they're committed (and emitted)
            // by the outer sink below, after the chain's journal index is rewritten.
            var chainEvents = EventSink()
            var chainJournal = journalSnapshot
            var chainTail: Asset?
            var chainError: Error?
            do {
                chainTail = try await runScope(rows: children, pathPrefix: pathPrefix, blockInputs: bundle,
                                               executor: executor, events: &chainEvents, journal: &chainJournal,
                                               onError: onError, start: chain[0], end: chain[chain.count - 1],
                                               eachIndex: eachIndex, eachItem: eachItem,
                                               definitions: definitions, presets: presets,
                                               usesGraph: usesGraph, usedContent: usedContent,
                                               triggerName: triggerName,
                                               occurrence: occurrence, answers: answers,
                                               maxSteps: maxSteps, parkOnTimeout: parkOnTimeout)
            } catch {
                chainError = error
            }
            // Commit this chain's events in order; a journal append lands on the *real*
            // journal with its entry_index rewritten (it was only valid against the chain's
            // private copy). The failing chain's events still commit before its error
            // re-raises (the Python's join semantics).
            for var ev in chainEvents.events {
                if ev.kind == .journalAppended, let entry = ev.entry {
                    journal.append(JournalEntry(label: entry.label, content: entry.content))
                    ev = PathEvent.journalAppended(ev.path, journal.count - 1, entry.label, entry.content)
                }
                events.append(ev)
            }
            if chainError != nil {
                throw chainError!
            }
            if let chainTail {
                tails.append(chainTail)
            }
        }
        return Asset(items: tails.flatMap { $0.items })
    }

    // MARK: - Input gathering

    /// `_gather_inputs` — explicit refs (row/input/param), with the SPEC-Q72 edge-arrival
    /// override of position 1; otherwise the auto-chain shape gate.
    private static func gatherInputs(_ row: Row, scope: [UUID: Int],
                                     activations: [Int: [Activation]],
                                     lastOutput: Asset?, blockInputs: [Asset],
                                     edgeArrival: Bool, edgeSource: Int?) throws -> ([Asset], (Int, String)?) {
        if !row.refs.isEmpty {
            var bundle: [Asset] = []
            for ref in row.refs {
                switch ref {
                case .rowRef(let id):
                    guard let number = scope[id], let acts = activations[number], let act = acts.last else {
                        throw NotReadyYet()
                    }
                    bundle.append(act.output)
                case .inputRef(let position):
                    guard position >= 1, position <= blockInputs.count else {
                        throw NotReadyYet()
                    }
                    bundle.append(blockInputs[position - 1])
                case .paramRef:
                    throw NotReadyYet()
                }
            }
            var override: (Int, String)?
            if edgeArrival, let lastOutput, let edgeSource {
                override = (edgeSource, refDesc(row.refs[0], scope: scope))
                if !bundle.isEmpty { bundle[0] = lastOutput }
            }
            return (bundle, override)
        }
        guard let lastOutput else { return ([], nil) }
        guard let accepts = rowAcceptsAndRefKind(row)?.accepts else { return ([], nil) }
        guard let given = assetShape(lastOutput) else { return ([], nil) }
        let compatible = autoChainCompatible(accepts: accepts, given: given,
                                             refKind: rowAcceptsAndRefKind(row)?.refKind)
        return (compatible ? [lastOutput] : [], nil)
    }

    /// `_asset_shape` — `Single(kind)` for one item, `ListOf(kind)` for many.
    private static func assetShape(_ asset: Asset) -> Shape? {
        guard let first = asset.items.first else { return nil }
        return asset.items.count == 1 ? .single(first.kind) : .listOf(first.kind)
    }

    /// `_auto_chain_compatible` — `single_compatible`, plus the variadic-text leniency.
    private static func autoChainCompatible(accepts: Shape, given: Shape, refKind: RefKind?) -> Bool {
        if Shape.singleCompatible(accepts: accepts, given: given, rk: refKind) { return true }
        if case .listOf(let ak) = accepts, case .single(let gk) = given, gk == ak { return true }
        return false
    }

    /// `_row_accepts_and_ref_kind` — the traveling-asset gate for a row or block.
    private static func rowAcceptsAndRefKind(_ row: Row) -> (accepts: Shape, refKind: RefKind?)? {
        if row.blockKind != nil {
            guard let first = row.children.first else { return nil }
            // SPEC-Q37: a block that references its own external input (`(input:1)`) anywhere
            // among its direct children is explicit evidence it wants the traveling asset
            // regardless of what its first child happens to accept (`react_agent`'s `Think`
            // reads the block's external input; its first row `Read Index` is unrelated
            // setup) — mirror `core/model.py::any_child_references_block_input` (direct
            // children only; a nested block opens its own `(input:K)` scope).
            if row.children.contains(where: { $0.refs.contains(.inputRef(1)) }) {
                return (.anyKind, nil)
            }
            guard let firstAccepts = rowAcceptsAndRefKind(first)?.accepts else { return nil }
            if row.blockKind == .each {
                if case .single(let k) = firstAccepts { return (.listOf(k), nil) }
            }
            return (firstAccepts, nil)
        }
        guard let desc = TaskCatalog.get(row.task ?? "") else { return nil }
        // A decider's auto-chain uses `ref_kind=None` (the Python: `desc.ref_kind if row.task
        // in CATALOG else None`) — so a `.anyKind` decider accepts any traveling asset, not
        // the frame-refKind's text-only gate.
        let refKind = TaskCatalog.deciderTasks[row.task ?? ""] != nil ? nil : desc.refKind
        return (desc.accepts, refKind)
    }

    // MARK: - Clauses

    /// `_advance` — single-target advancement (fork handled by the caller). Returns
    /// `(next, popped, viaEdge, newCallStack)` — the call stack is a value, so a pushed
    /// (`call N`, or a decider's decide-edge tool call) or popped (`resume`) stack comes back
    /// for the caller to queue.
    private static func advance(_ row: Row, firedTag: String?, position: Int, stop: Int,
                                callStack: [CallFrame], output: Asset?)
        throws -> (Int?, CallFrame?, Bool, [CallFrame]) {
        guard let clause = row.clause else {
            return (position < stop ? position + 1 : nil, nil, false, callStack)
        }
        switch clause {
        case .goto(let target):
            let (next, popped, stack) = advanceTarget(target, callStack: callStack, callerPosition: nil,
                                                      tool: nil, toolInput: nil)
            return (next, popped, true, stack)
        case .call(let number):
            var stack = callStack
            stack.append(CallFrame(returnTo: position, owner: nil, tool: nil, toolInput: nil))
            return (number, nil, true, stack)
        case .resume:
            if let frame = callStack.last {
                var stack = callStack
                stack.removeLast()
                return (frame.returnTo, frame, false, stack)
            }
            return (nil, nil, false, callStack)
        case .fork(let targets):
            // Fork is handled by the caller — defensive fallback.
            return (targets.first.map { targetNumber($0) }, nil, false, callStack)
        case .decide(let edges):
            guard let firedTag else { return (nil, nil, false, callStack) }
            for edge in edges where edge.tag == firedTag {
                // A transcript-keeping decider's fired edge carries the decider's own output
                // text as the tool argument (SPEC-Q36); the edge's target being a call marks
                // `via_edge` (SPEC-Q72).
                let isToolCall = isTranscriptKeepingDecider(row.task)
                let (next, popped, stack) = advanceTarget(edge.target, callStack: callStack,
                                                          callerPosition: position,
                                                          tool: isToolCall ? firedTag : nil,
                                                          toolInput: isToolCall ? assetText(output) : nil)
                return (next, popped, isCallTarget(edge.target), stack)
            }
            // Minor fix: a fired tag naming no edge is a bug, not a silent end-of-run — the
            // Python raises `RuntimeError` here.
            throw FlowError.stageFailure(row: "\(position)",
                                         message: "decider fired tag `\(firedTag)`, which has no edge")
        }
    }

    /// `_advance_target` — one continuation target; returns the (possibly pushed) stack.
    private static func advanceTarget(_ target: ClauseTarget, callStack: [CallFrame],
                                      callerPosition: Int?, tool: String?, toolInput: String?)
        -> (next: Int?, popped: CallFrame?, callStack: [CallFrame]) {
        switch target {
        case .done:
            return (nil, nil, callStack)
        case .resume:
            if let frame = callStack.last {
                var stack = callStack
                stack.removeLast()
                return (frame.returnTo, frame, stack)
            }
            return (nil, nil, callStack)
        case .call(let number):
            if let callerPosition {
                var stack = callStack
                stack.append(CallFrame(returnTo: callerPosition,
                                       owner: tool != nil ? callerPosition : nil,
                                       tool: tool, toolInput: toolInput))
                return (number, nil, stack)
            }
            return (number, nil, callStack)
        case .row(let number):
            return (number, nil, callStack)
        }
    }

    /// `_forced_target` — a row past its `visits_leq` cap routes to its `on_budget` edge,
    /// returning `(edgeTag, target, priorActivationOutput)`.
    private static func forcedTarget(_ row: Row, position: Int, path: String,
                                     activations: [Int: [Activation]],
                                     visits: [Int: Int]) throws -> (String, ClauseTarget, Asset)? {
        guard let cap = row.visitsLeq, (visits[position] ?? 0) + 1 > cap else { return nil }
        // `on_budget=fail` → BudgetExceededError (the Python raises; the run does *not*
        // proceed normally or re-run the row until maxSteps).
        if row.onBudget == "fail" {
            throw FlowError.budgetExceeded(row: path, visitsLeq: cap)
        }
        guard case .decide(let edges) = row.clause, let onBudget = row.onBudget else { return nil }
        guard let edge = edges.first(where: { $0.tag == onBudget }) else { return nil }
        guard let prior = activations[position]?.last else { return nil }
        return (onBudget, edge.target, prior.output)
    }

    // MARK: - Small helpers

    /// `_each_on_error` — an `<each>` row's `on_error=` marker.
    private static func eachOnError(_ row: Row) -> String {
        guard let settings = row.settings,
              let range = settings.range(of: #"on_error\s*=\s*(\w+)"#, options: .regularExpression),
              let value = settings[range].split(separator: "=").last?.trimmingCharacters(in: .whitespaces) else {
            return "fail"
        }
        return value == "skip" ? "skip" : "fail"
    }

    /// `_item_value` — an `<each>` item's `{item}` substitution string.
    private static func itemValue(_ item: Item) -> String {
        if let value = item.value { return value }
        if let path = item.path { return path.lastPathComponent }
        return ""
    }

    /// `_ref_desc` — a ref's literal description (F007's `{ref}`).
    private static func refDesc(_ ref: Ref, scope: [UUID: Int]) -> String {
        switch ref {
        case .rowRef(let id):
            return String(scope[id] ?? -1)
        case .inputRef(let position):
            return "input:\(position)"
        case .paramRef(let name):
            return "param:\(name)"
        }
    }

    private static func targetNumber(_ target: ClauseTarget) -> Int {
        if case .row(let number) = target { return number }
        if case .call(let number) = target { return number }
        return 1
    }

    // MARK: - Transcripts (CFM-R7-1, P2-M13-01)

    /// `TRANSCRIPT_KEEPING_DECIDERS` — only `Think` keeps a tool-call transcript.
    private static func isTranscriptKeepingDecider(_ task: String?) -> Bool {
        task == "Think"
    }

    /// `isinstance(target, CallTarget)`.
    private static func isCallTarget(_ target: ClauseTarget) -> Bool {
        if case .call = target { return true }
        return false
    }

    /// `_asset_text` — a transcript entry's text: join every item's value (file-backed falls
    /// back to `<kind>`), never reads a file.
    private static func assetText(_ asset: Asset?) -> String {
        guard let asset, !asset.items.isEmpty else { return "" }
        return asset.items.map { $0.value ?? "<\($0.kind.rawValue)>" }.joined(separator: " ")
    }

    /// `_record_transcript` — a `resume` that pops a transcript-keeping decider's frame emits
    /// one `transcript_appended` entry *and* accumulates it on the per-position store
    /// (`transcripts`), so the decider's later activations see their own growing "So far"
    /// (FIX-12, the cache key's `transcript_version` ingredient). Anything else is a no-op.
    private static func recordTranscript(_ transcripts: inout [Int: [TranscriptEntry]],
                                         _ popped: CallFrame, output: Asset,
                                         pathPrefix: String) -> [PathEvent] {
        guard let owner = popped.owner else { return [] }
        let entry = TranscriptEntry(tool: popped.tool ?? "", input: popped.toolInput ?? "",
                                    observation: assetText(output))
        transcripts[owner, default: []].append(entry)
        return [PathEvent.transcriptAppended("\(pathPrefix)\(owner)",
                                             entry.tool, entry.input, entry.observation)]
    }

    // MARK: - Context journal (P3-MB-05, Spec §9.1)

    /// `_RE_CTX_READ` / `_RE_CTX_APPEND` — `· ctx` / `· ctx+` (or `;`) markers in settings.
    private static let reCtxRead = NSRegularExpression.compiled(#"[·;]\s*ctx(?!\+)\b"#)
    private static let reCtxAppend = NSRegularExpression.compiled(#"[·;]\s*ctx\+"#)

    /// `_row_ctx_markers` — (reads, appends) for a row's own marker text. Widened from
    /// `private` for FV-2 (`RSI/DelegateFrameViewBacklog.md` §4.1 point 3, the `; ctx` splice
    /// row): the Properties-tab frame preview needs `reads` to know whether to render a
    /// stand-in "Shared context so far:" block, and this regex pair is the one place that
    /// decision is made.
    static func rowCtxMarkers(_ row: Row) -> (reads: Bool, appends: Bool) {
        guard let settings = row.settings, !settings.isEmpty else { return (false, false) }
        let ns = NSRange(settings.startIndex..<settings.endIndex, in: settings)
        return (reCtxRead.firstMatch(in: settings, range: ns) != nil,
                reCtxAppend.firstMatch(in: settings, range: ns) != nil)
    }

    /// `entries_to_json` — a journal's inline `context`-kind text.
    private static func entriesToJSON(_ journal: [JournalEntry]) -> String {
        let dicts = journal.map { ["label": $0.label, "content": $0.content] }
        let data = try? JSONSerialization.data(withJSONObject: dicts)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "[]"
    }

    /// `entries_from_json` — a `Read Context` row's output parsed back into journal entries
    /// (P3-MB-05, Spec §9.3), loaded into the run's live journal. Unparseable text extends
    /// nothing.
    private static func entriesFromJSON(_ text: String) -> [JournalEntry] {
        guard let data = text.data(using: .utf8),
              let dicts = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return []
        }
        return dicts.compactMap { dict in
            guard let label = dict["label"] as? String, let content = dict["content"] as? String else {
                return nil
            }
            return JournalEntry(label: label, content: content)
        }
    }

    /// `truncate` — CONTENT_PREVIEW_LIMIT (200) chars, `.rstrip()`ed, then `…`.
    private static func truncate(_ text: String, limit: Int = 200) -> String {
        guard text.count > limit else { return text }
        let prefix = String(text.prefix(limit)).trimmingCharacters(in: .whitespaces)
        return prefix + "…"
    }

    /// `_asset_text_truncated` — a journal entry's content.
    private static func assetTextTruncated(_ asset: Asset) -> String {
        truncate(assetText(asset))
    }

    // MARK: - Triggers (P3-MD-01, R19)

    /// `_trigger_name` — the flow-wide `{name}` value, from row 1 + occurrence.
    private static func triggerName(rows: [Row], occurrence: Occurrence?) -> String {
        guard let first = rows.first, let occurrence, first.task == "On File",
              let path = occurrence.path else { return "" }
        return path.deletingPathExtension().lastPathComponent
    }

    /// `_occurrence_asset` — the trigger row's payload per §11.1.
    private static func occurrenceAsset(task: String, occurrence: Occurrence) -> Asset {
        if task == "On File", let path = occurrence.path {
            return Asset(items: [Item(kind: .file, value: nil, path: path, sourceText: nil)])
        }
        let text = task == "On Schedule" ? occurrence.tick : occurrence.payload
        return Asset(items: [Item(kind: .text, value: text ?? "", path: nil, sourceText: nil)])
    }

    // MARK: - Human rows (P3-MC-02)

    /// `_waits_forever` — `wait=forever` in the settings.
    private static func waitsForever(_ row: Row) -> Bool {
        guard let settings = row.settings else { return false }
        return settings.range(of: #"\bwait\s*=\s*forever\b"#, options: .regularExpression) != nil
    }

    /// Whether the row declares a `timeout=` — it parks (like `wait=forever`) and falls back
    /// to its default when the deadline passes unanswered.
    private static func hasTimeout(_ row: Row) -> Bool {
        guard let settings = row.settings else { return false }
        return settings.range(of: #"\btimeout\s*="#, options: .regularExpression) != nil
    }

    /// The parked row's waiting policy — `wait=forever`, or `timeout=<raw>` so the session can
    /// parse the deadline for the prompt's countdown + fallback.
    private static func parkPolicy(_ row: Row) -> String {
        if let timeout = FlowSettings(row.settings).value(for: "timeout") {
            return "timeout=\(timeout)"
        }
        return "wait=forever"
    }

    /// The F002 disclosure for a timeout fallback, or nil when none applies (an Ask Human row
    /// with no declared `default=`) — mirrors `RealExecutor`'s sentence.
    private static func timeoutDefaultSentence(_ row: Row) -> String? {
        let s = FlowSettings(row.settings)
        guard let timeout = s.value(for: "timeout") else { return nil }
        if row.task == "Ask Human" {
            guard let dflt = s.value(for: "default"), !dflt.isEmpty else { return nil }
            return "Nobody answered by \(timeout) — proceeded as `\(dflt)`, unreviewed."
        }
        return "Nobody answered by \(timeout) — proceeded as `unchanged`, unreviewed."
    }

    /// `_resolve_human_answer` — a live answer, resolved without touching an executor.
    private static func resolveHumanAnswer(row: Row, inputs: [Asset],
                                           answer: HumanAnswer) -> (Asset, String?) {
        if row.task == "Ask Human" {
            // A timeout fallback fires the row's declared `default=` tag (the F002 sentence
            // names it); a live answer fires the chosen tag.
            let tag = answer.isTimeoutDefault
                ? FlowSettings(row.settings).value(for: "default")
                : answer.tag
            return (inputs.first ?? Asset(items: []), tag)
        }
        // Human Input's timeout fallback is "unchanged": pass the input through.
        if answer.isTimeoutDefault {
            return (inputs.first ?? Asset(items: []), nil)
        }
        return (Asset(items: [Item(kind: .text, value: answer.text ?? "", path: nil, sourceText: nil)]), nil)
    }

    /// `_row_prompt_text` — a human row's leading quoted criterion, unquoted. Not `private`:
    /// RT-2 (`RSI/DelegateRowTextBacklog.md`) needs this exact reading, not a re-derivation, to
    /// verify the Properties tab's human-question box never edits a token this runtime displays
    /// differently (hazard 8 — this doesn't process `\n`/`\t` escapes the way `FlowSettings
    /// .unquote` does).
    static func rowPromptText(_ row: Row) -> String {
        let settings = row.settings ?? ""
        var inQuotes = false
        var chars: [Character] = []
        for c in settings {
            if c == "\"" {
                inQuotes.toggle()
                chars.append(c)
            } else if (c == "·" || c == ";") && !inQuotes {
                break
            } else {
                chars.append(c)
            }
        }
        let text = String(chars).trimmingCharacters(in: .whitespaces)
        if text.count >= 2, text.first == "\"", text.last == "\"" {
            return String(text.dropFirst().dropLast())
        }
        return text
    }

    // MARK: - Presets (Q199, P6-PRESET-02)

    /// `_split_tokens` — the raw `_TOKEN_RE` token boundaries (shared with `FlowSettings`).
    private static func splitTokens(_ text: String?) -> [String] {
        FlowSettings.rawTokens(text)
    }

    /// `_is_param_ref` — `(param:name)`, deferring expansion until a caller binds the param.
    private static func isParamRef(_ name: String) -> Bool {
        let stripped = name.trimmingCharacters(in: .whitespaces)
        guard stripped.hasPrefix("("), stripped.hasSuffix(")") else { return false }
        return stripped.range(of: #"\(param:[A-Za-z_][A-Za-z0-9_-]*\)"#, options: .regularExpression) != nil
    }

    /// `_strip_preset_tokens` — drop every `preset=` token, keeping the rest byte-for-byte.
    private static func stripPresetTokens(_ text: String) -> String {
        let kept = splitTokens(text).filter { !$0.hasPrefix("preset=") }
        return kept.isEmpty ? "" : kept.joined(separator: "; ")
    }

    /// `resolve_preset_settings` — expand a row's settings against the pipeline's presets.
    private static func resolvePresetSettings(_ settings: String,
                                              presets: [String: PresetDecl]) -> (settings: String, unknown: String?) {
        let tokens = splitTokens(settings)
        let presetTokens = tokens.filter { $0.hasPrefix("preset=") }
        if presetTokens.isEmpty { return (settings, nil) }
        let name = String(presetTokens.last!.dropFirst("preset=".count)).trimmingCharacters(in: .whitespaces)
        let others = tokens.filter { !$0.hasPrefix("preset=") }
        if presets.isEmpty { return (settings, nil) }   // a .cat — `preset=` is an ordinary setting
        if isParamRef(name) { return (stripPresetTokens(settings), nil) }   // deferred
        guard let preset = presets[name] else { return (stripPresetTokens(settings), name) }   // E715
        var explicit = Set<String>()
        for token in others where token.contains("=") && !token.hasPrefix("\"") {
            explicit.insert(String(token.split(separator: "=", maxSplits: 1)[0]).trimmingCharacters(in: .whitespaces))
        }
        var added: [String] = []
        for binding in preset.bindings where !explicit.contains(binding.key) {
            added.append("\(binding.key)=\(binding.value)")
        }
        let combined = others + added
        return (combined.isEmpty ? "" : combined.joined(separator: "; "), nil)
    }

    // MARK: - Composites (`definitions:`) — `_expand_composite_call` and friends

    /// `_split_on_dot_quoted` — quote-aware split on `·` or `;`.
    private static func splitOnDotQuoted(_ text: String) -> [String] {
        var parts: [String] = []
        var current = ""
        var inQuotes = false
        for c in text {
            if c == "\"" {
                inQuotes.toggle()
                current.append(c)
            } else if (c == "·" || c == ";") && !inQuotes {
                parts.append(current)
                current = ""
            } else {
                current.append(c)
            }
        }
        parts.append(current)
        return parts
    }

    /// `_RE_KV_PARAM` — `key=value` at the start of a settings part.
    private static func kvParam(_ part: String) -> (String, String)? {
        let regex = NSRegularExpression.compiled("^([\\w-]+)\\s*=\\s*(.+)$")
        guard let m = regex.firstMatch(in: part, range: NSRange(part.startIndex..<part.endIndex, in: part)),
              m.numberOfRanges >= 3,
              let keyRange = Range(m.range(at: 1), in: part),
              let valRange = Range(m.range(at: 2), in: part) else { return nil }
        return (String(part[keyRange]), String(part[valRange]))
    }

    /// `_bind_composite_params` — param name → raw bound text (positional first, then
    /// `key=value`, then defaults).
    private static func bindCompositeParams(row: Row, comp: CompositeDef) -> [String: String] {
        var values: [String: String] = [:]
        let parts = splitOnDotQuoted(row.settings ?? "")
        let firstPart = parts.first?.trimmingCharacters(in: .whitespaces) ?? ""
        let positionalBound = row.model != nil || (!firstPart.isEmpty && kvParam(firstPart) == nil)
        if let first = comp.params.first, positionalBound {
            values[first.name] = row.model ?? firstPart
        }
        for part in parts {
            if let kv = kvParam(part.trimmingCharacters(in: .whitespaces)) {
                values[kv.0] = kv.1
            }
        }
        for p in comp.params where values[p.name] == nil {
            if let d = p.defaultValue { values[p.name] = d }
        }
        return values
    }

    /// `_substitute_param_tokens` — `{name}` and `(param:name)` in settings/model text.
    private static func substituteParamTokens(_ text: String?, values: [String: String]) -> String? {
        guard let text else { return nil }
        let brace = NSRegularExpression.compiled(#"\{([A-Za-z_][A-Za-z0-9_-]*)\}"#)
        let paren = NSRegularExpression.compiled(#"\((?:param:)([A-Za-z_][A-Za-z0-9_-]*)\)"#)
        var out = text
        out = brace.replace(in: out) { values[$0] ?? $0 }
        out = paren.replace(in: out) { values[$0] ?? $0 }
        return out
    }

    /// `_substitute_composite_row` — drop `(param:name)` refs (fold into settings), bind
    /// `{name}`/`(param:name)` in model/settings, recurse into children.
    private static func substituteCompositeRow(_ row: Row, values: [String: String]) -> Row {
        var newRefs: [Ref] = []
        var extraSettings: [String] = []
        for ref in row.refs {
            if case .paramRef(let name) = ref {
                extraSettings.append(values[name] ?? "")
            } else {
                newRefs.append(ref)
            }
        }
        var settings = substituteParamTokens(row.settings, values: values)
        if !extraSettings.isEmpty {
            let base = extraSettings + (settings.map { [$0] } ?? [])
            settings = base.joined(separator: " · ")
        }
        return Row(id: row.id, task: row.task, blockKind: row.blockKind, blockName: row.blockName,
                   model: substituteParamTokens(row.model, values: values), settings: settings,
                   refs: newRefs, chainBreak: row.chainBreak,
                   children: row.children.map { substituteCompositeRow($0, values: values) },
                   clause: row.clause, tags: row.tags, visitsLeq: row.visitsLeq,
                   onBudget: row.onBudget, declaredSignature: row.declaredSignature,
                   comment: row.comment, leadingComments: row.leadingComments)
    }

    /// `_expand_composite_call` — a composite-calling row becomes an ordinary `<list>` block.
    private static func expandCompositeCall(row: Row, comp: CompositeDef) -> Row {
        let values = bindCompositeParams(row: row, comp: comp)
        let body = comp.rows.map { substituteCompositeRow($0, values: values) }
        return Row(id: row.id, task: row.task, blockKind: .list, blockName: row.task,
                   model: row.model, settings: row.settings, refs: row.refs,
                   chainBreak: row.chainBreak, children: body, clause: row.clause,
                   tags: row.tags, visitsLeq: row.visitsLeq, onBudget: row.onBudget,
                   declaredSignature: row.declaredSignature, comment: row.comment,
                   leadingComments: row.leadingComments)
    }
}

/// A non-throwing `{name}`/`(param:name)` substitution: each match's first capture group is
/// replaced via `replacement` (the group's own text when the map has no entry — the Python's
/// `values.get(name, group(0))`). Replaces right-to-left so earlier ranges stay valid.
private extension NSRegularExpression {
    nonisolated func replace(in string: String, replacement: (String) -> String) -> String {
        var output = string
        let matches = self.matches(in: string, range: NSRange(string.startIndex..<string.endIndex, in: string))
        for match in matches.reversed() {
            guard let fullRange = Range(match.range, in: output) else { continue }
            let group: String
            let g = match.range(at: 1)
            if g.location != NSNotFound, let gr = Range(g, in: output) {
                group = String(output[gr])
            } else {
                group = ""
            }
            output.replaceSubrange(fullRange, with: replacement(group))
        }
        return output
    }
}
