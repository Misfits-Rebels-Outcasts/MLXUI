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
}

/// Executes one row's transform given its gathered inputs — the seam `FlowRunner` runs
/// against, mirroring `catflow-mlx/src/catflow/engines/mock.py::MockExecutor.execute` (and,
/// later, `engines/real.py`'s real dispatch). The linear runner never decides *how* a row
/// runs, only *what feeds it* and in what order events fire.
nonisolated protocol FlowExecutor: Sendable {
    /// Run `row`, whose gathered inputs are `inputs`. `path` is the row's dotted path
    /// (e.g. `"1"`, `"2"` — the Python's fingerprint identity). Throws a `FlowError`.
    func execute(path: String, row: Row, inputs: [Asset]) async throws -> Asset
}

/// The execution engine for the **linear subset** (CFM-R2-6): straight-line rows top to
/// bottom, auto-chain (a row with no explicit reference takes the previous row's output),
/// explicit `(N)` / `(N,M)` references, and chain breaks. Everything else is refused by
/// `canRun`. The full interpreter port (R6) replaces this behind the same `FlowEvent` API.
///
/// Stateless: per-run state lives in an explicit `RunContext` value (the documented
/// divergence from the Python's thread-locals), never in this type. A **task** is the unit
/// of execution (each run creates one); cancellation is Swift `Task` cancellation.
nonisolated struct FlowRunner {

    /// Everything the linear runner needs per run, passed explicitly rather than stored on
    /// the runner — the `RunContext` divergence (SPEC-Q): `last_frame`/`last_tag`/
    /// `last_duration_ms` thread-locals become this value, passed alongside.
    struct RunContext: Sendable {
        let flowID: String
        let workspace: FlowWorkspace
        let blobDirectory: URL
        /// The executor that turns rows into assets (mock in CI, real in the app).
        let executor: any FlowExecutor
        /// True when running off the main actor's registry (real dispatch); the linear
        /// runner itself never touches the registry — the executor does.
        var realism: String = "mock"

        init(flowID: String, workspace: FlowWorkspace, blobDirectory: URL,
             executor: any FlowExecutor) {
            self.flowID = flowID
            self.workspace = workspace
            self.blobDirectory = blobDirectory
            self.executor = executor
        }
    }

    /// Whether `doc` uses only the linear subset. Every out-of-scope feature produces a
    /// named refusal, not a wrong answer.
    static func canRun(_ doc: FlowDocument) -> Runnability {
        var indexByID: [UUID: Int] = [:]
        for (i, row) in doc.rows.enumerated() { indexByID[row.id] = i + 1 }

        for (i, row) in doc.rows.enumerated() {
            let n = i + 1

            if row.blockKind != nil {
                return .notRunnable(reason:
                    "Row \(n) is a \(row.blockKind?.rawValue ?? "block"), which needs a newer version of Flows to run.")
            }
            if row.clause != nil {
                return .notRunnable(reason:
                    "Row \(n) uses a continuation clause, which needs a newer version of Flows to run.")
            }
            for ref in row.refs {
                if case .inputRef = ref {
                    return .notRunnable(reason:
                        "Row \(n) references a block input, which needs a newer version of Flows to run.")
                }
            }
            if row.task == nil {
                return .notRunnable(reason:
                    "Row \(n) is a block with no task, which needs a newer version of Flows to run.")
            }
            guard let desc = TaskCatalog.get(row.task!) else {
                return .notRunnable(reason:
                    "Row \(n) uses '\(row.task!)', which isn't a task this version of Flows knows.")
            }
            switch desc.taskClass {
            case .instant, .model:
                break   // in scope
            case .human, .trigger, .staged, .net, .agent:
                return .notRunnable(reason:
                    "Row \(n) is a \(desc.taskClass.rawValue) row, which needs a newer version of Flows to run.")
            }
        }
        return .runnable
    }

    /// Run `doc` with the linear engine. The caller drains the returned stream; each row
    /// emits `.started` → (`.progress`…) → `.finished` or `.failed`, and the run halts on
    /// the first failure (no later `.started`s). A cancelled `Task` stops the stream.
    func run(_ doc: FlowDocument, context: RunContext) -> AsyncStream<FlowEvent> {
        AsyncStream { continuation in
            let task = Task {
                do {
                    try await Self.executeLinear(doc, context: context, continuation: continuation)
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish()
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    private static func executeLinear(_ doc: FlowDocument, context: RunContext,
                                      continuation: AsyncStream<FlowEvent>.Continuation) async {
        var outputs: [UUID: Asset] = [:]
        var previousOutput: Asset?
        let rows = doc.rows

        for (i, row) in rows.enumerated() {
            // Every per-row failure — setup or stage — yields `.failed` and halts; nothing
            // throws out of the loop, so no later `.started` can follow a failure.
            do {
                guard TaskCatalog.get(row.task ?? "") != nil else {
                    throw FlowError.unknownTask(row: row.task ?? "?")
                }
                continuation.yield(.queued(rowID: row.id))

                // Gather inputs: explicit refs (in order), else auto-chain the previous
                // row's output unless this row starts a new chain.
                var inputs: [Asset] = []
                if !row.refs.isEmpty {
                    for ref in row.refs {
                        guard case .rowRef(let id) = ref else {
                            throw FlowError.unsupportedReference(row: String(i + 1))
                        }
                        guard let asset = outputs[id] else {
                            throw FlowError.referenceNotFound(row: String(i + 1))
                        }
                        inputs.append(asset)
                    }
                } else if !row.chainBreak, let previousOutput {
                    inputs = [previousOutput]
                }

                continuation.yield(.started(rowID: row.id))

                // Dispatch (instant tool vs frame-backed model vs plain model) is the
                // **executor's** job — the linear runner only feeds inputs and relays
                // events. MockExecutor ignores frames (kind-driven content); the real
                // executor (R2-8) renders the frame and drives the engine.
                let asset: Asset
                do {
                    asset = try await context.executor.execute(path: "\(i + 1)", row: row, inputs: inputs)
                } catch let error as FlowError {
                    continuation.yield(.failed(rowID: row.id, error))
                    return
                } catch {
                    // A non-FlowError (e.g. a `StageError` from an engine). Route it through
                    // the exhaustive `FlowErrorDisplay` mapping so the user sees the real
                    // sentence, never `error.localizedDescription` ("MLXUI.StageError error 4").
                    continuation.yield(.failed(rowID: row.id,
                                               FlowError.stageFailure(row: String(i + 1),
                                                                      message: FlowErrorDisplay.sentence(for: error))))
                    return
                }

                outputs[row.id] = asset
                previousOutput = asset
                continuation.yield(.finished(rowID: row.id, asset))
            } catch let error as FlowError {
                continuation.yield(.failed(rowID: row.id, error))
                return
            } catch {
                continuation.yield(.failed(rowID: row.id,
                                           FlowError.stageFailure(row: String(i + 1),
                                                                  message: error.localizedDescription)))
                return
            }
        }
    }
}
