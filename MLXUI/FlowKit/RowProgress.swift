//
//  RowProgress.swift
//  MLXUI — Model Tools, MT1-3b.
//
//  The progress seam. A row run through the *progress entry* (`FlowExecutor.execute(…, progress:)`)
//  can report a short line ("Transcribing…") and an optional fraction from anywhere underneath it
//  by calling `RowProgress.report`. The sink reaches the emitter by `@TaskLocal`, bound for the
//  duration of the progress entry only.
//
//  Flows never use the progress entry: the six-argument `execute` (the protocol requirement) binds
//  nothing, so inside a flow row `RowProgress.report` is a no-op and the runtime is unchanged.
//  A `Task.detached` inside a stage does not inherit the binding (child tasks and `async let` do).
//

import Foundation

nonisolated struct RowProgress: Sendable, Equatable {
    /// 0…1 when known; `nil` = indeterminate.
    let fraction: Double?
    /// A short line for a chip, e.g. "Transcribing…".
    let message: String

    init(message: String, fraction: Double? = nil) {
        self.message = message
        self.fraction = fraction.map { min(1, max(0, $0)) }
    }

    typealias Sink = @Sendable (RowProgress) -> Void

    /// The sink of the progress entry in flight, if any.
    @TaskLocal static var sink: Sink?

    /// Reports to the sink bound by the progress entry; does nothing when none is bound.
    static func report(_ message: String, fraction: Double? = nil) {
        sink?(RowProgress(message: message, fraction: fraction))
    }
}

nonisolated extension FlowExecutor {
    /// The six-argument `execute`, with a progress sink bound for its duration. `progress == nil`
    /// binds nothing and is exactly the six-argument call. Not a protocol requirement: no
    /// conformer changes, and `FlowRunner` keeps calling the requirement.
    func execute(path: String, row: Row, inputs: [Asset],
                 transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?,
                 usedFlowContent: String?,
                 progress: RowProgress.Sink? = nil) async throws -> Asset {
        guard let progress else {
            return try await execute(path: path, row: row, inputs: inputs, transcript: transcript,
                                     context: context, usedFlowContent: usedFlowContent)
        }
        return try await RowProgress.$sink.withValue(progress) {
            try await execute(path: path, row: row, inputs: inputs, transcript: transcript,
                              context: context, usedFlowContent: usedFlowContent)
        }
    }
}
