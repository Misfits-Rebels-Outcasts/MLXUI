import Foundation

/// Progress signal emitted as a pipeline runs, so a UI can drive a per-stage view.
nonisolated enum PipelineEvent: Sendable {
    case stageStarted(index: Int, name: String)
    case stageProgress(index: Int, fraction: Double)
    case stageFinished(index: Int, produced: MediaKind)
    case failed(index: Int, error: StageError)
}

/// An ordered chain of stages. Threads each stage's output into the next, validating
/// that adjacent kinds line up. See `Design/pipeline-stage-sketch.md` §the pipeline.
nonisolated struct Pipeline {
    var stages: [any PipelineStage]

    init(stages: [any PipelineStage]) {
        self.stages = stages
    }

    /// Static check: each stage must consume what the previous one produced. Rejects a
    /// malformed chain (e.g. TTS→ASR with the kinds swapped) *before* any model loads.
    func validate() throws {
        for (a, b) in zip(stages, stages.dropFirst()) where a.produces != b.accepts {
            throw StageError.kindMismatch(expected: b.accepts, got: a.produces)
        }
    }

    /// Run end-to-end, threading each stage's output into the next and emitting
    /// per-stage events. Throws (after emitting `.failed`) if a stage fails.
    func run(
        _ input: Media,
        onEvent: @Sendable @escaping (PipelineEvent) -> Void
    ) async throws -> Media {
        try validate()
        var current = input
        for (i, stage) in stages.enumerated() {
            onEvent(.stageStarted(index: i, name: stage.name))
            do {
                current = try await stage.run(current) { fraction in
                    onEvent(.stageProgress(index: i, fraction: fraction))
                }
                onEvent(.stageFinished(index: i, produced: current.kind))
            } catch let error as StageError {
                onEvent(.failed(index: i, error: error))
                throw error
            }
        }
        return current
    }
}
