import Foundation

/// One node in a pipeline: declares the media it consumes and produces, and runs
/// a single transform (a model or a tool). Concrete stages own their loaded engine.
///
/// See `Design/pipeline-stage-sketch.md`. Connections are validated by `MediaKind`
/// rather than compile-time generics because the chain is assembled at runtime.
/// `Sendable` (and the `nonisolated` members below) so stages run off the main
/// actor — the app target otherwise defaults to main-actor isolation.
protocol PipelineStage: Identifiable, Sendable {
    var id: String { get }            // stable, e.g. the model id
    var name: String { get }          // display name
    var accepts: MediaKind { get }
    var produces: MediaKind { get }

    /// Execute the stage. Reports `0...1` progress. Throws `StageError`.
    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media
}

extension PipelineStage {
    /// Shared guard so every `run` starts the same way: reject input whose kind
    /// doesn't match what this stage accepts.
    nonisolated func require(_ input: Media, _ expected: MediaKind) throws {
        guard input.kind == expected else {
            throw StageError.kindMismatch(expected: expected, got: input.kind)
        }
    }
}

/// Errors a stage (or the pipeline that drives it) can throw.
nonisolated enum StageError: Error {
    case kindMismatch(expected: MediaKind, got: MediaKind)
    case unsupportedModel(id: String, kind: RunnerKind)
    case modelNotInstalled(id: String)
    case insufficientRAM(requiredGB: Double, availableGB: Double)
    case engineFailure(stage: String, underlying: Error)
}
