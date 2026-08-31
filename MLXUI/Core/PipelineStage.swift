import Foundation

/// One node in a pipeline: declares the media it consumes and produces, and runs
/// a single transform (a model or a tool). Concrete stages own their loaded engine.
///
/// See `Design/pipeline-stage-sketch.md`. Connections are validated by `MediaKind`
/// rather than compile-time generics because the chain is assembled at runtime.
/// `Sendable` (and the `nonisolated` members below) so stages run off the main
/// actor — the app target otherwise defaults to main-actor isolation.
protocol PipelineStage: Identifiable, Sendable {
    nonisolated var id: String { get }            // stable, e.g. the model id
    nonisolated var name: String { get }          // display name
    nonisolated var accepts: MediaKind { get }
    nonisolated var produces: MediaKind { get }

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
///
/// `CustomStringConvertible` so a `String(describing:)` yields a plain sentence — the
/// flow interpreter stringifies a thrown error into the row's `.failed` message, and a
/// raw enum dump is not error voice (CFM-R16-1).
nonisolated enum StageError: Error, CustomStringConvertible {
    case kindMismatch(expected: MediaKind, got: MediaKind)
    case unsupportedModel(id: String, kind: RunnerKind)
    case modelNotInstalled(id: String)
    case insufficientRAM(requiredGB: Double, availableGB: Double)
    case engineFailure(stage: String, underlying: Error)
    case unsupportedSetting(setting: String)

    var description: String {
        switch self {
        case .kindMismatch(let expected, let got):
            return "This step expected \(expected.rawValue) but got \(got.rawValue) — check what feeds it."
        case .unsupportedModel(let id, let kind):
            return "The model '\(id)' isn't runnable as \(kind.rawValue) in this version — install a supported build."
        case .modelNotInstalled(let id):
            return "The model '\(id)' isn't installed yet — install it, then run again."
        case .insufficientRAM(let required, let available):
            return String(format: "This model needs %.2f GB of RAM but this Mac has %.2f GB available.",
                          required, available)
        case .engineFailure(let stage, let underlying):
            let cause = (underlying as? CustomStringConvertible)?.description
                ?? (underlying as NSError).localizedDescription
            if !cause.isEmpty && !cause.contains("couldn't be completed") {
                return "The \(stage) engine failed — \(cause). Check the model files and try again."
            }
            return "The \(stage) engine failed — check the model files and try again."
        case .unsupportedSetting(let setting):
            return "This step's \(setting) setting isn't supported by this engine yet — remove it or use a model that supports it."
        }
    }
}
