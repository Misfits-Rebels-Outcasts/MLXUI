import Testing
import Foundation
@testable import MLXUI

/// Covers M4 `Pipeline`: `validate()` adjacency checks, end-to-end threading, the
/// per-stage event sequence, and failure propagation. See RSI/evals/eval-plan.md G2.
struct PipelineTests {

    // MARK: validate()

    @Test func validatePassesForCompatibleChain() throws {
        let p = Pipeline(stages: [
            StubStage.textToText(name: "A", suffix: "-a"),
            StubStage.textToText(name: "B", suffix: "-b"),
        ])
        try p.validate()   // must not throw
    }

    @Test func validateThrowsOnKindMismatch() {
        // A produces .audio but B accepts .text → invalid adjacency.
        let p = Pipeline(stages: [
            StubStage(name: "A", accepts: .text, produces: .audio) { _ in
                .audio(AudioBuffer(samples: [], sampleRate: 16_000))
            },
            StubStage.textToText(name: "B", suffix: "-b"),
        ])
        #expect(throws: StageError.self) { try p.validate() }
    }

    // MARK: run

    @Test func runThreadsOutputThroughStagesAndReturnsFinal() async throws {
        let p = Pipeline(stages: [
            StubStage.textToText(name: "A", suffix: "-a"),
            StubStage.textToText(name: "B", suffix: "-b"),
        ])
        let out = try await p.run(.text("x")) { _ in }
        guard case let .text(s) = out else { Issue.record("expected text"); return }
        #expect(s == "x-a-b")
    }

    @Test func runEmitsStartedAndFinishedForEachStageInOrder() async throws {
        let p = Pipeline(stages: [
            StubStage.textToText(name: "A", suffix: "-a"),
            StubStage.textToText(name: "B", suffix: "-b"),
        ])
        let events = EventBox()
        _ = try await p.run(.text("x")) { events.record($0) }

        // Two starts and two finishes, indices in order; at least one progress event.
        #expect(events.startedIndices == [0, 1])
        #expect(events.finishedIndices == [0, 1])
        #expect(events.hasProgress)
        #expect(events.failedIndices.isEmpty)
    }

    @Test func runEmitsFailedAndRethrowsWhenAStageFails() async {
        let p = Pipeline(stages: [
            StubStage.textToText(name: "A", suffix: "-a"),
            StubStage(name: "B", accepts: .text, produces: .text) { _ in
                throw StageError.engineFailure(stage: "B", underlying: StubError.boom)
            },
        ])
        let events = EventBox()
        await #expect(throws: StageError.self) {
            _ = try await p.run(.text("x")) { events.record($0) }
        }
        #expect(events.failedIndices == [1])   // failure reported for stage 1
    }

    @Test func runOnEmptyPipelineReturnsInputUnchanged() async throws {
        let out = try await Pipeline(stages: []).run(.text("same")) { _ in }
        guard case let .text(s) = out else { Issue.record("expected text"); return }
        #expect(s == "same")
    }
}

// MARK: - Test doubles

private enum StubError: Error { case boom }

/// A configurable stage: validates input kind, reports progress, applies `transform`.
private nonisolated struct StubStage: PipelineStage {
    let id: String
    let name: String
    let accepts: MediaKind
    let produces: MediaKind
    let transform: @Sendable (Media) throws -> Media

    init(
        name: String,
        accepts: MediaKind,
        produces: MediaKind,
        transform: @escaping @Sendable (Media) throws -> Media
    ) {
        self.id = name
        self.name = name
        self.accepts = accepts
        self.produces = produces
        self.transform = transform
    }

    static func textToText(name: String, suffix: String) -> StubStage {
        StubStage(name: name, accepts: .text, produces: .text) { input in
            guard case let .text(t) = input else { return input }
            return .text(t + suffix)
        }
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, accepts)
        progress(0.5)
        let output = try transform(input)
        progress(1.0)
        return output
    }
}

/// Thread-safe collector for the `@Sendable` event callback.
private final class EventBox: @unchecked Sendable {
    private let lock = NSLock()
    private var events: [PipelineEvent] = []

    func record(_ e: PipelineEvent) { lock.lock(); events.append(e); lock.unlock() }

    private func snapshot() -> [PipelineEvent] { lock.lock(); defer { lock.unlock() }; return events }

    var startedIndices: [Int] {
        snapshot().compactMap { if case let .stageStarted(i, _) = $0 { return i } else { return nil } }
    }
    var finishedIndices: [Int] {
        snapshot().compactMap { if case let .stageFinished(i, _) = $0 { return i } else { return nil } }
    }
    var failedIndices: [Int] {
        snapshot().compactMap { if case let .failed(i, _) = $0 { return i } else { return nil } }
    }
    var hasProgress: Bool {
        snapshot().contains { if case .stageProgress = $0 { return true } else { return false } }
    }
}
