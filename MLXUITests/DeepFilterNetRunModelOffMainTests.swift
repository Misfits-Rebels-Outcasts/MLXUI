import Testing
import Foundation
@testable import MLXUI

/// MP5-3g: the Run UI's cleanup must leave the main actor. `SWIFT_APPROACHABLE_CONCURRENCY` makes a `nonisolated async` call run on
/// its caller's actor, so a stage awaited from the `@MainActor` run model's `Task` ran its synchronous window loop on the main
/// thread: beachball, 0% progress, Cancel unclickable (smoke row 125 step 11). These tests drive `DeepFilterNetRunModel.start`
/// with stub stages whose `run` is a synchronous loop, and need no weights.

/// A stage whose `run` is a synchronous window loop (blocking about 50 ms a window, `Task.checkCancellation()` between), the
/// shape of `DeepFilterNetEngine.runWindows`.
nonisolated private final class ThreadLog: @unchecked Sendable {
    private let lock = NSLock()
    private var onMain: [Bool] = []
    func record() { lock.lock(); onMain.append(Thread.isMainThread); lock.unlock() }
    var sawMainThread: Bool { lock.lock(); defer { lock.unlock() }; return onMain.contains(true) }
    var count: Int { lock.lock(); defer { lock.unlock() }; return onMain.count }
}

nonisolated private struct BlockingWindowStage: PipelineStage {
    let log = ThreadLog()
    let id = "blocking"
    let name = "blocking"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }
    let windows: Int

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        for index in 0..<windows {
            try Task.checkCancellation()
            log.record()
            BlockingWindowStage.block(seconds: 0.05)
            progress(Double(index + 1) / Double(windows))
        }
        return input
    }

    nonisolated static func block(seconds: Double) { Thread.sleep(forTimeInterval: seconds) }
}

/// A stage that waits (as `LeasingStage` does for a busy model) and, once its task is cancelled, throws the wait error the way a
/// cancelled `MemoryBudget` wait does. `throwWithoutCancel` throws it at once instead (a wait that ran out).
nonisolated private struct WaitingStage: PipelineStage {
    let id = "waiting"
    let name = "waiting"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }
    var throwWithoutCancel = false

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        if !throwWithoutCancel {
            while !Task.isCancelled { try? await Task.sleep(for: .milliseconds(10)) }
        }
        throw MemoryBudget.WaitError(label: "deepfilternet")
    }
}

@MainActor
struct DeepFilterNetRunModelOffMainTests {

    private func wav(seconds: Int = 1) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("dfn-offmain-\(UUID().uuidString).wav")
        let samples = (0..<(48_000 * seconds)).map { 0.3 * Float(sin(2 * Double.pi * 440 * Double($0) / 48_000)) }
        try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: 48_000), to: url)
        return url
    }

    private func settle(_ model: DeepFilterNetRunModel, timeout: Duration = .seconds(5)) async throws {
        let deadline = ContinuousClock.now + timeout
        while model.isRunning, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
    }

    /// The main actor stays free while the stage's synchronous loop runs: a ticker on it keeps advancing, and progress is
    /// observable before the run ends.
    @Test func theMainActorStaysFreeWhileTheStageLoopRuns() async throws {
        let url = try wav()
        defer { try? FileManager.default.removeItem(at: url) }
        let model = DeepFilterNetRunModel()
        let stage = BlockingWindowStage(windows: 60)   // about 3 s of blocking windows
        model.start(audioURL: url, stage: stage)
        var ticksWhileRunning = 0
        var sawProgress = false
        for _ in 0..<20 {
            try await Task.sleep(for: .milliseconds(50))
            if model.isRunning { ticksWhileRunning += 1 }
            if model.isRunning && model.progress > 0 { sawProgress = true }
        }
        #expect(ticksWhileRunning >= 10, "the main actor ticked \(ticksWhileRunning) of 20 times while the run was going")
        #expect(sawProgress, "progress above 0 must be observable before the run ends")
        model.cancel()
        try await settle(model)
        #expect(stage.log.count > 0 && !stage.log.sawMainThread, "every window ran off the main thread")
    }

    /// Cancel is delivered mid-run and the run ends within a second with no error and no buffers kept.
    @Test func cancelDuringTheStageLoopEndsTheRunQuietly() async throws {
        let url = try wav()
        defer { try? FileManager.default.removeItem(at: url) }
        let model = DeepFilterNetRunModel()
        model.start(audioURL: url, stage: BlockingWindowStage(windows: 200))   // about 10 s if left alone
        for _ in 0..<40 where !(model.isRunning && model.progress > 0) { try await Task.sleep(for: .milliseconds(25)) }
        #expect(model.isRunning && model.progress > 0, "cleaning, with progress")
        let started = ContinuousClock.now
        model.cancel()
        try await settle(model, timeout: .seconds(1))
        #expect(ContinuousClock.now - started < .seconds(1))
        #expect(!model.isRunning && model.errorText == nil && model.original == nil && !model.hasResult)
        #expect(model.progress == 0)
    }

    /// A cancelled run is never an error: a stage that throws the memory wait error once cancelled takes the cancellation branch.
    @Test func aWaitErrorThrownAfterCancelShowsNoError() async throws {
        let url = try wav()
        defer { try? FileManager.default.removeItem(at: url) }
        let model = DeepFilterNetRunModel()
        model.start(audioURL: url, stage: WaitingStage())
        for _ in 0..<100 where model.original == nil { try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.isRunning, "waiting")
        model.cancel()
        try await settle(model, timeout: .seconds(2))
        #expect(!model.isRunning && model.errorText == nil && model.original == nil && !model.hasResult)
    }

    /// The same error with no cancel is still shown: only a cancelled run is silent.
    @Test func aWaitErrorWithoutCancelIsStillShown() async throws {
        let url = try wav()
        defer { try? FileManager.default.removeItem(at: url) }
        let model = DeepFilterNetRunModel()
        model.start(audioURL: url, stage: WaitingStage(throwWithoutCancel: true))
        try await settle(model, timeout: .seconds(2))
        #expect(!model.isRunning)
        #expect(model.errorText == MemoryBudget.waitSentence([]))
    }
}
