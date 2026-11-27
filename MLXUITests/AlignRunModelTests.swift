import Testing
import Foundation
@testable import MLXUI

/// FA-5 (issue #13 decision 7): the Run screen's view-model — Stop, and the two refusals it makes before any work.
/// No weights: the stage is a stub.
@MainActor
struct AlignRunModelTests {

    private func wavURL(seconds: Double = 1.0) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fa-model-\(UUID().uuidString).wav")
        let samples = (0..<Int(16_000 * seconds)).map { Float(sin(Double($0) / 16_000.0 * 2 * .pi * 440) * 0.2) }
        try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: 16_000), to: url)
        return url
    }

    /// A stage that takes long enough to be stopped, and records that it was entered.
    private struct SlowStage: PipelineStage, @unchecked Sendable {
        let calls = AlignRunCallLog()
        var id: String { "slow-align-stub" }
        var name: String { "Qwen3 Aligner 0.6B" }
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            calls.enter()
            try await Task.sleep(for: .seconds(30))
            return .text("text,start,end\nhello,0.000,0.500\n")
        }
    }

    @Test func stopCancelsTheRunWithoutAnError() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = SlowStage()
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "hello", language: "en", stage: stage)
        // Wait until the stage is actually running, then press Stop (the button's call).
        for _ in 0..<200 where stage.calls.count == 0 {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(stage.calls.count == 1, "the run reached the stage")
        model.stop()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(!model.isRunning, "Stop ends the run")
        #expect(model.errorText == nil, "a stopped run is never an error")
        #expect(model.result == nil)
    }

    @Test func anEmptyTranscriptIsRefusedBeforeTheStageIsBuilt() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = SlowStage()
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: " \n\t ", language: "en", stage: stage)
        #expect(!model.isRunning)
        #expect(model.errorText == ForcedAlignerEngine.EngineError.emptyTranscript.errorDescription)
        #expect(stage.calls.count == 0)
    }

    @Test func anUnsupportedLanguageIsRefusedBeforeTheStageIsBuilt() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = SlowStage()
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "hello", language: "xx", stage: stage)
        #expect(!model.isRunning)
        #expect(model.errorText?.contains("xx") == true)
        #expect(stage.calls.count == 0)
    }

    /// An audio-in, fixed-CSV-out stage (the real stage is `ForcedAlignerStage`).
    private struct TextStubStage: PipelineStage, @unchecked Sendable {
        let csv: String
        var id: String { "align-text-stub" }
        var name: String { "Qwen3 Aligner 0.6B" }
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            progress(1)
            return .text(csv)
        }
    }

    @Test func aSuccessfulRunFoldsTheCSVBackIntoRows() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = TextStubStage(csv: "text,start,end\nHello,0.100,0.500\nworld,0.500,1.000\n")
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "Hello world", language: "en", stage: stage)
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(25))
        }
        let result = try #require(model.result)
        #expect(result.units.map(\.text) == ["Hello", "world"])
        #expect(abs(result.duration - 1.0) < 0.01)
        #expect(result.transcript == "Hello world")
        #expect(model.errorText == nil)
    }

    /// The Export SRT/VTT buttons end cues at the sentences of the transcript the run aligned.
    @Test func theExportsUseTheRunsTranscript() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = TextStubStage(csv: "text,start,end\nBonjour,0.100,0.400\nÀ,0.400,0.600\ndemain,0.600,0.900\n")
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: " Bonjour.\nÀ demain ! ", language: "fr", stage: stage)
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(25))
        }
        let result = try #require(model.result)
        #expect(result.transcript == "Bonjour. À demain !", "the text the run aligned")
        #expect(ForcedAlignerRunView.ExportFormat.srt.render(result)
                == "1\n00:00:00,100 --> 00:00:00,400\nBonjour\n\n2\n00:00:00,400 --> 00:00:00,900\nÀ demain\n")
        #expect(ForcedAlignerRunView.ExportFormat.vtt.render(result)
                == "WEBVTT\n\n00:00:00.100 --> 00:00:00.400\nBonjour\n\n00:00:00.400 --> 00:00:00.900\nÀ demain\n")
        #expect(ForcedAlignerRunView.ExportFormat.csv.render(result) == AlignmentExport.csv(result.units))
    }

    // MARK: - A late Stop, a replaced run, repeated rows

    /// A stage that ignores cancellation, as the engine's synchronous `generate` does: it waits for the test to release
    /// it, reports progress, then returns its CSV (or fails).
    private struct GatedStage: PipelineStage, @unchecked Sendable {
        let csv: String
        var fails = false
        let gate = AlignRunGate()
        var id: String { "align-gated-stub" }
        var name: String { "Qwen3 Aligner 0.6B" }
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            await gate.wait()
            progress(0.4)
            if fails { throw StageError.engineFailure(stage: "Forced aligner", underlying: CocoaError(.fileNoSuchFile)) }
            return .text(csv)
        }
    }

    private func waitUntilEntered(_ gate: AlignRunGate) async throws {
        for _ in 0..<200 where !gate.isEntered {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(gate.isEntered, "the run reached the stage")
    }

    /// Releases a gated run and waits for its task to end, then for any progress hop it queued.
    private func release(_ gate: AlignRunGate, _ task: Task<Void, Never>) async throws {
        gate.open()
        await task.value
        try await Task.sleep(for: .milliseconds(100))
    }

    private static let oldCSV = "text,start,end\nalt,0.100,0.500\n"
    private static let newCSV = "text,start,end\nneu,0.200,0.600\n"

    @Test func aStopDuringTheForwardPassDropsTheResult() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = GatedStage(csv: Self.oldCSV)
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: stage)
        try await waitUntilEntered(stage.gate)
        let task = try #require(model.task)
        model.stop()
        try await release(stage.gate, task)
        #expect(model.result == nil, "a stopped run shows no result")
        #expect(model.errorText == nil, "a stopped run is never an error")
        #expect(!model.isRunning)
        #expect(model.progress == 0)
    }

    @Test func aReplacedRunDoesNotOverwriteTheNewerResult() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let old = GatedStage(csv: Self.oldCSV)
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: old)
        try await waitUntilEntered(old.gate)
        let oldTask = try #require(model.task)
        model.start(audioURL: url, transcript: "neu", language: "de", stage: TextStubStage(csv: Self.newCSV))
        await model.task?.value
        #expect(model.result?.units.map(\.text) == ["neu"])
        try await release(old.gate, oldTask)
        #expect(model.result?.units.map(\.text) == ["neu"])
        #expect(model.result?.transcript == "neu")
        #expect(model.errorText == nil)
        #expect(!model.isRunning)
        #expect(model.progress == 1)
    }

    @Test(arguments: [false, true])
    func aReplacedRunLeavesTheRunningOneAlone(oldRunFails: Bool) async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let old = GatedStage(csv: Self.oldCSV, fails: oldRunFails)
        let new = GatedStage(csv: Self.newCSV)
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: old)
        try await waitUntilEntered(old.gate)
        let oldTask = try #require(model.task)
        model.start(audioURL: url, transcript: "neu", language: "de", stage: new)
        try await waitUntilEntered(new.gate)
        let newTask = try #require(model.task)
        try await release(old.gate, oldTask)
        #expect(model.isRunning, "the newer run is still going")
        #expect(model.result == nil)
        #expect(model.errorText == nil)
        #expect(model.progress == 0)
        try await release(new.gate, newTask)
        #expect(model.result?.units.map(\.text) == ["neu"])
        #expect(!model.isRunning)
    }

    /// The library's neighbour fill can give a repeated word equal times; the table's rows still need their own ids.
    @Test func repeatedUnitsGetDistinctRowIDs() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = TextStubStage(csv: "text,start,end\nthe,0.080,0.080\nthe,0.080,0.080\n")
        let model = AlignRunModel()
        model.start(audioURL: url, transcript: "the the", language: "en", stage: stage)
        await model.task?.value
        let result = try #require(model.result)
        #expect(result.units.count == 2)
        #expect(Set(result.rows.map(\.id)).count == 2, "the Table's ids are unique")
        #expect(result.rows.map(\.unit) == result.units)
    }
}

/// Holds a stub stage until the test opens it; deliberately deaf to cancellation.
nonisolated final class AlignRunGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var entered = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    var isEntered: Bool { lock.lock(); defer { lock.unlock() }; return entered }

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            entered = true
            let resumeNow = opened
            if !resumeNow { waiters.append(continuation) }
            lock.unlock()
            if resumeNow { continuation.resume() }
        }
    }

    func open() {
        lock.lock()
        opened = true
        let pending = waiters
        waiters = []
        lock.unlock()
        for continuation in pending { continuation.resume() }
    }
}

nonisolated final class AlignRunCallLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = 0
    func enter() { lock.lock(); entered += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return entered }
}
