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
        #expect(model.errorText == nil)
    }
}

nonisolated final class AlignRunCallLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = 0
    func enter() { lock.lock(); entered += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return entered }
}
