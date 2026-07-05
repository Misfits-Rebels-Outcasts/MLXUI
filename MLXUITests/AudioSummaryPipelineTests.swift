import Testing
import Foundation
@testable import MLXUI

/// M11 integration: the full MVP chain (ReadAudio → Resample → ASR → Template → LLM →
/// SaveText) runs end-to-end with mock ASR + LLM (no model downloads), reading a real
/// WAV and writing a real summary file. See RSI/evals/eval-plan.md G2.
struct AudioSummaryPipelineTests {

    /// Write a tiny silent WAV so `ReadAudioStage` has a real file to read.
    private func makeTempWAV() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("in-\(UUID().uuidString).wav")
        try AudioWriter.writeWAV(
            AudioBuffer(samples: [Float](repeating: 0, count: 1_600), sampleRate: 16_000),
            to: url)
        return url
    }

    @Test func chainTranscribesTemplatesSummarizesAndSaves() async throws {
        let audioURL = try makeTempWAV()
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("out-\(UUID().uuidString).txt")
        defer {
            try? FileManager.default.removeItem(at: audioURL)
            try? FileManager.default.removeItem(at: outURL)
        }

        let mockASR = ASRStage(id: "asr", name: "Mock ASR") { _ in "the meeting transcript" }
        let mockLLM = LLMStage(id: "llm", name: "Mock LLM", systemPrompt: nil) { prompt in
            "SUMMARY<<\(prompt)>>"
        }
        let pipeline = AudioSummaryPipeline.pipeline(saveTo: outURL, asr: mockASR, llm: mockLLM)

        let result = try await pipeline.run(.text(audioURL.path)) { _ in }

        let expectedPrompt =
            "Summarize the following transcript in 3 bullet points:\n\nthe meeting transcript"
        guard case let .text(summary) = result else { Issue.record("expected text"); return }
        #expect(summary == "SUMMARY<<\(expectedPrompt)>>")
        #expect(try String(contentsOf: outURL, encoding: .utf8) == summary)
    }

    @Test func chainValidatesAcrossAllSixStages() throws {
        let asr = ASRStage(id: "asr", name: "Mock") { _ in "" }
        let llm = LLMStage(id: "llm", name: "Mock", systemPrompt: nil) { $0 }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("x.txt")
        // ReadAudio(.text→.audio) · Resample(.audio→.audio) · ASR(.audio→.text) ·
        // Template(.text→.text) · LLM(.text→.text) · Save(.text→.text) — all adjacent kinds line up.
        try AudioSummaryPipeline.pipeline(saveTo: url, asr: asr, llm: llm).validate()
    }

    @Test func chainEmitsSixStageStarts() async throws {
        let audioURL = try makeTempWAV()
        let outURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("out-\(UUID().uuidString).txt")
        defer {
            try? FileManager.default.removeItem(at: audioURL)
            try? FileManager.default.removeItem(at: outURL)
        }
        let asr = ASRStage(id: "asr", name: "Mock") { _ in "t" }
        let llm = LLMStage(id: "llm", name: "Mock", systemPrompt: nil) { $0 }
        let starts = Counter()
        _ = try await AudioSummaryPipeline.pipeline(saveTo: outURL, asr: asr, llm: llm)
            .run(.text(audioURL.path)) { if case .stageStarted = $0 { starts.bump() } }
        #expect(starts.count == 6)
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    func bump() { lock.lock(); n += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return n }
}
