import Testing
import Foundation
@testable import MLXUI

/// Covers M8 `ASRStage` with a mock transcriber (no model download): the kind contract,
/// transcript passthrough, internal resample-to-16k, and non-audio rejection.
/// See RSI/evals/eval-plan.md G2.
struct ASRStageTests {

    @Test func stageAcceptsAudioProducesText() {
        let stage = ASRStage(id: "m", name: "Mock") { _ in "" }
        #expect(stage.accepts == .audio)
        #expect(stage.produces == .text)
    }

    @Test func runReturnsTranscribedText() async throws {
        let stage = ASRStage(id: "m", name: "Mock") { _ in "hello world" }
        let out = try await stage.run(
            .audio(AudioBuffer(samples: [0, 0, 0], sampleRate: 16_000))) { _ in }
        guard case let .text(s) = out else { Issue.record("expected text"); return }
        #expect(s == "hello world")
    }

    @Test func runResamplesTo16kBeforeTranscribing() async throws {
        // Feed 32 kHz; the transcriber should receive ~16 kHz worth of samples.
        let captured = SampleCountBox()
        let stage = ASRStage(id: "m", name: "Mock") { samples in
            captured.set(samples.count)
            return "ok"
        }
        let input = Media.audio(AudioBuffer(samples: [Float](repeating: 0, count: 32_000),
                                            sampleRate: 32_000))
        _ = try await stage.run(input) { _ in }
        let n = captured.value ?? 0
        #expect(n > 14_000 && n < 18_000)   // ~half of 32k after 32k→16k
    }

    @Test func runRejectsNonAudioInput() async {
        let stage = ASRStage(id: "m", name: "Mock") { _ in "" }
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.text("not audio")) { _ in }
        }
    }
}

/// Thread-safe holder for the count captured inside the `@Sendable` transcriber.
private final class SampleCountBox: @unchecked Sendable {
    private let lock = NSLock()
    private var n: Int?
    func set(_ value: Int) { lock.lock(); n = value; lock.unlock() }
    var value: Int? { lock.lock(); defer { lock.unlock() }; return n }
}
