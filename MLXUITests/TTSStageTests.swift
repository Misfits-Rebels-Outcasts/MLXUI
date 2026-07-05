import Testing
@testable import MLXUI

/// Covers the TTS stage logic (backlog K1a) via an injected synthesizer — no model
/// download. The real Kokoro path (`KokoroEngine`) is exercised by human runtime smoke.
struct TTSStageTests {
    private func mockStage(
        _ synth: @escaping @Sendable (String) async throws -> AudioBuffer
    ) -> TTSStage {
        TTSStage(id: "tts.test", name: "TTS Test", synthesize: synth)
    }

    @Test func declaresTextToAudio() {
        let stage = mockStage { _ in AudioBuffer(samples: [0], sampleRate: 24000) }
        #expect(stage.accepts == .text)
        #expect(stage.produces == .audio)
    }

    @Test func runSynthesizesAudioFromText() async throws {
        let stage = mockStage { text in
            AudioBuffer(samples: Array(repeating: 0.5, count: text.count), sampleRate: 24000)
        }
        let output = try await stage.run(.text("hello")) { _ in }
        guard case let .audio(buffer) = output else {
            Issue.record("Expected .audio output")
            return
        }
        #expect(buffer.samples.count == 5)
        #expect(buffer.sampleRate == 24000)
    }

    @Test func runReportsProgressToCompletion() async throws {
        let stage = mockStage { _ in AudioBuffer(samples: [0.1], sampleRate: 24000) }
        var last: Double = -1
        _ = try await stage.run(.text("hi")) { last = $0 }
        #expect(last == 1.0)
    }

    @Test func rejectsNonTextInput() async {
        let stage = mockStage { _ in AudioBuffer(samples: [], sampleRate: 24000) }
        await #expect(throws: StageError.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 16000))) { _ in }
        }
    }

    // MARK: - Voice listing (K2)

    @Test func voiceNamesDropsExtensionSortsAndIgnoresNonVoices() {
        let names = KokoroEngine.voiceNames(inDirectoryContents: [
            "am_adam.safetensors", "af_heart.safetensors", "af_heart.pt", "README.md",
        ])
        #expect(names == ["af_heart", "am_adam"])
    }

    @Test func voiceNamesEmptyForNoSafetensors() {
        #expect(KokoroEngine.voiceNames(inDirectoryContents: ["README.md", "config.json"]).isEmpty)
    }
}
