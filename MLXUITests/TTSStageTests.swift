import Testing
import Foundation
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

    // MARK: - Long-text segmentation (the Python's split_pattern=r"\n+")

    @Test func textSegmentsSplitsOnNewlines() {
        #expect(KokoroEngine.textSegments("first\nsecond\nthird") == ["first", "second", "third"])
    }

    @Test func textSegmentsDropsBlankLines() {
        #expect(KokoroEngine.textSegments("first\n\n\nsecond") == ["first", "second"])
    }

    @Test func textSegmentsKeepsSingleParagraphWhole() {
        // A single unbroken paragraph stays one segment — the Python would refuse it too
        // if it exceeded the model's cap; we never silently truncate.
        let one = "A single paragraph with no newlines at all."
        #expect(KokoroEngine.textSegments(one) == [one])
    }

    // MARK: - Speak row settings → voice (Python's `voice or first_bare or "af_heart"`)

    @Test func speakBareTokenBecomesVoice() {
        // The RealExecutor derives a StageConfig from the row settings; the bare token is
        // the Kokoro voice. Assert via the descriptor-driven helper path: Speak + "af_heart".
        let desc = try! #require(TaskCatalog.get("Speak"))
        let row = Row(task: "Speak", model: "Kokoro 82M", settings: "af_heart")
        let executor = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                    flowID: "t",
                                    blobDirectory: FileManager.default.temporaryDirectory,
                                    makeModelStage: { _, _ in TTSStubStage() },
                                    installedModelIDs: [],
                                    catalog: [])
        let config = executor.stageConfig(for: desc, row: row)
        #expect(config.voice == "af_heart")
    }

    @Test func speakVoiceSettingBeatsBareToken() {
        let desc = try! #require(TaskCatalog.get("Speak"))
        let row = Row(task: "Speak", model: "Kokoro 82M", settings: "af_bella; voice=am_adam")
        let executor = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                    flowID: "t",
                                    blobDirectory: FileManager.default.temporaryDirectory,
                                    makeModelStage: { _, _ in TTSStubStage() },
                                    installedModelIDs: [],
                                    catalog: [])
        let config = executor.stageConfig(for: desc, row: row)
        #expect(config.voice == "am_adam")
    }
}

/// A local text→text stub for RealExecutor config tests (no model needed).
private struct TTSStubStage: PipelineStage {
    let id = "stub.tts"
    let name = "Stub TTS"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        return input
    }
}
