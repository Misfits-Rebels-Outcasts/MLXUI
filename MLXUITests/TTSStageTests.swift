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

    @Test func textSegmentsOnlySplitsOnNewlineRuns() {
        // CFM-FIX-4b/c: the Python splits on `\n+` after `text.strip()` — CR/LF, VT, FF and
        // U+2028/9 are NOT split points (the old `\.isNewline` split them), and leading or
        // trailing whitespace is stripped from the whole text first.
        #expect(KokoroEngine.textSegments("  hello world  ") == ["hello world"])
        #expect(KokoroEngine.textSegments("one\rtwo") == ["one\rtwo"])
        // CRLF still splits at the `\n` (with `\r` kept on the first part, as Python does).
        #expect(KokoroEngine.textSegments("a\r\nb") == ["a\r", "b"])
        #expect(KokoroEngine.textSegments("first\u{2028}second") == ["first\u{2028}second"])
        #expect(KokoroEngine.textSegments("first\nsecond") == ["first", "second"])
    }

    @Test func textSegmentsKeepsSingleParagraphWhole() {
        // A single unbroken paragraph stays one segment.
        let one = "A single paragraph with no newlines at all."
        #expect(KokoroEngine.textSegments(one) == [one])
    }

    // MARK: - Sub-chunking (CFM-FIX-4d)

    @Test func subChunksKeepsShortTextWhole() {
        let short = "A short sentence fits in one chunk."
        #expect(KokoroEngine.subChunks(short) == [short])
    }

    @Test func subChunksSplitsLongTextAtSentences() {
        let text = "First sentence here. Second sentence also here. "
            + "Third sentence keeps going and going and going, making this paragraph long "
            + "enough to need a boundary somewhere in the middle of its run."
        let chunks = KokoroEngine.subChunks(text, maxLength: 40)
        #expect(chunks.count > 1)
        // Nothing is dropped and order is preserved.
        #expect(chunks.joined(separator: " ").trimmingCharacters(in: .whitespaces) == text.trimmingCharacters(in: .whitespaces))
        // Every chunk fits the budget.
        #expect(chunks.allSatisfy { $0.unicodeScalars.count <= 40 })
    }

    // MARK: - Speak row settings → voice (Python's `voice or first_bare or "af_heart"`)

    @Test func speakBareTokenBecomesVoice() throws {
        // The RealExecutor derives a StageConfig from the row settings; the bare token is
        // the Kokoro voice. Assert via the descriptor-driven helper path: Speak + "af_heart".
        let desc = try #require(TaskCatalog.get("Speak"))
        let row = Row(task: "Speak", model: "Kokoro 82M", settings: "af_heart")
        let executor = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                    flowID: "t",
                                    blobDirectory: FileManager.default.temporaryDirectory,
                                    makeModelStage: { _, _ in TTSStubStage() },
                                    installedModelIDs: [],
                                    catalog: [])
        let config = try executor.stageConfig(for: desc, row: row)
        #expect(config.voice == "af_heart")
    }

    @Test func speakVoiceSettingBeatsBareToken() throws {
        let desc = try #require(TaskCatalog.get("Speak"))
        let row = Row(task: "Speak", model: "Kokoro 82M", settings: "af_bella; voice=am_adam")
        let executor = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                    flowID: "t",
                                    blobDirectory: FileManager.default.temporaryDirectory,
                                    makeModelStage: { _, _ in TTSStubStage() },
                                    installedModelIDs: [],
                                    catalog: [])
        let config = try executor.stageConfig(for: desc, row: row)
        #expect(config.voice == "am_adam")
    }

    // MARK: - CFM-R14-FIX: Transcribe row settings → language (the ASR repetition-loop fix)

    @Test func transcribeLangSettingBecomesLanguage() throws {
        // Python's `lang = s.get("lang")`: `Transcribe Whisper Small; lang=en` pins "en".
        let desc = try #require(TaskCatalog.get("Transcribe"))
        let row = Row(task: "Transcribe", model: "Whisper Small", settings: "lang=en")
        let executor = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                    flowID: "t",
                                    blobDirectory: FileManager.default.temporaryDirectory,
                                    makeModelStage: { _, _ in TTSStubStage() },
                                    installedModelIDs: [],
                                    catalog: [])
        let config = try executor.stageConfig(for: desc, row: row)
        #expect(config.language == "en")
    }

    @Test func transcribeWithoutLangDefaultsToEnglish() throws {
        // The app's MLXAudioSTT has no real auto-detection — `nil` makes a weak model loop
        // ("mother mother mother…"). The standalone run sheet defaults to "en" for the same
        // reason; the flow now matches it instead of handing the engine a nil language.
        let desc = try #require(TaskCatalog.get("Transcribe"))
        let row = Row(task: "Transcribe", model: "Whisper Small", settings: "")
        let executor = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
                                    flowID: "t",
                                    blobDirectory: FileManager.default.temporaryDirectory,
                                    makeModelStage: { _, _ in TTSStubStage() },
                                    installedModelIDs: [],
                                    catalog: [])
        let config = try executor.stageConfig(for: desc, row: row)
        #expect(config.language == "en")
    }
}

/// A local text→text stub for RealExecutor config tests (no model needed).
private nonisolated struct TTSStubStage: PipelineStage {
    let id = "stub.tts"
    let name = "Stub TTS"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        return input
    }
}
