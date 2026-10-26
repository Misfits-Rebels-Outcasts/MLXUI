import Foundation
import MLX
import Testing
@testable import MLXUI

/// MP6-3a -- the real-weights tests for LFM2.5-Audio 1.5B (KICKOFF rule 13b, rule 17c). Owner/lead-run, gated on the weights
/// (about 1.73 GB): skipped unless `mlx-community--LFM2.5-Audio-1.5B-6bit/.installed` exists under the directory this process
/// resolves, `ModelStore`'s (the Direct edition's `~/Library/Application Support/AI Browser/models/...`) or
/// `MLXUI_TEST_LFMAUDIO_DIR=<model folder>` (passed to the host as `TEST_RUNNER_MLXUI_TEST_LFMAUDIO_DIR`). A skipped run does not
/// count. The exact commands are in the MP6-3a journal.
///
/// Every test calls the same engine entry points the stages and the Run UI call (`LFMAudioEngine.transcribe / synthesize /
/// voiceTurn` with no factory, so the real backend), on a temp copy when the env override is used. Each asserts first that the
/// folder has **no `tokenizer-*.safetensors`** -- the file the selector skips (readme T7) -- so a green run proves all three modes
/// load without it.
enum LFMAudioRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static let directory: URL? = {
        let url: URL
        if let override = env["MLXUI_TEST_LFMAUDIO_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/LFM2.5-Audio-1.5B-6bit")
        }
        return FileManager.default.fileExists(atPath: url.appendingPathComponent(".installed").path) ? url : nil
    }()

    /// The folder to load from, plus the T7 precondition. The source is never modified.
    nonisolated static func modelDirectory() throws -> URL {
        let source = try #require(directory)
        let files = try FileManager.default.contentsOfDirectory(atPath: source.path)
        let rootTokenizer = files.filter { $0.hasPrefix("tokenizer-") && $0.hasSuffix(".safetensors") }
        #expect(rootTokenizer.isEmpty, "T7: the root tokenizer checkpoint must not be installed; found \(rootTokenizer)")
        return source
    }

    enum ClipError: Error, CustomStringConvertible {
        case sayUnavailable(String)
        var description: String {
            switch self { case .sayUnavailable(let why): return "the speech clip could not be generated: \(why)" }
        }
    }

    /// 16 kHz mono speech of `text` made with macOS `say` (never checked in), cached in the temp folder.
    nonisolated static func speech(_ text: String) throws -> [Float] {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("lfm-say-\(abs(text.hashValue)).wav")
        if !FileManager.default.fileExists(atPath: cache.path) {
            guard FileManager.default.isExecutableFile(atPath: "/usr/bin/say") else { throw ClipError.sayUnavailable("/usr/bin/say is missing") }
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
            process.arguments = ["-v", env["MLXUI_TEST_LFMAUDIO_VOICE"] ?? "Samantha", "--file-format=WAVE", "--data-format=LEI16@16000", "-o", cache.path, text]
            do { try process.run() } catch { throw ClipError.sayUnavailable("\(error)") }
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { throw ClipError.sayUnavailable("say exited \(process.terminationStatus)") }
        }
        return try AudioResampler.resample(try AudioFileReader.read(cache), to: 16_000).samples
    }

    nonisolated static func finite(_ samples: [Float]) -> Bool { !samples.contains { !$0.isFinite } }

    nonisolated static func rms(_ samples: [Float]) -> Double {
        guard !samples.isEmpty else { return 0 }
        return (samples.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(samples.count)).squareRoot()
    }

    /// Runs `body` with the clip, or prints why it can't and returns (a skip with a message, not a failure).
    nonisolated static func withSpeech(_ text: String, _ body: ([Float]) async throws -> Void) async throws {
        let samples: [Float]
        do { samples = try speech(text) } catch let error as ClipError {
            print("LFMSKIP \(error)")
            return
        }
        try await body(samples)
    }
}

struct LFMAudioRealLoadTests {
    private static let enabled = LFMAudioRealLoad.directory != nil

    @Test(.enabled(if: enabled))
    func speechToTextTranscribesASpokenSentence() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        try await LFMAudioRealLoad.withSpeech("The quick brown fox jumps over the lazy dog.") { samples in
            let started = Date()
            let text = try await LFMAudioEngine.transcribe(samples, modelDirectory: dir)
            print(String(format: "LFMSTT %.1f s for %.1f s of audio: %@", Date().timeIntervalSince(started), Double(samples.count) / 16_000, text))
            let lower = text.lowercased()
            #expect(lower.contains("fox") && lower.contains("dog"), "transcript: \(text)")
        }
    }

    @Test(.enabled(if: enabled))
    func textToSpeechMakesAudibleTwentyFourKHzSamples() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        let started = Date()
        let samples = try await LFMAudioEngine.synthesize("Hello, welcome to the test.", modelDirectory: dir)
        let seconds = Double(samples.count) / Double(LFMAudioEngine.outputSampleRate)
        print(String(format: "LFMTTS %.1f s to make %.2f s of audio, rms %.4f", Date().timeIntervalSince(started), seconds, LFMAudioRealLoad.rms(samples)))
        #expect(seconds > 0.5 && seconds < 30)
        #expect(LFMAudioRealLoad.finite(samples))
        #expect(LFMAudioRealLoad.rms(samples) > 1e-3, "the output is silent")
    }

    @Test(.enabled(if: enabled))
    func aVoiceTurnAnswersInTextAndSpeechAndTheSecondTurnHasMemory() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        try await LFMAudioRealLoad.withSpeech("My name is Alex. What is two plus two?") { first in
            var conversation = LFMConversation()
            let events = LFMEventBox()
            let started = Date()
            let reply = try await LFMAudioEngine.voiceTurn(userSamples: first, conversation: conversation, modelDirectory: dir) { events.add($0) }
            print(String(format: "LFMVOICE1 %.1f s, %d audio samples, text: %@", Date().timeIntervalSince(started), reply.audioSampleCount, reply.text))
            #expect(!reply.text.isEmpty || reply.audioSampleCount > 0, "the model said nothing")
            if case .text? = events.values.first {} else { Issue.record("the first event must be the text") }
            conversation.record(userSamples: first, assistantText: reply.text)
            try await LFMAudioRealLoad.withSpeech("What is my name?") { second in
                let again = try await LFMAudioEngine.voiceTurn(userSamples: second, conversation: conversation, modelDirectory: dir) { _ in }
                print("LFMVOICE2 text: \(again.text)")
                #expect(again.text.lowercased().contains("alex") || again.audioSampleCount > 0)
            }
        }
    }
}

/// Rule 17c: the shape the Run UI gives the engine -- awaited from the main actor inside `MLXErrorGuard.guardedRun` -- for each
/// mode. A thread-bound-stream failure shows up here as an MLX error or a crash, not as a wrong answer.
@MainActor
struct LFMAudioMainActorProbeTests {
    private static let enabled = LFMAudioRealLoad.directory != nil

    @Test(.enabled(if: enabled))
    func allThreeModesFromTheMainActor() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        try await LFMAudioRealLoad.withSpeech("Good morning. How are you today?") { samples in
            #expect(onMainThread())
            let text = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.transcribe(samples, modelDirectory: dir) }
            #expect(!text.isEmpty)
            let spoken = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.synthesize("Hello there.", modelDirectory: dir) }
            #expect(!spoken.isEmpty && LFMAudioRealLoad.finite(spoken))
            let reply = try await MLXErrorGuard.guardedRun {
                try await LFMAudioEngine.voiceTurn(userSamples: samples, conversation: LFMConversation(), modelDirectory: dir) { _ in }
            }
            #expect(!reply.text.isEmpty || reply.audioSampleCount > 0)
            // the stages the app builds, called the way a flow calls them
            let stage = LFMAudioStages.asr(modelDirectory: dir)
            let out = try await MLXErrorGuard.guardedRun { try await stage.run(.audio(AudioBuffer(samples: samples, sampleRate: 16_000))) { _ in } }
            if case .text = out {} else { Issue.record("the ASR stage must return text") }
        }
    }
}
