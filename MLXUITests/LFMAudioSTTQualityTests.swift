import Foundation
import Testing
@testable import MLXUI

/// MP6-3f -- speech to text uses the model card's recipe ("Perform ASR." system turn, audio alone, sequential generation). The
/// measurements behind it are in journal 2026-614: with the old prompt the model answered clips under ~15 s with a chatty
/// "Sure, here's the transcription:" and stopped (WER > 1 on 10 of 10 runs of a 3.7 s clip), and was 5-10x slower.
///
/// Part 1 (no weights): the stage a Transcribe row and the Run UI build sends that recipe.
/// Part 2 (real weights, `LFMAudioRealLoad.directory`): production settings (the library's random dither ON, no seed), every
/// clip at least `MLXUI_TEST_LFMAUDIO_STT_RUNS` (default 5) times, **every run** word error rate <= 0.10, no chatty preamble, and no
/// window hitting its token cap. No seeding, no best-of: a pass has to be repeatable on its own.
enum STTScore {
    nonisolated static func words(_ s: String) -> [String] {
        s.lowercased().components(separatedBy: CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "'")).inverted).filter { !$0.isEmpty }
    }

    /// Word-level Levenshtein distance over the reference length.
    nonisolated static func wer(reference: String, hypothesis: String) -> Double {
        let ref = words(reference), hyp = words(hypothesis)
        if ref.isEmpty { return hyp.isEmpty ? 0 : 1 }
        var prev = Array(0...hyp.count)
        for i in 1...ref.count {
            var cur = [i] + Array(repeating: 0, count: hyp.count)
            if !hyp.isEmpty {
                for j in 1...hyp.count {
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] == hyp[j - 1] ? 0 : 1))
                }
            }
            prev = cur
        }
        return Double(prev[hyp.count]) / Double(ref.count)
    }

    nonisolated static func hasPreamble(_ s: String) -> Bool {
        let low = s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return ["sure", "here's", "here is", "certainly", "okay,"].contains { low.hasPrefix($0) } && low.contains(":")
    }

    nonisolated static let sentences: [String: String] = [
        "S": "The quick brown fox jumps over the lazy dog.",
        "M": "Please remind me to call the dentist tomorrow morning at nine thirty.",
        "L": "Yesterday I went to the market to buy some vegetables for dinner. The tomatoes were fresh, but the carrots looked a little old, so I bought potatoes and onions instead. Then I walked home along the river.",
        "XL": "Last weekend my family and I drove to the coast for a short holiday. We stayed in a small hotel near the harbor, and every morning we walked along the beach before breakfast. In the afternoon we visited a lighthouse, took a lot of photographs, and ate fresh fish at a quiet restaurant. On Sunday evening we packed the car and drove back, tired but very happy.",
        // the MP5 noisy-voice passage (RSI/notes/claude-code/modelpack/mp5-smoke), read aloud with noise mixed in at 5 dB
        "harbour": "The harbour was quiet before sunrise. A few fishing boats rocked against their ropes, and the only sound was the slow knock of wood on wood. Maria walked down the stone steps with a thermos of coffee and a small notebook. She had promised herself that this year she would write down everything the old fishermen said, every proverb, every warning about the weather, every story about the storms that never made the newspapers.",
    ]
}

struct LFMAudioSTTRecipeTests {
    /// What a Transcribe row (SDK -> `LFMAudioStages.asr`) and the Run UI's Speech to text tab both call: one sequential call per
    /// window with the system line and the audio alone.
    @Test func theAsrStageSendsTheModelCardRecipe() async throws {
        let backend = MockLFMBackend { _ in LFMGenerated(text: "hello there", audioFrames: []) }
        let stage = LFMAudioStages.asr(modelDirectory: URL(fileURLWithPath: "/tmp/lfm-model"), factory: { _ in backend })
        let out = try await stage.run(.audio(AudioBuffer(samples: [Float](repeating: 0.01, count: 16_000), sampleRate: 16_000))) { _ in }
        guard case let .text(text) = out else { Issue.record("expected text"); return }
        #expect(text == "hello there")
        let call = try #require(backend.calls.first)
        #expect(call.mode == .sequential)
        #expect(call.sampling.temperature == 0)
        #expect(call.prompt.turns.count == 2)
        #expect(call.prompt.turns[0].parts == [.text("Perform ASR.")])
        #expect(call.prompt.turns[1].parts.count == 1)
        #expect(!call.prompt.assistantStartsAudio)
    }

    @Test func theScoringHelpers() {
        #expect(STTScore.wer(reference: "a b c d", hypothesis: "a b c d") == 0)
        #expect(STTScore.wer(reference: "a b c d", hypothesis: "a x c") == 0.5)
        #expect(STTScore.wer(reference: "Hello, World!", hypothesis: "hello world") == 0)
        #expect(STTScore.hasPreamble("Sure, here's the transcription:"))
        #expect(!STTScore.hasPreamble("The quick brown fox"))
    }
}

struct LFMAudioSTTQualityTests {
    private static let enabled = LFMAudioRealLoad.directory != nil
    private static let runs = max(Int(LFMAudioRealLoad.env["MLXUI_TEST_LFMAUDIO_STT_RUNS"] ?? "") ?? 5, 1)

    private struct Case { var name: String; var text: String; var voice: String; var rate: Int? }

    private static func say(_ text: String, voice: String, rate: Int?) throws -> [Float] {
        let key = abs((text + voice + String(rate ?? 0)).hashValue)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("lfm-stt-q-\(key).wav")
        if !FileManager.default.fileExists(atPath: url.path) {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/say")
            var arguments = ["-v", voice]
            if let rate { arguments += ["-r", String(rate)] }
            arguments += ["--file-format=WAVE", "--data-format=LEI16@16000", "-o", url.path, text]
            process.arguments = arguments
            try process.run()
            process.waitUntilExit()
        }
        return try AudioResampler.resample(try AudioFileReader.read(url), to: 16_000).samples
    }

    /// Runs `samples` `runs` times through the production entry point and checks every run.
    /// `maxWER` is 0.10 for every clean clip. The noisy clip has its own bar, 0.20, by owner ruling (Boo, 2026-10-27): it is a voice
    /// mixed with 50 Hz hum and broadband hiss at 5 dB SNR, the old recipe scored 0.41-1.54 on it, and this recipe scored
    /// 0.041-0.122 (one word or two changing per pass, e.g. "ropes" heard as "moat"). No other threshold differs.
    private static func check(_ name: String, samples: [Float], reference: String, dir: URL, maxWER: Double = 0.10) async throws {
        let probe = LFMTurnProbe()
        let factory = LFMMeasuringBackend.factory(probe)
        var worst = 0.0
        for run in 1...runs {
            probe.reset()
            let started = Date()
            let text = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.transcribe(samples, modelDirectory: dir, factory: factory) }
            let wer = STTScore.wer(reference: reference, hypothesis: text)
            worst = max(worst, wer)
            let snap = probe.snapshot
            print(String(format: "LFMSTTQ %@ run%d WER %.3f %.1fs %d tokens: %@", name, run, wer, Date().timeIntervalSince(started), snap.tokens, text))
            #expect(wer <= maxWER, "\(name) run \(run): WER \(wer): \(text)")
            #expect(!STTScore.hasPreamble(text), "\(name) run \(run) starts with a chatty preamble: \(text)")
            #expect(!snap.cap, "\(name) run \(run) hit the token cap")
        }
        print(String(format: "LFMSTTQ %@ (%.1f s audio): %d runs, worst WER %.3f", name, Double(samples.count) / 16_000, runs, worst))
    }

    @Test(.enabled(if: enabled))
    func everyClipTranscribesCleanlyOnEveryRun() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        let s = STTScore.sentences
        let cases = [
            Case(name: "S Samantha", text: s["S"]!, voice: "Samantha", rate: nil),
            Case(name: "M Samantha", text: s["M"]!, voice: "Samantha", rate: nil),
            Case(name: "M Daniel", text: s["M"]!, voice: "Daniel", rate: nil),
            Case(name: "M Samantha -r140", text: s["M"]!, voice: "Samantha", rate: 140),
            Case(name: "M Samantha -r220", text: s["M"]!, voice: "Samantha", rate: 220),
            Case(name: "L Samantha", text: s["L"]!, voice: "Samantha", rate: nil),
            Case(name: "L Daniel", text: s["L"]!, voice: "Daniel", rate: nil),
            Case(name: "L Samantha -r140", text: s["L"]!, voice: "Samantha", rate: 140),
            Case(name: "L Samantha -r220", text: s["L"]!, voice: "Samantha", rate: 220),
            Case(name: "XL Samantha", text: s["XL"]!, voice: "Samantha", rate: nil),
        ]
        for c in cases {
            try await Self.check(c.name, samples: try Self.say(c.text, voice: c.voice, rate: c.rate), reference: c.text, dir: dir)
        }
    }

    /// 45-60 s: the second 30 s window starts mid-sentence, so a join that drops or repeats words shows as WER.
    @Test(.enabled(if: enabled))
    func aClipCrossingTheWindowBoundaryJoinsWithoutLossOrRepeats() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        let s = STTScore.sentences
        let text = [s["XL"]!, s["L"]!, s["harbour"]!].joined(separator: " ")
        let samples = try Self.say(text, voice: "Samantha", rate: nil)
        let seconds = Double(samples.count) / 16_000
        #expect(seconds > 45 && seconds < 75, "the clip is \(seconds) s; it must cross the 30 s window boundary")
        try await Self.check("crossing", samples: samples, reference: text, dir: dir)
    }

    /// The MP5 noisy-voice clip (voice + hum + hiss at 5 dB SNR), 16 kHz. The test host cannot read the main folder, so copy it
    /// to `/private/tmp` or point `MLXUI_TEST_LFMAUDIO_NOISY` at it; skipped with a message when absent.
    @Test(.enabled(if: enabled))
    func theNoisyClipTranscribes() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        let path = LFMAudioRealLoad.env["MLXUI_TEST_LFMAUDIO_NOISY"] ?? "/private/tmp/lfm-smoke/noisy-voice-16k.wav"
        guard FileManager.default.isReadableFile(atPath: path) else {
            print("LFMSKIP no noisy clip at \(path) (copy RSI/notes/claude-code/modelpack/mp5-smoke/noisy-voice-16k.wav there)")
            return
        }
        let samples = try AudioResampler.resample(try AudioFileReader.read(URL(fileURLWithPath: path)), to: 16_000).samples
        try await Self.check("noisy", samples: samples, reference: STTScore.sentences["harbour"]!, dir: dir, maxWER: 0.20)
    }

    /// A real human recording with a reference: `me.m4a` plus `me.txt` (the words as spoken) in `MLXUI_TEST_LFMAUDIO_ME_DIR`
    /// or `/private/tmp/lfm-smoke/`. Skipped with a message when either is missing, never failed and never substituted.
    @Test(.enabled(if: enabled))
    func aRealHumanRecordingTranscribes() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        let folder = LFMAudioRealLoad.env["MLXUI_TEST_LFMAUDIO_ME_DIR"] ?? "/private/tmp/lfm-smoke"
        let audio = URL(fileURLWithPath: folder).appendingPathComponent("me.m4a")
        let reference = URL(fileURLWithPath: folder).appendingPathComponent("me.txt")
        guard FileManager.default.isReadableFile(atPath: audio.path), FileManager.default.isReadableFile(atPath: reference.path) else {
            print("LFMSKIP no human recording: need \(audio.path) and \(reference.path) (the words as spoken)")
            return
        }
        let samples = try AudioResampler.resample(try AudioFileReader.read(audio), to: 16_000).samples
        try await Self.check("me.m4a", samples: samples, reference: try String(contentsOf: reference, encoding: .utf8), dir: dir)
    }
}
