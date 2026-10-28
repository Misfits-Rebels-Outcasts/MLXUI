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

    /// 4-grams that occur three or more times.
    nonisolated static func repeatedFourGrams(_ words: [String]) -> Set<String> {
        guard words.count >= 4 else { return [] }
        var counts: [String: Int] = [:]
        for i in 0...(words.count - 4) { counts[words[i..<(i + 4)].joined(separator: " "), default: 0] += 1 }
        return Set(counts.filter { $0.value >= 3 }.keys)
    }

    /// The words that differ, as a short list ("-ref +hyp"), by a longest-common-subsequence diff, for the report.
    nonisolated static func differences(reference ref: [String], hypothesis hyp: [String]) -> [String] {
        var table = Array(repeating: Array(repeating: 0, count: hyp.count + 1), count: ref.count + 1)
        for i in stride(from: ref.count - 1, through: 0, by: -1) {
            for j in stride(from: hyp.count - 1, through: 0, by: -1) {
                table[i][j] = ref[i] == hyp[j] ? table[i + 1][j + 1] + 1 : max(table[i + 1][j], table[i][j + 1])
            }
        }
        var out: [String] = [], i = 0, j = 0
        while i < ref.count || j < hyp.count {
            if i < ref.count, j < hyp.count, ref[i] == hyp[j] { i += 1; j += 1 }
            else if j < hyp.count, i == ref.count || table[i][j + 1] >= table[i + 1][j] { out.append("+" + hyp[j]); j += 1 }
            else { out.append("-" + ref[i]); i += 1 }
        }
        return out
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

    /// What `check` measured: one WER and one transcript per run.
    private struct Outcome { var wers: [Double]; var texts: [String]; var mean: Double { wers.reduce(0, +) / Double(max(wers.count, 1)) } }

    /// Runs `samples` `count` times (default `runs`) through the production entry point and checks every run.
    /// - `maxWER` (default 0.10, every clean clip) bounds EVERY run; nil leaves single runs unbounded (the lossy-codec clip, below).
    /// - `meanWER`, when given, bounds the mean over the runs; `ceiling` bounds any single run in that case.
    /// - `noTail`: nothing past the end of the audio, i.e. the transcript has at most the reference's words plus 5 % and 2.
    /// The noisy clip has its own bar, 0.20, by owner ruling (Boo, 2026-10-27): a voice mixed with 50 Hz hum and broadband hiss at
    /// 5 dB SNR; the old recipe scored 0.41-1.54 on it, this one 0.041-0.122. No preamble and no window at the token cap, always.
    @discardableResult
    private static func check(_ name: String, samples: [Float], reference: String, dir: URL, count: Int? = nil, maxWER: Double? = 0.10,
                              meanWER: Double? = nil, ceiling: Double? = nil, noTail: Bool = false) async throws -> Outcome {
        let probe = LFMTurnProbe()
        let factory = LFMMeasuringBackend.factory(probe)
        var outcome = Outcome(wers: [], texts: [])
        let n = count ?? runs
        for run in 1...n {
            probe.reset()
            let started = Date()
            let text = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.transcribe(samples, modelDirectory: dir, factory: factory) }
            let wer = STTScore.wer(reference: reference, hypothesis: text)
            outcome.wers.append(wer); outcome.texts.append(text)
            let snap = probe.snapshot
            print(String(format: "LFMSTTQ %@ run%d WER %.3f %.1fs %d tokens: %@", name, run, wer, Date().timeIntervalSince(started), snap.tokens, text))
            if let maxWER { #expect(wer <= maxWER, "\(name) run \(run): WER \(wer): \(text)") }
            if let ceiling { #expect(wer <= ceiling, "\(name) run \(run): WER \(wer) above the single-run ceiling \(ceiling): \(text)") }
            #expect(!STTScore.hasPreamble(text), "\(name) run \(run) starts with a chatty preamble: \(text)")
            #expect(!snap.cap, "\(name) run \(run) hit the token cap")
            if noTail {
                let allowed = Double(STTScore.words(reference).count) * 1.05 + 2
                #expect(Double(STTScore.words(text).count) <= allowed, "\(name) run \(run): \(STTScore.words(text).count) words against a reference of \(STTScore.words(reference).count): text past the end of the audio")
            }
        }
        if let meanWER { #expect(outcome.mean <= meanWER, "\(name): mean WER \(outcome.mean) over \(n) runs") }
        print(String(format: "LFMSTTQ %@ (%.1f s audio): %d runs, mean WER %.3f, worst %.3f", name, Double(samples.count) / 16_000, n, outcome.mean, outcome.wers.max() ?? 0))
        return outcome
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

    // MARK: Real human recordings (owner ruling, Boo, 2026-10-27: "Use this set for the real-recording condition")

    /// The set lives in the main folder's `RSI/notes/claude-code/modelpack/mp6-stt-real/` and is **not** part of the repo or the test
    /// target (JFK 1961 is public domain; the TED clip is TED-licensed and stays local). The test host cannot read that folder, so
    /// copy it to `/private/tmp/lfm-real` and point `MLXUI_TEST_LFMAUDIO_REAL_DIR` at the copy
    /// (`TEST_RUNNER_MLXUI_TEST_LFMAUDIO_REAL_DIR`). Unset or missing: `LFMSKIP`, never a failure and never a substitute.
    private static var realDirectory: URL? {
        guard let path = LFMAudioRealLoad.env["MLXUI_TEST_LFMAUDIO_REAL_DIR"],
              FileManager.default.isReadableFile(atPath: path) else { return nil }
        return URL(fileURLWithPath: path)
    }

    private static func realClip(_ file: String) throws -> (samples: [Float], reference: String)? {
        guard let real = realDirectory else {
            print("LFMSKIP MLXUI_TEST_LFMAUDIO_REAL_DIR is unset or unreadable (copy RSI/notes/claude-code/modelpack/mp6-stt-real to /private/tmp/lfm-real)")
            return nil
        }
        let audio = real.appendingPathComponent(file)
        let reference = real.appendingPathComponent((file as NSString).deletingPathExtension + ".txt")
        guard FileManager.default.isReadableFile(atPath: audio.path), FileManager.default.isReadableFile(atPath: reference.path) else {
            print("LFMSKIP \(file) or its .txt is missing from \(real.path)")
            return nil
        }
        return (try AudioResampler.resample(try AudioFileReader.read(audio), to: 16_000).samples, try String(contentsOf: reference, encoding: .utf8))
    }

    /// Two clean JFK recordings with exact references, every run <= 0.10: a real voice at 16 kHz, and the same with pink noise at
    /// about 11 dB SNR.
    @Test(.enabled(if: enabled))
    func theRealJFKRecordingsTranscribe() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        for file in ["jfk-16k.wav", "jfk-noisy-16k.wav"] {
            guard let clip = try Self.realClip(file) else { return }
            try await Self.check(file, samples: clip.samples, reference: clip.reference, dir: dir)
        }
    }

    /// The same speech as 44.1 kHz AAC (the lossy-codec clip). Owner ruling (Boo, 2026-10-27): mean WER <= 0.12 over 20 runs, with the
    /// single-run ceiling of 0.30, no preamble and no loop, for **this clip only**. The first ruling was 0.10; the mean of 20 runs
    /// measured 0.098 and 0.107 on two batches, so 0.10 sat on the noise. Diagnosis (journal 2026-616): it is the AAC codec, not the
    /// sample rate or our resampler -- the same speech written as 44.1 kHz / 48 kHz WAV (no codec) and resampled by the app's path
    /// scores 0.014 / 0.025, afconvert's 16 kHz conversion of the AAC scores like ours (10 of 20 runs over 0.10 against 11 of 20),
    /// and clean 16 kHz audio is 0.000. One word flipping ("Americans"/"American", "ask"/"asked") is 0.045 of a 22-word reference.
    @Test(.enabled(if: enabled))
    func theLossyCodecClipMeetsItsMeanBar() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        guard let clip = try Self.realClip("jfk-44k1.m4a") else { return }
        try await Self.check("jfk-44k1.m4a", samples: clip.samples, reference: clip.reference, dir: dir, count: 20, maxWER: nil,
                             meanWER: 0.12, ceiling: 0.30)
    }

    /// 54.3 s of distinct content whose 30 s window cut falls mid-sentence (owner's clip, `mixed-distinct-54s`): the window-crossing
    /// pass/fail test with a real voice. Every run <= 0.10, at least 5 runs, no window at the 512-token cap, nothing past the end of
    /// the audio.
    @Test(.enabled(if: enabled))
    func aDistinctClipCrossingTheWindowBoundaryJoinsCleanly() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        guard let clip = try Self.realClip("mixed-distinct-54s.wav") else { return }
        try await Self.check("mixed-distinct-54s", samples: clip.samples, reference: clip.reference, dir: dir, count: max(Self.runs, 5), noTail: true)
    }

    /// KNOWN LIMITATION, measurement only (owner ruling, Boo, 2026-10-27): `jfk-x5-59s` is the same sentence five times back to
    /// back. The model loses takes from verbatim back-to-back repeats (3 of 5 found on every run; each take alone and two in one
    /// window are perfect; a cut in the silence does not help), and the same cause makes tiled audio overrun: a sentence tiled 8x
    /// into 24 s loops ~40x past the end of the audio (2176 chars at the 512 cap). Real speech does not repeat itself verbatim, and
    /// the window join is covered by `aDistinctClipCrossingTheWindowBoundaryJoinsCleanly` and the TED comparison. This test never
    /// fails on the content; it prints how many takes were found so the limitation stays visible.
    @Test(.enabled(if: enabled))
    func repeatedTakesAreAKnownLimitationMeasuredNotAsserted() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        guard let clip = try Self.realClip("jfk-x5-59s.wav") else { return }
        let probe = LFMTurnProbe()
        let factory = LFMMeasuringBackend.factory(probe)
        for run in 1...max(Self.runs, 1) {
            let text = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.transcribe(clip.samples, modelDirectory: dir, factory: factory) }
            let takes = STTScore.words(text).filter { $0 == "fellow" }.count
            print("LFMSTTQ KNOWN-LIMITATION jfk-x5-59s run\(run): \(takes) of 5 takes found, WER \(STTScore.wer(reference: clip.reference, hypothesis: text))")
        }
    }

    /// Report only: is the lossy-codec miss the codec or the sample rate? `jfk-16k` is resampled up to 44.1 kHz and to 48 kHz,
    /// written as 16-bit WAV (no AAC), read back and resampled to 16 kHz by the app's own path, then transcribed 20 times.
    /// If both come out ~0 the cause is AAC, not the rate. Nothing asserted.
    @Test(.enabled(if: enabled))
    func reportOnlyTheRateRoundTripWithoutACodec() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        guard let clip = try Self.realClip("jfk-16k.wav") else { return }
        for rate in [44_100, 48_000] {
            let up = try AudioResampler.resample(AudioBuffer(samples: clip.samples, sampleRate: 16_000), to: rate)
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("lfm-rate-\(rate).wav")
            try Self.wav(up.samples, sampleRate: rate).write(to: url)
            defer { try? FileManager.default.removeItem(at: url) }
            let back = try AudioResampler.resample(try AudioFileReader.read(url), to: 16_000).samples
            let outcome = try await Self.check("jfk-16k via \(rate) Hz WAV", samples: back, reference: clip.reference, dir: dir, count: 20, maxWER: nil)
            print(String(format: "LFMSTTQ REPORT %d Hz WAV round trip: mean WER %.3f, runs over 0.10: %d of 20, perfect: %d", rate, outcome.mean,
                         outcome.wers.filter { $0 > 0.10 }.count, outcome.wers.filter { $0 == 0 }.count))
        }
    }

    private static func wav(_ samples: [Float], sampleRate: Int) -> Data {
        var pcm = Data()
        for x in samples { var v = Int16(max(-1, min(1, x)) * 32767).littleEndian; withUnsafeBytes(of: &v) { pcm.append(contentsOf: $0) } }
        func le32(_ v: Int) -> Data { var x = UInt32(v).littleEndian; return Data(bytes: &x, count: 4) }
        func le16(_ v: Int) -> Data { var x = UInt16(v).littleEndian; return Data(bytes: &x, count: 2) }
        var d = Data("RIFF".utf8); d += le32(36 + pcm.count); d += Data("WAVEfmt ".utf8); d += le32(16); d += le16(1); d += le16(1)
        d += le32(sampleRate); d += le32(sampleRate * 2); d += le16(2); d += le16(16); d += Data("data".utf8); d += le32(pcm.count)
        return d + pcm
    }

    /// `ted-60s.m4a` has no reference. LFM's transcript is compared with Parakeet TDT 0.6B v3 and Qwen3-ASR 1.7B on the same file;
    /// the disagreeing words are printed. Pass condition (no hallucinated or looping passage versus the other two): no 4-gram that
    /// LFM repeats three or more times unless both others repeat it too, its length within 15 % of the other two's mean, and its
    /// word error rate against each of them no more than 0.20 above their own against each other. (Those three numbers are the
    /// implementer's heuristics for "no runaway passage", pending owner confirmation.) Needs the two models via
    /// `MLXUI_TEST_PARAKEET_DIR` and `MLXUI_TEST_QWEN3ASR_DIR` (copies under `/private/tmp`).
    @Test(.enabled(if: enabled))
    func theTedExcerptAgreesWithParakeetAndQwen3ASR() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        guard let real = Self.realDirectory, FileManager.default.isReadableFile(atPath: real.appendingPathComponent("ted-60s.m4a").path) else {
            print("LFMSKIP no ted-60s.m4a (set MLXUI_TEST_LFMAUDIO_REAL_DIR)")
            return
        }
        guard let parakeetDir = ParakeetRealLoad.directory, Qwen3ASRRealLoad.directory != nil else {
            print("LFMSKIP the TED comparison needs Parakeet (MLXUI_TEST_PARAKEET_DIR) and Qwen3-ASR (MLXUI_TEST_QWEN3ASR_DIR) installed where the host can read them")
            return
        }
        let qwen = try Qwen3ASRRealLoad.workingCopy()
        defer { qwen.remove() }
        let samples = try AudioResampler.resample(try AudioFileReader.read(real.appendingPathComponent("ted-60s.m4a")), to: 16_000).samples
        let parakeet = try await MLXErrorGuard.guardedRun { try await ParakeetEngine.transcribe(samples, modelDirectory: parakeetDir) }
        let qwenText = try await MLXErrorGuard.guardedRun { try await Qwen3ASREngine.transcribe(samples, modelDirectory: qwen.url) }
        print("LFMTED parakeet: \(parakeet)")
        print("LFMTED qwen3asr: \(qwenText)")
        let others = STTScore.wer(reference: parakeet, hypothesis: qwenText)
        print(String(format: "LFMTED parakeet vs qwen3asr WER %.3f", others))
        let probe = LFMTurnProbe()
        let factory = LFMMeasuringBackend.factory(probe)
        for run in 1...Self.runs {
            probe.reset()
            let lfm = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.transcribe(samples, modelDirectory: dir, factory: factory) }
            let vsParakeet = STTScore.wer(reference: parakeet, hypothesis: lfm), vsQwen = STTScore.wer(reference: qwenText, hypothesis: lfm)
            let lengths = [STTScore.words(parakeet).count, STTScore.words(qwenText).count]
            let mean = Double(lengths[0] + lengths[1]) / 2, mine = Double(STTScore.words(lfm).count)
            print(String(format: "LFMTED lfm run%d: WER vs parakeet %.3f, vs qwen3asr %.3f, words %d (others %d / %d), tokens %d: %@", run,
                         vsParakeet, vsQwen, Int(mine), lengths[0], lengths[1], probe.snapshot.tokens, lfm))
            print("LFMTED disagreements run\(run) (lfm vs parakeet): \(STTScore.differences(reference: STTScore.words(parakeet), hypothesis: STTScore.words(lfm)))")
            print("LFMTED disagreements run\(run) (lfm vs qwen3asr): \(STTScore.differences(reference: STTScore.words(qwenText), hypothesis: STTScore.words(lfm)))")
            #expect(!STTScore.hasPreamble(lfm), "run \(run) starts with a chatty preamble")
            #expect(!probe.snapshot.cap, "run \(run) hit the token cap")
            #expect(abs(mine - mean) <= 0.15 * mean, "run \(run): \(Int(mine)) words against \(lengths)")
            #expect(vsParakeet <= others + 0.20 && vsQwen <= others + 0.20, "run \(run) disagrees with the others far more than they do with each other")
            for gram in STTScore.repeatedFourGrams(STTScore.words(lfm)) {
                let inOthers = STTScore.repeatedFourGrams(STTScore.words(parakeet)).contains(gram) && STTScore.repeatedFourGrams(STTScore.words(qwenText)).contains(gram)
                #expect(inOthers, "run \(run) loops on \"\(gram)\"")
            }
        }
    }
}
