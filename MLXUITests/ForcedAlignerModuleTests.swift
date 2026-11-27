import Testing
import Foundation
import AVFoundation
import MLXAudioSTT
@testable import MLXUI

/// FA-5 for issue #13: the pure `text,start,end` / SRT / VTT writers and the cue rule (decision 7), and the module
/// wiring — claim (trap 3), stage shape, resolution, and the 16 kHz/limit contract with no weights needed. The real load,
/// determinism, the main-actor probe and the parity clips are in `ForcedAlignerRealLoadTests` (FA-4/FA-5, gated).
struct AlignmentExportTests {

    private func units(_ pairs: [(String, Double, Double)]) -> [AlignedUnit] {
        pairs.map { AlignedUnit(text: $0.0, start: $0.1, end: $0.2) }
    }

    // MARK: - CSV (the table and the flow row's output)

    @Test func csvIsTheTextStartEndTable() {
        let table = units([("The", 0.24, 0.48), ("quick", 0.48, 0.72), ("周", 1, 1.08)])
        #expect(AlignmentExport.csv(table) == "text,start,end\nThe,0.240,0.480\nquick,0.480,0.720\n周,1.000,1.080\n")
        #expect(AlignmentExport.columns == ["text", "start", "end"])
        #expect(AlignmentExport.csv([]) == "text,start,end\n", "an empty result is the header only")
    }

    @Test func parseCSVIsTheInverse() throws {
        let table = units([("The", 0.24, 0.48), ("quick", 0.48, 0.72)])
        let parsed = try AlignmentExport.parseCSV(AlignmentExport.csv(table))
        #expect(parsed == table)
        #expect(try AlignmentExport.parseCSV("text,start,end\n") == [])
        #expect(throws: AlignmentExport.CSVError.badHeader) {
            _ = try AlignmentExport.parseCSV("word,start,end\nA,0.000,1.000\n")
        }
        #expect(throws: AlignmentExport.CSVError.badRow(2)) {
            _ = try AlignmentExport.parseCSV("text,start,end\nA,0.000\n")
        }
        #expect(throws: AlignmentExport.CSVError.badRow(2)) {
            _ = try AlignmentExport.parseCSV("text,start,end\nA,1.000,0.500\n")   // ends before it starts
        }
        #expect(throws: AlignmentExport.CSVError.badRow(2)) {
            _ = try AlignmentExport.parseCSV("text,start,end\nA,nan,1.000\n")
        }
    }

    @Test func tableRowsAreNumbersNotStrings() {
        let rows = AlignmentExport.tableRows(units([("The", 0.24, 0.48)]))
        #expect(rows.count == 1)
        #expect(rows[0].count == 3)
        #expect(rows[0][0] as? String == "The")
        #expect(rows[0][1] as? Double == 0.24)
        #expect(rows[0][2] as? Double == 0.48)
    }

    // MARK: - SRT / VTT

    @Test func srtAndVTTTimestampsUseTheirSeparators() {
        #expect(AlignmentExport.srtTimestamp(1.24) == "00:00:01,240")
        #expect(AlignmentExport.srtTimestamp(0) == "00:00:00,000")
        #expect(AlignmentExport.srtTimestamp(3723.5) == "01:02:03,500")
        #expect(AlignmentExport.vttTimestamp(1.24) == "00:00:01.240")
        #expect(AlignmentExport.vttTimestamp(3723.5) == "01:02:03.500")
    }

    @Test func srtNumbersAndFormatsItsCues() {
        let srt = AlignmentExport.srt(units([("The", 0.24, 0.48), ("quick", 0.48, 0.72), ("brown", 0.72, 1.04)]))
        #expect(srt.hasPrefix("1\n00:00:00,240 --> 00:00:01,040\nThe quick brown\n"))
        #expect(!srt.contains("WEBVTT"))
        #expect(AlignmentExport.srt([]) == "")
    }

    @Test func vttHasTheHeaderAndDotTimestamps() {
        let vtt = AlignmentExport.vtt(units([("The", 0.24, 0.48), ("quick", 0.48, 0.72)]))
        #expect(vtt.hasPrefix("WEBVTT\n\n00:00:00,240".replacingOccurrences(of: ",", with: ".")))
        #expect(vtt.contains("00:00:00.240 --> 00:00:00.720\nThe quick"))
        #expect(AlignmentExport.vtt([]) == "WEBVTT\n", "an empty VTT is the header")
    }

    // MARK: - The cue rule (decision 7)

    @Test func aCueEndsBeforeItWouldPass42Characters() {
        // "abcdefghij" is 10 characters; three of them plus two spaces is 32, the fourth would make 43 > 42, so it
        // starts a new cue.
        let letters = units((0..<8).map { ("abcdefghij", Double($0), Double($0) + 0.5) })
        let cues = AlignmentExport.cues(letters)
        #expect(cues.map { $0.text.split(separator: " ").count } == [3, 3, 2], "\(cues)")
        for cue in cues {
            #expect(cue.text.count <= AlignmentExport.cueMaxCharacters, "\(cue.text.count): \(cue.text)")
        }
    }

    @Test func aCueEndsBeforeItWouldPassSixSeconds() {
        let units = self.units([("one", 0, 2), ("two", 2, 4), ("three", 4, 6.4), ("four", 6.4, 8)])
        let cues = AlignmentExport.cues(units)
        #expect(cues.count == 2)
        #expect(cues[0].text == "one two")
        #expect(cues[0].end == 4)
        #expect(cues[1].text == "three four")
    }

    @Test func aCueEndsAtSentenceEndingPunctuation() {
        let units = self.units([("Hello.", 0, 0.5), ("Next", 0.5, 1.0), ("one", 1.0, 1.5)])
        let cues = AlignmentExport.cues(units)
        #expect(cues.map(\.text) == ["Hello.", "Next one"])
    }

    /// The units the pinned library really makes from a transcript (its own splitter: no punctuation), 0.3 s each.
    private func libraryUnits(_ transcript: String, _ language: String) -> [AlignedUnit] {
        ForceAlignProcessor().encodeTimestamp(text: transcript, language: language).0.enumerated().map { index, text in
            AlignedUnit(text: text, start: Double(index) * 0.3, end: Double(index) * 0.3 + 0.3)
        }
    }

    /// Decision 7 on real output: the splitter drops the punctuation, so the sentence end comes from the transcript.
    @Test func anEnglishSentenceEndsItsCueOnLibraryUnits() {
        let transcript = "The river runs past the mill. Every morning the boats leave."
        let srt = AlignmentExport.srt(libraryUnits(transcript, "English"), transcript: transcript)
        #expect(srt == "1\n00:00:00,000 --> 00:00:01,800\nThe river runs past the mill\n\n"
                + "2\n00:00:01,800 --> 00:00:03,300\nEvery morning the boats leave\n")
    }

    @Test func aChineseSentenceEndsItsCueOnLibraryUnits() {
        let transcript = "今天天气很好。我们去公园。"
        let srt = AlignmentExport.srt(libraryUnits(transcript, "Chinese"), transcript: transcript)
        #expect(srt == "1\n00:00:00,000 --> 00:00:01,800\n今天天气很好\n\n2\n00:00:01,800 --> 00:00:03,300\n我们去公园\n")
    }

    @Test func aGermanSentencePairMakesTwoCuesInVTTToo() {
        let transcript = "Der Zug fährt um acht Uhr ab! Wir warten am Gleis."
        let vtt = AlignmentExport.vtt(libraryUnits(transcript, "German"), transcript: transcript)
        #expect(vtt == "WEBVTT\n\n00:00:00.000 --> 00:00:02.100\nDer Zug fährt um acht Uhr ab\n\n"
                + "00:00:02.100 --> 00:00:03.300\nWir warten am Gleis\n")
    }

    @Test func aTranscriptWithoutPunctuationChangesNothing() {
        let transcript = "The river runs past the mill every morning the boats leave"
        let units = libraryUnits(transcript, "English")
        #expect(AlignmentExport.sentenceEnds(in: transcript, for: units).allSatisfy { !$0 })
        #expect(AlignmentExport.srt(units, transcript: transcript) == AlignmentExport.srt(units))
        #expect(AlignmentExport.vtt(units, transcript: transcript) == AlignmentExport.vtt(units))
    }

    @Test func theMarksFollowTheUnitsThroughRepeatedWords() {
        let transcript = "„Ja.“ Ja, gern … Ja"
        #expect(AlignmentExport.sentenceEnds(in: transcript, for: libraryUnits(transcript, "German"))
                == [true, false, true, false])
    }

    @Test func aUnitThatIsNotInTheTranscriptGetsNoMarkAndTheWalkCarriesOn() {
        let units = self.units([("Guten", 0, 0.3), ("Morgen", 0.3, 0.6), ("Xylophon", 0.6, 0.9),
                                ("Bis", 0.9, 1.2), ("bald", 1.2, 1.5)])
        #expect(AlignmentExport.sentenceEnds(in: "Guten Morgen. Bis bald.", for: units) == [false, true, false, false, true])
        #expect(AlignmentExport.sentenceEnds(in: "", for: units) == [false, false, false, false, false])
        #expect(AlignmentExport.sentenceEnds(in: "Ja.", for: units) == [false, false, false, false, false])
        #expect(AlignmentExport.sentenceEnds(in: "Ja.", for: []).isEmpty)
    }

    /// A match far ahead is not taken: the units after the stray one still find their own places.
    @Test func aStrayUnitDoesNotJumpToAFarAwayMatch() {
        let transcript = "Wir gehen. Danach essen wir zusammen im Garten hinter dem alten Haus Brot."
        let units = self.units([("Wir", 0, 0.3), ("Brot", 0.3, 0.6), ("gehen", 0.6, 0.9), ("Danach", 0.9, 1.2)])
        #expect(AlignmentExport.sentenceEnds(in: transcript, for: units) == [false, false, true, false])
    }

    @Test func aDecimalPointIsNotASentenceEnd() {
        let transcript = "全长3.5公里。"
        let units = libraryUnits(transcript, "Chinese")
        #expect(units.map(\.text) == ["全", "长", "3", "5", "公", "里"])
        #expect(AlignmentExport.sentenceEnds(in: transcript, for: units) == [false, false, false, false, false, true])
    }

    /// Known limitation: there is no abbreviation list, so an abbreviation's period ends a cue.
    @Test func anAbbreviationsPeriodEndsACue() {
        let transcript = "Dr. Schmidt kommt."
        let cues = AlignmentExport.cues(libraryUnits(transcript, "German"),
                                        sentenceEnds: AlignmentExport.sentenceEnds(
                                            in: transcript, for: libraryUnits(transcript, "German")))
        #expect(cues.map(\.text) == ["Dr", "Schmidt kommt"])
    }

    @Test func cjkUnitsJoinWithoutSpaces() {
        let units = self.units([("你", 0, 0.2), ("好", 0.2, 0.4), ("world", 0.4, 0.8), ("世", 0.8, 1.0), ("界", 1.0, 1.2)])
        #expect(AlignmentExport.cueText(units) == "你好world世界", "a space only between non-CJK words")
        #expect(AlignmentExport.isCJK("你") && AlignmentExport.isCJK("こ") && AlignmentExport.isCJK("한"))
        #expect(!AlignmentExport.isCJK("A") && !AlignmentExport.isCJK("1"))
    }

    @Test func noUnitsIsNoCues() {
        #expect(AlignmentExport.cues([]).isEmpty)
        #expect(AlignmentExport.srt([]).isEmpty)
        #expect(AlignmentExport.vtt([]) == "WEBVTT\n")
    }
}

/// FA-5, trap 3: the aligner resolves to its own module and never to Qwen3-ASR's (or the chat sheet's, or the
/// `Transcribe` pool), and the module/stage shape is the `audio → text` one. The real run is gated in
/// `ForcedAlignerRealLoadTests`.
@MainActor
struct ForcedAlignerModuleTests {

    private func alignerEntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Qwen3-ForcedAligner-0.6B-8bit", family: "Qwen3-ForcedAligner",
                  displayName: "Qwen3 Aligner 0.6B", modelType: .align, source: .mlx,
                  hfRepo: "mlx-community", hfModelId: "mlx-community/Qwen3-ForcedAligner-0.6B-8bit")
    }

    private func qwen3ASREntry() -> ModelEntry {
        makeEntry(id: "mlx-community--Qwen3-ASR-1.7B-8bit", family: "Qwen3-ASR",
                  displayName: "Qwen3-ASR 1.7B", modelType: .asr, source: .mlx,
                  hfRepo: "mlx-community", hfModelId: "mlx-community/Qwen3-ASR-1.7B-8bit")
    }

    private func bundledCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    @Test func theAlignerResolvesToItsOwnModuleAndQwen3ASRToItsOwn() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let alignerResolved = try #require(registry.bestModule(for: alignerEntry()))
        let asrResolved = try #require(registry.bestModule(for: qwen3ASREntry()))
        #expect(alignerResolved.descriptor.id == "forced-aligner")
        // Trap 3: the shared qwen3_asr tag must not move the Qwen3-ASR entry.
        #expect(asrResolved.descriptor.id == "qwen3-asr")
    }

    @Test func claimsOnlyItsOwnFamilyKindAndSource() {
        let sdk = ForcedAlignerSDK()
        #expect(sdk.claim(alignerEntry()) == .exact)
        #expect(sdk.claim(makeEntry(family: "Qwen3-ForcedAligner", modelType: .align, source: .coreml)) == .no)
        #expect(sdk.claim(makeEntry(family: "Qwen3-ForcedAligner", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(family: "Qwen3-ASR", modelType: .align, source: .mlx)) == .no)
        // Every other SDK declines the aligner, Qwen3-ASR's included (its family test is `contains("qwen3-asr")`).
        for module in installedModules where module.descriptor.id != "forced-aligner" {
            let registry = ModelRegistry()
            module.register(into: registry)
            #expect(registry.bestModule(for: alignerEntry()) == nil,
                    "\(module.descriptor.id) must decline the aligner")
        }
        #expect(Qwen3ASRSDK().claim(alignerEntry()) == .no, "Qwen3-ASR must not claim the aligner (trap 3)")
        #expect(ForcedAlignerSDK().claim(qwen3ASREntry()) == .no, "the aligner must not claim Qwen3-ASR")
    }

    @Test func descriptorAndStageShape() throws {
        #expect(ForcedAlignerModule.descriptor.id == "forced-aligner")
        #expect(ForcedAlignerModule.descriptor.modelTypes == [.align])
        let stage = try ForcedAlignerSDK().makeStage(for: alignerEntry(), config: .default)
        #expect(stage is ForcedAlignerStage)
        #expect(stage.accepts == .audio && stage.produces == .text)
    }

    /// Trap 3's "never in the Transcribe pool or the Chat sheet": the model is not a derived candidate for any LLM or
    /// ASR task (until FA-6 gives `Align` its pool entry, it is in no pool at all).
    @Test func theAlignerIsInNoOtherTaskPool() throws {
        let catalog = try bundledCatalog()
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        for task in ["Transcribe", "Generate", "Speak", "Describe Image"] {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(!derived.contains { $0.id == "mlx-community--Qwen3-ForcedAligner-0.6B-8bit" }, "\(task)")
        }
    }

    // MARK: - The stage, with an injected engine (no weights)

    /// Trap 5: a 44.1 kHz stereo recording is mixed to mono and resampled to 16 kHz before the engine sees it.
    @Test func theStageResamplesAStereoRecordingTo16k() async throws {
        let url = try makeStereoWAV(seconds: 1.0)
        defer { try? FileManager.default.removeItem(at: url) }
        let buffer = try AudioFileReader.read(url)
        #expect(buffer.sampleRate == 44_100)
        let seen = SampleLog()
        let stage = ForcedAlignerStage(id: "stub", name: "stub", transcript: "hello", language: "en") { samples, _, _ in
            seen.add(samples.count)
            return [AlignedUnit(text: "hello", start: 0.1, end: 0.5)]
        }
        let output = try await stage.run(.audio(buffer)) { _ in }
        guard case let .text(csv) = output else { Issue.record("expected text"); return }
        #expect(csv == "text,start,end\nhello,0.100,0.500\n")
        let count = try #require(seen.values.first)
        #expect(abs(Double(count) - 16_000) < 200, "the engine sees ~1 s at 16 kHz, got \(count)")
    }

    @Test func theStageRefusesPastTheLimitBeforeTheEngineRuns() async throws {
        let buffer = AudioBuffer(samples: [Float](repeating: 0, count: 16_000 * 301), sampleRate: 16_000)
        let called = SampleLog()
        let stage = ForcedAlignerStage(id: "stub", name: "stub", transcript: "hello", language: "en") { samples, _, _ in
            called.add(samples.count)
            return []
        }
        do {
            _ = try await stage.run(.audio(buffer)) { _ in }
            Issue.record("a 301 s recording must be refused")
        } catch let error as ForcedAlignerEngine.EngineError {
            guard case .overLimit = error else {
                Issue.record("expected overLimit, got \(error)")
                return
            }
        }
        #expect(called.values.isEmpty, "the engine is never called for an over-long recording")
    }

    private func makeStereoWAV(seconds: Double) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fa-stereo-\(UUID().uuidString).wav")
        let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 44_100,
                                                channels: 2, interleaved: false))
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let frames = AVAudioFrameCount(44_100 * seconds)
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames))
        buffer.frameLength = frames
        if let channels = buffer.floatChannelData {
            for frame in 0..<Int(frames) {
                let value = Float(sin(Double(frame) / 44_100.0 * 2 * .pi * 440) * 0.2)
                channels[0][frame] = value
                channels[1][frame] = value
            }
        }
        try file.write(from: buffer)
        return url
    }
}

/// A tiny `@unchecked Sendable` recorder for what an injected stage's engine received (the closures are `@Sendable`).
final class SampleLog: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [Int] = []
    func add(_ value: Int) { lock.lock(); storage.append(value); lock.unlock() }
    var values: [Int] { lock.lock(); defer { lock.unlock() }; return storage }
}
