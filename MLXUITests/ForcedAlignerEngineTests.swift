import Testing
import Foundation
import MLXAudioSTT
@testable import MLXUI

/// FA-3 for issue #13: the forced-aligner engine's contract with no weights needed — the language table, the input
/// refusals (before any model work), decision 4's clean-up on hand-made library results, and the pinned library's own
/// text splitter (trap 6). The real load, parity and the main-actor probe are in `ForcedAlignerRealLoadTests`
/// (FA-4/FA-5, gated on the installed copy).
struct ForcedAlignerEngineTests {

    // MARK: - The kind's constants

    @Test func constantsMatchTheModelCard() {
        #expect(ForcedAlignerEngine.sampleRate == 16_000, "the model takes 16 kHz mono (trap 5)")
        #expect(ForcedAlignerEngine.maxSeconds == 300, "about five minutes per call (owner decision 5)")
        #expect(ForcedAlignerEngine.timestampStepSeconds == 0.08, "one class = 80 ms (config timestamp_segment_time)")
        #expect(ForcedAlignerEngine.usesAudioThread == false)
    }

    // MARK: - Languages (owner decision 6)

    @Test func theElevenLanguageCodesMapToTheLibraryNames() {
        #expect(ForcedAlignerEngine.languageCodes == ["zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"])
        #expect(ForcedAlignerEngine.languages.map(\.code) == ForcedAlignerEngine.languageCodes)
        #expect(ForcedAlignerEngine.languages.map(\.name)
                == ["Chinese", "English", "Cantonese", "French", "German", "Italian", "Japanese", "Korean",
                    "Portuguese", "Russian", "Spanish"])
        #expect(ForcedAlignerEngine.defaultLanguageCode == "en")
        for code in ForcedAlignerEngine.languageCodes {
            #expect(ForcedAlignerEngine.libraryLanguageName(for: code) != nil, "\(code)")
        }
        #expect(ForcedAlignerEngine.libraryLanguageName(for: "EN") == "English", "codes are matched case-insensitively")
        #expect(ForcedAlignerEngine.libraryLanguageName(for: "auto") == nil, "SPEC-Q248's lang=auto is not this ticket's")
        #expect(ForcedAlignerEngine.libraryLanguageName(for: "xx") == nil)
    }

    // MARK: - Input refusals (owner decision 5)

    @Test func aRecordingOverTheLimitIsRefusedBeforeAnyModelWork() async throws {
        try ForcedAlignerEngine.requireWithinLimit(seconds: 300)   // exactly the limit is allowed
        let over = ForcedAlignerEngine.EngineError.overLimit(seconds: 301)
        let sentence = try #require(over.errorDescription)
        #expect(sentence.contains("5 minutes"))
        #expect(sentence.contains("trim") && sentence.contains("split"))

        // A non-effective engine call proves the refusal happens before the (nonexistent) model is loaded.
        let samples = [Float](repeating: 0, count: Int(ForcedAlignerEngine.maxSeconds + 1) * ForcedAlignerEngine.sampleRate)
        let bogus = URL(fileURLWithPath: "/private/tmp/fa-no-such-model-\(UUID().uuidString)")
        await #expect(throws: ForcedAlignerEngine.EngineError.overLimit(seconds: 301)) {
            _ = try await ForcedAlignerEngine.align(samples, transcript: "hello", language: "en", modelDirectory: bogus)
        }
    }

    @Test func anEmptyTranscriptIsRefusedWithASentence() async throws {
        let bogus = URL(fileURLWithPath: "/private/tmp/fa-no-such-model-\(UUID().uuidString)")
        await #expect(throws: ForcedAlignerEngine.EngineError.emptyTranscript) {
            _ = try await ForcedAlignerEngine.align([0, 0.1, -0.1], transcript: "  \n\t ", modelDirectory: bogus)
        }
        #expect(ForcedAlignerEngine.EngineError.emptyTranscript.errorDescription?.contains("does not transcribe") == true)
    }

    @Test func anUnsupportedLanguageIsRefusedByName() async throws {
        let bogus = URL(fileURLWithPath: "/private/tmp/fa-no-such-model-\(UUID().uuidString)")
        await #expect(throws: ForcedAlignerEngine.EngineError.unsupportedLanguage(code: "xx")) {
            _ = try await ForcedAlignerEngine.align([0.1], transcript: "hello", language: "xx", modelDirectory: bogus)
        }
        let sentence = try #require(ForcedAlignerEngine.EngineError.unsupportedLanguage(code: "xx").errorDescription)
        #expect(sentence.contains("'xx'"))
        #expect(sentence.contains("zh, en, yue, fr, de, it, ja, ko, pt, ru, es"))
    }

    @Test func anEmptyRecordingIsRefused() async throws {
        let bogus = URL(fileURLWithPath: "/private/tmp/fa-no-such-model-\(UUID().uuidString)")
        await #expect(throws: ForcedAlignerEngine.EngineError.emptyAudio) {
            _ = try await ForcedAlignerEngine.align([], transcript: "hello", modelDirectory: bogus)
        }
    }

    /// Trap 1: loading reads the installed folder and nothing else. An empty folder throws instead of downloading.
    @Test func aMissingFileThrowsInsteadOfDownloading() async throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("fa-missing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        // The library's own loader, first: no config, no weights, and no repo id to fetch from.
        await #expect(throws: (any Error).self) {
            _ = try await Qwen3ForcedAlignerModel.fromModelDirectory(folder)
        }
        // Through the engine: a plain engine-failure sentence, not a crash and not a download.
        await #expect(throws: StageError.self) {
            _ = try await ForcedAlignerEngine.align([0, 0.1], transcript: "hello", modelDirectory: folder)
        }
    }

    // MARK: - Decision 4's clean-up

    private func items(_ pairs: [(String, Double, Double)]) -> [ForcedAlignItem] {
        pairs.map { ForcedAlignItem(text: $0.0, startTime: $0.1, endTime: $0.2) }
    }

    @Test func timesAreRoundedClippedAndOrdered() {
        let raw = items([
            ("The", -0.5, 0.1234),          // negative start → 0; 0.1234 → 0.123
            ("quick", 0.1234, 0.5678),       // 0.5678 → 0.568
            ("brown", 9.0, 9.5),             // past a 5 s clip → both 5
            ("fox", 2.0, 2.5),               // a late pair after a clipped one: starts never decrease
        ])
        let units = ForcedAlignerEngine.units(from: raw, duration: 5)
        #expect(units.map(\.text) == ["The", "quick", "brown", "fox"], "transcript order is kept")
        #expect(units.map(\.start) == [0, 0.123, 5, 5])
        #expect(units.map(\.end) == [0.123, 0.568, 5, 5])
        for unit in units {
            #expect(unit.end >= unit.start, "\(unit)")
            #expect(unit.start >= 0 && unit.end <= 5, "\(unit)")
            #expect((unit.start * 1000).rounded() == unit.start * 1000, "rounded to the millisecond: \(unit)")
        }
    }

    @Test func anInvertedPairNeverEndsBeforeItStarts() {
        let raw = items([("a", 2.0, 2.5), ("b", 1.0, 1.5)])
        let units = ForcedAlignerEngine.units(from: raw, duration: 10)
        #expect(units[0] == AlignedUnit(text: "a", start: 2.0, end: 2.5))
        #expect(units[1] == AlignedUnit(text: "b", start: 2.0, end: 2.0))
        #expect(zip(units, units.dropFirst()).allSatisfy { $0.start <= $1.start })
    }

    @Test func nonFiniteTimesBecomeZero() {
        let raw = items([("a", .nan, .infinity), ("b", 0.25, 0.5)])
        let units = ForcedAlignerEngine.units(from: raw, duration: 1)
        #expect(units[0] == AlignedUnit(text: "a", start: 0, end: 0))
        #expect(units[1] == AlignedUnit(text: "b", start: 0.25, end: 0.5))
    }

    @Test func noItemsIsAnEmptyResult() {
        #expect(ForcedAlignerEngine.units(from: [], duration: 12).isEmpty)
    }

    // MARK: - The transcript (trap 6)

    /// Trap 6: words, case, punctuation and numbers are never normalised. Only the whitespace layout is (owner ruling 3,
    /// 2026-11-27, quoted on the next test): the library splits on the ASCII space alone, so a pasted newline would
    /// merge two words into one unit.
    @Test func theTranscriptIsTrimmedAndItsWhitespaceLayoutNormalised() {
        #expect(ForcedAlignerEngine.libraryTranscript("  Hello,  World!  ") == "Hello, World!")
        #expect(ForcedAlignerEngine.libraryTranscript("\n\tThe quick brown fox.\n") == "The quick brown fox.")
        #expect(ForcedAlignerEngine.libraryTranscript("Hello\nworld\there") == "Hello world here")
        #expect(ForcedAlignerEngine.libraryTranscript("2024 was 25% better") == "2024 was 25% better")
        #expect(ForcedAlignerEngine.libraryTranscript("Hello, WORLD! 42") == "Hello, WORLD! 42")
        // A whitespace-only transcript is empty after the trim — refused by the empty-transcript sentence.
        #expect(ForcedAlignerEngine.libraryTranscript("   \n\t  ").isEmpty)
    }

    /// Owner ruling 3 (Boo, 2026-11-27): "Keep it only if a test shows the library's units are identical with and without
    /// it, for the English, Chinese and Japanese clips. Otherwise remove it and pass the transcript as given, trimmed only
    /// (Trap 6)." Each committed clip's transcript goes through the pinned library's splitter as written and after
    /// `libraryTranscript`; both give the Swift-side units (the golden's items, except Japanese, which the library splits
    /// finer than the Python reference: owner ruling 8).
    @Test func theWhitespaceCollapseLeavesTheClipUnitsUnchanged() throws {
        let clips = try ForcedAlignerRealLoad.golden().clips
        #expect(clips.map(\.language) == ["en", "zh", "ja"])
        let processor = ForceAlignProcessor()
        for clip in clips {
            let name = try #require(ForcedAlignerEngine.libraryLanguageName(for: clip.language))
            let asGiven = processor.encodeTimestamp(text: clip.text, language: name).0
            let collapsed = processor.encodeTimestamp(text: ForcedAlignerEngine.libraryTranscript(clip.text),
                                                      language: name).0
            let expected = clip.language == "ja" ? ["こんにちは", "世", "界"] : clip.items.map(\.text)
            #expect(asGiven == collapsed, "\(clip.file)")
            #expect(collapsed == expected, "\(clip.file)")
        }
    }

    /// The pinned library's own splitter (public `ForceAlignProcessor.encodeTimestamp` at 5c4c12b): words in
    /// space-separated languages, one unit per Chinese character. Punctuation and casing are dropped/normalised by the
    /// splitter — the engine never re-splits or rewrites what it returns.
    @Test func theLibrarySplitsWordsAndChineseCharacters() {
        let processor = ForceAlignProcessor()
        let english = processor.encodeTimestamp(text: "The quick brown fox jumps.", language: "English")
        #expect(english.0 == ["The", "quick", "brown", "fox", "jumps"], "punctuation is dropped, casing kept")
        #expect(english.1.contains("The<timestamp><timestamp>quick"))
        let chinese = processor.encodeTimestamp(text: "你好世界。", language: "Chinese")
        #expect(chinese.0 == ["你", "好", "世", "界"], "one unit per Chinese character")
        // Japanese has no external tokenizer in the library (its own comment): kana runs stay whole and each kanji is
        // its own unit — a difference from Python's MeCab-style split, recorded here and in the PR.
        let japanese = processor.encodeTimestamp(text: "こんにちは世界", language: "Japanese")
        #expect(japanese.0 == ["こんにちは", "世", "界"])
    }

    /// The committed Japanese clip's units as the pinned library splits them today. Owner ruling 8 (Boo, 2026-11-27):
    /// "The engine and the app's output stay as the library splits them (issue #13 decision 4: don't re-split). Add a test
    /// that pins Swift's current Japanese units (こんにちは, 世, 界), so a future upstream splitter shows up as a deliberate
    /// change." A failure here is an upstream splitter change: revisit the parity test's Japanese join with it.
    @Test func theLibrarySplitsTheJapaneseClipIntoThreeUnits() {
        let japanese = ForceAlignProcessor().encodeTimestamp(text: "こんにちは、世界。", language: "Japanese")
        #expect(japanese.0 == ["こんにちは", "世", "界"])
    }

    @Test func theLibraryLanguageNameSelectsTheChineseSplitter() {
        let processor = ForceAlignProcessor()
        let asChinese = processor.encodeTimestamp(text: "你好", language: ForcedAlignerEngine.libraryLanguageName(for: "zh") ?? "")
        let asEnglish = processor.encodeTimestamp(text: "你好", language: ForcedAlignerEngine.libraryLanguageName(for: "en") ?? "")
        #expect(asChinese.0 == ["你", "好"])
        #expect(asEnglish.0 == ["你", "好"], "even the space path splits CJK characters, but only after a space split")
        let mixed = processor.encodeTimestamp(text: "hello 世界", language: "English")
        #expect(mixed.0 == ["hello", "世", "界"])
    }
}
