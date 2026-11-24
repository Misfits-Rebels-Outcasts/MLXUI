import Foundation
import Testing
@testable import MLXUI

/// PF-6 — span reconstruction, whitespace trimming, non-overlap selection and redaction.
///
/// The embedded cases are opf-derived (`opf/_core/spans.py` + `runtime._select_non_overlapping_spans`
/// + `predict_text`'s typed redaction, `opf` commit `f7f00ca…`) and cover the Unicode shapes from
/// Trap 1: emoji with ZWJ (including a token boundary inside the grapheme), CJK, combining accents,
/// CRLF, tabs, leading/trailing whitespace, and a category subset. The opt-in corpus suite runs the
/// same pipeline through the real tokenizer on every fixture text and compares with `opf` byte for
/// byte.
struct PrivacyFilterRedactionTests {

    struct EmbeddedSpan: Equatable {
        let label: String
        let utf8Start: Int
        let utf8End: Int
        let text: String
    }

    struct EmbeddedSpanCase {
        let name: String
        let text: String
        let tokenByteStarts: [Int]
        let tokenByteEnds: [Int]
        /// BIOES token labels for a sharply peaked synthetic score matrix.
        let tokenLabels: [String]
        let expected: [EmbeddedSpan]
        let redactedAll: String
        let onlyCategories: Set<PrivacyFilterCategory>?
        let redactedOnly: String?
    }

    static let embeddedCases: [EmbeddedSpanCase] = [
        EmbeddedSpanCase(
            name: "ascii_trim_and_email",
            text: "  John Smith  met Ann Lee at x@example.com.",
            tokenByteStarts: [0, 2, 6, 12, 13, 17, 21, 25, 28, 30, 38, 42],
            tokenByteEnds: [2, 6, 12, 13, 17, 21, 25, 28, 30, 38, 42, 43],
            tokenLabels: [
                "O", "B-private_person", "E-private_person", "O", "O",
                "B-private_person", "E-private_person", "O",
                "B-private_email", "I-private_email", "E-private_email", "O",
            ],
            expected: [
                EmbeddedSpan(label: "private_person", utf8Start: 2, utf8End: 12, text: "John Smith"),
                EmbeddedSpan(label: "private_person", utf8Start: 18, utf8End: 25, text: "Ann Lee"),
                EmbeddedSpan(label: "private_email", utf8Start: 29, utf8End: 42, text: "x@example.com"),
            ],
            redactedAll: "  <PRIVATE_PERSON>  met <PRIVATE_PERSON> at <PRIVATE_EMAIL>.",
            onlyCategories: nil,
            redactedOnly: nil),
        EmbeddedSpanCase(
            name: "emoji_zwj_split",
            text: "Hi 👨\u{200d}👩\u{200d}👧 there 👋!",
            tokenByteStarts: [0, 3, 10, 17, 28, 32],
            tokenByteEnds: [3, 10, 17, 28, 32, 33],
            tokenLabels: ["O", "B-private_person", "I-private_person", "E-private_person", "O", "O"],
            expected: [
                EmbeddedSpan(
                    label: "private_person", utf8Start: 3, utf8End: 27,
                    text: "👨\u{200d}👩\u{200d}👧 there"),
            ],
            redactedAll: "Hi <PRIVATE_PERSON> 👋!",
            onlyCategories: nil,
            redactedOnly: nil),
        EmbeddedSpanCase(
            name: "cjk_person_email",
            text: "联系人 张伟，邮箱 zhang@example.com。",
            tokenByteStarts: [0, 9, 16, 19, 25, 31, 43],
            tokenByteEnds: [9, 16, 19, 25, 31, 43, 46],
            tokenLabels: [
                "O", "B-private_person", "E-private_person", "O", "O",
                "B-private_email", "E-private_email",
            ],
            expected: [
                EmbeddedSpan(label: "private_person", utf8Start: 10, utf8End: 19, text: "张伟，"),
                EmbeddedSpan(label: "private_email", utf8Start: 31, utf8End: 46, text: "@example.com。"),
            ],
            redactedAll: "联系人 <PRIVATE_PERSON>邮箱 zhang<PRIVATE_EMAIL>",
            onlyCategories: nil,
            redactedOnly: nil),
        EmbeddedSpanCase(
            name: "combining_accent_split",
            text: "Cafe\u{0301} 555-1234 ok",
            tokenByteStarts: [0, 4, 6, 10, 15],
            tokenByteEnds: [4, 6, 10, 15, 18],
            tokenLabels: ["B-private_person", "E-private_person", "B-private_phone", "E-private_phone", "O"],
            expected: [
                EmbeddedSpan(label: "private_person", utf8Start: 0, utf8End: 6, text: "Cafe\u{0301}"),
                EmbeddedSpan(label: "private_phone", utf8Start: 7, utf8End: 15, text: "555-1234"),
            ],
            redactedAll: "<PRIVATE_PERSON> <PRIVATE_PHONE> ok",
            onlyCategories: nil,
            redactedOnly: nil),
        EmbeddedSpanCase(
            name: "crlf_tab_date",
            text: "Plan\r\n\tDue 2024-01-05 \r\n\tend",
            tokenByteStarts: [0, 4, 6, 10, 15, 18, 21, 24],
            tokenByteEnds: [4, 6, 10, 15, 18, 21, 24, 28],
            tokenLabels: ["O", "O", "O", "B-private_date", "I-private_date", "E-private_date", "O", "O"],
            expected: [
                EmbeddedSpan(label: "private_date", utf8Start: 11, utf8End: 21, text: "2024-01-05"),
            ],
            redactedAll: "Plan\r\n\tDue <PRIVATE_DATE> \r\n\tend",
            onlyCategories: nil,
            redactedOnly: nil),
        EmbeddedSpanCase(
            name: "only_subset",
            text: "Kim at kim@example.com key AKIA1234END",
            tokenByteStarts: [0, 3, 6, 10, 22, 26, 31, 35],
            tokenByteEnds: [3, 6, 10, 22, 26, 31, 35, 38],
            tokenLabels: [
                "S-private_person", "O", "B-private_email", "E-private_email", "O",
                "B-secret", "I-secret", "E-secret",
            ],
            expected: [
                EmbeddedSpan(label: "private_person", utf8Start: 0, utf8End: 3, text: "Kim"),
                EmbeddedSpan(label: "private_email", utf8Start: 7, utf8End: 22, text: "kim@example.com"),
                EmbeddedSpan(label: "secret", utf8Start: 27, utf8End: 38, text: "AKIA1234END"),
            ],
            redactedAll: "<PRIVATE_PERSON> at <PRIVATE_EMAIL> key <SECRET>",
            onlyCategories: Set([.email, .person]),
            redactedOnly: "<PRIVATE_PERSON> at <PRIVATE_EMAIL> key AKIA1234END"),
        EmbeddedSpanCase(
            name: "no_spans",
            text: "nothing personal here",
            tokenByteStarts: [0, 7, 16],
            tokenByteEnds: [7, 16, 21],
            tokenLabels: ["O", "O", "O"],
            expected: [],
            redactedAll: "nothing personal here",
            onlyCategories: nil,
            redactedOnly: nil),
    ]

    // MARK: - Helpers

    private static func peakedLogProbs(tokenLabels: [String]) -> [[Float]] {
        let names = PrivacyFilterLabelInfo.canonicalLabelNames
        return tokenLabels.map { name in
            var row = [Float](repeating: -3.0, count: 33)
            if let index = names.firstIndex(of: name) { row[index] = 2.0 }
            return row
        }
    }

    private func tokenization(_ testCase: EmbeddedSpanCase) -> PrivacyFilterTokenizer.Tokenization {
        PrivacyFilterTokenizer.Tokenization(
            tokenIDs: Array(0 ..< testCase.tokenLabels.count),
            utf8ByteStarts: testCase.tokenByteStarts,
            utf8ByteEnds: testCase.tokenByteEnds)
    }

    private func detectedSpans(
        _ testCase: EmbeddedSpanCase,
        decoder: PrivacyFilterDecoder = PrivacyFilterDecoder(labelInfo: .standard, calibration: .zero)
    ) throws -> [PrivacyFilterSpan] {
        try PrivacyFilterRedactor.detectedSpans(
            testCase.text,
            logProbs: Self.peakedLogProbs(tokenLabels: testCase.tokenLabels),
            tokenization: tokenization(testCase),
            decoder: decoder)
    }

    // MARK: - Embedded opf cases

    @Test func everyEmbeddedCaseMatchesTheReferenceSpans() throws {
        for testCase in Self.embeddedCases {
            let spans = try detectedSpans(testCase)
            #expect(spans.count == testCase.expected.count, "\(testCase.name): span count")
            if spans.count == testCase.expected.count {
                for (span, expected) in zip(spans, testCase.expected) {
                    #expect(span.category.modelLabel == expected.label, "\(testCase.name): label")
                    #expect(span.utf8Start == expected.utf8Start, "\(testCase.name): start")
                    #expect(span.utf8End == expected.utf8End, "\(testCase.name): end")
                    #expect(span.text == expected.text, "\(testCase.name): covered text")
                }
            }
        }
    }

    @Test func everyEmbeddedCaseRedactsToTheReferenceText() throws {
        for testCase in Self.embeddedCases {
            let redaction = try PrivacyFilterRedactor.redact(
                testCase.text,
                logProbs: Self.peakedLogProbs(tokenLabels: testCase.tokenLabels),
                tokenization: tokenization(testCase),
                decoder: PrivacyFilterDecoder(labelInfo: .standard, calibration: .zero),
                categories: nil)
            #expect(redaction.redactedText == testCase.redactedAll, "\(testCase.name): all categories")
            #expect(redaction.tokenCount == testCase.tokenLabels.count)
            if let categories = testCase.onlyCategories, let expected = testCase.redactedOnly {
                let subset = try PrivacyFilterRedactor.redact(
                    testCase.text,
                    logProbs: Self.peakedLogProbs(tokenLabels: testCase.tokenLabels),
                    tokenization: tokenization(testCase),
                    decoder: PrivacyFilterDecoder(labelInfo: .standard, calibration: .zero),
                    categories: categories)
                #expect(subset.redactedText == expected, "\(testCase.name): category subset")
                // The subset must not change the detected spans, and unchecked categories must leave
                // their original text in place.
                #expect(subset.spans == redaction.spans)
                #expect(subset.redactedText.contains("AKIA1234END"))
            }
        }
    }

    @Test func onlySubsetLeavesOtherCategoriesTextInPlace() throws {
        let testCase = try #require(Self.embeddedCases.first { $0.name == "only_subset" })
        let spans = try detectedSpans(testCase)
        let redacted = PrivacyFilterRedactor.redactedText(
            testCase.text, spans: spans, categories: Set([.email, .person]))
        #expect(redacted.contains("AKIA1234END"))
        #expect(!redacted.contains("<SECRET>"))
        #expect(redacted.contains("<PRIVATE_PERSON>"))
        #expect(redacted.contains("<PRIVATE_EMAIL>"))
        // nil means all eight.
        let all = PrivacyFilterRedactor.redactedText(testCase.text, spans: spans, categories: nil)
        #expect(all.contains("<SECRET>"))
        #expect(!all.contains("AKIA1234END"))
        // An empty set replaces nothing.
        #expect(PrivacyFilterRedactor.redactedText(testCase.text, spans: spans, categories: [])
                == testCase.text)
    }

    @Test func characterRangesSnapOutwardToGraphemeBoundaries() throws {
        let testCase = try #require(Self.embeddedCases.first { $0.name == "emoji_zwj_split" })
        let spans = try detectedSpans(testCase)
        #expect(spans.count == 1)
        let span = try #require(spans.first)
        // The ZWJ family emoji is one Swift Character; the span must not split it.
        #expect(span.characterRange == 3 ..< 10)
        let characters = Array(testCase.text)
        let covered = String(characters[span.characterRange])
        #expect(covered.hasPrefix("👨\u{200d}👩\u{200d}👧"))
    }

    @Test func graphemeRangeExpandsBeyondATokenThatSplitsAGrapheme() {
        let text = "a👨\u{200d}👩\u{200d}👧b"
        let starts = PrivacyFilterSpans.graphemeByteStarts(text)
        // Bytes 3…? are inside the family emoji (bytes 1…19); a token boundary there must snap out.
        let range = PrivacyFilterSpans.graphemeRange(
            utf8Start: 3, utf8End: 10, graphemeStarts: starts)
        #expect(range == 1 ..< 2)
        #expect(String(Array(text)[range]) == "👨\u{200d}👩\u{200d}👧")
    }

    @Test func whitespaceTrimMatchesPythonIsspace() {
        #expect(PrivacyFilterSpans.isPythonWhitespace(" "))
        #expect(PrivacyFilterSpans.isPythonWhitespace("\t"))
        #expect(PrivacyFilterSpans.isPythonWhitespace("\r"))
        #expect(PrivacyFilterSpans.isPythonWhitespace("\n"))
        #expect(PrivacyFilterSpans.isPythonWhitespace("\u{00a0}"))
        #expect(PrivacyFilterSpans.isPythonWhitespace("\u{3000}"))
        #expect(PrivacyFilterSpans.isPythonWhitespace("\u{001c}")) // CPython-only information separator
        #expect(!PrivacyFilterSpans.isPythonWhitespace("\u{200b}")) // zero-width space is not whitespace
        #expect(!PrivacyFilterSpans.isPythonWhitespace("\u{180e}")) // Mongolian vowel separator
        #expect(!PrivacyFilterSpans.isPythonWhitespace("a"))
    }

    // MARK: - Non-overlap selection

    private func span(
        _ category: PrivacyFilterCategory, _ start: Int, _ end: Int, in text: String
    ) -> PrivacyFilterSpan {
        let bytes = Array(text.utf8)
        let graphemes = PrivacyFilterSpans.graphemeByteStarts(text)
        return PrivacyFilterSpan(
            category: category,
            utf8Start: start,
            utf8End: end,
            text: String(decoding: bytes[start ..< end], as: UTF8.self),
            characterRange: PrivacyFilterSpans.graphemeRange(
                utf8Start: start, utf8End: end, graphemeStarts: graphemes))
    }

    @Test func selectionKeepsTheLongestSpanStartingFirst() {
        let text = "abcdefghij"
        let long = span(.person, 0, 6, in: text)
        let shortEarlier = span(.secret, 0, 4, in: text)
        let overlapping = span(.email, 2, 4, in: text)
        let later = span(.date, 8, 10, in: text)
        let kept = PrivacyFilterRedactor.selectNonOverlappingSpans([shortEarlier, later, overlapping, long])
        #expect(kept.map { "\($0.category.rawValue):\($0.utf8Start)-\($0.utf8End)" }
            == ["person:0-6", "date:8-10"])
    }

    @Test func selectionSplitsTiesByLabelName() {
        let text = "abcdefgh"
        // Same start and length: the label string decides ("private_email" < "private_person").
        let person = span(.person, 0, 3, in: text)
        let email = span(.email, 0, 3, in: text)
        let kept = PrivacyFilterRedactor.selectNonOverlappingSpans([person, email])
        #expect(kept.map(\.category) == [.email])
    }

    @Test func selectionKeepsAdjacentAndDisjointSpans() {
        let text = "abcdef"
        let first = span(.person, 0, 3, in: text)
        let adjacent = span(.email, 3, 6, in: text)
        #expect(PrivacyFilterRedactor.selectNonOverlappingSpans([adjacent, first]) == [first, adjacent])
    }

    // MARK: - Input limits (decision 5, pure validation)

    @Test func emptyAndWhitespaceOnlyTextIsRefused() {
        #expect(throws: PrivacyFilterInferenceError.self) {
            try PrivacyFilterInference.validateInput("")
        }
        #expect(throws: PrivacyFilterInferenceError.self) {
            try PrivacyFilterInference.validateInput(" \t\r\n \u{00a0}")
        }
        // A single visible character is enough.
        try? PrivacyFilterInference.validateInput("a")
    }

    @Test func overTheTokenLimitIsRefused() {
        #expect(PrivacyFilterInference.maxTokens == 128_000)
        try? PrivacyFilterInference.validateTokenCount(128_000)
        do {
            try PrivacyFilterInference.validateTokenCount(128_001)
            Issue.record("128,001 tokens were accepted")
        } catch let error as PrivacyFilterInferenceError {
            let sentence = error.errorDescription ?? ""
            #expect(sentence.contains("128000"))
            #expect(sentence.contains("Shorten it"))
        } catch {
            Issue.record("wrong error: \(error)")
        }
    }

    @Test func refusalSentencesNeverQuoteInput() {
        for error in [PrivacyFilterInferenceError.emptyText,
                      PrivacyFilterInferenceError.tooManyTokens(200_000)] {
            let sentence = error.errorDescription ?? ""
            #expect(!sentence.contains("John"))
            #expect(!sentence.isEmpty)
        }
    }

    // MARK: - Corpus parity (opt-in)

    @Suite(.enabled(
        if: PrivacyFilterDecodeTestSupport.canRunCorpus,
        "needs MLXUI_TEST_PRIVACY=1, MLXUI_TEST_PRIVACY_DIR and the checked-in fixtures"))
    struct CorpusRedactionTests {

        private func loadPipeline() async throws -> (PrivacyFilterTokenizer, PrivacyFilterDecoder) {
            let directory = try #require(PrivacyFilterDecodeTestSupport.tokenizerDirectory)
            let tokenizer = try await PrivacyFilterTokenizer(modelDirectory: directory)
            let calibration = try PrivacyFilterCalibration.load(from: directory)
            return (tokenizer, PrivacyFilterDecoder(labelInfo: .standard, calibration: calibration))
        }

        /// Spans **and** redacted text equal opf's on every corpus text (decision 4 / acceptance).
        @Test func everyCorpusTextRedactsByteForByteLikeOpf() async throws {
            let (tokenizer, decoder) = try await loadPipeline()
            let golden = try PrivacyFilterDecodeTestSupport.loadGolden()
            var cases = PrivacyFilterDecodeTestSupport.shortCases(golden).map { (name: $0.name, value: $0.value) }
            if let longDocument = golden.long_document { cases.append((name: "long_document", value: longDocument)) }
            var checked = 0
            var totalSpans = 0
            for (name, testCase) in cases {
                let tokenization = try tokenizer.tokenize(testCase.text)
                guard let logProbsFile = testCase.log_probs else { continue }
                let logProbs = try PrivacyFilterDecodeTestSupport.floatRows(logProbsFile)
                #expect(logProbs.count == tokenization.tokenCount, "\(name): log-prob rows vs tokens")
                let redaction = try PrivacyFilterRedactor.redact(
                    testCase.text,
                    logProbs: logProbs,
                    tokenization: tokenization,
                    decoder: decoder,
                    categories: nil)
                #expect(redaction.redactedText == testCase.redacted_text, "\(name): redacted text")
                #expect(redaction.spans.count == testCase.spans.count,
                        "\(name): \(redaction.spans.count) spans vs \(testCase.spans.count)")
                if redaction.spans.count == testCase.spans.count {
                    for (span, expected) in zip(redaction.spans, testCase.spans) {
                        #expect(span.category.modelLabel == expected.label, "\(name): label")
                        #expect(span.utf8Start == expected.start_byte, "\(name): start byte")
                        #expect(span.utf8End == expected.end_byte, "\(name): end byte")
                        #expect(span.text == expected.text, "\(name): covered text")
                        totalSpans += 1
                    }
                }
                checked += 1
            }
            print("PF6 corpus: \(checked)/\(cases.count) texts redact byte-for-byte like opf (\(totalSpans) spans)")
            PrivacyFilterDecodeTestSupport.record("PF6 corpus: \(checked)/\(cases.count) texts redact byte-for-byte like opf (\(totalSpans) spans)")
        }

        @Test func categorySubsetOnlyChangesTheRedactedText() async throws {
            let (tokenizer, decoder) = try await loadPipeline()
            let golden = try PrivacyFilterDecodeTestSupport.loadGolden()
            // The case with the most categories exercises the subset on several spans.
            let candidates = PrivacyFilterDecodeTestSupport.shortCases(golden)
                .filter { $0.value.log_probs != nil }
            let mixed = try #require(candidates.max { $0.value.spans.count < $1.value.spans.count })
            let tokenization = try tokenizer.tokenize(mixed.value.text)
            let logProbs = try PrivacyFilterDecodeTestSupport.floatRows(
                try #require(mixed.value.log_probs))

            let all = try PrivacyFilterRedactor.redact(
                mixed.value.text, logProbs: logProbs, tokenization: tokenization,
                decoder: decoder, categories: nil)
            let emailOnly = try PrivacyFilterRedactor.redact(
                mixed.value.text, logProbs: logProbs, tokenization: tokenization,
                decoder: decoder, categories: Set([.email, .person]))
            #expect(all.spans == emailOnly.spans, "the category subset must not change the spans")
            #expect(all.tokenCount == emailOnly.tokenCount)

            // Rebuild the subset text independently from the original bytes: unchecked spans keep
            // their original text, checked spans get their placeholder.
            let bytes = Array(mixed.value.text.utf8)
            var expected: [UInt8] = []
            var cursor = 0
            for span in all.spans {
                expected.append(contentsOf: bytes[cursor ..< span.utf8Start])
                if span.category == .email || span.category == .person {
                    expected.append(contentsOf: span.category.placeholder.utf8)
                } else {
                    expected.append(contentsOf: bytes[span.utf8Start ..< span.utf8End])
                }
                cursor = span.utf8End
            }
            expected.append(contentsOf: bytes[cursor...])
            #expect(emailOnly.redactedText == String(decoding: expected, as: UTF8.self))
        }

        @Test func emptyAndWhitespaceOnlyCorpusCasesHaveNoSpans() async throws {
            let (tokenizer, decoder) = try await loadPipeline()
            let golden = try PrivacyFilterDecodeTestSupport.loadGolden()
            for name in ["empty_string", "whitespace_only"] {
                guard let testCase = golden.cases[name] else { continue }
                let tokenization = try tokenizer.tokenize(testCase.text)
                let logProbs: [[Float]]
                if let file = testCase.log_probs, file.shape.first ?? 0 > 0 {
                    logProbs = try PrivacyFilterDecodeTestSupport.floatRows(file)
                } else {
                    logProbs = []
                }
                if logProbs.isEmpty {
                    #expect(testCase.spans.isEmpty, "\(name)")
                    continue
                }
                let redaction = try PrivacyFilterRedactor.redact(
                    testCase.text, logProbs: logProbs, tokenization: tokenization,
                    decoder: decoder, categories: nil)
                #expect(redaction.spans.isEmpty, "\(name)")
                #expect(redaction.redactedText == testCase.text, "\(name)")
            }
        }
    }
}
