import Foundation
import Testing
import MLX
@testable import MLXUI

/// PF-7 — the real-weights tests for OpenAI Privacy Filter. They are **opt-in twice over**: skipped unless
/// `MLXUI_TEST_PRIVACY=1` (it reaches the host as `TEST_RUNNER_MLXUI_TEST_PRIVACY=1`) **and**
/// `MLXUI_TEST_PRIVACY_DIR` points at an installed copy — normally a `/private/tmp` copy, because the sandboxed test
/// host cannot read `~/Library/Application Support` (`SAMAudioRealLoadTests`' pattern). A skipped test is not evidence.
///
/// The checkpoint is `openai/privacy-filter` at revision `7ffa9a043d54d1be65afb281eddf0ffbe629385b`, copied to
/// `/private/tmp/pf-model`. All text here is synthetic (decision 10); nothing prints input or span text.
enum PrivacyFilterRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static let hfID = "openai/privacy-filter"

    nonisolated static let directory: URL? = {
        let url: URL
        if let override = env["MLXUI_TEST_PRIVACY_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: hfID)
        }
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.appendingPathComponent("model.safetensors").path),
              fm.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path),
              fm.fileExists(atPath: url.appendingPathComponent("viterbi_calibration.json").path) else { return nil }
        return url
    }()

    nonisolated static var optedIn: Bool {
        (env["MLXUI_TEST_PRIVACY"] ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY"]) == "1"
    }

    nonisolated static var canRun: Bool { optedIn && directory != nil }

    /// One synthetic business-letter sentence covering four categories.
    nonisolated static let letterSentence =
        "Alex Sample lives at 12 Example Street. Email alex.sample@example.com or call 555 010 1234."

    /// A short synthetic sentence with no obvious personal data (the company name is a negative by design).
    nonisolated static let negativeSentence =
        "The Example Company published its quarterly summary for the whole industry to read."

    /// The reference corpus (`privacy-golden.json`): 25 short synthetic texts with `opf`'s spans (UTF-8 byte
    /// offsets) and redacted text, plus one 6,019-token document. Every text is synthetic (decision 10).
    nonisolated static var corpusURL: URL? {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/PrivacyFilter/privacy-golden.json")
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The long document takes ~5 minutes of real inference; it is part of the corpus suite only when asked for.
    nonisolated static var corpusLongDocument: Bool {
        (env["MLXUI_TEST_PRIVACY_CORPUS_LONG"] ?? env["TEST_RUNNER_MLXUI_TEST_PRIVACY_CORPUS_LONG"]) == "1"
    }

    /// The acceptance lines, also written beside the run (the harness cannot read a test host's stdout; the same
    /// evidence-file pattern `PrivacyFilterFootprintTests` uses). Metadata only — no input or span text.
    nonisolated static func record(_ line: String) {
        let url = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("privacy-filter-corpus.txt")
        let previous = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
        try? (previous + line + "\n").write(to: url, atomically: true, encoding: .utf8)
    }
}

/// One fixture case, decoded from `privacy-golden.json`. Snake-case keys are decoded with `.convertFromSnakeCase`;
/// `text`/span `text` stay in memory only and are never printed (decision 10).
private nonisolated struct PrivacyFilterGoldenCase: Decodable {
    struct Span: Decodable {
        let startByte: Int
        let endByte: Int
        let label: String
        let text: String
        let placeholder: String
    }
    struct Agreement: Decodable {
        let transformersDerivedSpansEqualOpf: Bool
        let transformersDerivedRedactedEqualsOpf: Bool
    }
    let text: String
    let spans: [Span]
    let redactedText: String
    let agreement: Agreement
    let tokenCount: Int
}

private nonisolated struct PrivacyFilterGoldenFixture: Decodable {
    let cases: [String: PrivacyFilterGoldenCase]
    let longDocument: PrivacyFilterGoldenCase
}

@Suite(.serialized)
struct PrivacyFilterRealLoadTests {

    private func directory() throws -> URL { try #require(PrivacyFilterRealLoad.directory) }

    // MARK: - Load + redact

    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func redactsASyntheticBusinessSentence() async throws {
        let outcome = try await PrivacyFilterEngine.redact(
            PrivacyFilterRealLoad.letterSentence, modelDirectory: try directory(), holder: .runWindow)
        print("PRIVACYFILTER spans=\(outcome.spans.count) categories=\(outcome.spans.map(\.category.rawValue).joined(separator: ",")) " +
              "tokens=\(outcome.tokenCount) changed=\(outcome.redactedText != PrivacyFilterRealLoad.letterSentence)")
        #expect(!outcome.cancelled)
        #expect(!outcome.spans.isEmpty, "a letter with a person, address, email and phone must find something")
        #expect(outcome.redactedText != PrivacyFilterRealLoad.letterSentence)
        #expect(!outcome.redactedText.contains("alex.sample@example.com"),
                "the email span is replaced by its placeholder")
        #expect(outcome.redactedText.contains("<PRIVATE_"), "the placeholder form is decision 4's")
        #expect(outcome.tokenCount > 0)
        // Every span's byte offsets slice the original UTF-8 to its own text (decision 4, Trap 1).
        let bytes = Array(PrivacyFilterRealLoad.letterSentence.utf8)
        for span in outcome.spans {
            #expect(span.utf8Start >= 0 && span.utf8End <= bytes.count && span.utf8Start < span.utf8End)
            #expect(String(decoding: bytes[span.utf8Start ..< span.utf8End], as: UTF8.self) == span.text)
        }
    }

    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func aNegativeSentenceStillRoundTripsEveryByte() async throws {
        let text = PrivacyFilterRealLoad.negativeSentence
        let outcome = try await PrivacyFilterEngine.redact(text, modelDirectory: try directory(), holder: .runWindow)
        print("PRIVACYFILTER negative spans=\(outcome.spans.count) tokens=\(outcome.tokenCount)")
        #expect(!outcome.cancelled)
        // Whatever it decides, the contract holds: every byte outside a span is unchanged. With no spans that is the
        // whole text; with spans it is the text minus those spans plus placeholders — never anything else.
        if outcome.spans.isEmpty {
            #expect(outcome.redactedText == text)
        } else {
            let bytes = Array(text.utf8)
            for span in outcome.spans {
                #expect(String(decoding: bytes[span.utf8Start ..< span.utf8End], as: UTF8.self) == span.text)
            }
        }
    }

    /// Decision 6: the eight checkboxes are a re-render over cached spans — the redactor can be re-run on the same
    /// spans with a different subset and gives exactly the runs' own text back for the full set.
    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func cachedSpansReRenderWithoutTheModel() async throws {
        let text = PrivacyFilterRealLoad.letterSentence
        let outcome = try await PrivacyFilterEngine.redact(text, modelDirectory: try directory(), holder: .runWindow)
        let allAgain = PrivacyFilterRedactor.redactedText(text, spans: outcome.spans, categories: nil)
        #expect(allAgain == outcome.redactedText)
        // A subset keeps at least the bytes outside the excluded spans: no excluded-category placeholder appears.
        if let firstCategory = outcome.spans.first?.category {
            let subset = Set(PrivacyFilterCategory.allCases).subtracting([firstCategory])
            let subsetText = PrivacyFilterRedactor.redactedText(text, spans: outcome.spans, categories: subset)
            let excludedCategoryStillThere = outcome.spans.contains { $0.category == firstCategory }
            if excludedCategoryStillThere {
                let placeholder = firstCategory.placeholder
                let occurrencesInFull = outcome.redactedText.components(separatedBy: placeholder).count - 1
                let occurrencesInSubset = subsetText.components(separatedBy: placeholder).count - 1
                #expect(occurrencesInSubset < occurrencesInFull,
                        "excluding \(firstCategory.rawValue) removes its placeholders only")
            }
        }
    }

    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func isDeterministicAcrossTwoRuns() async throws {
        let text = PrivacyFilterRealLoad.letterSentence
        let dir = try directory()
        let first = try await PrivacyFilterEngine.redact(text, modelDirectory: dir, holder: .runWindow)
        let second = try await PrivacyFilterEngine.redact(text, modelDirectory: dir, holder: .runWindow)
        print("PRIVACYFILTER determinism: \(first.spans == second.spans ? "identical" : "DIFFERENT") " +
              "spans=\(first.spans.count)")
        #expect(first.spans == second.spans)
        #expect(first.redactedText == second.redactedText)
        #expect(first.tokenCount == second.tokenCount)
    }

    /// Trap 7 / decision 10: the whole Run path awaited from the main actor records no MLX fallback error, and a second
    /// run keeps doing so (a failed array read would abort the host, hence the probe).
    @MainActor
    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func redactFromTheMainActorRecordsNoMLXError() async throws {
        let dir = try directory()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for text in [PrivacyFilterRealLoad.letterSentence, PrivacyFilterRealLoad.negativeSentence] {
            let outcome = try await MLXErrorGuard.guardedRun {
                try await PrivacyFilterEngine.redact(text, modelDirectory: dir, holder: .runWindow)
            }
            #expect(!outcome.cancelled)
            #expect(!outcome.redactedText.isEmpty)
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    // MARK: - Stop

    /// Stop is a cancellation between windows: an already-cancelled callback returns `.cancelled`, never a failure and
    /// never a partial redaction.
    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func aCancelledRunIsReportedAsCancelledNotAsAFailure() async throws {
        let outcome = try await PrivacyFilterEngine.redact(
            PrivacyFilterRealLoad.letterSentence, modelDirectory: try directory(), holder: .runWindow,
            isCancelled: { true })
        print("PRIVACYFILTER stop cancelled=\(outcome.cancelled) spans=\(outcome.spans.count)")
        #expect(outcome.cancelled)
        #expect(outcome.redactedText.isEmpty)
        #expect(outcome.spans.isEmpty)
    }

    // MARK: - The input limit (decision 5), on the real tokenizer

    // MARK: - The acceptance corpus: the real model must equal opf's golden exactly

    /// The acceptance criterion PF-5/PF-6/PF-7 together: **with the real Swift model**, every fixture text's spans
    /// (category + UTF-8 byte ranges) and redacted text equal `opf`'s, byte for byte. The two refusal cases (empty and
    /// whitespace-only, decision 5) must be refused before any forward pass instead. Prints `PFCORPUS … N/N`.
    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func everyCorpusTextMatchesTheOpfGoldenWithTheRealModel() async throws {
        let dir = try directory()
        let url = try #require(PrivacyFilterRealLoad.corpusURL, "the corpus fixture is missing")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let fixture = try decoder.decode(PrivacyFilterGoldenFixture.self, from: Data(contentsOf: url))

        var matched = 0
        var refusals = 0
        var mismatches: [String] = []
        for name in fixture.cases.keys.sorted() {
            guard let golden = fixture.cases[name] else { continue }
            #expect(golden.agreement.transformersDerivedSpansEqualOpf,
                    "\(name): the fixture itself records a transformers-vs-opf span mismatch")
            #expect(golden.agreement.transformersDerivedRedactedEqualsOpf,
                    "\(name): the fixture itself records a transformers-vs-opf redaction mismatch")
            let text = golden.text
            let trimmedEmpty = text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            if trimmedEmpty {
                // Decision 5: at least one non-whitespace character; refused with one plain sentence.
                do {
                    _ = try await PrivacyFilterEngine.redact(text, modelDirectory: dir)
                    mismatches.append("\(name): empty/whitespace-only text was not refused")
                } catch let error as PrivacyFilterInferenceError {
                    if error == .emptyText { refusals += 1; matched += 1 }
                    else { mismatches.append("\(name): wrong refusal \(error)") }
                }
                continue
            }
            let outcome = try await PrivacyFilterEngine.redact(text, modelDirectory: dir, holder: .runWindow)
            // Spans: same count, same categories, same UTF-8 byte ranges (ordering normalised by start/end).
            let gotSpans = outcome.spans.sorted {
                $0.utf8Start == $1.utf8Start ? $0.utf8End < $1.utf8End : $0.utf8Start < $1.utf8Start
            }
            let wantSpans = golden.spans.sorted {
                $0.startByte == $1.startByte ? $0.endByte < $1.endByte : $0.startByte < $1.startByte
            }
            var spansOK = gotSpans.count == wantSpans.count
            if spansOK {
                for (index, pair) in zip(gotSpans, wantSpans).enumerated() {
                    let wantCategory = PrivacyFilterCategory(modelLabel: pair.1.label)
                    if pair.0.category != wantCategory
                        || pair.0.utf8Start != pair.1.startByte
                        || pair.0.utf8End != pair.1.endByte
                        || pair.0.category.placeholder != pair.1.placeholder {
                        spansOK = false
                        mismatches.append("\(name): span \(index) differs (bytes \(pair.0.utf8Start)..<\(pair.0.utf8End)"
                            + " vs \(pair.1.startByte)..<\(pair.1.endByte), category \(pair.0.category.rawValue)"
                            + " vs \(wantCategory?.rawValue ?? "?"))")
                        break
                    }
                }
            } else {
                mismatches.append("\(name): \(gotSpans.count) spans vs golden \(wantSpans.count)")
            }
            // The redacted text: byte-for-byte equal (String equality over the same UTF-8).
            let redactedOK = outcome.redactedText == golden.redactedText
            if !redactedOK {
                mismatches.append("\(name): redacted text differs (\(outcome.redactedText.utf8.count) B vs "
                    + "\(golden.redactedText.utf8.count) B)")
            }
            if spansOK && redactedOK { matched += 1 }
        }
        let total = fixture.cases.count
        let line = "PFCORPUS real-model spans/redacted: \(matched)/\(total) (\(refusals) refusal cases)"
        print(line)
        PrivacyFilterRealLoad.record(line)
        #expect(mismatches.isEmpty, "real-model corpus mismatches:\n\(mismatches.prefix(10).joined(separator: "\n"))")
        #expect(matched == total, "real-model corpus: \(matched)/\(total)")
    }

    /// The 6,019-token long document (425 opf spans) through the real model's windowed path. Opt-in: ~5 minutes.
    @Test(.enabled(if: PrivacyFilterRealLoad.canRun && PrivacyFilterRealLoad.corpusLongDocument))
    func theLongDocumentMatchesTheOpfGoldenWithTheRealModel() async throws {
        let dir = try directory()
        let url = try #require(PrivacyFilterRealLoad.corpusURL, "the corpus fixture is missing")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let fixture = try decoder.decode(PrivacyFilterGoldenFixture.self, from: Data(contentsOf: url))
        let golden = fixture.longDocument
        let outcome = try await PrivacyFilterEngine.redact(golden.text, modelDirectory: dir, holder: .runWindow)
        let gotSpans = outcome.spans.sorted {
            $0.utf8Start == $1.utf8Start ? $0.utf8End < $1.utf8End : $0.utf8Start < $1.utf8Start
        }
        let wantSpans = golden.spans.sorted {
            $0.startByte == $1.startByte ? $0.endByte < $1.endByte : $0.startByte < $1.startByte
        }
        var spanMatches = 0
        for (got, want) in zip(gotSpans, wantSpans) where got.category == PrivacyFilterCategory(modelLabel: want.label)
            && got.utf8Start == want.startByte && got.utf8End == want.endByte {
            spanMatches += 1
        }
        let redactedOK = outcome.redactedText == golden.redactedText
        let line = "PFCORPUS long-document: tokens=\(outcome.tokenCount) spans=\(spanMatches)/\(wantSpans.count) "
            + "redacted=\(redactedOK ? "equal" : "DIFFERENT") spanCount=\(gotSpans.count)/\(wantSpans.count)"
        print(line)
        PrivacyFilterRealLoad.record(line)
        #expect(gotSpans.count == wantSpans.count)
        #expect(spanMatches == wantSpans.count)
        #expect(redactedOK)
    }

    @Test(.enabled(if: PrivacyFilterRealLoad.canRun))
    func anOverLimitInputIsRefusedBeforeTheModelRuns() async throws {
        let dir = try directory()
        // The refusal needs the tokenizer, not a forward pass: build a text over 128,000 tokens and ask the inference
        // directly. Nothing reaches the model's logits path.
        let tokenizer = try await PrivacyFilterTokenizer(modelDirectory: dir)
        let paragraph = "The quarterly report records ordinary business details for the regional office. "
        var paragraphs = 1
        var text = paragraph
        while true {
            let count = try tokenizer.tokenize(text).tokenIDs.count
            if count > PrivacyFilterInference.maxTokens { break }
            paragraphs *= 2
            text = String(repeating: paragraph, count: paragraphs)
        }
        let tokens = try tokenizer.tokenize(text).tokenIDs.count
        print("PRIVACYFILTER over-limit refusal tokens=\(tokens) limit=\(PrivacyFilterInference.maxTokens)")
        #expect(tokens > PrivacyFilterInference.maxTokens)
        let inference = try await PrivacyFilterEngineCache.shared.inference(for: dir)
        do {
            _ = try inference.redact(text, categories: nil, progress: nil, isCancelled: { false })
            Issue.record("an over-limit input must be refused")
        } catch let error as PrivacyFilterInferenceError {
            #expect(error == .tooManyTokens(tokens))
            if case .tooManyTokens(let refused) = error {
                #expect(refused == tokens, "the sentence carries the real token count")
            }
        }
    }
}
