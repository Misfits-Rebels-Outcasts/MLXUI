import Foundation
import SwiftUI
import Testing
@testable import MLXUI

/// PF-7 — the Redact Run screen's view model, without weights. Every test drives ``RedactRunModel`` through its stub
/// stage seam, so none of these needs the checkpoint. Text is synthetic only (decision 10).
@MainActor
struct PrivacyFilterRunTests {

    // MARK: - Seams

    /// A stage that records the config it was built for and hands one canned outcome back. `calls` counts stage runs,
    /// which is how "the checkboxes never re-run the model" is proven.
    private final class StubRun: @unchecked Sendable {
        private let lock = NSLock()
        private var runs = 0
        private var configs: [StageConfig] = []
        var outcome: PrivacyFilterEngine.Outcome?
        var failure: Error?
        /// One second per run, so the Stop test can cancel a run that is genuinely in flight.
        var sleeps = false

        var runCount: Int { lock.lock(); defer { lock.unlock() }; return runs }
        var seenConfigs: [StageConfig] { lock.lock(); defer { lock.unlock() }; return configs }
        func note(_ config: StageConfig) { lock.lock(); configs.append(config); runs += 1; lock.unlock() }

        func make(config: StageConfig, box: RedactRunModel.RedactionBox) -> any PipelineStage {
            note(config)
            let outcome = outcome
            let failure = failure
            let sleeps = sleeps
            return PrivacyFilterStage(id: "stub", name: "Privacy Filter") { _ in
                if sleeps { try await Task.sleep(for: .seconds(2)) }
                if let failure { throw failure }
                guard let outcome else { throw StageError.engineFailure(stage: "stub", underlying: StubRunNoOutcome()) }
                box.store(outcome)
                return outcome
            }
        }
    }

    private struct StubRunNoOutcome: Error {}

    private func makeModel(_ run: StubRun) -> RedactRunModel {
        let model = RedactRunModel(modelID: "openai--privacy-filter", displayName: "Privacy Filter",
                                   modelDirectory: URL(fileURLWithPath: "/nonexistent/pf"),
                                   footprintBytes: 3_000_000_000)
        model.makeStage = { config, box in run.make(config: config, box: box) }
        return model
    }

    private func waitUntil(timeout: Duration = .seconds(5), _ condition: @MainActor () -> Bool) async {
        let deadline = ContinuousClock.now + timeout
        while !condition() && ContinuousClock.now < deadline {
            try? await Task.sleep(for: .milliseconds(10))
        }
    }

    /// UTF-8 byte offsets and Character offsets of `sub` inside `text`, computed independently of the redactor.
    private func offsets(_ text: String, _ sub: String) -> (bytes: Range<Int>, characters: Range<Int>) {
        guard let range = text.range(of: sub) else { preconditionFailure("test text lacks \(sub)") }
        let startByte = text.utf8.distance(from: text.utf8.startIndex, to: range.lowerBound.samePosition(in: text.utf8)!)
        let endByte = text.utf8.distance(from: text.utf8.startIndex, to: range.upperBound.samePosition(in: text.utf8)!)
        let startChar = text.distance(from: text.startIndex, to: range.lowerBound)
        let endChar = text.distance(from: text.startIndex, to: range.upperBound)
        return (startByte ..< endByte, startChar ..< endChar)
    }

    /// The two synthetic spans (a person and an email) and the text they were found in.
    private func syntheticCase() -> (text: String, spans: [PrivacyFilterSpan]) {
        let text = "Alex Sample wrote to alex.sample@example.com today."
        let person = offsets(text, "Alex Sample")
        let email = offsets(text, "alex.sample@example.com")
        return (text, [
            PrivacyFilterSpan(category: .person, utf8Start: person.bytes.lowerBound,
                              utf8End: person.bytes.upperBound, text: "Alex Sample",
                              characterRange: person.characters),
            PrivacyFilterSpan(category: .email, utf8Start: email.bytes.lowerBound,
                              utf8End: email.bytes.upperBound, text: "alex.sample@example.com",
                              characterRange: email.characters),
        ])
    }

    private func outcome(for text: String, spans: [PrivacyFilterSpan],
                         categories: Set<PrivacyFilterCategory>?) -> PrivacyFilterEngine.Outcome {
        .init(redactedText: PrivacyFilterRedactor.redactedText(text, spans: spans, categories: categories),
              spans: spans, tokenCount: 12, cancelled: false)
    }

    // MARK: - Decision 6's caption

    @Test func theCaptionIsVerbatim() {
        #expect(RedactRunModel.caption
                == "Finds common personal data, mainly in English. It can miss things, so review before you share. "
                + "Not an anonymisation or compliance guarantee.")
    }

    // MARK: - Input-limit refusals

    @Test func emptyInputIsRefusedBeforeAnyStageIsBuilt() {
        let run = StubRun()
        let model = makeModel(run)
        for text in ["", "   ", "\n\t \n"] {
            model.inputText = text
            model.start(handedStage: PrivacyFilterStage(id: "handed", name: "Privacy Filter") { _ in
                throw StageError.engineFailure(stage: "handed", underlying: StubRunNoOutcome())
            })
            #expect(model.errorText == PrivacyFilterInferenceError.emptyText.errorDescription)
            #expect(!model.isRunning)
            #expect(model.redactedText == nil)
        }
        #expect(run.runCount == 0, "an empty input must not build or run a stage")
    }

    /// Decision 5: an over-limit input is refused with one plain sentence that carries the token count and never the
    /// input. The engine refuses before the forward pass (PF-6's `PrivacyFilterInference`), and the view shows its
    /// sentence verbatim.
    @Test func overLimitInputShowsTheOneSentenceRefusal() async {
        let run = StubRun()
        run.failure = PrivacyFilterInferenceError.tooManyTokens(128_001)
        let model = makeModel(run)
        model.inputText = String(repeating: "Synthetic sentence. ", count: 10)
        model.start(handedStage: PrivacyFilterStage(id: "handed", name: "Privacy Filter") { _ in
            throw StageError.engineFailure(stage: "handed", underlying: StubRunNoOutcome())
        })
        await waitUntil { !model.isRunning }
        let sentence = try? #require(model.errorText)
        #expect(sentence == PrivacyFilterInferenceError.tooManyTokens(128_001).errorDescription)
        #expect(sentence?.contains("128000") == true, "the sentence carries the limit")
        #expect(model.redactedText == nil)
        #expect(model.spans.isEmpty)
    }

    // MARK: - One run: side-by-side result

    @Test func redactReturnsTheRedactedTextAndAllDetectedSpans() async throws {
        let run = StubRun()
        let (text, spans) = syntheticCase()
        run.outcome = outcome(for: text, spans: spans, categories: nil)
        let model = makeModel(run)
        model.inputText = text
        model.start(handedStage: PrivacyFilterStage(id: "handed", name: "Privacy Filter") { _ in
            throw StageError.engineFailure(stage: "handed", underlying: StubRunNoOutcome())
        })
        await waitUntil { !model.isRunning && model.hasResult }
        #expect(model.redactedText == "<PRIVATE_PERSON> wrote to <PRIVATE_EMAIL> today.")
        #expect(model.spans.map(\.category) == [.person, .email])
        #expect(model.tokenCount == 12)
        #expect(model.errorText == nil)
        #expect(!model.didStop)
        // The run carried all eight categories (decision 4's default).
        #expect(run.seenConfigs.count == 1)
        #expect(run.seenConfigs.first?.redactOnly == Set(PrivacyFilterCategory.allCases))
    }

    // MARK: - Category checkboxes (decision 6: re-render, never re-run)

    @Test func categoryCheckboxesReRenderWithoutASecondModelCall() async throws {
        let run = StubRun()
        let (text, spans) = syntheticCase()
        run.outcome = outcome(for: text, spans: spans, categories: nil)
        let model = makeModel(run)
        model.inputText = text
        model.start(handedStage: PrivacyFilterStage(id: "handed", name: "Privacy Filter") { _ in
            throw StageError.engineFailure(stage: "handed", underlying: StubRunNoOutcome())
        })
        await waitUntil { !model.isRunning && model.hasResult }
        let runsAfterRedact = run.runCount
        #expect(runsAfterRedact == 1)

        // Keep only email: the person span is detected but no longer replaced.
        model.selectedCategories = [.email]
        model.categoriesChanged()
        #expect(model.redactedText == "Alex Sample wrote to <PRIVATE_EMAIL> today.")

        // Keep only person: the email is left in place.
        model.selectedCategories = [.person]
        model.categoriesChanged()
        #expect(model.redactedText == "<PRIVATE_PERSON> wrote to alex.sample@example.com today.")

        // All eight again: the effective set is `nil` (the engine's "all" default).
        model.selectedCategories = Set(PrivacyFilterCategory.allCases)
        model.categoriesChanged()
        #expect(model.effectiveCategories == nil)
        #expect(model.redactedText == "<PRIVATE_PERSON> wrote to <PRIVATE_EMAIL> today.")

        // None: nothing is replaced at all, and the text is unchanged.
        model.selectedCategories = []
        model.categoriesChanged()
        #expect(model.redactedText == text)

        #expect(run.runCount == runsAfterRedact, "the checkboxes must never re-run the model")
    }

    /// The re-render works on the text the spans were detected in, even if the editor was edited afterwards.
    @Test func reRenderIgnoresLaterEditsToTheEditor() async throws {
        let run = StubRun()
        let (text, spans) = syntheticCase()
        run.outcome = outcome(for: text, spans: spans, categories: nil)
        let model = makeModel(run)
        model.inputText = text
        model.start(handedStage: PrivacyFilterStage(id: "handed", name: "Privacy Filter") { _ in
            throw StageError.engineFailure(stage: "handed", underlying: StubRunNoOutcome())
        })
        await waitUntil { !model.isRunning && model.hasResult }
        model.inputText = "A completely different text."
        model.selectedCategories = [.email]
        model.categoriesChanged()
        #expect(model.redactedText == "Alex Sample wrote to <PRIVATE_EMAIL> today.")
    }

    // MARK: - Export spans .json (UTF-8 byte offsets)

    /// Decision 4: exported JSON carries **UTF-8 byte** offsets, and slicing the original bytes with them gives exactly
    /// each span's text. The text has a non-ASCII character, so a Character-offset implementation would fail this.
    @Test func exportedJSONByteOffsetsSliceTheOriginalUTF8() throws {
        let text = "Zoë Sample emailed zoe.sample@example.com."
        let person = offsets(text, "Zoë Sample")
        let email = offsets(text, "zoe.sample@example.com")
        let spans = [
            PrivacyFilterSpan(category: .person, utf8Start: person.bytes.lowerBound,
                              utf8End: person.bytes.upperBound, text: "Zoë Sample",
                              characterRange: person.characters),
            PrivacyFilterSpan(category: .email, utf8Start: email.bytes.lowerBound,
                              utf8End: email.bytes.upperBound, text: "zoe.sample@example.com",
                              characterRange: email.characters),
        ]
        let data = RedactRunModel.spansJSON(spans)
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["encoding"] as? String == "utf-8-bytes")
        let records = try #require(object["spans"] as? [[String: Any]])
        #expect(records.count == 2)
        let bytes = Array(text.utf8)
        for (record, span) in zip(records, spans.sorted { $0.utf8Start < $1.utf8Start }) {
            let start = try #require(record["start"] as? Int)
            let end = try #require(record["end"] as? Int)
            #expect(start == span.utf8Start)
            #expect(end == span.utf8End)
            #expect(record["category"] as? String == span.category.rawValue)
            #expect(record["label"] as? String == span.category.modelLabel)
            #expect(record["text"] as? String == span.text)
            let slice = String(decoding: bytes[start ..< end], as: UTF8.self)
            #expect(slice == span.text, "byte offsets must slice the original UTF-8 to the span text")
        }
        // Proof it is bytes, not Characters: the ë is two bytes, so the person span ends at byte 11 but Character 10.
        let first = try #require(records.first)
        #expect(first["end"] as? Int == 11)
        #expect(person.characters.upperBound == 10)
    }

    @Test func theExportIsEmptyAndStableWithoutSpans() throws {
        let data = RedactRunModel.spansJSON([])
        let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect((object["spans"] as? [Any])?.isEmpty == true)
    }

    // MARK: - Stop

    @Test func stopEndsTheRunWithNoResultAndNoError() async throws {
        let run = StubRun()
        run.sleeps = true
        let model = makeModel(run)
        model.inputText = "Synthetic text that will be stopped."
        model.start(handedStage: PrivacyFilterStage(id: "handed", name: "Privacy Filter") { _ in
            throw StageError.engineFailure(stage: "handed", underlying: StubRunNoOutcome())
        })
        await waitUntil { model.isRunning }
        #expect(model.isRunning)
        model.stop()
        await waitUntil { !model.isRunning }
        #expect(!model.isRunning)
        #expect(model.didStop)
        #expect(model.redactedText == nil)
        #expect(model.errorText == nil)
    }

    // MARK: - File input

    @Test func openingANonUTF8FileIsOnePlainSentence() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("pf-non-utf8-\(UUID().uuidString).txt")
        // 0xFF is not valid UTF-8.
        try Data([0x48, 0x69, 0xFF, 0x0A]).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let model = makeModel(StubRun())
        model.open(url)
        #expect(model.errorText == "That file couldn't be read as UTF-8 text. Save it as UTF-8 and try again.")
        #expect(model.inputText.isEmpty)
        #expect(model.redactedText == nil)
    }

    @Test func droppingAnUnsupportedFileTypeIsRefusedWithASentence() {
        let model = makeModel(StubRun())
        model.drop([URL(fileURLWithPath: "/tmp/synthetic.pdf")])
        #expect(model.errorText == "Drop a .txt or .md file — other file types aren't read here.")
        #expect(model.inputText.isEmpty)
    }

    @Test func suggestedFileNamesKeepTheSourceStem() {
        #expect(RedactRunModel.suggestedFileName(sourceName: "letter.txt", suffix: "redacted", ext: "txt")
                == "letter-redacted.txt")
        #expect(RedactRunModel.suggestedFileName(sourceName: "notes.md", suffix: "spans", ext: "json")
                == "notes-spans.json")
        #expect(RedactRunModel.suggestedFileName(sourceName: nil, suffix: "redacted", ext: "txt")
                == "redacted.txt")
    }

    // MARK: - The highlight the original pane renders

    @Test func theOriginalHighlightsEveryDetectedSpan() {
        let text = "Alex Sample wrote to alex.sample@example.com today."
        let (_, spans) = syntheticCase()
        let attributed = PrivacyFilterHighlight.attributed(text, spans: spans)
        #expect(String(attributed.characters) == text)
        var highlighted = 0
        for run in attributed.runs where run.backgroundColor != nil { highlighted += 1 }
        #expect(highlighted == spans.count, "every detected span is tinted, whatever its checkbox says")
    }
}
