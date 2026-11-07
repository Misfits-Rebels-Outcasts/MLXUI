import Foundation
import Testing
@testable import MLXUI

/// MC2-5 — the Translate screen's view model with a stub stage and a stub token counter: swap, Auto detection shown,
/// the low-confidence prompt, the refusal sentences, and Copy being unavailable while a run is going. No weights, no
/// GUI. The real screen is `TranslateGemmaRunView`.
///
/// A stage that records whether it ran and returns/throws what the test asks.
private nonisolated final class TranslateRunStubStage: PipelineStage, @unchecked Sendable {
    let id = "stub"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    private let lock = NSLock()
    private var calls = 0
    private let body: @Sendable (String) async throws -> String

    init(_ body: @escaping @Sendable (String) async throws -> String) { self.body = body }

    var callCount: Int { lock.lock(); defer { lock.unlock() }; return calls }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        lock.lock(); calls += 1; lock.unlock()
        guard case .text(let text) = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
        return .text(try await body(text))
    }
}

@MainActor
struct TranslateGemmaRunModelTests {

    private func model() -> TranslateGemmaRunModel {
        TranslateGemmaRunModel(modelID: TranslateGemmaCatalogTests.cardID,
                               displayName: "TranslateGemma 4B",
                               modelDirectory: URL(fileURLWithPath: "/tmp/nonexistent-translategemma"),
                               footprintBytes: 1)
    }

    /// The full suite can be heavily loaded (parallel test processes); the run's own work is a few main-actor hops, so a
    /// generous bound is a no-op when healthy and keeps a loaded machine from recording a false timeout.
    private func waitUntilIdle(_ model: TranslateGemmaRunModel, timeout: Duration = .seconds(30)) async throws {
        let deadline = ContinuousClock.now + timeout
        while model.isRunning, ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(5))
        }
        if model.isRunning {
            Issue.record("waitUntilIdle timed out after \(timeout) waiting for the Translate run to go idle")
        }
    }

    private nonisolated static func german() -> TranslateGemmaLanguageDetector.Detection {
        .init(code: "de", name: "German", confidence: 0.93)
    }

    // MARK: - Swap

    @Test func swapExchangesTwoChosenLanguages() {
        let m = model()
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.swap()
        #expect(m.sourceSelection == "ms")
        #expect(m.targetSelection == "en")
        #expect(m.targetLabel == "English")
    }

    @Test func swapUnderAutoUsesTheDetection() {
        let m = model()
        m.detect = { _ in Self.german() }
        m.targetSelection = "ms"
        m.sourceText = "Guten Morgen, wie geht es dir?"
        m.sourceTextChanged()
        #expect(m.detected?.code == "de")
        #expect(m.canSwap)
        m.swap()
        #expect(m.sourceSelection == "ms")
        #expect(m.targetSelection == "de")
        #expect(m.detected == nil)
    }

    @Test func swapIsUnavailableUntilAutoDetectsSomething() {
        let m = model()
        m.detect = { _ in nil }
        m.sourceText = "..."
        m.sourceTextChanged()
        #expect(!m.canSwap)
        let before = m.sourceSelection
        m.swap()
        #expect(m.sourceSelection == before, "swap must not invent a language")
    }

    // MARK: - Auto

    @Test func aDetectedLanguageIsShown() {
        let m = model()
        m.detect = { _ in Self.german() }
        m.sourceText = "Guten Morgen, wie geht es dir?"
        m.sourceTextChanged()
        #expect(m.detected?.name == "German")
        #expect(m.sourceLabel == "Auto (German)")
        #expect(m.resolvedSourceCode == "de")
    }

    @Test func lowConfidenceAsksThePersonToPick() {
        let m = model()
        m.detect = { _ in nil }
        m.sourceText = "xyz zzz qqq"
        m.sourceTextChanged()
        let stage = TranslateRunStubStage { text in "should not run: \(text)" }
        m.start(stage: stage)
        let sentence = m.errorText ?? ""
        #expect(sentence.contains("couldn't tell which language"))
        #expect(sentence.contains("Pick the source language"))
        #expect(stage.callCount == 0, "no run starts from a guessed language")
        #expect(!m.isRunning)
    }

    // MARK: - Refusals (decision 8 and the plain sentences)

    @Test func anOverLimitInputIsRefusedBeforeAnyWork() {
        let m = model()
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.sourceText = "A long passage."
        m.promptTokenCount = 2049
        #expect(m.isOverLimit)
        #expect(m.tokenCountLabel == "2049 / 2048 tokens")
        #expect(!m.canTranslate)
        let stage = TranslateRunStubStage { _ in "no" }
        m.start(stage: stage)
        let sentence = m.errorText ?? ""
        #expect(sentence.contains("2049"))
        #expect(sentence.contains("2,048"))
        #expect(stage.callCount == 0, "over-limit is refused before the engine")
    }

    @Test func anEmptyTextIsASentence() {
        let m = model()
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.sourceText = "   \n"
        let stage = TranslateRunStubStage { _ in "no" }
        m.start(stage: stage)
        #expect(m.errorText == TranslateGemmaError.emptyText.description)
        #expect(stage.callCount == 0)
    }

    @Test func theTokenCountIsTheRenderedCountAndClearsWhenItCannotBeRendered() async {
        let m = model()
        m.countTokens = { request in
            request.text == "Hello" ? 37 : 0
        }
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.sourceText = "Hello"
        await m.refreshTokenCount()
        #expect(m.promptTokenCount == 37)
        #expect(m.tokenCountLabel == "37 / 2048 tokens")
        #expect(!m.isOverLimit)
        // An empty text cannot be rendered, so there is no count to show.
        m.sourceText = ""
        await m.refreshTokenCount()
        #expect(m.promptTokenCount == nil)
        #expect(m.tokenCountLabel == "— / 2048 tokens")
    }

    // MARK: - Running, Copy, Stop

    @Test func aSuccessfulRunFillsTheResultAndEnablesCopy() async throws {
        let m = model()
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.sourceText = "The weather is lovely today."
        m.promptTokenCount = 30
        #expect(!m.canCopy)
        let stage = TranslateRunStubStage { _ in "Cuaca hari ini sangat baik." }
        m.start(stage: stage)
        #expect(m.isRunning, "the run is live until the stage returns")
        #expect(!m.canCopy, "Copy is unavailable while translating")
        #expect(!m.canTranslate)
        try await waitUntilIdle(m)
        #expect(m.resultText == "Cuaca hari ini sangat baik.")
        #expect(m.canCopy)
        #expect(!m.isRunning)
        #expect(m.errorText == nil)
    }

    @Test func stopEndsTheRunWithoutATranslation() async throws {
        let m = model()
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.sourceText = "The weather is lovely today."
        m.promptTokenCount = 30
        let stage = TranslateRunStubStage { _ in
            try await Task.sleep(for: .seconds(5))
            return "never"
        }
        m.start(stage: stage)
        #expect(m.isRunning)
        m.cancel()
        try await waitUntilIdle(m)
        #expect(!m.isRunning)
        #expect(m.didStop, "a stopped run is marked stopped, not returned as a translation")
        #expect(m.resultText == nil)
        #expect(m.errorText == nil, "Stop is not an error")
        #expect(!m.canCopy)
    }

    @Test func anEngineRefusalBecomesASentence() async throws {
        let m = model()
        m.sourceSelection = "en"
        m.targetSelection = "ms"
        m.sourceText = "Hello"
        m.promptTokenCount = 12
        let stage = TranslateRunStubStage { _ in throw TranslateGemmaError.promptTooLong(tokens: 3000) }
        m.start(stage: stage)
        try await waitUntilIdle(m)
        let sentence = m.errorText ?? ""
        #expect(sentence.contains("3000"))
        #expect(sentence.contains("Shorten it and try again."))
        #expect(m.resultText == nil)
    }

    @Test func thePickersShowTheFavouritesAndEveryOtherLanguageOnce() {
        let m = model()
        #expect(TranslateGemmaLanguages.favourites.count == 8)
        #expect(m.otherLanguages.count == TranslateGemmaLanguages.all.count - 8)
        #expect(Set(m.otherLanguages.map(\.code)).isDisjoint(with: Set(TranslateGemmaLanguages.favourites.map(\.code))))
        #expect(m.otherLanguages.contains { $0.name == "Zulu" })
        // No duplicated tag between the two picker sections.
        let tags = TranslateGemmaLanguages.favourites.map(\.code) + m.otherLanguages.map(\.code)
        #expect(Set(tags).count == tags.count)
    }
}
