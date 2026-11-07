import Foundation
import Testing
import MLX
import MLXLMCommon
import Tokenizers
@testable import MLXUI

/// MC2-4 — the real-weights tests for TranslateGemma 4B. They are **opt-in twice over**: skipped unless
/// `MLXUI_TEST_TRANSLATEGEMMA=1` (it reaches the host as `TEST_RUNNER_MLXUI_TEST_TRANSLATEGEMMA=1`) **and**
/// `MLXUI_TEST_TRANSLATEGEMMA_DIR` points at an installed copy — normally a `/private/tmp` copy, because the
/// sandboxed test host cannot read `~/Library/Application Support`. Without the variable, `ModelStore`'s own folder is
/// tried. A skipped test does not count as evidence (the issue says so).
///
/// The downloaded model for this suite was `mlx-community/translategemma-4b-it-4bit` at revision
/// `5788ec08c047f3f2e17808101b8d9566ac930d58`, copied to `/private/tmp/translategemma`.
enum TranslateGemmaRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static let hfID = "mlx-community/translategemma-4b-it-4bit"

    nonisolated static let directory: URL? = {
        let url: URL
        if let override = env["MLXUI_TEST_TRANSLATEGEMMA_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_TRANSLATEGEMMA_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: hfID)
        }
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.appendingPathComponent("model.safetensors").path),
              fm.fileExists(atPath: url.appendingPathComponent("chat_template.jinja").path),
              fm.fileExists(atPath: url.appendingPathComponent("tokenizer.json").path) else { return nil }
        return url
    }()

    nonisolated static var optedIn: Bool {
        (env["MLXUI_TEST_TRANSLATEGEMMA"] ?? env["TEST_RUNNER_MLXUI_TEST_TRANSLATEGEMMA"]) == "1"
    }

    nonisolated static var canRun: Bool { optedIn && directory != nil }

    /// Fixed sentences at temperature 0 (the issue's review pairs; EN→MS and EN→ZH-Hans here).
    nonisolated static let englishSentence = "The weather is lovely today, and the river is calm."
    nonisolated static let malaySentence = "Cuaca hari ini sangat baik. Saya suka membaca buku di tepi sungai."
    nonisolated static let chineseSentence = "今天天气很好。我们沿着河边散步。"

    /// CJK ideographs (Simplified Chinese output is expected to contain some).
    nonisolated static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value) || (0x3400...0x4DBF).contains(scalar.value)
        }
    }

    nonisolated static func makeRequest(_ text: String, _ source: String, _ target: String) -> TranslateGemmaEngine.Request {
        .init(text: text, sourceCode: source, targetCode: target)
    }
}

@Suite(.serialized)
struct TranslateGemmaRealLoadTests {

    private func directory() throws -> URL {
        try #require(TranslateGemmaRealLoad.directory)
    }

    // MARK: - Load + translate

    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func translatesEnglishIntoMalay() async throws {
        let dir = try directory()
        let outcome = try await TranslateGemmaEngine.translate(
            TranslateGemmaRealLoad.makeRequest(TranslateGemmaRealLoad.englishSentence, "en", "ms"),
            modelDirectory: dir)
        print("TRANSLATEGEMMA en→ms: \(outcome.text) [prompt=\(outcome.promptTokens) generated=\(outcome.generatedTokens)]")
        #expect(!outcome.cancelled)
        #expect(!outcome.text.isEmpty)
        #expect(outcome.text != TranslateGemmaRealLoad.englishSentence)
        // It ends at the turn: no turn markers in the output and nothing like a runaway generation.
        #expect(!outcome.text.contains("<end_of_turn>"))
        #expect(!outcome.text.contains("<eos>"))
        #expect(!outcome.text.contains("<bos>"))
        #expect(!outcome.text.contains("<start_of_turn>"))
        #expect(outcome.generatedTokens < 200, "the turn ended; it did not run on")
        #expect(outcome.generatedTokens < TranslateGemmaLanguages.maxOutputTokens)
        #expect(outcome.promptTokens < TranslateGemmaLanguages.maxPromptTokens)
    }

    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func translatesEnglishIntoSimplifiedChinese() async throws {
        let dir = try directory()
        let outcome = try await TranslateGemmaEngine.translate(
            TranslateGemmaRealLoad.makeRequest(TranslateGemmaRealLoad.englishSentence, "en", "zh-Hans"),
            modelDirectory: dir)
        print("TRANSLATEGEMMA en→zh-Hans: \(outcome.text) [prompt=\(outcome.promptTokens) generated=\(outcome.generatedTokens)]")
        #expect(!outcome.cancelled)
        #expect(!outcome.text.isEmpty)
        #expect(TranslateGemmaRealLoad.containsCJK(outcome.text), "the output is in Chinese script")
        #expect(!outcome.text.contains("<end_of_turn>"))
        #expect(outcome.generatedTokens < 200)
    }

    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func isDeterministicAtTemperatureZero() async throws {
        let dir = try directory()
        let request = TranslateGemmaRealLoad.makeRequest(TranslateGemmaRealLoad.englishSentence, "en", "ms")
        let first = try await TranslateGemmaEngine.translate(request, modelDirectory: dir)
        let second = try await TranslateGemmaEngine.translate(request, modelDirectory: dir)
        print("TRANSLATEGEMMA determinism: \(first.text == second.text ? "identical" : "DIFFERENT")")
        #expect(first.text == second.text, "temperature 0 must give the same translation twice")
        #expect(first.generatedTokens == second.generatedTokens)
    }

    /// The four pairs the PR asks the owner to read (the issue's "Read these"): fixed sentences, temperature 0, source
    /// and output pasted into the PR. Not asserted for content — the owner judges the translation.
    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func printsTheFourReviewPairs() async throws {
        let dir = try directory()
        let pairs: [(String, String, String)] = [
            ("EN→MS", "en", TranslateGemmaRealLoad.englishSentence),
            ("EN→ZH-Hans", "en", TranslateGemmaRealLoad.englishSentence),
            ("MS→EN", "ms", TranslateGemmaRealLoad.malaySentence),
            ("ZH-Hans→EN", "zh-Hans", TranslateGemmaRealLoad.chineseSentence),
        ]
        var report = "TranslateGemma 4B — the four review pairs (temperature 0)\n\n"
        for (label, source, text) in pairs {
            let target = source == "en" ? (label.contains("ZH") ? "zh-Hans" : "ms") : "en"
            let outcome = try await TranslateGemmaEngine.translate(
                .init(text: text, sourceCode: source, targetCode: target), modelDirectory: dir)
            let line = "TGEMPAIR \(label)\n  source: \(text)\n  output: \(outcome.text)\n  prompt=\(outcome.promptTokens) generated=\(outcome.generatedTokens)"
            print(line)
            report += line + "\n\n"
            #expect(!outcome.cancelled)
            #expect(!outcome.text.isEmpty)
            #expect(!outcome.text.contains("<end_of_turn>"))
        }
        // The harness cannot read a test host's stdout, so the PR's evidence is also written beside it.
        try? report.write(to: FileManager.default.temporaryDirectory
            .appendingPathComponent("translategemma-four-pairs.txt"), atomically: true, encoding: .utf8)
    }

    // MARK: - The template on the real tokenizer (Traps 3 and 4)

    /// The tokenizer the app loads, for the checks that do not need the weights at all.
    private func tokenizer() async throws -> any Tokenizers.Tokenizer {
        try await AutoTokenizer.from(modelFolder: try directory())
    }

    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func theRenderedPromptStartsWithExactlyOneBOS() async throws {
        let tokenizer = try await tokenizer()
        let messages = try TranslateGemmaPrompt.messages(sourceCode: "en", targetCode: "ms",
                                                         text: TranslateGemmaRealLoad.englishSentence)
        let ids = try tokenizer.applyChatTemplate(messages: messages)
        print("TRANSLATEGEMMA prompt ids: first=\(ids.first ?? -1) count=\(ids.count) bosCount=\(ids.filter { $0 == 2 }.count)")
        #expect(ids.first == 2, "the template's own <bos> is the first token")
        #expect(ids.filter { $0 == 2 }.count == 1, "exactly one <bos> (Trap 4)")
        #expect(ids.count > 1, "the template rendered (not the silent [BOS]-only fallback)")
        // The real tokenizer's `applyChatTemplate` is the run's prompt; the pure renderer is the preview's prompt.
        // They must be the same ids (FV-2: a run and its preview never differ).
        for (source, target, text) in [("en", "ms", TranslateGemmaRealLoad.englishSentence),
                                       ("zh-Hans", "en", TranslateGemmaRealLoad.chineseSentence),
                                       ("en_GB", "ms", TranslateGemmaRealLoad.englishSentence)] {
            let rendered = try TranslateGemmaPrompt.render(sourceCode: source, targetCode: target, text: text)
            let renderedIds = tokenizer.encode(text: rendered, addSpecialTokens: false)
            let templateIds = try tokenizer.applyChatTemplate(
                messages: try TranslateGemmaPrompt.messages(sourceCode: source, targetCode: target, text: text))
            #expect(renderedIds == templateIds, "\(source)→\(target): the pure renderer and the real template differ")
        }
    }

    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func theEndOfTurnTokenIsOneHundredAndSix() async throws {
        let tokenizer = try await tokenizer()
        #expect(tokenizer.convertTokenToId("<end_of_turn>") == TranslateGemmaEngine.endOfTurnTokenId)
        #expect(tokenizer.convertTokenToId("<eos>") == TranslateGemmaEngine.eosTokenId)
        #expect(TranslateGemmaEngine.stopTokenIds == [1, 106])
    }

    // MARK: - Stop, on the first generated token (an event, not a delay)

    /// A box the `onToken` callback writes the first token into, and that cancels the run when it sees it.
    private final class StopOnFirstToken: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<TranslateGemmaEngine.Outcome, Error>?
        private var firstTokenClock: ContinuousClock.Instant?
        private var firstTokenId: Int?
        private var fired = false

        var firstToken: (id: Int, at: ContinuousClock.Instant)? {
            lock.lock(); defer { lock.unlock() }
            guard let id = firstTokenId, let at = firstTokenClock else { return nil }
            return (id, at)
        }

        func attach(_ task: Task<TranslateGemmaEngine.Outcome, Error>) {
            lock.lock(); defer { lock.unlock() }
            self.task = task
        }

        func sawToken(_ id: Int) {
            lock.lock()
            if !fired {
                fired = true
                firstTokenId = id
                firstTokenClock = ContinuousClock.now
                let task = self.task
                lock.unlock()
                task?.cancel()      // the event itself stops the run
            } else {
                lock.unlock()
            }
        }
    }

    @Test(.enabled(if: TranslateGemmaRealLoad.canRun))
    func stoppingOnTheFirstGeneratedTokenEndsTheRunPromptly() async throws {
        let dir = try directory()
        let box = StopOnFirstToken()
        let request = TranslateGemmaRealLoad.makeRequest(TranslateGemmaRealLoad.englishSentence, "en", "ms")
        let task = Task { try await TranslateGemmaEngine.translate(request, modelDirectory: dir, onToken: { id in
            box.sawToken(id)
        }) }
        box.attach(task)
        let outcome = try await task.value
        let first = try #require(box.firstToken)
        let margin = ContinuousClock.now - first.at
        print("TRANSLATEGEMMA stop: firstToken=\(first.id) outcomeCancelled=\(outcome.cancelled) " +
              "text=\(outcome.text.isEmpty ? "<empty>" : outcome.text) margin=\(margin)")
        #expect(outcome.cancelled, "a stopped run is marked cancelled, not returned as a translation")
        #expect(outcome.text.isEmpty, "a cancelled run must not surface partial text as a translation")
        #expect(margin < .seconds(2), "generation stopped within a short margin of the first token")
    }
}
