import Foundation
import Testing
@testable import MLXUI

/// AM-T — TranslateGemma 4B (`mlx-community/translategemma-4b-it-4bit`), owner ruling R8 (MC2-11): install, the four fixed
/// language pairs, determinism, a clean stop, one flow row through `CachingExecutor(RealExecutor)`, and memory at the
/// engine's 2,048-token limit. Text only — no image phase. Run with:
///
///     bash RSI/evals/model-smoke.sh run mlx-community/translategemma-4b-it-4bit --suite TranslateGemmaSmokeTests
///
/// Gated on MLXUI_SMOKE_REPO == hfID, so the normal suite (and CI) skips it; a skipped run does not count.
///
/// Every stage is built on the **checkpoint directory** `ModelSmoke.checkpoint(for:)` returns, never `ModelStore.shared`.
/// The four pairs drive `TranslateGemmaRunModel` the way the Translate screen's button does (select languages, count the
/// rendered tokens, `start(stage:)`); the stop/limit/memory checks use `TranslateGemmaEngine` directly because they need
/// the outcome's token counts.
@Suite(.enabled(if: ModelSmoke.enabled("mlx-community/translategemma-4b-it-4bit"), "run through RSI/evals/model-smoke.sh"), .serialized)
@MainActor
struct TranslateGemmaSmokeTests {
    static let hfID = "mlx-community/translategemma-4b-it-4bit"
    static let display = "TranslateGemma 4B"      // the CatalogBridge display name
    static let auto = "MC2-11"                     // the AUTO smoke row this suite fills
    static let review = "MC2-11r"                  // the owner's REVIEW row (artifacts)

    // Fixed natural sentences, 15-25 words each.
    static let english = "We reached the trailhead just after sunrise, and the weather looked clear, so we packed water and set off up the hill."
    static let malay = "Doktor meminta saya datang semula pada hari Selasa depan untuk memeriksa keputusan ujian darah dan berbincang tentang ubat yang baru."
    static let chinese = "今天早上天气很好，我们沿着河边慢慢走到山脚下，然后在小店里喝了一杯热茶。"

    struct Pair {
        let label: String, slug: String, source: String, target: String, text: String
    }
    static let pairs: [Pair] = [
        Pair(label: "EN→MS", slug: "en-ms", source: "en", target: "ms", text: english),
        Pair(label: "EN→ZH-Hans", slug: "en-zh-hans", source: "en", target: "zh-Hans", text: english),
        Pair(label: "MS→EN", slug: "ms-en", source: "ms", target: "en", text: malay),
        Pair(label: "ZH-Hans→EN", slug: "zh-hans-en", source: "zh-Hans", target: "en", text: chinese),
    ]

    // MARK: Oracles

    static let malayStopWords: Set<String> = ["yang", "dan", "di", "itu", "ini", "untuk", "dengan", "pada", "adalah", "saya", "ke", "dari"]
    static let englishStopWords: Set<String> = ["the", "and", "is", "to", "of", "in", "a", "for", "with", "on"]
    static let commentaryPrefixes = ["here is", "here's", "here are", "translation:", "sure", "okay", "ok,", "certainly"]

    static func isHan(_ s: Unicode.Scalar) -> Bool {
        (0x4E00...0x9FFF).contains(s.value) || (0x3400...0x4DBF).contains(s.value) || (0x20000...0x2A6DF).contains(s.value)
    }

    /// Han characters / non-space, non-punctuation, non-symbol characters.
    static func hanRatio(_ text: String) -> Double {
        let scalars = text.unicodeScalars.filter {
            !CharacterSet.whitespacesAndNewlines.contains($0) && !CharacterSet.punctuationCharacters.contains($0)
                && !CharacterSet.symbols.contains($0)
        }
        guard !scalars.isEmpty else { return 0 }
        return Double(scalars.filter(isHan).count) / Double(scalars.count)
    }

    static func hanCount(_ text: String) -> Int { text.unicodeScalars.filter(isHan).count }

    static func stopWordHits(_ text: String, _ list: Set<String>) -> Int { Set(ModelSmoke.words(text)).intersection(list).count }

    static func hasCommentaryPrefix(_ text: String) -> Bool {
        let lower = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return commentaryPrefixes.contains { lower.hasPrefix($0) }
    }

    static func hasTurnMarkers(_ text: String) -> Bool {
        ["<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>"].contains { text.contains($0) }
    }

    /// The per-pair language / no-commentary oracle (R8), recorded as SMOKE lines under `name`.
    static func judge(_ p: Pair, _ out: String, name: String) {
        let a = auto
        ModelSmoke.check(a, "\(name).non-empty", !out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, out)
        switch p.target {
        case "zh-Hans":
            let r = hanRatio(out)
            ModelSmoke.check(a, "\(name).han-script", r >= 0.5, String(format: "Han ratio %.2f: %@", r, out))
        case "ms":
            let hits = stopWordHits(out, malayStopWords)
            ModelSmoke.check(a, "\(name).malay-stop-words", hits >= 3, "\(hits) of the Malay list: \(out)")
        default:
            let hits = stopWordHits(out, englishStopWords)
            ModelSmoke.check(a, "\(name).english-stop-words", hits >= 3, "\(hits) of the English list: \(out)")
        }
        // Length: latin→latin uses the words ratio; a Chinese target uses characters (it is shorter, so this only catches a
        // run-on); a Chinese source has no word count, so its English output is compared in words against the Han
        // characters (a Chinese character is roughly one English word or less). The brief's 2.5x character rule would
        // fail every ZH→EN line by construction, which is why it is not used there.
        let ok: Bool, detail: String
        if p.target == "zh-Hans" {
            ok = out.count <= Int(2.5 * Double(p.text.count)); detail = "\(out.count) chars vs input \(p.text.count)"
        } else if p.source == "zh-Hans" {
            let w = ModelSmoke.words(out).count, h = hanCount(p.text)
            ok = Double(w) <= 2.5 * Double(h); detail = "\(w) words vs \(h) Han characters"
        } else {
            let w = ModelSmoke.words(out).count, i = ModelSmoke.words(p.text).count
            ok = Double(w) <= 2.5 * Double(i); detail = "\(w) words vs \(i) input words"
        }
        ModelSmoke.check(a, "\(name).no-added-commentary-length", ok, detail)
        ModelSmoke.check(a, "\(name).no-commentary-prefix", !hasCommentaryPrefix(out), String(out.prefix(40)))
    }

    // MARK: Install (R8-1)

    @Test func a_installs() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)       // prints the INSTALL line when it installs
        let entry = try ModelSmoke.catalogEntry(Self.hfID)
        let fm = FileManager.default
        ModelSmoke.check(Self.auto, "install.files", ["config.json", "model.safetensors", "tokenizer.json", "chat_template.jinja"]
            .allSatisfy { fm.fileExists(atPath: dir.appendingPathComponent($0).path) })
        if ModelSmoke.weights == nil {   // a --weights copy need not carry the marker
            ModelSmoke.check(Self.auto, "install.marker", fm.fileExists(atPath: dir.appendingPathComponent(".installed").path))
        }
        let gb = Double(ModelStore.directorySize(at: dir)) / 1e9
        let card = entry.downloadSizeGB
        ModelSmoke.check(Self.auto, "install.size-within-5pct-of-card", card > 0 && abs(gb - card) / card <= 0.05,
                         String(format: "on disk %.2f GB, card %.2f GB", gb, card))
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        ModelSmoke.check(Self.auto, "install.resolves-to-module", registry.bestModule(for: entry) != nil)
    }

    // MARK: Run screen driver (the buttons' path)

    private struct RunResult { let text: String?; let error: String?; let stopped: Bool; let tokens: Int? }

    /// Selects the languages, counts the rendered tokens and starts the stage as the Translate screen does, on a stage
    /// built over the checkpoint folder.
    private func runScreen(_ dir: URL, _ source: String, _ target: String, _ text: String) async throws -> RunResult {
        let entry = try ModelSmoke.catalogEntry(Self.hfID)
        let model = TranslateGemmaRunModel(modelID: entry.id, displayName: Self.display, modelDirectory: dir,
                                           footprintBytes: Int64(entry.ramGB * 1_073_741_824))
        model.makeStage = { s, t in
            TranslateGemmaStage(modelDirectory: dir, id: entry.id, name: Self.display, sourceCode: s, targetCode: t,
                                footprintBytes: Int64(entry.ramGB * 1_073_741_824))
        }
        model.sourceSelection = source
        model.targetSelection = target
        model.sourceText = text
        await model.refreshTokenCount()
        guard let s = model.resolvedSourceCode, let t = model.targetLanguage?.code else {
            return RunResult(text: nil, error: "languages did not resolve", stopped: false, tokens: model.promptTokenCount)
        }
        model.start(stage: model.makeStage(s, t))
        while model.isRunning { try await Task.sleep(for: .milliseconds(50)) }
        return RunResult(text: model.resultText, error: model.errorText, stopped: model.didStop, tokens: model.promptTokenCount)
    }

    // MARK: The four pairs (R8-2) + REVIEW (R8-7)

    @Test func b_fourPairs() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        var report = "TranslateGemma 4B — four fixed pairs (temperature 0)\n"
        for p in Self.pairs {
            let r = try await runScreen(dir, p.source, p.target, p.text)
            ModelSmoke.check(Self.auto, "pair.\(p.slug).run-finished", r.error == nil && !r.stopped && r.text != nil,
                             r.error ?? (r.stopped ? "stopped" : "ok"))
            let out = r.text ?? ""
            Self.judge(p, out, name: "pair.\(p.slug)")
            report += "\n== \(p.label) ==\nSOURCE: \(p.text)\nOUTPUT: \(out)\n"
            try ModelSmoke.artifact(Self.review, "pair-\(p.slug).txt", text: "\(p.label)\n\nSOURCE: \(p.text)\n\nOUTPUT: \(out)\n",
                                    judge: "\(p.label): is the translation faithful and fluent (Malay / Simplified Chinese quality is the owner's call)?")
        }
        try ModelSmoke.artifact(Self.review, "pairs-all.txt", text: report)
    }

    // MARK: Determinism (R8-3)

    @Test func c_deterministicAtTemperatureZero() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let first = try await runScreen(dir, "en", "ms", Self.english)
        let second = try await runScreen(dir, "en", "ms", Self.english)
        ModelSmoke.check(Self.auto, "determinism.en-ms-twice-identical", first.text != nil && first.text == second.text,
                         "1: \(first.text ?? "nil") | 2: \(second.text ?? "nil")")
    }

    // MARK: Stops cleanly (R8-4)

    @Test func d_stopsCleanly() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        for p in Self.pairs {
            let o = try await TranslateGemmaEngine.translate(.init(text: p.text, sourceCode: p.source, targetCode: p.target),
                                                             modelDirectory: dir)
            ModelSmoke.check(Self.auto, "stop.\(p.slug).finished-not-cancelled", !o.cancelled)
            ModelSmoke.check(Self.auto, "stop.\(p.slug).no-turn-markers", !Self.hasTurnMarkers(o.text), o.text)
            ModelSmoke.check(Self.auto, "stop.\(p.slug).ended-before-max-tokens",
                             o.generatedTokens > 0 && o.generatedTokens < TranslateGemmaLanguages.maxOutputTokens,
                             "generated \(o.generatedTokens) of \(TranslateGemmaLanguages.maxOutputTokens), prompt \(o.promptTokens)")
            ModelSmoke.check(Self.auto, "stop.\(p.slug).not-a-run-on", o.generatedTokens < 200, "\(o.generatedTokens) tokens")
        }
    }

    // MARK: One flow row through RealExecutor (R8-5)

    private func catalog() throws -> [ModelEntry] {
        let data = try Data(contentsOf: ModelSmoke.repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    @Test func e_flowRowThroughRealExecutor() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let catalog = try catalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("mc2-11-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))
        let make: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            guard let s = config.sourceLanguage, let t = config.targetLanguage else { throw TranslateGemmaError.missingLanguages }
            return TranslateGemmaStage(modelDirectory: dir, id: model.id, name: model.displayName, sourceCode: s, targetCode: t,
                                       footprintBytes: Int64(model.ramGB * 1_073_741_824))
        }
        let real = RealExecutor(workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
                                makeModelStage: make, installedModelIDs: [entry.id], catalog: catalog)
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real", catalog: catalog, runSeed: 0,
                                      workspace: ws, flowID: "f")
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Template   "\(Self.english)"
        2. Translate  \(Self.display); to=Malay
        """)

        func run(_ phase: String) async throws -> (text: String, events: [FlowInterpreter.PathEvent]) {
            let events = try await ModelSmoke.measure("flow row \(phase)") { try await FlowInterpreter.run(doc, executor: caching) }
            let failed = events.filter { $0.kind == .rowFailed }
            ModelSmoke.check(Self.auto, "flow.\(phase).no-row-failed", failed.isEmpty,
                             failed.map { "\($0.path): \($0.error ?? $0.message ?? "")" }.joined(separator: "; "))
            let text = events.first { $0.kind == .rowCompleted && $0.path == "2" }?.output?.items.first?.value ?? ""
            return (text, events)
        }
        let first = try await run("run1")
        Self.judge(Self.pairs[0], first.text, name: "flow.run1")
        ModelSmoke.check(Self.auto, "flow.run1.ran-the-model", !first.events.contains { $0.kind == .cacheHit && $0.path == "2" })
        let second = try await run("run2")
        ModelSmoke.check(Self.auto, "flow.run2.cache-hit", second.events.contains { $0.kind == .cacheHit && $0.path == "2" })
        ModelSmoke.check(Self.auto, "flow.run2.identical", !first.text.isEmpty && first.text == second.text,
                         "1: \(first.text) | 2: \(second.text)")
    }

    // MARK: Memory (R8-6) — load+first run, typical run, worst = the 2,048-token input

    /// A text of `n` repeats of one fixed English sentence.
    private static func filler(_ n: Int) -> String { String(repeating: "The river was calm, and we walked slowly along the bank. ", count: n) }

    @Test func f_memoryAndTheTokenLimit() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let counter = TranslateGemmaTokenCounter.shared
        func tokens(_ n: Int) async throws -> Int {
            try await counter.tokenCount(for: .init(text: Self.filler(n), sourceCode: "en", targetCode: "ms"), modelDirectory: dir)
        }
        // The largest repeat count the engine accepts (the rendered prompt, template text included, <= 2,048), found
        // with the same counter the Translate screen shows.
        var lo = 1, hi = 400
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if try await tokens(mid) <= TranslateGemmaLanguages.maxPromptTokens { lo = mid } else { hi = mid - 1 }
        }
        let worstCount = try await tokens(lo)
        let overCount = try await tokens(lo + 1)
        ModelSmoke.check(Self.auto, "memory.worst-input-near-limit", (1_900...2_048).contains(worstCount), "\(worstCount) rendered tokens")
        ModelSmoke.check(Self.auto, "memory.over-limit-input-is-over", overCount > TranslateGemmaLanguages.maxPromptTokens, "\(overCount) rendered tokens")

        // A private pool, so "load + first run" really loads (the other tests share `.shared`).
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let request = TranslateGemmaEngine.Request(text: Self.english, sourceCode: "en", targetCode: "ms")
        let first = try await ModelSmoke.measure("load + first run", detail: "EN→MS, private pool") {
            try await TranslateGemmaEngine.translate(request, modelDirectory: dir, pool: pool)
        }
        ModelSmoke.check(Self.auto, "memory.load-first-run-translates", !first.cancelled && !first.text.isEmpty, first.text)
        let typical = try await ModelSmoke.measure("typical run", detail: "EN→MS, loaded") {
            try await TranslateGemmaEngine.translate(request, modelDirectory: dir, pool: pool)
        }
        ModelSmoke.check(Self.auto, "memory.typical-run-translates", !typical.cancelled && !typical.text.isEmpty, typical.text)
        let worst = try await ModelSmoke.measure("worst = 2,048-token input", detail: "\(worstCount) prompt tokens, EN→MS") {
            try await TranslateGemmaEngine.translate(.init(text: Self.filler(lo), sourceCode: "en", targetCode: "ms"),
                                                     modelDirectory: dir, pool: pool)
        }
        ModelSmoke.check(Self.auto, "memory.worst-run-translates", !worst.cancelled && !worst.text.isEmpty,
                         "prompt \(worst.promptTokens), generated \(worst.generatedTokens)")
        ModelSmoke.check(Self.auto, "memory.worst-prompt-count-matches-screen", worst.promptTokens == worstCount,
                         "engine \(worst.promptTokens), screen counter \(worstCount)")

        // 2,049+ tokens: refused with the engine's sentence, nothing run. The Translate screen refuses before any work…
        let over = try await runScreen(dir, "en", "ms", Self.filler(lo + 1))
        ModelSmoke.check(Self.auto, "limit.screen-refuses-over-2048",
                         over.text == nil && over.error == TranslateGemmaError.promptTooLong(tokens: overCount).description,
                         over.error ?? "no error")
        // …and so does the engine's own check (the rendered count against the same limit).
        var engineSentence = ""
        do { try TranslateGemmaEngine.requireWithinLimit(promptTokens: overCount) } catch { engineSentence = "\(error)" }
        ModelSmoke.check(Self.auto, "limit.engine-refuses-over-2048",
                         engineSentence == TranslateGemmaError.promptTooLong(tokens: overCount).description, engineSentence)
    }
}
