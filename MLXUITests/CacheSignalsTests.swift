import Testing
import Foundation
@testable import MLXUI

/// CACHE-SIGNALS-1 — `CachingExecutor` used to forward none of `lastTag`/`lastTimeoutFlag`/
/// `lastProviderDeciderFlag`/`lastStaged`/`lastDeciderDetail` from its wrapped executor:
/// every conformer relied on `FlowExecutor`'s protocol-default `nil`, and `CachingExecutor`
/// itself never overrode any of them (only `lastCacheHit`). Since every real Run wraps
/// `RealExecutor` in `CachingExecutor` (`AppFlowExecutorFactory.cachingContext`), a decider
/// row's fired tag never reached `FlowInterpreter.advance()`'s `.decide` clause on any real
/// run — the block just ended silently right after the decider (smoke row 97's "Classify
/// completes, 3.3–3.7 never fire" report; not Laya-specific, any decider-routed flow was
/// affected). Ported from `catflow-mlx/src/catflow/core/cache.py:655-983`.
struct CacheSignalsTests {

    private func makeStore() throws -> (FlowCacheStore, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("cache-signals-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowCacheStore(root: base.appendingPathComponent("cache")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    /// A lock-guarded counter a `@Sendable` fake closure can report back through — a plain
    /// captured `var` isn't Sendable-safe across an escaping `@Sendable` closure boundary
    /// (same reasoning as `LayaDeciderTests.StageFlag`).
    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var _count = 0
        func increment() { lock.withLock { _count += 1 } }
        var count: Int { lock.withLock { _count } }
    }

    // MARK: - Unit level: tag recovery on a hit, reset on a miss, decider-realism gating

    /// `cache.py:669-676`/`:960-965` — a decider's fired tag is recovered from the cache
    /// entry itself on a hit, not left `nil` like every other signal.
    @Test func tagIsRecoveredFromTheStoreOnAHit() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))

        let laya = makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                             modelType: .decision, hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
        let calls = Counter()
        var real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [laya.id], catalog: [laya])
        real.askLaya = { _, _, _ in
            calls.increment()
            var answer = LayaAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1],
                                    optionLabels: [], stateTruncated: false)
            answer.choiceLabel = "billing"
            return [answer]
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [laya], runSeed: 0, workspace: ws, flowID: "f")

        let row = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "Who owns this?",
                     tags: ["billing", "technical"])
        let input = Asset(items: [Item(kind: .text, value: "my invoice was charged twice", path: nil, sourceText: nil)])

        let first = try await caching.execute(path: "1", row: row, inputs: [input])
        #expect(!caching.lastCacheHit)
        #expect(caching.lastTag == "billing")
        #expect(first.items.first?.value == "my invoice was charged twice")
        #expect(calls.count == 1)

        let second = try await caching.execute(path: "1", row: row, inputs: [input])
        #expect(caching.lastCacheHit, "identical (task, model, settings, inputs) must hit")
        #expect(caching.lastTag == "billing", "the tag must be recoverable from the cache entry, not just the payload")
        #expect(second.items.first?.value == "my invoice was charged twice")
        #expect(calls.count == 1, "a hit must never call askLaya again")
    }

    /// Every signal resets to "nothing ran" at the start of every `execute` — a stale tag
    /// from a previous row (or a previous item of the same row inside an `<each>`) must never
    /// leak into the next activation, hit or miss.
    @Test func signalsResetBetweenActivationsRegardlessOfOutcome() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))

        let laya = makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                             modelType: .decision, hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
        var real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [laya.id], catalog: [laya])
        real.askLaya = { _, state, _ in
            var answer = LayaAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1],
                                    optionLabels: [], stateTruncated: false)
            answer.choiceLabel = state.contains("first") ? "a" : "b"
            return [answer]
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [laya], runSeed: 0, workspace: ws, flowID: "f")

        let rowA = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "x", tags: ["a", "b"])
        let inputA = Asset(items: [Item(kind: .text, value: "first ticket", path: nil, sourceText: nil)])
        _ = try await caching.execute(path: "1", row: rowA, inputs: [inputA])
        #expect(caching.lastTag == "a")

        // A different row/input (never cached before) must not see the previous tag.
        let rowB = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "x", tags: ["a", "b"])
        let inputB = Asset(items: [Item(kind: .text, value: "second ticket", path: nil, sourceText: nil)])
        _ = try await caching.execute(path: "2", row: rowB, inputs: [inputB])
        #expect(caching.lastTag == "b", "the second activation's own tag, never the first's leaking forward")
    }

    /// `cache.py:901-912` (`_skip_cache`): a decider skips the cache entirely unless its own
    /// realism is genuinely `"real"` — under `"mock"`/`"hybrid"` the fired tag is driven by
    /// visit count, not `(task, model, settings, inputs)` alone, so caching it would replay a
    /// stale tag. `AppFlowExecutorFactory.cachingContext` always passes `"real"`, so this is a
    /// no-op for the shipped app today, but must not silently regress if that ever changes.
    @Test func aDeciderNeverCachesUnderMockOrHybridRealism() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))

        let laya = makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                             modelType: .decision, hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
        let calls = Counter()
        var real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [laya.id], catalog: [laya])
        real.askLaya = { _, _, _ in
            calls.increment()
            var answer = LayaAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1],
                                    optionLabels: [], stateTruncated: false)
            answer.choiceLabel = "a"
            return [answer]
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "mock",
                                      catalog: [laya], runSeed: 0, workspace: ws, flowID: "f")

        let row = Row(id: UUID(), task: "Classify", model: "Laya 0.4B", settings: "x", tags: ["a", "b"])
        let input = Asset(items: [Item(kind: .text, value: "same ticket", path: nil, sourceText: nil)])
        _ = try await caching.execute(path: "1", row: row, inputs: [input])
        _ = try await caching.execute(path: "1", row: row, inputs: [input])
        #expect(calls.count == 2, "a decider under mock/hybrid realism must never serve a cache hit")
        #expect(!caching.lastCacheHit)
    }

    // MARK: - The other signals, forwarded on the NEVER_CACHE bypass (miss path)

    @Test func timeoutFlagForwardsThroughTheCachingWrapper() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [], catalog: [])
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [], runSeed: 0, workspace: ws, flowID: "f")

        // "Ask Human" is NEVER_CACHE — every call takes the skip-cache bypass, never the
        // hit/miss branches, so this exercises the bypass's own signal-copy step.
        let row = Row(id: UUID(), task: "Ask Human", settings: "timeout=4h; default=edit")
        let input = Asset(items: [Item(kind: .text, value: "draft", path: nil, sourceText: nil)])
        let output = try await caching.execute(path: "2", row: row, inputs: [input])
        #expect(caching.lastTag == "edit")
        #expect(caching.lastTimeoutFlag?.code == "F002")
        #expect(caching.lastTimeoutFlag?.message == "Nobody answered by 4h — proceeded as `edit`, unreviewed.")
        #expect(output.items.first?.value == "draft")
    }

    @Test func stagedEffectForwardsThroughTheCachingWrapper() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [], catalog: [])
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [], runSeed: 0, workspace: ws, flowID: "f")

        // "Stage Send" is NEVER_CACHE too — a second, identical call must queue a second
        // outbox entry, never replay the first from a cache hit (cache.py's own reasoning).
        let row = Row(id: UUID(), task: "Stage Send", settings: "outbox")
        let input = Asset(items: [Item(kind: .text, value: "ship the minutes", path: nil, sourceText: nil)])
        let output = try await caching.execute(path: "1", row: row, inputs: [input])
        #expect(!caching.lastCacheHit)
        let staged = try #require(caching.lastStaged)
        #expect(staged.kind == "send")
        #expect(staged.summary.contains("ship the minutes"))
        #expect(output.items.first?.kind == .status)
    }

    @Test func providerDeciderDisclosureForwardsThroughTheCachingWrapper() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))

        ProviderAvailability.executorOverride = nil
        defer { ProviderAvailability.executorOverride = nil }
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        defer {
            if let original { KeychainHelper.save(original, account: account) }
            else { KeychainHelper.delete(account: account) }
        }
        let mock = MockProviderExecutorStub()
        mock.textToReturn = "ship"
        ProviderAvailability.executorOverride = mock

        let real = RealExecutor(
            workspace: ws, flowID: "f", blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [], catalog: [])
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [], runSeed: 0, workspace: ws, flowID: "f")

        // "Gate" is a decider (cacheable under cacheTier "real") — this is a genuine first-
        // time miss, so the copy-on-miss path is what's under test here.
        let row = Row(task: "Gate", model: "claude-sonnet @ anthropic",
                     settings: "\"Ready to ship?\"", tags: ["ship", "hold"])
        let input = Asset(items: [Item(kind: .text, value: "looks solid", path: nil, sourceText: nil)])
        let output = try await caching.execute(path: "1", row: row, inputs: [input])
        #expect(!caching.lastCacheHit)
        #expect(caching.lastTag == "ship")
        #expect(output.items.first?.value == "looks solid")
        let flag = try #require(caching.lastProviderDeciderFlag)
        #expect(flag.code == "F010")
        #expect(flag.message.contains("claude-sonnet @ anthropic"))
    }

    private final class MockProviderExecutorStub: ProviderExecuting, @unchecked Sendable {
        var textToReturn = "a reply"
        func generate(instructions: String, prompt: String, maxTokens: Int, temperature: Double) async throws -> String {
            textToReturn
        }
    }

    // MARK: - End to end, through the real interpreter: gallery flow 77 (Laya), cold + warm

    private final class Box<T>: @unchecked Sendable {
        var values: [T] = []
    }

    /// READ-UPSTREAM-1's own test proved the tool/interpreter wiring works with a bare
    /// `RealExecutor`. This proves the same flow through the composition every real Run
    /// actually uses — `RealExecutor` wrapped in `CachingExecutor`
    /// (`AppFlowExecutorFactory.cachingContext`) — cold (every row a miss) and then warm
    /// (every decider row a hit), asserting every branch still fires both times.
    @Test func flow77ReachesTheRightBranchColdAndWarmThroughTheCachingWrapper() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowID = "77-TicketRouter"
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let doc = try GalleryLoader.loadDocument(flowID: flowID)

        let laya = makeEntry(id: "aac6fef--laya-mlx", family: "Laya", displayName: "Laya 0.4B",
                             modelType: .decision, hfRepo: "aac6fef", hfModelId: "aac6fef/laya-mlx")
        let calls = Box<String>()
        var real = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [laya.id], catalog: [laya])
        real.askLaya = { _, state, _ in
            calls.values.append(state)
            let tag: String
            if state.contains("Charged twice") || state.contains("VAT number") { tag = "billing" }
            else if state.contains("crashing") { tag = "technical" }
            else if state.contains("log in") { tag = "account" }
            else { tag = "other" }
            var answer = LayaAnswer(type: .choice, confidence: 0.9, probabilities: [0.9, 0.1],
                                    optionLabels: [], stateTruncated: false)
            answer.choiceLabel = tag
            return [answer]
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [laya], runSeed: 0, workspace: ws, flowID: flowID)

        func assertRun(_ events: [FlowInterpreter.PathEvent], label: String) throws {
            let f003 = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
            #expect(f003.isEmpty, "\(label): no ticket should fail: \(f003.map(\.message))")
            // Every branch (billing/technical/account/other) plus the merge row must fire —
            // this is the exact symptom: Classify (2.2) completed but 2.3–2.7 never did.
            for branch in ["2.3", "2.4", "2.5", "2.6", "2.7"] {
                #expect(events.contains { $0.kind == .rowCompleted && $0.path == branch },
                        "\(label): row \(branch) never completed")
            }
            let saved = try String(contentsOf: ws.directory(for: flowID).appendingPathComponent("routed-tickets.md"),
                                   encoding: .utf8)
            for prefix in ["BILLING", "TECHNICAL", "ACCOUNT", "OTHER"] {
                #expect(saved.contains(prefix), "\(label): routed-tickets.md is missing the \(prefix) group")
            }
        }

        let cold = try await FlowInterpreter.run(doc, executor: caching)
        try assertRun(cold, label: "cold")
        #expect(calls.values.count == 5, "cold: every ticket must ask Laya for real")

        let warm = try await FlowInterpreter.run(doc, executor: caching)
        try assertRun(warm, label: "warm")
        #expect(calls.values.count == 5, "warm: a cache hit must never call askLaya again")
        // `Read Text` (2.1) is also cacheable now (post READ-UPSTREAM-1, not NEVER_CACHE), so
        // it hits too on a warm run — filter to Classify's own path (2.2) specifically.
        let hits = warm.filter { $0.kind == .cacheHit && $0.path == "2.2" }
        #expect(hits.count == 5, "warm: every ticket's Classify should be a cache hit")
    }

    // MARK: - End to end: gallery flow 37 (an ordinary LLM decider, not Laya-specific)

    private nonisolated final class RecordingStage: PipelineStage, @unchecked Sendable {
        let id = "test.recording-stage"
        let name = "Recording Stage"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        let seen: Box<String>

        init(seen: Box<String>) { self.seen = seen }

        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            guard case .text(let prompt) = input else {
                throw FlowError.stageFailure(row: "test", message: "expected a text prompt")
            }
            seen.values.append(prompt)
            progress(1.0)
            if prompt.contains("Answer with exactly one of these words") {
                return .text("read")
            }
            return .text("(summary)")
        }
    }

    /// Same proof as flow 77, through `runDecider`/`fireTag`'s ordinary-LLM path instead of
    /// Laya's — the caching gap affected every decider task, not just Laya's.
    @Test func flow37ReachesTheRightBranchColdAndWarmThroughTheCachingWrapper() async throws {
        let (store, base) = try makeStore()
        defer { teardown(base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowID = "37-InboxTriage"
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let doc = try GalleryLoader.loadDocument(flowID: flowID)

        let ministral = makeEntry(id: "mlx-community--Ministral-3-3B-Instruct-2512-4bit",
                                  family: "Ministral", displayName: "Ministral 3B", modelType: .llm,
                                  hfRepo: "mlx-community/Ministral-3-3B-Instruct-2512",
                                  hfModelId: "mlx-community/Ministral-3-3B-Instruct-2512-4bit")
        let seenPrompts = Box<String>()
        let stage = RecordingStage(seen: seenPrompts)
        let stageCalls = Counter()
        let real = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in
                stageCalls.increment()
                return stage
            },
            installedModelIDs: [ministral.id], catalog: [ministral])
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [ministral], runSeed: 0, workspace: ws, flowID: flowID)

        func assertRun(_ events: [FlowInterpreter.PathEvent], label: String) throws {
            let f003 = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
            #expect(f003.isEmpty, "\(label): no mail should fail: \(f003.map(\.message))")
            // Row 2.5 ("READ") and the merge row 2.7 must both fire for every mail — the
            // `<each>` block is row 2 at top level, so its body's own paths are "2.N".
            for branch in ["2.5", "2.7"] {
                #expect(events.contains { $0.kind == .rowCompleted && $0.path == branch },
                        "\(label): row \(branch) never completed")
            }
            let saved = try String(contentsOf: ws.directory(for: flowID).appendingPathComponent("inbox-brief.md"),
                                   encoding: .utf8)
            #expect(saved.contains("(summary)"), "\(label): inbox-brief.md should carry the (faked) summaries")
        }

        let cold = try await FlowInterpreter.run(doc, executor: caching)
        try assertRun(cold, label: "cold")
        #expect(stageCalls.count > 0, "cold: the fake model stage must actually be built and called")

        let coldStageCalls = stageCalls.count
        let warm = try await FlowInterpreter.run(doc, executor: caching)
        try assertRun(warm, label: "warm")
        #expect(stageCalls.count == coldStageCalls, "warm: a cache hit must never build/call the model stage again")
    }
}
