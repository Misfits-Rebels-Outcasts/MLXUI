import Testing
import Foundation
@testable import MLXUI

/// MoC-3-3 (`RSI/DelegateMoCBacklog.md`) — `RerankStage`'s orchestration, ported from the
/// shape `catflow-mlx/tests/test_engines_rerank.py` uses: stub the scorer, assert the
/// ordering. No weights, no model — the scorer is an injected function.
struct RerankStageTests {

    private func item(_ text: String) -> Item {
        Item(kind: .text, value: text, path: nil, sourceText: nil)
    }

    private func stage(query: String = "q", topK: Int? = nil,
                       scorer: @escaping @Sendable (String, String) async throws -> Double) -> RerankStage {
        RerankStage(id: "test.rerank", name: "Test Rerank", rowLabel: "Row 1",
                    query: query, topK: topK, scorer: scorer)
    }

    // MARK: - Shape

    @Test func declaresListOfTextToListOfText() {
        let s = stage { _, _ in 0 }
        #expect(s.accepts == .listOf(.text))
        #expect(s.produces == .listOf(.text))
    }

    // MARK: - Basic scoring and ordering

    @Test func sortsDescendingByScore() async throws {
        let scores: [String: Double] = ["a": 0.2, "b": 0.9, "c": 0.5]
        let s = stage { _, candidate in scores[candidate] ?? 0 }
        let input = Asset(items: ["a", "b", "c"].map(item))
        let result = try await s.run(input) { _ in }
        #expect(result.items.map(\.value) == ["b", "c", "a"])
    }

    @Test func scorerReceivesTheQueryAndEachCandidate() async throws {
        var seen: [(String, String)] = []
        let s = stage(query: "what is MLX") { query, candidate in
            seen.append((query, candidate))
            return 0
        }
        let input = Asset(items: ["x", "y"].map(item))
        _ = try await s.run(input) { _ in }
        #expect(seen.map(\.0) == ["what is MLX", "what is MLX"])
        #expect(seen.map(\.1) == ["x", "y"])
    }

    // MARK: - Tie-stability: the property a naive sort loses

    @Test func tiedScoresKeepInputOrder() async throws {
        // np.argsort(-scores, kind="stable") — two equal scores must keep their original
        // relative order. A naive (unstable) sort would be free to swap them; this test
        // fails against that, not just passes vacuously.
        let s = stage { _, _ in 0.5 }   // every candidate scores identically
        let input = Asset(items: ["first", "second", "third", "fourth"].map(item))
        let result = try await s.run(input) { _ in }
        #expect(result.items.map(\.value) == ["first", "second", "third", "fourth"])
    }

    @Test func tiesAmongSomeCandidatesKeepTheirRelativeOrder() async throws {
        // "b" and "d" tie at 0.5; "a" and "c" are unambiguous. The tied pair must come out
        // in the order they went in (b before d), not sorted by any other criterion.
        let scores: [String: Double] = ["a": 0.9, "b": 0.5, "c": 0.1, "d": 0.5]
        let s = stage { _, candidate in scores[candidate] ?? 0 }
        let input = Asset(items: ["a", "b", "c", "d"].map(item))
        let result = try await s.run(input) { _ in }
        #expect(result.items.map(\.value) == ["a", "b", "d", "c"])
    }

    // MARK: - top_k truncates AFTER sorting, never before

    @Test func topKTruncatesAfterSorting() async throws {
        let scores: [String: Double] = ["a": 0.2, "b": 0.9, "c": 0.5, "d": 0.1]
        let s = stage(topK: 2) { _, candidate in scores[candidate] ?? 0 }
        let input = Asset(items: ["a", "b", "c", "d"].map(item))
        let result = try await s.run(input) { _ in }
        #expect(result.items.map(\.value) == ["b", "c"])
    }

    // MoC-FIX-1: a tie-break comparator with no index tie-break can still pass
    // `tiedScoresKeepInputOrder`/`tiesAmongSomeCandidatesKeepTheirRelativeOrder` above
    // vacuously (Swift's `sorted(by:)` happens to be stable today) without the *property*
    // actually being structural. This test forces the distinction: a three-way tie whose
    // members straddle the `topK` boundary — reordering *within* the tie group changes
    // which items survive truncation, so this fails if the tie-break is ever removed,
    // even on a standard library that stops being incidentally stable.
    @Test func threeWayTieSpanningTopKBoundaryKeepsInputOrderSurvivors() async throws {
        // b, c, d tie at 0.5; a is highest, e is lowest. topK=3 must keep exactly
        // [a, b, c] — b and c are the first two tied candidates in input order — and
        // drop d, never e.g. [a, c, d] or [a, d, b].
        let scores: [String: Double] = ["a": 0.9, "b": 0.5, "c": 0.5, "d": 0.5, "e": 0.1]
        let s = stage(topK: 3) { _, candidate in scores[candidate] ?? 0 }
        let input = Asset(items: ["a", "b", "c", "d", "e"].map(item))
        let result = try await s.run(input) { _ in }
        #expect(result.items.map(\.value) == ["a", "b", "c"])
    }

    @Test func topKLargerThanCandidateCountKeepsEverything() async throws {
        let s = stage(topK: 10) { _, _ in 0.5 }
        let input = Asset(items: ["a", "b"].map(item))
        let result = try await s.run(input) { _ in }
        #expect(result.items.count == 2)
    }

    @Test func noTopKKeepsEveryCandidate() async throws {
        let s = stage { _, _ in 0.5 }
        let input = Asset(items: (1...5).map { item("c\($0)") })
        let result = try await s.run(input) { _ in }
        #expect(result.items.count == 5)
    }

    // MARK: - A text item with no inline value is an error, not a skipped row

    @Test func itemWithNoValueThrowsRatherThanSkipping() async throws {
        let s = stage { _, _ in 0 }
        let input = Asset(items: [item("a"), Item(kind: .text, value: nil, path: nil, sourceText: nil), item("c")])
        await #expect(throws: FlowError.self) {
            _ = try await s.run(input) { _ in }
        }
    }

    @Test func emptyCandidateListProducesAnEmptyResultNotAnError() async throws {
        // No candidates to score is not the same failure as a candidate with no value —
        // an upstream row that legitimately produced zero items should not be treated as
        // malformed input.
        let s = stage { _, _ in 0 }
        let result = try await s.run(Asset(items: [])) { _ in }
        #expect(result.items.isEmpty)
    }

    // MARK: - One forward pass per candidate, never batched

    @Test func scorerIsCalledExactlyOncePerCandidate() async throws {
        let count = Locked(0)
        let s = stage { _, _ in
            count.increment()
            return 0
        }
        let input = Asset(items: (1...4).map { item("c\($0)") })
        _ = try await s.run(input) { _ in }
        #expect(count.value == 4)
    }

    // MARK: - The real seam: RealExecutor.execute builds a RerankStage, not a SingleMediaStage

    /// Exercises the whole path — `execute` → `runModel`'s `engines.rerank.` branch →
    /// `stageConfig` (the real function) → the scorer bridge (joins query+candidate,
    /// parses the registry stage's text reply as the score) → `RerankStage` (the real
    /// stage). Not a mock echoing the config back — the same discipline
    /// `generateImageSeedDrivesDeterministicOutput` (`CatFlowR16DiffusionSettingsTests`)
    /// applies to diffusion settings, applied here to confirm the branch actually builds
    /// what MoC-3-3 says it must (a `RerankStage`, never `SingleMediaStage`, which would
    /// reject any asset that isn't exactly one item).
    @Test func realExecutorBuildsARerankStageThroughTheRerankBranch() async throws {
        let entry = makeEntry(modelType: .rerank, source: .mlx)
        let scores: [String: Double] = ["low": 0.1, "high": 0.9, "mid": 0.5]
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "moc3-integration",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in StubScoringStage(scores: scores) },
            installedModelIDs: [entry.id],
            catalog: [entry])
        // CFM-R14-FIX-1: an unbridged hfModelId resolves `.same` with no manifest — Rerank
        // has no bridge entry yet (MoC-4-4), so the row names the raw id directly.
        let row = Row(task: "Rerank", model: entry.hfModelId, settings: "query=\"q\"; top_k=2")
        let input = Asset(items: ["low", "high", "mid"].map(item))
        let result = try await executor.execute(
            path: "1", row: row, inputs: [input],
            transcript: nil, context: nil, usedFlowContent: nil)
        #expect(result.items.map(\.value) == ["high", "mid"])
    }
}

/// A stub registry stage standing in for MoC-4's `RerankSDK`: `.text → .text`, parsing the
/// bridge's `"query\ncandidate"` input and replying with the candidate's score as text —
/// the seam contract `RealExecutor.runModel`'s `engines.rerank.` branch relies on.
private nonisolated struct StubScoringStage: PipelineStage {
    let id = "stub.rerank.model"
    let name = "Stub Reranker"
    let scores: [String: Double]
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        guard case .text(let combined) = input else {
            throw StageError.unsupportedModel(id: id, kind: .rerank)
        }
        let candidate = combined.split(separator: "\n", maxSplits: 1).last.map(String.init) ?? ""
        return .text(String(scores[candidate] ?? 0))
    }
}

/// A tiny actor-free counter for the "called once per candidate" assertion — the scorer
/// closure is `@Sendable` and may run on any executor, so a plain `var` capture would race.
private final class Locked: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Int
    init(_ value: Int) { _value = value }
    var value: Int { lock.withLock { _value } }
    func increment() { lock.withLock { _value += 1 } }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock(); defer { unlock() }
        return body()
    }
}
