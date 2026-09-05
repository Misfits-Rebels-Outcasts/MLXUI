import Foundation

/// Rerank: scores a list of text candidates against one query and returns them sorted,
/// most-relevant first, optionally truncated to `topK`. List-shaped
/// (`.listOf(.text) → .listOf(.text)`) — the first model task that cannot be a
/// `PipelineStage`: `SingleMediaStage` requires exactly one item and must not be widened
/// (`AssetStage.swift`'s own header), while a reranker needs the whole candidate list at
/// once. Written directly on `AssetStage`, per `RSI/DelegateMoCBacklog.md` MoC-3-3.
///
/// Semantics ported verbatim from `catflow-mlx/src/catflow/engines/rerank.py::rerank`, per
/// the backlog's own description of it (the sibling repo wasn't reachable this cycle — see
/// the MoC-3-3 journal entry): score every candidate against the one query; sort
/// **descending, stable** (`np.argsort(-scores, kind="stable")` — ties keep input order);
/// truncate to `topK` **after** sorting, never before. A text item with no inline value is
/// an error, not a silently-skipped row.
///
/// The `scorer` is injected so this orchestration is testable without weights — the same
/// split `tests/test_engines_rerank.py` uses (`monkeypatch` the scorer, assert the
/// ordering). `RealExecutor.runModel`'s `engines.rerank.` branch supplies the real one,
/// bridging the registry-resolved model's `PipelineStage` (`.text → .text`, called once per
/// candidate, output parsed as the score) into this signature.
nonisolated struct RerankStage: AssetStage {
    let id: String
    let name: String
    /// The flow row this stage belongs to, for error voice ("Row 2 …").
    let rowLabel: String
    /// The query every candidate is scored against. `stageConfig` already refused a missing
    /// query with a named error before this stage is ever constructed.
    let query: String
    /// Truncate the sorted result to this many, or keep everything when `nil`.
    let topK: Int?
    let scorer: @Sendable (_ query: String, _ candidate: String) async throws -> Double

    var accepts: Shape { .listOf(.text) }
    var produces: Shape { .listOf(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        var candidates: [String] = []
        candidates.reserveCapacity(input.items.count)
        for item in input.items {
            guard item.kind == .text, let value = item.value else {
                throw FlowError.missingInlineValue(row: rowLabel, kind: .text)
            }
            candidates.append(value)
        }

        // One forward pass per candidate — never batched. The Python docstring says why:
        // right-padding a batch corrupts the last-token read for every candidate shorter
        // than the longest.
        var scores: [Double] = []
        scores.reserveCapacity(candidates.count)
        for candidate in candidates {
            scores.append(try await scorer(query, candidate))
            progress(Double(scores.count) / Double(max(candidates.count, 1)))
        }

        // `Array.sorted(by:)` is a stable sort — ties keep their input order, matching
        // `np.argsort(-scores, kind="stable")`.
        let order = candidates.indices.sorted { scores[$0] > scores[$1] }
        let truncated = topK.map { Array(order.prefix($0)) } ?? order
        return Asset(items: truncated.map { Item(kind: .text, value: candidates[$0], path: nil, sourceText: nil) })
    }
}
