import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

/// Ported from `RealityCat/CLM-v0.1-8B-MLX-8bit`'s vendored `clm_mlx/encoder.py` — `Encoder`.
/// Reuses the app's existing `MLXEmbedders`/`EmbedderModelFactory` infra — `mlx-swift-lm`
/// already runs Qwen3 as an embedder (§0 of `RSI/DelegateCLMBacklog.md`, Step A) — via **its
/// own call site**, not `Modules/Embedding/EmbeddingEngine.swift`'s `embed(_:modelDirectory:)`:
/// that helper pads with `tokenizer.eosTokenId` and never passes a `mask` into
/// `context.pooling(...)` at all, so `Pooling.callAsFunction`'s `mask` parameter defaults to an
/// all-ones mask covering the **padded** length — `.last` then reads the last *padded*
/// position for any batch of mixed lengths, not each row's own last real token. CLM also must
/// **never** apply LayerNorm to the pooled vector (upstream's `embed_ids` does L2 only, no
/// norm) — `EmbeddingEngine`'s call always does.
///
/// Golden test: `MLXUITests/CLMEncoderTests.swift` against `Fixtures/CLM/{tokens,embeddings}.json`.
nonisolated enum CLMEncoder {
    /// `encoder.py::Encoder.max_tokens` — head truncation (the captured vLLM server keeps the
    /// **first** 2048 tokens: two texts sharing their first 2990 tokens got identical
    /// embeddings, cos 1.0000).
    static let maxTokens = 2048
    /// `encoder.py::Encoder.batch_tokens` — a batch grows while `rows × longest ≤ 4096`.
    static let batchTokens = 4096

    /// `encoder.py::Encoder.ids`. Empty text embeds as a single space (upstream: an empty
    /// token list would crash pooling downstream). `truncated` records whether the real text
    /// was cut to fit `maxTokens`.
    static func ids(for text: String, tokenizer: any MLXLMCommon.Tokenizer) -> (ids: [Int], truncated: Bool) {
        var encoded = tokenizer.encode(text: text)
        if encoded.isEmpty { encoded = tokenizer.encode(text: " ") }
        let truncated = encoded.count > maxTokens
        return (truncated ? Array(encoded.prefix(maxTokens)) : encoded, truncated)
    }

    /// `encoder.py::Encoder.embed_ids`. `[n, hidden]` float32, L2-normalised, in input order.
    /// Batching: sort ascending by length, then greedily grow a batch while
    /// `(count-if-included) × (candidate row's length) ≤ batchTokens` — since rows are visited
    /// in ascending order, the candidate being considered is always the longest in the batch
    /// so far. Right-pads with 0; the explicit length mask (not "!= 0", which would be wrong
    /// for a genuinely-empty-token row) is what lets pooling read each row's own last real
    /// token instead of the padded tail.
    static func embed(idLists: [[Int]], context: EmbedderModelContext) -> MLXArray {
        guard !idLists.isEmpty else { return MLXArray.zeros([0, 0]) }
        let order = (0 ..< idLists.count).sorted { idLists[$0].count < idLists[$1].count }
        var indexed: [(index: Int, vector: MLXArray)] = []
        let n = order.count
        var i = 0
        while i < n {
            var j = i + 1
            while j < n, (j - i + 1) * idLists[order[j]].count <= batchTokens {
                j += 1
            }
            let batchIndices = Array(order[i ..< j])
            let rows = batchIndices.map { idLists[$0] }
            let length = rows.last?.count ?? 0
            var idsBuffer = [Int32](repeating: 0, count: rows.count * length)
            var maskBuffer = [Int32](repeating: 0, count: rows.count * length)
            for (row, ids) in rows.enumerated() {
                for (col, id) in ids.enumerated() {
                    idsBuffer[row * length + col] = Int32(id)
                    maskBuffer[row * length + col] = 1
                }
            }
            let x = MLXArray(idsBuffer, [rows.count, length])
            let mask = MLXArray(maskBuffer, [rows.count, length])
            let tokenTypes = MLXArray.zeros(like: x)
            let output = context.model(
                x, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask.asType(.bool))
            guard let hiddenStates = output.hiddenStates else {
                fatalError("Qwen3Model (MLXEmbedders) always returns hiddenStates; got none")
            }
            // `Pooling.Strategy.last`'s own logic (`MLXEmbedders/Pooling.swift`), replicated
            // rather than called through `context.pooling(...)` so the float32 cast happens
            // **before** normalizing, matching `encoder.py::embed_ids` exactly
            // (`v = ...astype(mx.float32)` then `v = v / norm(v)`). `context.pooling`'s own
            // `normalize` runs in the model's native compute dtype — bfloat16 for this 8-bit
            // checkpoint — which is fine for the forward pass itself but produces a pooled
            // vector whose norm is off by ~0.1–0.4%, not the ~1e-6 float32 rounding a unit
            // vector should have. `EmbeddingModelOutput`'s init is `internal` to MLXEmbedders,
            // so there's no way to hand `context.pooling` an already-upcast output directly.
            let tokenCounts = sum(mask.asType(.int32), axis: -1)
            let tokenIndices = maximum(tokenCounts - MLXArray(Int32(1)), MLXArray(Int32(0)))
            let gatherIndex = tokenIndices.expandedDimensions(axes: [1, 2])
            let gathered = takeAlong(hiddenStates, gatherIndex, axis: 1)
            let lastHidden = gathered.squeezed(axis: 1).asType(.float32)
            let pooled = clmL2Normalize(lastHidden)
            eval(pooled)
            for (row, index) in batchIndices.enumerated() {
                // Bare-Int subscript on an MLXArray is a 0-dim-index `Gather` that Metal API
                // Validation aborts on for real (see `LayaHeadAttention`/`LayaDecisionModel` in
                // `Modules/Laya/LayaModel.swift`) — range-slice + squeeze instead.
                indexed.append((index, pooled[row ..< row + 1].squeezed(axis: 0)))
            }
            i = j
        }
        let ordered = indexed.sorted { $0.index < $1.index }.map(\.vector)
        return stacked(ordered)
    }

    /// `encoder.py::Encoder.embed`. Tokenizes every text, embeds the batch, and reports the
    /// total tokens spent (for `CLMEngine`'s `tokensSpent`) and which texts were truncated.
    static func embed(texts: [String], context: EmbedderModelContext) -> (
        embeddings: MLXArray, tokensSpent: Int, truncated: [Bool]
    ) {
        let results = texts.map { ids(for: $0, tokenizer: context.tokenizer) }
        let idLists = results.map(\.ids)
        let embeddings = embed(idLists: idLists, context: context)
        let tokensSpent = idLists.reduce(0) { $0 + $1.count }
        return (embeddings, tokensSpent, results.map(\.truncated))
    }
}
