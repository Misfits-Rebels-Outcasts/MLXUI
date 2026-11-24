import Foundation
import MLX

/// PF-5 — the Privacy Filter encoder: embedding → 8 bidirectional MoE blocks → final RMSNorm →
/// 33-label token-classification head.
///
/// The interface is frozen by `/tmp/pf-team/CONTRACT.md`:
/// - ``logits(tokenIDs:positionOffset:)`` is the only primitive the decoder needs;
/// - ``trace(tokenIDs:)`` exists for parity only;
/// - ``logProbs(tokenIDs:positionOffset:)`` gives the fp32 log-softmax the Viterbi decoder runs on.
///
/// **Trap 7 (thread-bound MLX arrays).** Everything here is synchronous: one call builds the graph,
/// consumes it and returns an array on the calling thread. `logProbs` is the one place an array
/// crosses back to Swift, and it goes through `MLXErrorGuard.readRows`, which evaluates under
/// `withError` first and throws instead of reading an array whose evaluation failed. No `await`
/// appears anywhere in this file, so there is no eval/await ordering hazard.
nonisolated final class PrivacyFilterModel {

    let config: PrivacyFilterConfig
    let weights: PrivacyFilterWeights
    let rotary: PrivacyFilterNetwork.Rotary

    /// Build from already-loaded weights (`PrivacyFilterWeights.load` performs the strict checks).
    init(weights: PrivacyFilterWeights, config: PrivacyFilterConfig) {
        self.weights = weights
        self.config = config
        self.rotary = PrivacyFilterNetwork.yarnRotary(config)
    }

    /// Read `config.json` + weights from an installed model folder.
    static func load(from directory: URL) throws -> PrivacyFilterModel {
        let config = try PrivacyFilterConfig.load(from: directory)
        let weights = try PrivacyFilterWeights.load(from: directory, config: config)
        return PrivacyFilterModel(weights: weights, config: config)
    }

    /// One forward pass over `tokenIDs`; RoPE positions start at `positionOffset` (absolute positions
    /// for the windowed path). Returns `[count, numLabels]` bf16 logits, no log-softmax. The mask is
    /// the band `abs(i - j) <= slidingWindow` over this slice.
    func logits(tokenIDs: [Int], positionOffset: Int = 0) -> MLXArray {
        forward(tokenIDs: tokenIDs, positionOffset: positionOffset, captureTrace: false).logits
    }

    /// For PF-5 parity only: the embedding output, the output of each block (pre final norm) and the
    /// final logits — the same quantities the reference fixture saves.
    func trace(tokenIDs: [Int])
        -> (embedding: MLXArray, blocks: [MLXArray], logits: MLXArray)
    {
        forward(tokenIDs: tokenIDs, positionOffset: 0, captureTrace: true)
    }

    /// fp32 log-softmax of ``logits(tokenIDs:positionOffset:)``, read back to Swift. Evaluated under
    /// `MLXErrorGuard` first, so an MLX failure is thrown rather than read (`MLXErrorGuard.read`).
    ///
    /// This is `throws` where the contract sketch showed a non-returning failure path: the read can
    /// fail (Trap 7) and the only honest options are a thrown sentence or a bogus row. The decoder
    /// and inference both live in `throws` code, so `try` costs nothing there.
    func logProbs(tokenIDs: [Int], positionOffset: Int = 0) throws -> [[Float]] {
        let logits = logits(tokenIDs: tokenIDs, positionOffset: positionOffset).asType(.float32)
        guard logits.ndim == 2, logits.dim(0) > 0 else {
            throw PrivacyFilterConfig.ModelError.inputInvalid("the model returned no rows")
        }
        // log_softmax in fp32: x - max - log(sum(exp(x - max))).
        let maximum = logits.max(axis: -1, keepDims: true)
        let shifted = logits - maximum
        let logDenominator = log(exp(shifted).sum(axis: -1, keepDims: true))
        return try MLXErrorGuard.readRows(shifted - logDenominator, as: Float.self)
    }

    // MARK: - Internals

    private func forward(
        tokenIDs: [Int], positionOffset: Int, captureTrace: Bool
    ) -> (embedding: MLXArray, blocks: [MLXArray], logits: MLXArray) {
        let count = tokenIDs.count
        guard count > 0 else {
            let zeros = MLXArray.zeros([0, config.numLabels], dtype: .bfloat16)
            return (zeros, [], zeros)
        }
        let ids = MLXArray(tokenIDs.map { Int32($0) })
        let embedding = MLX.take(weights.embedTokens, ids, axis: 0)          // [count, hidden] bf16
        var hidden = embedding
        var blocks: [MLXArray] = []
        if captureTrace { blocks.reserveCapacity(weights.layers.count) }
        for layer in weights.layers {
            hidden = PrivacyFilterNetwork.encoderLayer(
                hidden, layer: layer, rotary: rotary, config: config,
                positionOffset: positionOffset)
            if captureTrace { blocks.append(hidden) }
        }
        let normalized = PrivacyFilterNetwork.rmsNorm(
            hidden, weight: weights.norm, eps: config.rmsNormEps)
        let logits = PrivacyFilterNetwork.linear(
            normalized, weight: weights.scoreWeight, bias: weights.scoreBias)
        return (embedding, blocks, logits)
    }
}
