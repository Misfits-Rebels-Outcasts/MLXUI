import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Errors specific to the causal yes/no scorer that don't fit `StageError`'s existing cases.
nonisolated enum RerankEngineError: Error, CustomStringConvertible {
    /// The tokenizer has no id for "yes" or "no" — should never happen for Qwen3-Reranker,
    /// but a silently-wrong score is worse than a loud refusal.
    case noYesNoTokenIDs
    /// `RerankSDK.makeStage`'s stage received input that doesn't split into a query and a
    /// document on the first newline — the seam contract `RealExecutor.runModel`'s
    /// `engines.rerank.` branch is supposed to guarantee.
    case malformedScoringInput

    var description: String {
        switch self {
        case .noYesNoTokenIDs:
            return "This reranker's tokenizer has no \"yes\"/\"no\" token — it isn't a causal yes/no scorer."
        case .malformedScoringInput:
            return "The rerank stage received input with no query/document split — this is a wiring bug, not a model problem."
        }
    }
}

/// The Qwen3-Reranker causal yes/no scorer: one forward pass per `(query, document)` pair,
/// reading `softmax([logit("no"), logit("yes")])[1]` at the last prompt position. Ported
/// verbatim from the model card's own reference implementation (see `RerankPrompt`) — never
/// batched: right-padding a batch corrupts the last-token read for every candidate shorter
/// than the longest (MoC-4-3, `RSI/DelegateMoCBacklog.md`).
enum RerankEngine {
    /// Where `InstallManager` places a model's files.
    nonisolated static func modelDirectory(for modelID: String) -> URL {
        ModelStore.shared.directory(forModelID: modelID)
    }

    /// The relevance score of `document` against `query`, in `0...1`.
    nonisolated static func score(query: String, document: String, modelDir: URL) async throws -> Double {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        do {
            let loader = HFTokenizerLoader()
            let container = try await LLMModelFactory.shared.loadContainer(from: modelDir, using: loader)
            return try await container.perform { context in
                guard let yesID = context.tokenizer.convertTokenToId("yes"),
                      let noID = context.tokenizer.convertTokenToId("no") else {
                    throw RerankEngineError.noYesNoTokenIDs
                }
                let content = RerankPrompt.content(query: query, document: document)
                let ids = context.tokenizer.encode(text: RerankPrompt.prefix, addSpecialTokens: false)
                    + context.tokenizer.encode(text: content, addSpecialTokens: false)
                    + context.tokenizer.encode(text: RerankPrompt.suffix, addSpecialTokens: false)
                // One forward pass, no KV cache — this is scoring, not incremental generation.
                let input = MLXArray(ids, [1, ids.count])
                let logits = context.model(input, cache: nil).asType(.float32)
                let lastPosition = ids.count - 1
                let row = logits[0, lastPosition].asArray(Float.self)
                let logitYes = Double(row[yesID])
                let logitNo = Double(row[noID])
                // softmax([logit_no, logit_yes])[1], numerically stable.
                let m = max(logitYes, logitNo)
                let expYes = exp(logitYes - m)
                let expNo = exp(logitNo - m)
                return expYes / (expYes + expNo)
            }
        } catch let error as StageError {
            throw error
        } catch let error as RerankEngineError {
            throw StageError.engineFailure(stage: "Rerank", underlying: error)
        } catch {
            throw StageError.engineFailure(stage: "Rerank", underlying: error)
        }
    }
}
