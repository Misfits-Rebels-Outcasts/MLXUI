import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon

/// MLX-native text embedding via mlx-swift-lm's `MLXEmbedders`. Loads the **installed**
/// model directory (the A2 pattern) and returns L2-normalized vectors — so cosine similarity
/// between two vectors is just their dot product. The encode → pad → model → pool pipeline
/// mirrors the package's own reference usage. Isolates the `MLXEmbedders` import to one file.
enum EmbeddingEngine {
    /// Embed one or more texts into L2-normalized vectors (one per input).
    /// Loads the model per call (matches the other engines).
    nonisolated static func embed(_ texts: [String], modelDirectory: URL) async throws -> [[Float]] {
        guard FileManager.default.fileExists(atPath: modelDirectory.path) else {
            throw StageError.modelNotInstalled(id: modelDirectory.lastPathComponent)
        }
        // Architectures MLXEmbedders can't load run through their own standalone runner
        // (SUP-4: ModernBERT — its EmbeddingModelOutput init is internal, so it can't go through
        // the factory/registry). Route by the installed config's model_type.
        if configModelType(modelDirectory) == "modernbert" {
            return try await ModernBERTEmbedder.embed(texts, modelDirectory: modelDirectory)
        }
        // EmbeddingGemma (gemma3_text / gemma3): MLXEmbedders' port crashes on load — it reassigns
        // its `@ModuleInfo dense` after init (the "please use Model.update(modules:)…" fatalError) —
        // and applies causal rather than bidirectional attention. Use our own standalone runner.
        if let type = configModelType(modelDirectory), type == "gemma3_text" || type == "gemma3" {
            return try await EmbeddingGemmaEmbedder.embed(texts, modelDirectory: modelDirectory)
        }
        do {
            let loader = HFTokenizerLoader()
            let container = try await EmbedderModelFactory.shared.loadContainer(
                from: modelDirectory, using: loader)
            return try await container.perform { context in
                let tokenizer = context.tokenizer
                let encoded = texts.map { tokenizer.encode(text: $0, addSpecialTokens: true) }
                let maxLength = encoded.reduce(into: 1) { acc, elem in acc = max(acc, elem.count) }
                let pad = tokenizer.eosTokenId ?? 0
                let padded = stacked(encoded.map { elem in
                    MLXArray(elem + Array(repeating: pad, count: maxLength - elem.count))
                })
                let mask = (padded .!= pad)
                let tokenTypes = MLXArray.zeros(like: padded)
                let output = context.model(
                    padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask)
                let pooled = context.pooling(output, normalize: true, applyLayerNorm: true)
                pooled.eval()
                return pooled.map { $0.asArray(Float.self) }
            }
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "Embedding", underlying: error)
        }
    }

    /// The `model_type` from an installed model's `config.json`, used to route to a runner.
    nonisolated static func configModelType(_ directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return obj["model_type"] as? String
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
