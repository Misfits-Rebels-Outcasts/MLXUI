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
        // Architectures `MLXEmbedders` cannot load run through their own standalone runner, routed
        // by the installed `config.json`'s model_type — the one place that dispatch decision is
        // made (see `standaloneEmbedder(forModelType:)`, pure so every entry's route is testable).
        switch standaloneEmbedder(forModelType: configModelType(modelDirectory)) {
        case "qwen3-vl-embedding":
            return try await Qwen3VLEmbedder.embedTexts(texts, modelDirectory: modelDirectory)
        case "modernbert":
            return try await ModernBERTEmbedder.embed(texts, modelDirectory: modelDirectory)
        case "embeddinggemma":
            return try await EmbeddingGemmaEmbedder.embed(texts, modelDirectory: modelDirectory)
        default:
            break
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
                return try MLXErrorGuard.readRows(pooled, as: Float.self)      // WW2-3-8
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

    /// The in-app standalone runner for an installed `model_type`, or nil when `MLXEmbedders` can
    /// load it. Pure, so the dispatch of every `.embedding` entry is unit-testable without weights
    /// (MC4-4, issue Trap 1). This is the *only* place that decides it: the Qwen3-VL 4B chat card
    /// shares `model_type` `qwen3_vl` and is a `.vision` entry, which never reaches this engine.
    nonisolated static func standaloneEmbedder(forModelType type: String?) -> String? {
        switch type {
        // Qwen3-VL Embedding 2B (MC4): the package keeps its vision/language internals internal,
        // `MLXEmbedders` has no qwen3_vl, and `MLXVLM` returns logits — the app carries a trimmed
        // final-norm forward pass under `Modules/Embedding/Qwen3VLEmbed/`.
        case "qwen3_vl": return "qwen3-vl-embedding"
        // SUP-4: `EmbeddingModelOutput`'s init is internal, so ModernBERT can't go through the
        // factory/registry.
        case "modernbert": return "modernbert"
        // EmbeddingGemma: `MLXEmbedders`' port reassigns its `@ModuleInfo dense` after init (the
        // "please use Model.update(modules:)…" fatalError) and applies causal rather than
        // bidirectional attention, so it has its own runner.
        case "gemma3_text", "gemma3": return "embeddinggemma"
        default: return nil
        }
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
