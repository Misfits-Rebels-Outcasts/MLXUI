import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The single place MLX LLM text generation happens. `LLMStage` (pipeline) calls it;
/// the chat sheet (`ModelRunner`) uses the same `MLXLMCommon.generate` call and is to
/// be unified onto this engine in the OP4/OP5 ChatSDK cycle. Returns the final text
/// (non-streaming) — stages thread a single `Media` value, not a token stream.
enum LLMEngine {
    /// Where `InstallManager` places a model's files.
    nonisolated static func modelDirectory(for modelID: String) -> URL {
        ModelStore.shared.directory(forModelID: modelID)
    }

    /// Load the model at `modelDir` and generate a reply to `prompt`.
    nonisolated static func generate(
        prompt: String,
        modelDir: URL,
        maxTokens: Int,
        temperature: Float = 0.7
    ) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        do {
            let loader = HFTokenizerLoader()
            let container = try await LLMModelFactory.shared.loadContainer(from: modelDir, using: loader)
            return try await container.perform { context in
                let input = try await context.processor.prepare(input: UserInput(prompt: prompt))
                // TCP-1 (`RSI/backlog.md`, journal `2026-230`): consume **raw token ids**, not
                // `.chunk` text. The decoded-text path runs every chunk through mlx-swift-lm's
                // `ToolCallProcessor`, which drops the text before a `<` whenever the `<…` tail
                // partially matches `<tool_call>` — so a reply containing `<div><span>` loses
                // the `>` between them. `LLMStage` (flow rows) never emits tool calls; skip
                // that scanner and detokenize the whole id stream once.
                let stream = try MLXLMCommon.generateTokens(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxTokens, temperature: temperature),
                    context: context)
                var tokenIds: [Int] = []
                for await generation in stream {
                    if case .token(let id) = generation { tokenIds.append(id) }
                }
                return context.tokenizer.decode(tokenIds: tokenIds)
            }
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "LLM", underlying: error)
        }
    }
}
