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
        let base = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("AI Browser/models/\(modelID)")
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
                let result = try MLXLMCommon.generate(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxTokens, temperature: temperature),
                    context: context
                ) { tokens in
                    tokens.count >= maxTokens ? .stop : .more
                }
                return result.output
            }
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "LLM", underlying: error)
        }
    }
}
