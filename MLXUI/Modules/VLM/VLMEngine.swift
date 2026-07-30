import Foundation
import MLX
import MLXVLM
import MLXLMCommon
import CoreImage

/// MLX-native vision-language generation via mlx-swift-lm's `VLMModelFactory`. Mirrors
/// `LLMEngine` (same `loadContainer` + `MLXLMCommon.generate` path) but feeds an image
/// alongside the prompt. Loads the **installed** mlx-community VLM safetensors (the files
/// `InstallManager` downloads) — **no Apple Vision, no temp file**: the `CGImage` goes in
/// directly as `UserInput.Image.ciImage`. Isolates the `MLXVLM` import to one file.
enum VLMEngine {
    /// Cap on input image pixels. VLM vision attention is O(patches²) and these repos ship
    /// `max_pixels: null`, so a 4K image (8 MP) OOMs Metal (~34 GB alloc). ~1 MP keeps the
    /// vision-token count safe while remaining legible for VQA/OCR (journal/2026-39).
    nonisolated static let maxInputPixels = 1_048_576

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    /// Load the VLM at `modelDir` and answer `prompt` about `image`. Loads per call
    /// (matches `LLMEngine`); instance caching is a later optimization.
    nonisolated static func generate(
        prompt: String,
        image: CGImage,
        modelDir: URL,
        maxTokens: Int
    ) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        do {
            // Some checkpoints (e.g. olmOCR-2, fine-tuned from Qwen2.5-VL) ship a config.json
            // missing the special-token ids the stock Qwen2.5-VL decoder requires. Fill them
            // in before the factory decodes it, else load fails with `keyNotFound`.
            VLMConfigNormalizer.normalizeConfigIfNeeded(at: modelDir)
            let loader = HFTokenizerLoader()
            let container = try await VLMModelFactory.shared.loadContainer(from: modelDir, using: loader)
            // Cap resolution before inference so vision attention can't OOM (see maxInputPixels).
            let bounded = ImageLoader.downscaled(image, maxPixels: maxInputPixels)
            return try await container.perform { context in
                // The image must ride on a chat *message* (not the flat `images:` list): the
                // per-model message generator reads `message.images` to inject the vision
                // placeholders the chat template needs. Passing `UserInput(prompt:images:)`
                // yields zero placeholders → an empty-array reshape crash in the vision merge.
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [
                        .user(prompt, images: [.ciImage(CIImage(cgImage: bounded))])
                    ]))
                let result = try MLXLMCommon.generate(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxTokens),
                    context: context
                ) { tokens in
                    tokens.count >= maxTokens ? .stop : .more
                }
                return result.output
            }
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "VLM", underlying: error)
        }
    }
}
