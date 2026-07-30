import Foundation
import MLX
import MLXVLM
import MLXLMCommon
import CoreImage

/// Runs DeepSeek-OCR-2 document OCR. Registers the custom `deepseekocr_2` model + `DeepseekVLV2Processor`
/// into MLXVLM (once), then uses the standard `VLMModelFactory` + `MLXLMCommon.generate` path — the
/// processor builds the tiled pixels + `<image>` token layout, so the engine just hands it the image.
/// Mirrors `DotsOCREngine`; isolates the DeepSeek MLXVLM/MLX imports here.
enum DeepSeekOCREngine {
    /// Cap input pixels before the processor's 1024² pad, so oversized scans don't blow up memory
    /// (this is the **bf16** checkpoint — no quantization, so it's already heavy).
    nonisolated static let maxInputPixels = 4_194_304   // ~2048²

    /// `<｜end▁of▁sentence｜>` = id 1 (checkpoint `config.json` `eos_token_id`). We stop on it explicitly
    /// because `VLMModelFactory` builds the run's `ModelContext` with a stand-in configuration that can
    /// drop the checkpoint's `eos_token_id`; if `tokenizer.eosTokenId` also isn't surfaced, the
    /// framework's stop set misses EOS and the model re-emits the whole page after each end token.
    nonisolated static let eosTokenId = 1

    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    nonisolated static func generate(image: CGImage, modelDir: URL, maxTokens: Int) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        await DeepSeekOCRRegistration.ensureRegistered()
        do {
            let loader = HFTokenizerLoader()
            let container = try await VLMModelFactory.shared.loadContainer(from: modelDir, using: loader)
            let bounded = ImageLoader.downscaled(image, maxPixels: maxInputPixels)
            return try await container.perform { context in
                // The DeepSeek processor builds the fixed OCR prompt + tiling from the image itself.
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [.user("", images: [.ciImage(CIImage(cgImage: bounded))])]))
                // Deterministic OCR decoding (temperature 0). No repetition penalty: it's a blunt tool
                // here — it damaged the legitimately-repeated lines while barely denting the page loop,
                // whose real cause is EOS not stopping generation (handled below).
                let result = try MLXLMCommon.generate(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxTokens, temperature: 0),
                    context: context
                ) { tokens in
                    // Stop on the checkpoint's end-of-sentence token even if the framework's stop set
                    // missed it (see `eosTokenId`), else the model re-reads the page until `maxTokens`.
                    if tokens.last == Self.eosTokenId { return .stop }
                    return tokens.count >= maxTokens ? .stop : .more
                }
                // Stopping via the closure includes the EOS token in the output; drop its literal form.
                return result.output
                    .replacingOccurrences(of: "<｜end▁of▁sentence｜>", with: "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "DeepSeekOCR", underlying: error)
        }
    }
}
