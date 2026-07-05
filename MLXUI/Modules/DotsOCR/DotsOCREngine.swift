import Foundation
import MLX
import MLXVLM
import MLXLMCommon
import CoreImage

/// Runs dots.ocr / dots.mocr document OCR. Registers the custom `dots_ocr` model + `DotsVLProcessor`
/// into MLXVLM (once), then uses the **standard** `VLMModelFactory` + `MLXLMCommon.generate` path —
/// identical to `VLMEngine`, just with our model type. Loads the **installed** mlx-community
/// safetensors (the files `InstallManager` downloads); the `CGImage` goes in as a `CIImage`, no
/// Apple Vision. Isolates the MLXVLM/MLX imports for dots to this file.
enum DotsOCREngine {
    /// Fixed OCR instruction (dots.ocr has no per-run prompt in the OCR run UI).
    static let ocrPrompt = "Extract all text from this image, preserving reading order."

    /// Cap input pixels before inference: dots' `max_pixels` is ~11 MP, which would OOM Metal on
    /// large scans. ~1 MP keeps the vision-token count safe while staying legible (mirrors `VLMEngine`).
    nonisolated static let maxInputPixels = 1_048_576

    /// Installed-model directory (`Application Support/AI Browser/models/{id}`), like `VLMEngine`.
    nonisolated static func installedModelDirectory(id: String) -> URL {
        FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("AI Browser/models/\(id)", isDirectory: true)
    }

    nonisolated static func generate(image: CGImage, modelDir: URL, maxTokens: Int) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        await DotsOCRRegistration.ensureRegistered()
        do {
            let loader = HFTokenizerLoader()
            let container = try await VLMModelFactory.shared.loadContainer(from: modelDir, using: loader)
            let bounded = ImageLoader.downscaled(image, maxPixels: maxInputPixels)
            return try await container.perform { context in
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [
                        .user(Self.ocrPrompt, images: [.ciImage(CIImage(cgImage: bounded))])
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
            throw StageError.engineFailure(stage: "DotsOCR", underlying: error)
        }
    }
}
