import Foundation
import CoreImage
import PaddleOCRVL

/// MLX-native document OCR via the vendored `PaddleOCRVL` package (a from-scratch Swift/MLX
/// port of PaddlePaddle's PaddleOCR-VL 0.9B — `Vendor/PaddleOCRVL`, MIT © lulzx). PaddleOCR-VL
/// uses a custom architecture `MLXVLM` doesn't support, so — unlike `OCRModule`/`VLMEngine` —
/// it runs on its own pipeline. Loads the **installed** mlx-community safetensors (the files
/// `InstallManager` downloads); the `CGImage` goes straight in as a `CIImage`, no Apple Vision.
/// Isolates the `PaddleOCRVL` import to this one file (mirrors `VLMEngine`'s `MLXVLM` isolation).
enum PaddleOCREngine {
    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`) — same as `VLMEngine`.
    nonisolated static func installedModelDirectory(id: String) -> URL {
        FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("AI Browser/models/\(id)", isDirectory: true)
    }

    /// Load the PaddleOCR-VL model at `modelDir` and transcribe `image`. Loads per call
    /// (matches `VLMEngine`/`LLMEngine`); instance caching is a later optimization. `.dynamic`
    /// mode enables NaViT-style tiling — best for full-page documents.
    nonisolated static func generate(
        image: CGImage,
        modelDir: URL,
        maxTokens: Int
    ) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        do {
            let pipeline = try await PaddleOCRVLPipeline(modelURL: modelDir, mode: .dynamic)
            return pipeline.recognize(image: CIImage(cgImage: image), task: .ocr, maxTokens: maxTokens)
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "PaddleOCR", underlying: error)
        }
    }
}
