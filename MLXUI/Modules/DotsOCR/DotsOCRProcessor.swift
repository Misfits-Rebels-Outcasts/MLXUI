import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// `UserInputProcessor` for dots.ocr / dots.mocr. dots ships a `Qwen2_5_VLProcessor`-derived
/// processor, so the image pipeline is identical to Qwen2.5-VL (smart-resize → sRGB → bicubic →
/// normalize → patchify with `grid_thw`) — reusing the **public** `MediaProcessing` helpers and
/// `Qwen2VLProcessorConfiguration` shape (`Qwen25VLProcessorConfiguration`). The one difference:
/// dots' image placeholder is `<|imgpad|>` (id 151665), so we expand that token (not
/// `<|image_pad|>`) to the per-image vision-token count. `QwenVL.targetSize`/`patchify` are
/// package-internal, so they're reimplemented here.
struct DotsOCRProcessor: UserInputProcessor {
    /// dots.ocr `image_token_id` (config.json). The chat template inserts a single `<|imgpad|>`
    /// per image; we expand it to `grid.product / mergeSize²` copies to match the vision tokens.
    static let imageTokenId = 151665

    private let config: Qwen25VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    init(config: Qwen25VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)

        if input.images.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        let pixelsAndFrames = try input.images.map {
            try preprocess(images: [$0.asCIImage()], processing: input.processing)
        }
        let pixels = concatenated(pixelsAndFrames.map { $0.0 })
        let frames = pixelsAndFrames.map { $0.1 }

        promptTokens = expandImageTokens(promptTokens, frames: frames)

        return LMInput(
            text: .init(tokens: MLXArray(promptTokens)),
            image: .init(pixels: pixels, frames: frames))
    }

    // MARK: - Image preprocessing (mirrors Qwen2.5-VL)

    private func preprocess(images: [CIImage], processing: UserInput.Processing?) throws
        -> (MLXArray, THW)
    {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }
        let size = images[0].extent.size
        let (resizedHeight, resizedWidth) = try Self.targetSize(
            height: Int(size.height), width: Int(size.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels, maxPixels: config.size.maxPixels)
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processed = images
            .map { MediaProcessing.inSRGBToneCurveSpace($0) }
            .map { MediaProcessing.resampleBicubic($0, to: resizedSize) }
            .map { MediaProcessing.normalize($0, mean: config.imageMeanTuple, std: config.imageStdTuple) }
            .map { MediaProcessing.asMLXArray($0) }

        return try Self.patchify(
            images: processed, mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }

    /// Expand each single `<|imgpad|>` (id `imageTokenId`) to the per-image vision-token count
    /// (`grid.product / mergeSize²`). Robust to whatever `<|vision_start|>`/`<|vision_end|>`
    /// wrapping the chat template uses — it only rewrites the pad token itself.
    private func expandImageTokens(_ tokens: [Int], frames: [THW]) -> [Int] {
        let mergeLength = config.mergeSize * config.mergeSize
        var result: [Int] = []
        result.reserveCapacity(tokens.count)
        var frameIndex = 0
        for token in tokens {
            if token == Self.imageTokenId, frameIndex < frames.count {
                let count = frames[frameIndex].product / mergeLength
                result.append(contentsOf: Array(repeating: Self.imageTokenId, count: count))
                frameIndex += 1
            } else {
                result.append(token)
            }
        }
        return result
    }

    // MARK: - QwenVL math reimplemented (package-internal upstream)

    /// `image_processing_qwen2_vl.smart_resize`.
    static func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws -> (Int, Int)
    {
        if height < factor || width < factor {
            throw StageError.engineFailure(
                stage: "DotsOCR",
                underlying: NSError(domain: "DotsOCR", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: "Image \(width)×\(height) smaller than patch factor \(factor)"]))
        }
        var hBar = max(factor, Int((Float(height) / Float(factor)).rounded()) * factor)
        var wBar = max(factor, Int((Float(width) / Float(factor)).rounded()) * factor)
        if hBar * wBar > maxPixels {
            let beta = (Float(height * width) / Float(maxPixels)).squareRoot()
            hBar = Int((Float(height) / beta / Float(factor)).rounded(.down)) * factor
            wBar = Int((Float(width) / beta / Float(factor)).rounded(.down)) * factor
        } else if hBar * wBar < minPixels {
            let beta = (Float(minPixels) / Float(height * width)).squareRoot()
            hBar = Int((Float(height) * beta / Float(factor)).rounded(.up)) * factor
            wBar = Int((Float(width) * beta / Float(factor)).rounded(.up)) * factor
        }
        hBar = (hBar / factor) * factor
        wBar = (wBar / factor) * factor
        return (hBar, wBar)
    }

    /// `image_processing_qwen2_vl._preprocess` patch flattening → `(pixels, grid_thw)`.
    static func patchify(images: [MLXArray], mergeSize: Int, patchSize: Int, temporalPatchSize: Int)
        throws -> (MLXArray, THW)
    {
        guard let first = images.first else {
            throw StageError.engineFailure(
                stage: "DotsOCR",
                underlying: NSError(domain: "DotsOCR", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "No image to patchify"]))
        }
        let resizedHeight = first.dim(-2)
        let resizedWidth = first.dim(-1)
        var patches = concatenated(images)

        let mod = patches.dim(0) % temporalPatchSize
        if mod != 0 {
            let last = patches[-1, .ellipsis]
            let repeated = tiled(last, repetitions: [temporalPatchSize - mod, 1, 1, 1])
            patches = concatenated([patches, repeated])
        }
        let channel = patches.dim(1)
        let gridT = patches.dim(0) / temporalPatchSize
        let gridH = resizedHeight / patchSize
        let gridW = resizedWidth / patchSize

        patches = patches.reshaped(
            gridT, temporalPatchSize, channel,
            gridH / mergeSize, mergeSize, patchSize,
            gridW / mergeSize, mergeSize, patchSize)
        patches = patches.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
        let flattened = patches.reshaped(
            gridT * gridH * gridW, channel * temporalPatchSize * patchSize * patchSize)

        return (flattened, THW(gridT, gridH, gridW))
    }
}
