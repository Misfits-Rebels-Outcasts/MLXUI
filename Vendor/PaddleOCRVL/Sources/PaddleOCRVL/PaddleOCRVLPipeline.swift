import CoreImage
import Foundation
import MLX
import Tokenizers
import Hub

public class PaddleOCRVLPipeline {
    public let model: PaddleOCRVLModel
    public let tokenizer: any Tokenizer
    public let imageProcessor: PaddleOCRVLImageProcessor
    public let generator: PaddleOCRVLGenerator
    public let config: PaddleOCRVLConfig
    public let processingMode: ProcessingMode

    public convenience init(modelPath: String) async throws {
        let modelURL = URL(fileURLWithPath: modelPath)
        try await self.init(modelURL: modelURL, mode: .base)
    }

    public convenience init(modelURL: URL) async throws {
        try await self.init(modelURL: modelURL, mode: .base)
    }

    public init(modelURL: URL, mode: ProcessingMode) async throws {
        self.processingMode = mode
        self.config = try PaddleOCRVLConfig.load(from: modelURL)
        self.model = try PaddleOCRVLModel.load(from: modelURL)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
        self.imageProcessor = PaddleOCRVLImageProcessor(mode: mode)
        self.generator = PaddleOCRVLGenerator(model: model, tokenizer: tokenizer, config: config)
    }

    public init(
        modelURL: URL,
        imageSize: Int,
        dynamicResolution: Bool
    ) async throws {
        self.processingMode = dynamicResolution ? .dynamic : .base
        self.config = try PaddleOCRVLConfig.load(from: modelURL)
        self.model = try PaddleOCRVLModel.load(from: modelURL)
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
        self.imageProcessor = PaddleOCRVLImageProcessor(
            imageSize: imageSize,
            dynamicResolution: dynamicResolution
        )
        self.generator = PaddleOCRVLGenerator(model: model, tokenizer: tokenizer, config: config)
    }

    public func recognize(imagePath: String, task: PaddleOCRTask = .ocr, maxTokens: Int = 1024) throws -> String {
        let processedImages = try imageProcessor.process(imageAt: imagePath)
        return recognize(processedImages: processedImages, task: task, maxTokens: maxTokens)
    }

    public func recognize(image: CIImage, task: PaddleOCRTask = .ocr, maxTokens: Int = 1024) -> String {
        let processedImages = imageProcessor.process(image)
        return recognize(processedImages: processedImages, task: task, maxTokens: maxTokens)
    }

    public func recognize(processedImages: ProcessedImages, task: PaddleOCRTask = .ocr, maxTokens: Int = 1024) -> String {
        let result = generator.generate(
            processedImages: processedImages,
            task: task,
            maxNewTokens: maxTokens,
            temperature: 0.0,
            topP: 1.0
        )
        return result.text
    }

    public func recognize(pixelValues: MLXArray, task: PaddleOCRTask = .ocr, maxTokens: Int = 1024) -> String {
        let result = generator.generate(
            pixelValues: pixelValues,
            task: task,
            maxNewTokens: maxTokens,
            temperature: 0.0,
            topP: 1.0
        )
        return result.text
    }

    public func recognizeBatch(imagePaths: [String], task: PaddleOCRTask = .ocr, maxTokens: Int = 1024) throws -> [String] {
        return try imagePaths.map { path in
            try recognize(imagePath: path, task: task, maxTokens: maxTokens)
        }
    }

    public func recognizeBatch(images: [CIImage], task: PaddleOCRTask = .ocr, maxTokens: Int = 1024) -> [String] {
        return images.map { image in
            recognize(image: image, task: task, maxTokens: maxTokens)
        }
    }
}

extension PaddleOCRVLPipeline {
    public static let supportedFormats = ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"]

    public static func isSupportedImage(_ path: String) -> Bool {
        let ext = URL(fileURLWithPath: path).pathExtension.lowercased()
        return supportedFormats.contains(ext)
    }
}
