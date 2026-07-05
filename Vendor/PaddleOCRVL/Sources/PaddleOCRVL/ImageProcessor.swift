import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import MLX

public enum ProcessingMode: String, CaseIterable, Sendable {
    case base
    case dynamic

    public var imageSize: Int {
        switch self {
        case .base: return 448
        case .dynamic: return 448
        }
    }

    public var dynamicResolution: Bool {
        switch self {
        case .dynamic: return true
        default: return false
        }
    }
}

public struct ProcessedImages: Sendable {
    public let pixelValues: MLXArray
    public let width: Int
    public let height: Int
    public let numImageTokens: Int

    public init(pixelValues: MLXArray, width: Int, height: Int, numImageTokens: Int) {
        self.pixelValues = pixelValues
        self.width = width
        self.height = height
        self.numImageTokens = numImageTokens
    }
}

public struct BatchProcessedImages: Sendable {
    public let items: [ProcessedImages]

    public var batchSize: Int { items.count }

    public var maxNumImageTokens: Int {
        items.map { $0.numImageTokens }.max() ?? 0
    }

    public var canBatchVision: Bool {
        guard let first = items.first else { return false }
        return items.allSatisfy { $0.numImageTokens == first.numImageTokens }
    }

    public func stackedPixelValues() -> MLXArray {
        let views = items.map { $0.pixelValues.squeezed(axis: 0) }
        return MLX.stacked(views, axis: 0)
    }
}

public struct PaddleOCRVLImageProcessor {
    // PaddleOCR-VL's SigLIP normalizes to [-1, 1] with mean/std 0.5 (per preprocessor_config.json),
    // NOT ImageNet stats. Wrong normalization corrupts the vision features → the model can't read
    // the image and falls back to hallucinating generic OCR-test text.
    public static let imageMean: (Float, Float, Float) = (0.5, 0.5, 0.5)
    public static let imageStd: (Float, Float, Float) = (0.5, 0.5, 0.5)

    public let imageSize: Int
    public let dynamicResolution: Bool
    public let patchSize: Int
    public let minPixels: Int
    public let maxPixels: Int

    private let context = CIContext()

    public init(
        imageSize: Int = 448,
        dynamicResolution: Bool = false,
        patchSize: Int = 14,
        minPixels: Int = 112_896,
        maxPixels: Int = 1_003_520
    ) {
        self.imageSize = imageSize
        self.dynamicResolution = dynamicResolution
        self.patchSize = patchSize
        self.minPixels = minPixels
        self.maxPixels = maxPixels
    }

    public init(mode: ProcessingMode) {
        self.imageSize = mode.imageSize
        self.dynamicResolution = mode.dynamicResolution
        self.patchSize = 14
        // Match preprocessor_config.json (min_pixels / max_pixels).
        self.minPixels = 112_896
        self.maxPixels = 1_003_520
    }

    public func process(imageAt path: String) throws -> ProcessedImages {
        let url = URL(fileURLWithPath: path)
        return try process(imageAt: url)
    }

    public func process(imageAt url: URL) throws -> ProcessedImages {
        guard let ciImage = CIImage(contentsOf: url) else {
            throw PaddleOCRVLError.imageLoadFailed(url.path)
        }
        return process(ciImage)
    }

    public func process(_ image: CIImage) -> ProcessedImages {
        if dynamicResolution {
            return processDynamicResolution(image)
        } else {
            return processFixedResolution(image)
        }
    }

    private func processFixedResolution(_ image: CIImage) -> ProcessedImages {
        let targetSize = CGSize(width: CGFloat(imageSize), height: CGFloat(imageSize))
        let resizedImage = resizeWithPadding(image, to: targetSize)
        let pixelValues = normalizeAndConvert(resizedImage)

        // The projector merges each 2×2 patch block into one token, so the LM sees
        // (gridH/2)·(gridW/2) image tokens, not gridH·gridW.
        let numPatches = imageSize / patchSize
        let numImageTokens = (numPatches / 2) * (numPatches / 2)

        return ProcessedImages(
            pixelValues: pixelValues,
            width: imageSize,
            height: imageSize,
            numImageTokens: numImageTokens
        )
    }

    private func processDynamicResolution(_ image: CIImage) -> ProcessedImages {
        let originalWidth = Int(image.extent.width)
        let originalHeight = Int(image.extent.height)

        let (targetWidth, targetHeight) = smartResize(
            width: originalWidth,
            height: originalHeight
        )

        let targetSize = CGSize(width: CGFloat(targetWidth), height: CGFloat(targetHeight))
        let resizedImage = resizeDirectly(image, to: targetSize)
        let pixelValues = normalizeAndConvert(resizedImage)

        // 2×2 projector merge → (gridH/2)·(gridW/2) image tokens (smartResize keeps both even).
        let numPatchesW = targetWidth / patchSize
        let numPatchesH = targetHeight / patchSize
        let numImageTokens = (numPatchesW / 2) * (numPatchesH / 2)

        return ProcessedImages(
            pixelValues: pixelValues,
            width: targetWidth,
            height: targetHeight,
            numImageTokens: numImageTokens
        )
    }

    private func smartResize(width: Int, height: Int) -> (Int, Int) {
        let currentPixels = width * height

        var scale: Double = 1.0
        if currentPixels > maxPixels {
            scale = sqrt(Double(maxPixels) / Double(currentPixels))
        } else if currentPixels < minPixels {
            scale = sqrt(Double(minPixels) / Double(currentPixels))
        }

        var newWidth = Int(Double(width) * scale)
        var newHeight = Int(Double(height) * scale)

        // Round to a multiple of 2·patchSize so the patch grid is even in both dims — required
        // by the projector's 2×2 spatial merge.
        let unit = patchSize * 2
        newWidth = max(unit, (newWidth / unit) * unit)
        newHeight = max(unit, (newHeight / unit) * unit)

        return (newWidth, newHeight)
    }

    private func resizeWithPadding(_ image: CIImage, to size: CGSize) -> CIImage {
        let originalWidth = image.extent.width
        let originalHeight = image.extent.height

        let scaleX = size.width / originalWidth
        let scaleY = size.height / originalHeight
        let scale = min(scaleX, scaleY)

        let scaledWidth = originalWidth * scale
        let scaledHeight = originalHeight * scale

        let scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        var scaledImage = image.transformed(by: scaleTransform)

        let padX = (size.width - scaledWidth) / 2
        let padY = (size.height - scaledHeight) / 2

        let translateTransform = CGAffineTransform(translationX: padX, y: padY)
        scaledImage = scaledImage.transformed(by: translateTransform)

        let fillColor = CIColor(red: 0.0, green: 0.0, blue: 0.0, alpha: 1.0)
        let backgroundRect = CGRect(origin: .zero, size: size)
        let background = CIImage(color: fillColor).cropped(to: backgroundRect)

        let composited = scaledImage.composited(over: background)
        return composited.cropped(to: backgroundRect)
    }

    private func resizeDirectly(_ image: CIImage, to size: CGSize) -> CIImage {
        let scaleX = size.width / image.extent.width
        let scaleY = size.height / image.extent.height

        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = image
        filter.scale = Float(scaleY)
        filter.aspectRatio = Float(scaleX / scaleY)
        let scaledImage = filter.outputImage!

        return scaledImage.cropped(to: CGRect(origin: .zero, size: size))
    }

    private func normalizeAndConvert(_ image: CIImage) -> MLXArray {
        let rawArray = asMLXArrayRaw(image)
        let mean = MLXArray([Self.imageMean.0, Self.imageMean.1, Self.imageMean.2])
        let std = MLXArray([Self.imageStd.0, Self.imageStd.1, Self.imageStd.2])
        let normalized = (rawArray - mean) / std
        return normalized.asType(.bfloat16)
    }

    private func asMLXArrayRaw(_ image: CIImage) -> MLXArray {
        let size = image.extent.size
        let w = Int(size.width.rounded())
        let h = Int(size.height.rounded())

        let format = CIFormat.RGBA8
        let bytesPerPixel = 4
        let bytesPerRow = w * bytesPerPixel

        var data = Data(count: w * h * bytesPerPixel)
        data.withUnsafeMutableBytes { ptr in
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: format,
                colorSpace: colorSpace
            )
            context.clearCaches()
        }

        let uint8Array = MLXArray(data, [h, w, 4], type: UInt8.self)
        var array = uint8Array.asType(.float32) / 255.0
        array = array[0..., 0..., ..<3]
        array = array.reshaped(1, h, w, 3)

        return array
    }

    public func processBatch(imagesAt paths: [String]) throws -> BatchProcessedImages {
        let items = try paths.map { try process(imageAt: $0) }
        return BatchProcessedImages(items: items)
    }

    public func processBatch(_ images: [CIImage]) -> BatchProcessedImages {
        let items = images.map { process($0) }
        return BatchProcessedImages(items: items)
    }
}

public enum PaddleOCRVLError: LocalizedError {
    case imageLoadFailed(String)
    case modelLoadFailed(String)
    case tokenizerLoadFailed(String)
    case imageRequired
    case configurationError(String)
    case generationError(String)

    public var errorDescription: String? {
        switch self {
        case .imageLoadFailed(let path):
            return "Failed to load image at: \(path)"
        case .modelLoadFailed(let message):
            return "Failed to load model: \(message)"
        case .tokenizerLoadFailed(let message):
            return "Failed to load tokenizer: \(message)"
        case .imageRequired:
            return "Image is required for OCR"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .generationError(let message):
            return "Generation error: \(message)"
        }
    }
}
