import Foundation
import CoreGraphics
import CoreImage
import MLX

/// Decode + preprocessing for Depth Anything V2 Small, ported from the reference `DPTImageProcessor` config
/// (`preprocessor_config.json` of the pinned checkpoint): keep the aspect ratio, scale by whichever factor is
/// closer to 1, round each side to a multiple of 14, bicubic resize, rescale 1/255, ImageNet normalisation.
///
/// This is the Swift side of Trap 2: the reference resizes with PIL's bicubic and the app resizes with
/// Core Graphics (`.high` interpolation), so the two resized tensors differ slightly. The model parity tests
/// feed the reference's preprocessed tensor; preprocessing has its own tolerance (see the fixture README).
nonisolated enum DepthAnythingPreprocessor {

    static let resizeSize = 518
    static let ensureMultipleOf = 14
    static let imageMean: [Float] = [0.485, 0.456, 0.406]
    static let imageStd: [Float] = [0.229, 0.224, 0.225]

    /// `get_resize_output_image_size` with `keep_aspect_ratio` from the pinned `transformers` processor.
    /// Float64 throughout, because the reference computes the scale in Python floats and rounds half to even.
    static func targetSize(inputHeight: Int, inputWidth: Int,
                           size: Int = resizeSize, multiple: Int = ensureMultipleOf) -> (height: Int, width: Int) {
        let output = Double(size)
        var scaleHeight = output / Double(inputHeight)
        var scaleWidth = output / Double(inputWidth)
        if abs(1 - scaleWidth) < abs(1 - scaleHeight) {
            scaleHeight = scaleWidth
        } else {
            scaleWidth = scaleHeight
        }
        return (constrain(scaleHeight * Double(inputHeight), multiple: multiple),
                constrain(scaleWidth * Double(inputWidth), multiple: multiple))
    }

    private static func constrain(_ value: Double, multiple: Int) -> Int {
        Int(pythonRound(value / Double(multiple))) * multiple
    }

    /// Python's `round` (half to even).
    private static func pythonRound(_ value: Double) -> Double {
        let floor = value.rounded(.down)
        let fraction = value - floor
        if fraction > 0.5 { return floor + 1 }
        if fraction < 0.5 { return floor }
        return Int(floor).isMultiple(of: 2) ? floor : floor + 1
    }

    /// Preprocess an already-decoded, oriented, sRGB image. Returns `[1, Hp, Wp, 3]` float32.
    static func preprocess(image: CGImage) throws -> MLXArray {
        let target = targetSize(inputHeight: image.height, inputWidth: image.width)
        guard let pixels = resizedRGBBytes(image: image, height: target.height, width: target.width) else {
            throw DepthAnythingError.badImage("could not resize the image to \(target.width) x \(target.height)")
        }
        return normalisedArray(rgb: pixels, height: target.height, width: target.width)
    }

    /// Decode + preprocess image data (applies EXIF orientation, converts to sRGB). Trap 6.
    static func preprocess(data: Data) throws -> MLXArray {
        guard let image = DepthAnythingImage.decoded(data) else {
            throw DepthAnythingError.badImage("the image could not be decoded")
        }
        return try preprocess(image: image)
    }

    static func preprocess(url: URL) throws -> MLXArray {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DepthAnythingError.badImage("the image could not be read: \(error.localizedDescription)")
        }
        return try preprocess(data: data)
    }

    /// Core Graphics bicubic (`interpolationQuality = .high`) into an sRGB RGBA bitmap, then RGB bytes.
    private static func resizedRGBBytes(image: CGImage, height: Int, width: Int) -> [UInt8]? {
        let bytesPerRow = width * 4
        var buffer = [UInt8](repeating: 0, count: bytesPerRow * height)
        let success = buffer.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let context = CGContext(data: base, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                                          space: colorSpace,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
                return false
            }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard success else { return nil }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< width * height {
            rgb[pixel * 3] = buffer[pixel * 4]
            rgb[pixel * 3 + 1] = buffer[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = buffer[pixel * 4 + 2]
        }
        return rgb
    }

    private static func normalisedArray(rgb: [UInt8], height: Int, width: Int) -> MLXArray {
        let values = rgb.map { Float($0) / 255.0 }
        let array = MLXArray(values).reshaped(1, height, width, 3)
        let mean = MLXArray(imageMean).reshaped(1, 1, 1, 3)
        let std = MLXArray(imageStd).reshaped(1, 1, 1, 3)
        return (array - mean) / std
    }
}

/// Image decoding with EXIF orientation applied and colour converted to sRGB, so the depth map lines up with
/// the picture the user sees.
nonisolated enum DepthAnythingImage {

    static func decoded(contentsOf url: URL) -> CGImage? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decoded(data)
    }

    static func decoded(_ data: Data) -> CGImage? {
        let sourceOptions = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let raw = CGImageSourceCreateImageAtIndex(source, 0, sourceOptions) else { return nil }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? UInt32) ?? 1
        // A CIContext is created per decode: the engines run this off the main actor, and a shared
        // non-Sendable CIContext would not be safe there. One context per image is cheap.
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        let context = CIContext(options: colorSpace.map {
            [.workingColorSpace: $0, .outputColorSpace: $0]
        } ?? [:])
        let image = CIImage(cgImage: raw).oriented(forExifOrientation: Int32(orientation))
        return context.createCGImage(image, from: image.extent)
    }
}
