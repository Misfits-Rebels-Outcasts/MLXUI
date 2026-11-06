import Foundation
import CoreGraphics
import MLX

/// The output contract (owner decision 4): an 8-bit grayscale PNG, near = bright, per-image min–max
/// normalised, at the **input image's pixel size**. The float prediction is upsampled bicubically first; a
/// prediction whose max equals its min becomes all black, never NaN.
nonisolated enum DepthAnythingPostprocessor {

    /// Row-major 8-bit grayscale pixels at `height` x `width`.
    static func grayscale(prediction: MLXArray, height: Int, width: Int) throws -> [UInt8] {
        guard height > 0, width > 0 else {
            throw DepthAnythingError.badInput("the output size must be positive")
        }
        var plane = prediction
        if plane.ndim == 3 {
            guard plane.dim(0) == 1 else {
                throw DepthAnythingError.badInput("expected one depth map, got \(plane.dim(0))")
            }
            plane = plane.reshaped(plane.dim(1), plane.dim(2))
        }
        guard plane.ndim == 2 else {
            throw DepthAnythingError.badInput("the depth prediction must be [1, H, W] or [H, W]")
        }
        let resized = DepthAnythingOps.resizeBicubic(plane.reshaped(1, plane.dim(0), plane.dim(1), 1),
                                                     outH: height, outW: width, alignCorners: false)
        let values = try MLXErrorGuard.read(resized.asType(.float32), as: Float.self)
        guard let low = values.min(), let high = values.max() else {
            throw DepthAnythingError.badInput("the depth prediction was empty")
        }
        if high <= low {
            return [UInt8](repeating: 0, count: height * width)
        }
        let scale = 255.0 / (high - low)
        return values.map { value in
            let level = ((value - low) * scale).rounded()
            return UInt8(max(0, min(255, Int(level))))
        }
    }

    /// A grayscale byte map as a `CGImage`.
    static func image(gray: [UInt8], width: Int, height: Int) throws -> CGImage {
        let data = Data(gray)
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: width, height: height,
                                  bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil,
                                  shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the depth map image")
        }
        return image
    }

    /// The grayscale map as a `CGImage` straight from a prediction.
    static func image(prediction: MLXArray, height: Int, width: Int) throws -> CGImage {
        let pixels = try grayscale(prediction: prediction, height: height, width: width)
        return try image(gray: pixels, width: width, height: height)
    }

    /// The gray bytes of an 8-bit grayscale image (the Run screen reuses the model's own output).
    static func grayBytes(from image: CGImage) -> [UInt8]? {
        let width = image.width, height = image.height
        var bytes = [UInt8](repeating: 0, count: width * height)
        let success = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let base = raw.baseAddress,
                  let context = CGContext(data: base, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                          bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return success ? bytes : nil
    }

    /// **Display-only** colour map: the saved PNG is always the grayscale map (owner decision 6). Bright
    /// (near) runs dark blue → cyan → green → yellow → orange → white, so the picture reads as depth.
    /// Implementer's call, pending owner confirmation.
    static func colourImage(gray: [UInt8], width: Int, height: Int) throws -> CGImage {
        let stops: [(Float, Float, Float)] = [
            (13, 24, 74),      // far: dark blue
            (15, 105, 172),
            (23, 178, 160),
            (150, 214, 90),
            (252, 191, 53),
            (255, 255, 240),   // near: white
        ]
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for index in 0 ..< width * height {
            let position = Float(gray[index]) / 255
            let scaled = position * Float(stops.count - 1)
            let lower = min(Int(scaled.rounded(.down)), stops.count - 1)
            let upper = min(lower + 1, stops.count - 1)
            let fraction = scaled - Float(lower)
            let low = stops[lower], high = stops[upper]
            rgba[index * 4] = UInt8(max(0, min(255, (low.0 + (high.0 - low.0) * fraction).rounded())))
            rgba[index * 4 + 1] = UInt8(max(0, min(255, (low.1 + (high.1 - low.1) * fraction).rounded())))
            rgba[index * 4 + 2] = UInt8(max(0, min(255, (low.2 + (high.2 - low.2) * fraction).rounded())))
        }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData),
              let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: colorSpace,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil,
                                  shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the colour depth preview")
        }
        return image
    }

    /// The 8-bit grayscale PNG `Save PNG` writes.
    static func pngData(prediction: MLXArray, height: Int, width: Int) throws -> Data {
        let image = try image(prediction: prediction, height: height, width: width)
        guard let data = PNGEncoder.pngData(from: image) else {
            throw DepthAnythingError.badInput("could not encode the depth map as PNG")
        }
        return data
    }
}
