import Foundation
import CoreGraphics
import ImageIO

/// Loads a fully-decoded, **memory-resident** `CGImage` for the run views. In the sandbox,
/// a file-picked URL is only readable while its security scope is held; `CGImageSource…`
/// creates a *lazy* image that memory-maps the file and decodes on first pixel access — which
/// happens later, inside model preprocessing, after the scope has closed (`open` fails →
/// 0 image frames → "placeholder tokens ≠ frames"). So we read the bytes and force a decode
/// into an owned bitmap up front. Shared by `ImageQARunView` (VLM) and `OCRRunView`.
enum ImageLoader {
    /// Read a security-scoped file URL into memory and return an image that no longer
    /// references the file — safe to use after the scope closes and off the main actor.
    static func decodedCGImage(fromSecurityScoped url: URL) -> CGImage? {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        guard let data = try? Data(contentsOf: url) else { return nil }
        return decodedCGImage(from: data)
    }

    /// Decode image bytes into an owned bitmap (used by the drag-and-drop path too).
    static func decodedCGImage(from data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(
                source, 0,
                [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
        else { return nil }
        return forceDecoded(image)
    }

    /// Downscale (preserving aspect ratio) so `width * height <= maxPixels`; returns the image
    /// unchanged if already within budget. VLMs (Qwen3-VL etc.) turn image pixels into vision
    /// patches, and their attention is O(patches²) — an un-capped 4K image (8 MP) explodes to a
    /// 34 GB allocation past Metal's ~10 GB buffer limit (journal/2026-39). The model's own
    /// `preprocessor_config.json` here has `max_pixels: null`, so we cap on the app side.
    static func downscaled(_ image: CGImage, maxPixels: Int) -> CGImage {
        let width = image.width, height = image.height
        let pixels = width * height
        guard pixels > maxPixels, width > 0, height > 0 else { return image }
        let scale = (Double(maxPixels) / Double(pixels)).squareRoot()
        let newWidth = max(1, Int((Double(width) * scale).rounded()))
        let newHeight = max(1, Int((Double(height) * scale).rounded()))
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: newWidth, height: newHeight, bitsPerComponent: 8,
                bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return image }
        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        return context.makeImage() ?? image
    }

    /// Redraw into a fresh sRGB RGBA bitmap so the result owns its pixels (no lazy/mmap
    /// dependency on the original source). Falls back to the input if a context can't be made.
    private static func forceDecoded(_ image: CGImage) -> CGImage {
        let width = image.width, height = image.height
        guard width > 0, height > 0,
              let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return image }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage() ?? image
    }
}
