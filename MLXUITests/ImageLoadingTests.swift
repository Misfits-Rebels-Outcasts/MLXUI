import Testing
import CoreGraphics
@testable import MLXUI

/// Covers `ImageLoader.downscaled` — the VLM/OCR input pixel cap that prevents the
/// vision-attention OOM (journal/2026-39).
struct ImageLoadingTests {
    private func makeImage(width: Int, height: Int) -> CGImage {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: 0, space: space,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return ctx.makeImage()!
    }

    @Test func downscalesOversizedImageWithinBudget() {
        let cap = 1_048_576
        let out = ImageLoader.downscaled(makeImage(width: 3840, height: 2160), maxPixels: cap)
        #expect(out.width * out.height <= cap)
        // Aspect ratio preserved (16:9 ≈ 1.777).
        let ratio = Double(out.width) / Double(out.height)
        #expect(abs(ratio - 3840.0 / 2160.0) < 0.02)
    }

    @Test func leavesSmallImageUnchanged() {
        let original = makeImage(width: 512, height: 512)
        let out = ImageLoader.downscaled(original, maxPixels: 1_048_576)
        #expect(out.width == 512 && out.height == 512)
    }
}
