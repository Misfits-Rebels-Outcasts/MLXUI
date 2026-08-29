import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// CFM-R15-1(b) — the flow boundary: the `Segment` stage must emit a **binary** mask
/// (white = mask, black = background, nothing of the source photo survives — the
/// `tools/media.py::save_mask` L-mode contract `63-CutOutSubject`'s `Save Image` expects),
/// never the run view's red-tinted overlay. Pins the pixel-buffer writer's values directly
/// and the stage's Media path end to end.
struct SegmentAnythingBinaryMaskTests {

    // MARK: - The pixel-buffer writer

    @Test func binaryMaskWriterEmitsTwoValuedPixels() throws {
        // 2×2 mask: top row background, bottom row foreground (image row 0 = mask row 0).
        let binary: [Float] = [0, 0, 1, 1]
        let image = try #require(SegmentAnythingEngine.renderBinaryMask(
            binary: binary, maskWidth: 2, maskHeight: 2, width: 2, height: 2))
        #expect(image.width == 2 && image.height == 2)
        let pixels = try grayPixels(image)
        #expect(pixels == [0, 0, 255, 255],
                "expected white = mask, black = background, got \(pixels)")
    }

    @Test func binaryMaskWriterIsTwoValuedAfterUpsample() throws {
        // A 2×1 mask upsampled to 4×2 — every output pixel must still be 0 or 255, and the
        // nearest-neighbour mapping must land where the mask says.
        let binary: [Float] = [1, 0]
        let image = try #require(SegmentAnythingEngine.renderBinaryMask(
            binary: binary, maskWidth: 2, maskHeight: 1, width: 4, height: 2))
        let pixels = try grayPixels(image)
        #expect(pixels.allSatisfy { $0 == 0 || $0 == 255 }, "mask must stay two-valued, got \(pixels)")
        #expect(pixels[0..<2].allSatisfy { $0 == 255 })   // mask column 0 → white
        #expect(pixels[2..<4].allSatisfy { $0 == 0 })     // mask column 1 → black
        #expect(pixels[4..<6].allSatisfy { $0 == 255 })
        #expect(pixels[6..<8].allSatisfy { $0 == 0 })
    }

    // MARK: - The stage the executor runs

    /// The stage's Media path: an all-red source photo in, a two-valued mask out, and not one
    /// source pixel (red) survives — the regression the run view's overlay used to break.
    @Test func stageEmitsAMaskAndNoSourcePixelSurvives() async throws {
        let source = try makeSourceImage(width: 4, height: 4)
        // The exact composition the real `init(modelID:)` performs after decode: highest-IoU
        // mask → `renderBinaryMask` at the source's size. The stage contract is image→image;
        // the renderer is the pure writer pinned above.
        let stage = SegmentAnythingStage(id: "segmentation.test", name: "SAM3 (test)") { input in
            SegmentAnythingEngine.renderBinaryMask(
                binary: [0, 0, 1, 1], maskWidth: 2, maskHeight: 2,
                width: input.width, height: input.height) ?? input
        }
        let out = try await stage.run(.image(ImageMedia(cgImage: source)), progress: { _ in })
        guard case .image(let media) = out else {
            Issue.record("stage must produce an image")
            return
        }
        let pixels = try grayPixels(media.cgImage)
        #expect(pixels == [0, 0, 0, 0,
                           0, 0, 0, 0,
                           255, 255, 255, 255,
                           255, 255, 255, 255],
                "expected the 2×2 mask upsampled onto the 4×4 source, got \(pixels)")
        #expect(!pixels.contains(where: { $0 > 0 && $0 < 255 }),
                "mask must be two-valued, got \(pixels)")
    }

    // MARK: - Helpers

    private enum TestError: Error {
        case contextFailed
    }

    /// An all-red, fully-opaque image — the strongest proof no source pixel survives.
    private func makeSourceImage(width: Int, height: Int) throws -> CGImage {
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw TestError.contextFailed
        }
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = ctx.makeImage() else { throw TestError.contextFailed }
        return image
    }

    /// The image's pixels as 8-bit gray values, top-to-bottom (mask row 0 = image row 0).
    private func grayPixels(_ image: CGImage) throws -> [UInt8] {
        let w = image.width, h = image.height
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let data = ctx.data else { throw TestError.contextFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return Array(UnsafeBufferPointer(
            start: data.assumingMemoryBound(to: UInt8.self), count: w * h))
    }
}
