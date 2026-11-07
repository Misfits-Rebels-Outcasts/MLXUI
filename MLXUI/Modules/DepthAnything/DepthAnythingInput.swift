import Foundation
import CoreGraphics

/// Input limits (owner decision 5). The reference processor keeps the aspect ratio, so a long panorama would
/// become an enormous attention sequence at the 518-pixel short side: refuse it with one plain sentence
/// **before any work** rather than let it run for minutes or exhaust memory.
nonisolated enum DepthAnythingInput {
    static let maximumLongSide = 8_192
    static let maximumAspectRatio = 3.0

    /// `nil` when the image is acceptable; otherwise the one sentence to show.
    static func refusal(width: Int, height: Int) -> String? {
        let longSide = max(width, height)
        if longSide > maximumLongSide {
            return "This photo is \(longSide) px on its longest side; Depth Anything takes pictures up to 8,192 px. Shrink it and try again."
        }
        let shortSide = min(width, height)
        if shortSide > 0 {
            let ratio = Double(longSide) / Double(shortSide)
            if ratio > maximumAspectRatio {
                return String(format: "This photo is %.1f : 1; Depth Anything takes pictures up to 3 : 1. Crop it and try again.", ratio)
            }
        }
        return nil
    }

    /// A decoded, oriented, sRGB image that has passed the limits.
    struct Prepared: @unchecked Sendable {
        let image: CGImage
        let width: Int
        let height: Int
    }

    static func prepare(data: Data) throws -> Prepared {
        guard let image = DepthAnythingImage.decoded(data) else {
            throw DepthAnythingError.badImage("this photo could not be opened")
        }
        return try prepare(image: image)
    }

    static func prepare(url: URL) throws -> Prepared {
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw DepthAnythingError.badImage("this photo could not be read: \(error.localizedDescription)")
        }
        return try prepare(data: data)
    }

    static func prepare(image: CGImage) throws -> Prepared {
        if let refusal = refusal(width: image.width, height: image.height) {
            throw DepthAnythingError.badInput(refusal)
        }
        return Prepared(image: image, width: image.width, height: image.height)
    }
}
