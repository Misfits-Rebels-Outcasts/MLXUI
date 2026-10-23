import Foundation
import Testing
import CoreGraphics
@testable import MLXUI

// WW2-3-8 — the SeedVR2 pre-flight (owner ruling 2026-10-23, journal 2026-560): an image the VAE decode can't fit is
// refused before anything loads, with a plain message. Calibrated on the measured branch ceiling (18 GiB Mac, 2×, padded
// output: 1.178 MP decoded, 1.229 MP ran out of memory), ~10 % under the passing size, scaled by the memory budget.

private let referenceCapacity = SeedVR2Preflight.referenceCapacityBytes

struct SeedVR2PreflightTests {

    @Test func theMeasuredPassingSizesAreAccepted() {
        // 512×512 → 1024×1024 is what the owner ran in the app; 576×448 and 512×384 passed in the probe.
        for (w, h) in [(512, 512), (576, 448), (512, 384), (256, 256), (64, 64)] {
            #expect(SeedVR2Preflight.check(width: w, height: h, scale: 2, capacityBytes: referenceCapacity) == nil, "\(w)×\(h)")
        }
    }

    @Test func theMeasuredFailingSizesAreRefused() {
        for (w, h) in [(640, 480), (768, 512), (1024, 768), (4000, 3000)] {
            #expect(SeedVR2Preflight.check(width: w, height: h, scale: 2, capacityBytes: referenceCapacity) != nil, "\(w)×\(h)")
        }
    }

    @Test func theLimitIsAboveEveryPassingAndBelowEveryFailingMeasurement() {
        let limit = SeedVR2Preflight.maxPaddedOutputPixels(capacityBytes: referenceCapacity)
        #expect(limit == SeedVR2Preflight.referenceMaxPaddedOutputPixels)
        #expect(limit >= SeedVR2Preflight.paddedOutputPixels(width: 512, height: 512, scale: 2))     // 1.049 MP, ran in the app
        #expect(limit < SeedVR2Preflight.paddedOutputPixels(width: 620, height: 465, scale: 2))      // 1.178 MP, decoded: the margin
        #expect(limit < SeedVR2Preflight.paddedOutputPixels(width: 640, height: 480, scale: 2))      // 1.229 MP, out of memory
    }

    @Test func paddingToMultiplesOfSixteenIsCounted() {
        // 620×465 at 2× is 1240×930, padded up to multiples of 16: 1248×944.
        #expect(SeedVR2Preflight.paddedOutputPixels(width: 620, height: 465, scale: 2) == 1248 * 944)
    }

    @Test func theSuggestedSizeFitsAndKeepsTheAspectRatio() throws {
        let tooLarge = try #require(SeedVR2Preflight.check(width: 4000, height: 3000, scale: 2, capacityBytes: referenceCapacity))
        #expect(tooLarge.scale == 2)
        #expect(SeedVR2Preflight.check(width: tooLarge.maxWidth, height: tooLarge.maxHeight, scale: 2,
                                       capacityBytes: referenceCapacity) == nil, "the suggestion itself must pass")
        let aspect = Double(tooLarge.maxWidth) / Double(tooLarge.maxHeight)
        #expect(abs(aspect - 4.0 / 3.0) < 0.05)
        // and it is close to the limit, not far below it (within ~15 % of the maximum area)
        let used = SeedVR2Preflight.paddedOutputPixels(width: tooLarge.maxWidth, height: tooLarge.maxHeight, scale: 2)
        #expect(Double(used) > 0.85 * Double(SeedVR2Preflight.referenceMaxPaddedOutputPixels))
    }

    @Test func theLimitScalesWithTheMemoryBudget() {
        let small = SeedVR2Preflight.maxPaddedOutputPixels(capacityBytes: referenceCapacity / 2)
        let big = SeedVR2Preflight.maxPaddedOutputPixels(capacityBytes: referenceCapacity * 2)
        #expect(abs(small - SeedVR2Preflight.referenceMaxPaddedOutputPixels / 2) <= 1)
        #expect(abs(big - SeedVR2Preflight.referenceMaxPaddedOutputPixels * 2) <= 1)
        // a 36 GiB-class Mac takes what the 18 GiB one refuses
        #expect(SeedVR2Preflight.check(width: 640, height: 480, scale: 2, capacityBytes: referenceCapacity * 2) == nil)
        #expect(SeedVR2Preflight.check(width: 512, height: 512, scale: 2, capacityBytes: referenceCapacity / 2) != nil)
    }

    @Test func theMessageIsPlainAndNamesTheSize() throws {
        let tooLarge = try #require(SeedVR2Preflight.check(width: 4000, height: 3000, scale: 2, capacityBytes: referenceCapacity))
        let message = SeedVR2Preflight.message(tooLarge)
        #expect(message == "This image is too large to upscale 2× on this Mac — try one up to \(tooLarge.maxWidth)×\(tooLarge.maxHeight).")
        #expect(SeedVR2EngineError.imageTooLarge(tooLarge).errorDescription == message)
    }

    @Test func aDegenerateImageOrScaleIsNotRefused() {
        #expect(SeedVR2Preflight.check(width: 0, height: 100, scale: 2, capacityBytes: referenceCapacity) == nil)
        #expect(SeedVR2Preflight.check(width: 100, height: 100, scale: 0, capacityBytes: referenceCapacity) == nil)
    }

    /// The engine refuses before it looks for weights: a large image with a model that isn't installed still gets the size
    /// message (nothing is loaded, nothing needs to be installed to see it).
    @Test func theEngineRefusesBeforeLoadingAnything() async throws {
        let ctx = try #require(CGContext(data: nil, width: 4000, height: 3000, bitsPerComponent: 8, bytesPerRow: 0,
                                         space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        let image = try #require(ctx.makeImage())
        do {
            _ = try await SeedVR2Engine.upscale(image: image, scale: 2, seed: 1, modelID: "no-such--model") { _ in }
            Issue.record("a 4000×3000 image must be refused")
        } catch let error as SeedVR2EngineError {
            guard case .imageTooLarge(let tooLarge) = error else { Issue.record("wrong error \(error)"); return }
            #expect(tooLarge.scale == 2)
            #expect(tooLarge.maxWidth * tooLarge.maxHeight < 4000 * 3000)
        }
    }
}
