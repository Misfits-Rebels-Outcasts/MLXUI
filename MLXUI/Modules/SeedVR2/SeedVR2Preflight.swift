import Foundation

/// WW2-3-8 (owner ruling 2026-10-23, journal 2026-560): refuse an image that is too large to upscale **before loading
/// anything**, with a plain message, instead of letting the VAE decode run out of GPU memory (B18).
///
/// **Why the limit exists.** The VAE is made of 3D convolutions, and MLX 0.32's `conv3d`/`convGeneral` need far more
/// temporary memory than 0.31's (T=3, 256×384, C=128: ~434 MB vs ~24 MB extra; measured, journal 2026-560), so the decode
/// peaks at ~16 kB per padded output pixel — ~1.9× `main`'s. It is not a setting: `Memory.cacheLimit`, `memoryLimit` and
/// the per-stage cache clears were all tried and leave the peak unchanged.
///
/// **Calibration** (18 GiB Mac, real weights, 2×, through the real `SeedVR2Engine.upscale`): the largest padded output that
/// decoded was 1.178 MP (1240×930 → 1248×944); 1.229 MP (1280×960) ran out of memory. The limit below keeps ~10 % under the
/// passing size, and scales with this Mac's `MemoryBudget` capacity (60 % of RAM: 10.8 GiB on the reference Mac).
///
/// **Remove / raise when** the decode is tiled (backlog: "SeedVR2 tiled VAE decode") or MLX's 3D-conv memory is fixed.
nonisolated enum SeedVR2Preflight {
    /// `MemoryBudget` capacity of the Mac the limit was measured on (18 GiB × 0.6).
    static let referenceCapacityBytes: Int64 = Int64(18.0 * 0.6 * 1_073_741_824)
    /// Largest padded (multiple-of-16) output, in pixels, that is accepted on the reference Mac.
    static let referenceMaxPaddedOutputPixels = 1_060_000

    /// **Tests only** (precedent: `PromptCacheAllowlist.overrideForTesting`): lets a test run an over-size image through the
    /// real decode, to prove a genuine out-of-memory becomes a thrown row error and not a read of the failed array.
    nonisolated(unsafe) static var disabledForTesting = false

    struct TooLarge: Equatable {
        /// The largest input size, at this image's aspect ratio, that is accepted.
        let maxWidth: Int
        let maxHeight: Int
        let scale: Int
    }

    static func maxPaddedOutputPixels(capacityBytes: Int64) -> Int {
        let ratio = Double(max(capacityBytes, 0)) / Double(referenceCapacityBytes)
        return Int(Double(referenceMaxPaddedOutputPixels) * ratio)
    }

    /// The decoder pads each side up to a multiple of 16 (`SeedVR2Engine.upscale`).
    static func paddedOutputPixels(width: Int, height: Int, scale: Int) -> Int {
        func up(_ n: Int) -> Int { (n + 15) / 16 * 16 }
        return up(width * scale) * up(height * scale)
    }

    /// `nil` when the image fits; otherwise the largest same-aspect input that does.
    static func check(width: Int, height: Int, scale: Int,
                      capacityBytes: Int64 = MemoryBudget.shared.capacityBytes) -> TooLarge? {
        if disabledForTesting { return nil }
        let limit = maxPaddedOutputPixels(capacityBytes: capacityBytes)
        guard width > 0, height > 0, scale > 0,
              paddedOutputPixels(width: width, height: height, scale: scale) > limit else { return nil }
        // Largest factor f in (0, 1) whose scaled-down input fits; round the result down to a tidy multiple of 8.
        var lo = 0.0, hi = 1.0
        for _ in 0 ..< 40 {
            let mid = (lo + hi) / 2
            let w = max(1, Int(Double(width) * mid)), h = max(1, Int(Double(height) * mid))
            if paddedOutputPixels(width: w, height: h, scale: scale) <= limit { lo = mid } else { hi = mid }
        }
        var w = max(1, Int(Double(width) * lo)), h = max(1, Int(Double(height) * lo))
        if w >= 64 && h >= 64 { w = w / 8 * 8; h = h / 8 * 8 }
        return TooLarge(maxWidth: w, maxHeight: h, scale: scale)
    }

    static func message(_ tooLarge: TooLarge) -> String {
        "This image is too large to upscale \(tooLarge.scale)× on this Mac — try one up to \(tooLarge.maxWidth)×\(tooLarge.maxHeight)."
    }
}
