import Foundation
import MLX

// FLUX.2 Klein latent layout, position ids and size rules (MP7-3a).
// Layouts and index math ported from VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin)
// `Pipeline/LatentUtils.swift`, and checked against diffusers 0.37.1 `Flux2KleinPipeline` static methods
// (MLXUITests/Fixtures/Flux2Klein/g3e_ids.safetensors, g3g_glue.safetensors).
//
// The picture. The VAE works on [B, 32, H/8, W/8]. The pipeline patchifies 2 x 2 to [B, 128, H/16, W/16] (channel = c*4 + p1*2 + p2),
// denoises that as a sequence [B, (H/16)*(W/16), 128] (row-major over the grid), and undoes it in reverse: unpack, BatchNorm
// denormalise on the 128 channels, unpatchify to 32 channels, VAE decode. Everything here is NCHW like diffusers, because the
// BatchNorm statistics and the 128-channel order are defined there.

nonisolated enum Flux2KleinSizeError: Error, Equatable, LocalizedError {
    case notPositive(width: Int, height: Int)
    case notMultipleOf16(width: Int, height: Int)

    var errorDescription: String? {
        switch self {
        case let .notPositive(w, h):
            return "FLUX.2 Klein needs a positive width and height; got \(w) x \(h)."
        case let .notMultipleOf16(w, h):
            // SPEC-Q242 wording: "Width 1000 isn't a multiple of 16. The nearest sizes are 992 and 1008."
            let (name, v) = w % 16 != 0 ? ("Width", w) : ("Height", h)
            let down = max(16, (v / 16) * 16), up = max(16, v % 16 == 0 ? v : (v / 16 + 1) * 16)
            let nearest = down == up ? "The nearest size is \(down)." : "The nearest sizes are \(down) and \(up)."
            return "\(name) \(v) isn't a multiple of 16. \(nearest)"
        }
    }
}

nonisolated enum Flux2KleinLatents {
    /// Pixels per packed-latent cell: VAE 8x downsample x 2 patch.
    static let sizeMultiple = Flux2KleinDims.vaeScale * Flux2KleinDims.vaePatch
    static let patchedChannels = Flux2KleinDims.transformerInChannels

    // MARK: Size

    /// Refuse, not round (owner ruling 2026-10-31, SPEC-Q242: "Refuse is the right reading of the size ruling"). A size that is not a
    /// multiple of 16 is an error naming the first offending setting, width before height; it is never silently changed. For the
    /// record, diffusers and mflux (catflow-mlx's engine) floor with a warning and the Swift reference rounds up; both runtimes
    /// here refuse instead.
    @discardableResult
    static func validateSize(width: Int, height: Int) throws -> (gridWidth: Int, gridHeight: Int) {
        guard width > 0, height > 0 else { throw Flux2KleinSizeError.notPositive(width: width, height: height) }
        // SPEC-Q242
        guard width % sizeMultiple == 0, height % sizeMultiple == 0 else {
            throw Flux2KleinSizeError.notMultipleOf16(width: width, height: height)
        }
        return (width / sizeMultiple, height / sizeMultiple)
    }

    /// Packed token count for a size already known to be valid.
    static func imageSequenceLength(width: Int, height: Int) throws -> Int {
        let g = try validateSize(width: width, height: height)
        return g.gridWidth * g.gridHeight
    }

    // MARK: Noise

    /// `[1, 128, H/16, W/16]` standard-normal noise, the patchified layout the pipeline draws (`prepare_latents`).
    /// The values are MLX's generator for the seed, not torch's; a Swift run is reproducible with itself, not with diffusers.
    static func makeNoise(width: Int, height: Int, seed: UInt64, dtype: DType = .float32) throws -> MLXArray {
        let g = try validateSize(width: width, height: height)
        return MLXRandom.normal([1, patchedChannels, g.gridHeight, g.gridWidth], dtype: dtype, key: MLXRandom.key(seed))
    }

    // MARK: Patchify / pack

    /// `[B, C, H, W]` -> `[B, 4C, H/2, W/2]`; channel index = c*4 + p1*2 + p2 (diffusers `_patchify_latents`).
    static func patchify(_ x: MLXArray) -> MLXArray {
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped([b, c, h / 2, 2, w / 2, 2]).transposed(0, 1, 3, 5, 2, 4).reshaped([b, c * 4, h / 2, w / 2])
    }

    /// `[B, 4C, H, W]` -> `[B, C, 2H, 2W]` (diffusers `_unpatchify_latents`).
    static func unpatchify(_ x: MLXArray) -> MLXArray {
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped([b, c / 4, 2, 2, h, w]).transposed(0, 1, 4, 2, 5, 3).reshaped([b, c / 4, h * 2, w * 2])
    }

    /// `[B, C, H, W]` -> `[B, H*W, C]`, row-major over the grid (diffusers `_pack_latents`).
    static func packToSequence(_ x: MLXArray) -> MLXArray {
        let (b, c, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped([b, c, h * w]).transposed(0, 2, 1)
    }

    /// `[B, H*W, C]` -> `[B, C, H, W]`. Equivalent to diffusers' scatter by position id for the grid ids `imageIDs` produces.
    static func unpackFromSequence(_ x: MLXArray, gridHeight: Int, gridWidth: Int) -> MLXArray {
        precondition(x.dim(1) == gridHeight * gridWidth, "sequence length \(x.dim(1)) is not \(gridHeight) x \(gridWidth)")
        return x.reshaped([x.dim(0), gridHeight, gridWidth, x.dim(2)]).transposed(0, 3, 1, 2)
    }

    // MARK: Position ids ([T, H, W, L], int32)

    /// Image tokens: T = 0, H and W the grid coordinates (row-major), L = 0. Shape `[gridHeight*gridWidth, 4]`.
    static func imageIDs(gridHeight: Int, gridWidth: Int) -> MLXArray {
        let n = gridHeight * gridWidth
        let h = MLXArray.arange(gridHeight, dtype: .int32).expandedDimensions(axis: 1)
        let w = MLXArray.arange(gridWidth, dtype: .int32).expandedDimensions(axis: 0)
        let hh = broadcast(h, to: [gridHeight, gridWidth]).reshaped([n])
        let ww = broadcast(w, to: [gridHeight, gridWidth]).reshaped([n])
        let zeros = MLXArray.zeros([n], dtype: .int32)
        return stacked([zeros, hh, ww, zeros], axis: 1)
    }

    /// Text tokens: T = H = W = 0, L = 0 ..< count. Shape `[count, 4]`.
    static func textIDs(count: Int) -> MLXArray {
        let zeros = MLXArray.zeros([count], dtype: .int32)
        return stacked([zeros, zeros, zeros, MLXArray.arange(count, dtype: .int32)], axis: 1)
    }

    // MARK: VAE BatchNorm

    /// `x * sqrt(var + eps) + mean` on the 128-channel patchified latents, before unpatchify (pipeline step before `vae.decode`).
    /// `mean` and `variance` are the VAE's `bn.running_mean` / `bn.running_var`, shape `[128]`.
    static func denormalize(_ x: MLXArray, mean: MLXArray, variance: MLXArray, eps: Double = Flux2KleinDims.vaeBatchNormEps) -> MLXArray {
        let c = mean.dim(0)
        let std = sqrt(variance.reshaped([1, c, 1, 1]) + Float(eps))
        return x * std + mean.reshaped([1, c, 1, 1])
    }
}
