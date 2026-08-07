import Foundation
import MLX

/// The flow-matching sampler for FLUX, ported from `ml-explore/mlx-examples/flux`
/// (`flux/sampler.py`) — `RSI/plan-flux.md` slice AM4b. Pure math, no model weights, so it's
/// unit-testable against the Python reference.
///
/// FLUX uses **rectified flow**: the denoise step is `x_t + (t_prev - t) * pred`, where `pred`
/// is the velocity the MMDiT predicts. `Flux-1.lite-8B` is a **dev-style** checkpoint
/// (`guidance_embeds: true`), so it uses the `_time_shift` timestep schedule and a guidance
/// strength (see the plan — it is NOT schnell's 4-step no-guidance schedule).
nonisolated enum FluxSampler {
    /// Shared schedule constants from the Python reference.
    static let baseShift = 0.5
    static let maxShift = 1.15

    /// Whether this checkpoint is the schnell variant (no `_time_shift`, no guidance).
    static func isSchnell(_ name: String) -> Bool {
        name.contains("schnell")
    }

    /// The FLUX.1 timestep shift: `exp((x-x1)·(t2-t1)/(x2-x1)+t1) / (exp(...) + (1/t - 1))`.
    /// Maps the linear schedule through the shift curve parameterized by the image sequence
    /// length. Vectorized over `t` exactly like the reference.
    static func timeShift(_ x: Double, _ t: [Double]) -> [Double] {
        let x1 = 256.0, x2 = 4096.0, t1 = baseShift, t2 = maxShift
        let expMu = x1 != x2
            ? (t2 - t1) / (x2 - x1) * (x - x1) + t1
            : t1
        return t.map { tt in
            let mu = Foundation.exp(expMu)
            let denom = mu + (1 / tt - 1)
            return mu / denom
        }
    }

    /// The denoise timesteps: `linspace(start, stop, num_steps+1)`, then (for non-schnell) the
    /// time shift applied elementwise. Returns the scalar list the denoise loop steps through.
    /// Mirrors the reference: linspace in float32 (MLX default), then the shift math in Double
    /// (as `tolist()` gives Python floats).
    static func timesteps(
        numSteps: Int,
        imageSequenceLength: Int,
        start: Double = 1,
        stop: Double = 0,
        schnell: Bool = false
    ) -> [Double] {
        let t = MLXArray.linspace(Float(start), Float(stop), count: numSteps + 1)
        let values = t.asArray(Float.self).map(Double.init)
        if schnell { return values }
        return timeShift(Double(imageSequenceLength), values)
    }

    /// Initial latent noise: standard normal of the given shape (b, h, w, 16).
    static func samplePrior(shape: [Int], dtype: DType = .float16, seed: UInt64? = nil) -> MLXArray {
        if let seed { MLXRandom.seed(seed) }
        return MLXRandom.normal(shape, dtype: dtype)
    }

    /// One rectified-flow step: `x_t + (t_prev - t) * pred`.
    static func step(pred: MLXArray, xT: MLXArray, t: Double, tPrev: Double) -> MLXArray {
        xT + (tPrev - t) * pred
    }

    /// Pack a `(b, h, w, 16)` latent into 2×2 patches: `(b, h·w/4, 64)` with the spatial axes
    /// interleaved exactly as the reference `_prepare_latent_images` does.
    static func packLatents(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let b = x.dim(0)
        // x.reshape(b, h//2, 2, w//2, 2, c).transpose(0,1,3,5,2,4).reshape(b, h*w//4, c*4)
        let c = 16
        let reshaped = x.reshaped([b, h / 2, 2, w / 2, 2, c])
        let transposed = reshaped.transposed(0, 1, 3, 5, 2, 4)
        return transposed.reshaped([b, h * w / 4, c * 4])
    }

    /// Position ids for the packed latents: `[0, j, k]` where `j`/`k` are the 2×2-patch grid
    /// coords (`_prepare_latent_images`). Shape `(b, h·w/4, 3)`, int32.
    static func packPositionIDs(h: Int, w: Int, batch: Int) -> MLXArray {
        // meshgrid indexing="ij": j → (h/2, 1) broadcast to (h/2, w/2); k → (1, w/2) broadcast.
        let j = MLXArray.arange(h / 2).reshaped([h / 2, 1])
        let k = MLXArray.arange(w / 2).reshaped([1, w / 2])
        let jGrid = MLX.broadcast(j, to: [h / 2, w / 2])
        let kGrid = MLX.broadcast(k, to: [h / 2, w / 2])
        let zero = MLXArray.zeros([h / 2, w / 2], dtype: .int32)
        let ids = MLX.stacked([zero, jGrid, kGrid], axis: -1)  // (h/2, w/2, 3)
        let flat = ids.reshaped([1, h * w / 4, 3])
        // Reference does `mx.repeat(x_ids.reshape(1, h*w//4, 3), b, 0)`: axis-0 is size 1, so
        // the repeat TILES the whole block `b` times → (b, h*w/4, 3).
        return MLXArray.repeated(flat, count: batch, axis: 0)
    }

    /// Unpack a decoded latent back to `(b, h, w, c)` (inverse of `packLatents`), as the
    /// reference `decode` does before the VAE.
    static func unpackLatents(_ x: MLXArray, h: Int, w: Int) -> MLXArray {
        let b = x.dim(0)
        // x.reshape(b, h//2, w//2, -1, 2, 2).transpose(0,1,4,2,5,3).reshape(b, h, w, -1)
        let reshaped = x.reshaped([b, h / 2, w / 2, -1, 2, 2])
        let transposed = reshaped.transposed(0, 1, 4, 2, 5, 3)
        return transposed.reshaped([b, h, w, -1])
    }
}
