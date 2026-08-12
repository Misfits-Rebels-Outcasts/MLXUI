import Foundation
import MLX

/// DDIM scheduler for SDXL-Turbo, ported from `ml-explore/mlx-examples/stable_diffusion`'s
/// `scheduler.py` (DDIMScheduler-style) per the SD-ENG4 spec. SDXL-Turbo runs 4 denoise steps
/// with **no CFG**, so the scheduler is a plain epsilon-DDIM integrator over the latent.
///
/// `alphas_cumprod` is **computed** from the repo's beta schedule (`scaled_linear`) — the
/// `stabilityai/sdxl-turbo` `scheduler/scheduler_config.json` stores `beta_start`/`beta_end`/
/// `beta_schedule`, not the cumprod array itself (differs from the backlog's assumption).
nonisolated struct SDScheduler {
    let alphasCumprod: [Float]

    /// `scaled_linear` schedule: `betas = linspace(sqrt(beta_start), sqrt(beta_end), N)²`,
    /// `alphas_cumprod = cumprod(1 - betas)` (diffusers `get_beta_schedule`).
    static func scaledLinearAlphasCumprod(
        betaStart: Float = 0.00085, betaEnd: Float = 0.012, numTrainSteps: Int = 1000
    ) -> [Float] {
        var betas = [Float](repeating: 0, count: numTrainSteps)
        for i in 0 ..< numTrainSteps {
            let frac = Float(i) / Float(numTrainSteps - 1)
            let b = (sqrt(betaStart) + (sqrt(betaEnd) - sqrt(betaStart)) * frac)
            betas[i] = b * b
        }
        var cumprod: [Float] = []
        cumprod.reserveCapacity(numTrainSteps)
        var running: Float = 1.0
        for b in betas {
            running *= 1 - b
            cumprod.append(running)
        }
        return cumprod
    }

    init(betaStart: Float = 0.00085, betaEnd: Float = 0.012, numTrainSteps: Int = 1000) {
        self.alphasCumprod = Self.scaledLinearAlphasCumprod(betaStart: betaStart, betaEnd: betaEnd, numTrainSteps: numTrainSteps)
    }

    /// Direct-array init for the unit test (a synthetic `alphas_cumprod`).
    init(alphasCumprod: [Float]) {
        self.alphasCumprod = alphasCumprod
    }

    /// `alpha_t` at a (possibly fractional) timestep: linear interpolation over the cumprod array,
    /// clamped to `[0, count-1]` (the reference `_interp`).
    func alpha(at t: Float) -> Float {
        let maxIndex = Float(alphasCumprod.count - 1)
        let idx = min(max(t, 0), maxIndex)
        let low = Int(floor(idx))
        let high = min(low + 1, alphasCumprod.count - 1)
        let frac = idx - Float(low)
        return alphasCumprod[low] * (1 - frac) + frac * alphasCumprod[high]
    }

    /// Evenly spaced denoise timesteps across `[numTrainSteps-1, 0]`, length `numSteps + 1`.
    /// SDXL-Turbo with 4 steps ≈ `[999, 749.25, 499.5, 249.75, 0]`. This matches the repo's
    /// declared `timestep_spacing: trailing` (diffusers: `round(arange(numTrain, 0, -step)) - 1`
    /// → `[999, 749, 499, 249]`, `steps_offset` is a `"leading"`-only adjustment and does NOT
    /// apply here) — the earlier concern that this used "leading" spacing (which would give
    /// `[751, 501, 251, 1]`) was a mislabeling in an earlier draft of this comment, not a real
    /// mismatch; the final `0` stands in for diffusers' `final_alpha_cumprod` (≈`alphasCumprod[0]`,
    /// used when `prevTimestep < 0` on the last DDIM step).
    func timesteps(numSteps: Int) -> [Float] {
        (0 ... numSteps).map { i in
            Float(alphasCumprod.count - 1) * (1 - Float(i) / Float(numSteps))
        }
    }

    /// One DDIM update (`prediction_type: epsilon`):
    /// `pred_x0 = (sample - sqrt(1-α_t)·noise) / sqrt(α_t)`;
    /// `x_prev = sqrt(α_{t-1})·pred_x0 + sqrt(1-α_{t-1})·noise`.
    func step(noisePred: MLXArray, sample: MLXArray, t: Float, tPrev: Float) -> MLXArray {
        let aT = alpha(at: t)
        let aPrev = alpha(at: tPrev)
        let sqrtA = sqrt(aT)
        let predX0 = (sample - sqrt(1 - aT) * noisePred) / sqrtA
        return sqrt(aPrev) * predX0 + sqrt(1 - aPrev) * noisePred
    }
}
