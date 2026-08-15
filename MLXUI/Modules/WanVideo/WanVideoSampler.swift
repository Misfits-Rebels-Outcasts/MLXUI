import Foundation
import MLX
import MLXNN

// MARK: - Euler sampler for WAN 2.1

/// Discrete sigma schedule (training noise levels) for WAN 2.1.
/// WAN uses a flow-matching objective similar to DDPM but with Euler ODE integration.
nonisolated enum WanSampler {

    // MARK: Schedule

    /// Build a linearly-spaced subset of the 1000-step training schedule.
    /// Returns sigmas in descending order (σ_max ... σ_min) plus a terminal 0.
    static func buildSigmas(numSteps: Int) -> [Float] {
        // WAN training schedule: σ_t = sqrt(t / (T - t)) for t in 0..T-1 (flow-matching variant)
        // Use simplified linear-beta schedule from the reference implementation.
        let T: Float = 1000
        let minSigma: Float = 0.001
        let maxSigma: Float = 1.0
        let sigmas = (0 ..< numSteps + 1).map { i -> Float in
            let frac = Float(numSteps - i) / Float(numSteps)
            return minSigma + (maxSigma - minSigma) * frac
        }
        return sigmas   // [σ_max, ..., σ_min, 0]
    }

    // MARK: Euler step

    /// Single Euler ODE step: x_{t-1} = x_t + (σ_{t-1} - σ_t) · noise_pred.
    static func eulerStep(
        x:        MLXArray,
        noisePred: MLXArray,
        sigmaFrom: Float,
        sigmaTo:   Float
    ) -> MLXArray {
        x + noisePred * (sigmaTo - sigmaFrom)
    }

    // MARK: Timestep schedule

    /// Convert sigma values to discrete timesteps in [0, 1000] for the model.
    static func sigmaToTimestep(_ sigma: Float) -> Float {
        // t = σ / (1 + σ) * 1000  (inverse of the WAN noise schedule)
        (sigma / (1.0 + sigma)) * 1000.0
    }
}
