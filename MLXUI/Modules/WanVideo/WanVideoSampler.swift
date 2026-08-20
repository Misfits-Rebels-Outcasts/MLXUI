import Foundation
import MLX
import MLXNN

// MARK: - Euler sampler for WAN 2.1

/// Discrete sigma schedule (training noise levels) for WAN 2.1.
/// WAN uses a flow-matching objective similar to DDPM but with Euler ODE integration.
nonisolated enum WanSampler {

    // MARK: Schedule

    /// Build flow-matching sigmas matching diffusers FlowMatchEulerDiscreteScheduler.
    /// Linspace from t=1 down to t=1/num_train_timesteps (NOT to t=0), apply the WAN 2.1
    /// shift formula, then append σ=0.  This keeps the penultimate denoising step at
    /// timestep≈1 rather than timestep≈136, matching the model's training distribution.
    static func buildSigmas(numSteps: Int, flowShift: Float = 3.0, numTrainTimesteps: Int = 1000) -> [Float] {
        let tMin = 1.0 / Float(numTrainTimesteps)
        var sigmas = (0 ..< numSteps).map { i -> Float in
            // numSteps==1 → division by (numSteps-1)=0 would be NaN; single step starts at t=1.
            let t = numSteps == 1 ? 1.0 : 1.0 - Float(i) / Float(numSteps - 1) * (1.0 - tMin)
            return flowShift * t / (1 + (flowShift - 1) * t)
        }
        sigmas.append(0.0)
        return sigmas
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
    /// Flow-matching: timestep = sigma * num_train_timesteps (linear, no DDPM inversion).
    static func sigmaToTimestep(_ sigma: Float) -> Float {
        sigma * 1000.0
    }
}
