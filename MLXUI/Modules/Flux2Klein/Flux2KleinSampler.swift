import Foundation
import MLX

// FLUX.2 Klein sampling schedule (MP7-3a): the empirical time-shift `mu`, the shifted sigma schedule, and the Euler step.
// Ported from VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `Scheduler/FlowMatchEulerScheduler.swift`,
// itself a port of diffusers' `compute_empirical_mu` (pipeline_flux2_klein.py) and `FlowMatchEulerDiscreteScheduler`; this
// version is checked against diffusers 0.37.1 values (MLXUITests/Fixtures/Flux2Klein/g3f_scalars.json), not against the reference.
//
// Differences from the reference, each deliberate: scalar math is Double (diffusers computes `mu` as a Python float and the
// shift in NumPy before storing float32; the reference used Float throughout), and the sigma list carries the terminal 0 only
// when asked for, so a caller can't run off the end.
//
// Klein is the distilled 4-step model (no guidance, no CFG): the schedule starts at linspace(1, 1/steps, steps).

nonisolated enum Flux2KleinSampler {
    /// diffusers `compute_empirical_mu(image_seq_len, num_steps)`; `imageSeqLen` is the number of packed latent tokens
    /// ((width / 16) x (height / 16)).
    static func empiricalMu(imageSeqLen: Int, steps: Int) -> Double {
        let a1 = 8.73809524e-05, b1 = 1.89833333
        let a2 = 0.00016927, b2 = 0.45666666
        let len = Double(imageSeqLen)
        if imageSeqLen > 4300 { return a2 * len + b2 }
        let m200 = a2 * len + b2
        let m10 = a1 * len + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Double(steps) + b
    }

    /// The `steps + 1` sigmas the loop walks: the exponentially time-shifted `linspace(1, 1/steps, steps)`, then the terminal 0.
    /// diffusers casts the input sigmas to float32 before shifting and stores the result as float32; the same happens here so the
    /// values agree to float32 rounding.
    static func sigmas(steps: Int, imageSeqLen: Int) -> [Float] {
        precondition(steps >= 1, "Klein needs at least one step")
        let mu = empiricalMu(imageSeqLen: imageSeqLen, steps: steps)
        let expMu = Foundation.exp(mu)
        var out: [Float] = []
        out.reserveCapacity(steps + 1)
        for i in 0..<steps {
            let linear = steps == 1 ? 1.0 : 1.0 + (1.0 / Double(steps) - 1.0) * Double(i) / Double(steps - 1)
            let t = Double(Float(linear))
            out.append(Float(expMu / (expMu + (1.0 / t - 1.0))))
        }
        out.append(0)
        return out
    }

    /// Scheduler timesteps: `sigmas x num_train_timesteps` (the transformer is fed `timestep / 1000`, i.e. the sigma itself).
    static func timesteps(steps: Int, imageSeqLen: Int, numTrainTimesteps: Int = 1000) -> [Float] {
        sigmas(steps: steps, imageSeqLen: imageSeqLen).dropLast().map { $0 * Float(numTrainTimesteps) }
    }

    /// One Euler step: `sample + (sigmaNext - sigma) x velocity`. Plain rectified-flow Euler, the same formula as
    /// `FluxSampler.step` (checked equal by test), kept separate so the two ports don't share a type before MP7-3h.
    static func step(sample: MLXArray, velocity: MLXArray, sigma: Float, sigmaNext: Float) -> MLXArray {
        sample + (sigmaNext - sigma) * velocity
    }
}
