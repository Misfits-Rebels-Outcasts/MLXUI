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
    ///
    /// Exactly the arithmetic of diffusers 0.37.1 (MP7-3h, owner done-when "sigmas exactly as diffusers' scheduler passes them"):
    /// `np.linspace(1.0, 1/steps, steps)` in float64 (numpy's formula: `i * step + start`, last element = stop), cast to float32
    /// (`set_timesteps`: `np.array(sigmas).astype(np.float32)`), then `_time_shift_exponential` evaluated on that float32 array with a
    /// Python float `exp(mu)`. NumPy keeps a Python scalar weak next to a float32 array (NEP 50, numpy 2.2.6 as pinned; value-based casting
    /// gave the same in 1.x), so `exp(mu) / (exp(mu) + (1/t - 1) ** 1.0)` is float32 division and addition throughout: the code below does
    /// the same in `Float`. Compared bit-for-bit with the golden sigmas in the tests (they used to be compared to 2e-7).
    static func sigmas(steps: Int, imageSeqLen: Int) -> [Float] {
        precondition(steps >= 1, "Klein needs at least one step")
        let mu = empiricalMu(imageSeqLen: imageSeqLen, steps: steps)
        let expMu = Float(Foundation.exp(mu))
        let start = 1.0, stop = 1.0 / Double(steps)
        let step = steps > 1 ? (stop - start) / Double(steps - 1) : 0.0
        var out: [Float] = []
        out.reserveCapacity(steps + 1)
        for i in 0..<steps {
            let linear = (steps > 1 && i == steps - 1) ? stop : Double(i) * step + start      // numpy linspace pins the endpoint
            let t = Float(linear)
            let shifted = expMu / (expMu + (Float(1) / t - Float(1)))
            out.append(shifted)
        }
        out.append(0)
        return out
    }

    /// The `sigma` value to hand `Flux2KleinTransformer` so that its own `sigma x 1000` reproduces what diffusers' transformer sees.
    ///
    /// diffusers 0.37.1 `Flux2KleinPipeline.__call__`: `timestep = t.expand(...).to(latents.dtype)` (t = float32 sigma x 1000, so the cast
    /// rounds it to bfloat16 in a bf16 pipeline), `transformer(timestep=timestep / 1000)`, and inside `Flux2Transformer2DModel.forward`
    /// `timestep.to(hidden_states.dtype) * 1000` (bf16 again). The sinusoid therefore sees `bf16(bf16(bf16(sigma x 1000) / 1000) x 1000)`;
    /// in a float32 pipeline the same chain without the roundings. Each bf16 operation here is computed in float32 and rounded once, as
    /// torch does. Our transformer multiplies by 1000 in float32 after receiving this value, so the product can differ from the diffusers
    /// value by one float32 ulp (a relative 6e-8): below anything the sinusoid can resolve.
    static func transformerSigma(_ sigma: Float, dtype: DType) -> Float {
        let t = sigma * 1000
        guard dtype == .bfloat16 else { return t / 1000 }
        let seen = roundToBF16(roundToBF16(roundToBF16(t) / 1000) * 1000)
        return seen / 1000
    }

    /// Round-to-nearest-even to bfloat16 (the upper 16 bits of the float32), returned as a Float. NaN and infinity pass through.
    static func roundToBF16(_ x: Float) -> Float {
        guard x.isFinite else { return x }
        let bits = x.bitPattern
        let rounded = (bits &+ 0x7FFF &+ ((bits >> 16) & 1)) & 0xFFFF_0000
        return Float(bitPattern: rounded)
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

    /// The step as diffusers' `FlowMatchEulerDiscreteScheduler.step` does it in a reduced-precision pipeline (MP7-3h): the sample is upcast to
    /// float32, `dt = sigma_next - sigma` is a float32 scalar, `dt * model_output` takes the prediction's dtype (a zero-dim float32 tensor does not
    /// promote a dimensioned bf16 one), the sum is float32, and the result goes back to the prediction's dtype. For float32 arrays this equals `step`.
    /// (Implementer's reading of torch's type promotion, not checked at operator level; the tiny fp32 golden checks the float32 case.)
    static func stepUpcasting(sample: MLXArray, velocity: MLXArray, sigma: Float, sigmaNext: Float) -> MLXArray {
        let dt = sigmaNext - sigma
        let scaled = (dt * velocity).asType(velocity.dtype)
        return (sample.asType(.float32) + scaled.asType(.float32)).asType(velocity.dtype)
    }
}
