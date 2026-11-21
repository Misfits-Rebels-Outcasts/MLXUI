import Foundation
import MLX

// MC5-3g: the ACE-Step turbo sampler on the no-LM, no-CFG path: Euler ODE over a linspace schedule (shift 1.0 = identity), with the optional
// DCW (differential correction in the wavelet domain) after every step. Truth, which agree: upstream's MLX loop `mlx_generate_diffusion`
// (`acestep/models/mlx/dit_generate.py`, with `dcw_correction_mlx.py`) and the torch loop in `modeling_acestep_v15_turbo.py` with
// `common/dcw_correction.py` + `dcw_primitives.py`. Initial noise is an INPUT (the engine draws it); there is no RNG parity with torch here.
// Not ported (not on this path): CFG/APG, SDE, Heun, velocity clamp/EMA, cover and repaint, non-Haar wavelets.

nonisolated struct AceStepDCW: Sendable, Equatable {
    enum Mode: String, Sendable { case low, high, double, pix }
    var mode: Mode = .double
    /// Low-band (or pix / high-only) scaler, `dcw_scaler`.
    var scaler: Double = 0.05
    /// High-band scaler in double mode, `dcw_high_scaler`.
    var highScaler: Double = 0.02

    /// Upstream's `dcw_active`: no-op when both effective scalers are zero.
    var isActive: Bool { scaler != 0 || (mode == .double && highScaler != 0) }

    /// `DCWCorrector.apply` / `apply_mlx_dcw`: `x` is the stepped latent, `denoised = x_before - v * t`, `t` the step's current timestep.
    /// Low scaler decays with t, the high scaler is the complementary schedule; `pix` uses the raw scaler. Haar only.
    func apply(_ x: MLXArray, denoised: MLXArray, t: Double) -> MLXArray {
        let lowS = t * scaler, highS = (1 - t) * scaler, doubleHighS = (1 - t) * highScaler
        switch mode {
        case .pix:
            return scaler == 0 ? x : x + Float(scaler) * (x - denoised)
        case .low:
            if lowS == 0 { return x }
            return correct(x, denoised, low: lowS, high: 0)
        case .high:
            if highS == 0 { return x }
            return correct(x, denoised, low: 0, high: highS)
        case .double:
            if lowS == 0 && doubleHighS == 0 { return x }
            return correct(x, denoised, low: lowS, high: doubleHighS)
        }
    }

    private func correct(_ x: MLXArray, _ y: MLXArray, low: Double, high: Double) -> MLXArray {
        var (xL, xH) = AceStepHaar.forward(x)
        let (yL, yH) = AceStepHaar.forward(y)
        if low != 0 { xL = xL + Float(low) * (xL - yL) }
        if high != 0 { xH = xH + Float(high) * (xH - yH) }
        return AceStepHaar.inverse(xL, xH, length: x.dim(1))
    }
}

/// Single-level Haar DWT along the time axis of `[B, T, C]` (`_haar_dwt_1d` / `_haar_idwt_1d`); odd T is zero-padded on the right.
nonisolated enum AceStepHaar {
    static let invSqrt2 = Float(1.0 / 2.0.squareRoot())

    static func forward(_ x: MLXArray) -> (low: MLXArray, high: MLXArray) {
        var x = x
        let (B, C) = (x.dim(0), x.dim(2))
        if x.dim(1) % 2 == 1 { x = concatenated([x, MLXArray.zeros([B, 1, C], dtype: x.dtype)], axis: 1) }
        let pairs = x.reshaped(B, x.dim(1) / 2, 2, C)
        let even = pairs[0..., 0..., 0, 0...], odd = pairs[0..., 0..., 1, 0...]
        return ((even + odd) * invSqrt2, (even - odd) * invSqrt2)
    }

    static func inverse(_ low: MLXArray, _ high: MLXArray, length: Int) -> MLXArray {
        let even = (low + high) * invSqrt2, odd = (low - high) * invSqrt2
        let (B, H, C) = (even.dim(0), even.dim(1), even.dim(2))
        let woven = stacked([even, odd], axis: 2).reshaped(B, H * 2, C)
        return woven[0..., 0 ..< length, 0...]
    }
}

nonisolated struct AceStepSamplerConfig: Sendable, Equatable {
    var steps = 8
    var shift = 1.0
    /// nil = DCW off.
    var dcw: AceStepDCW? = AceStepDCW()
}

nonisolated enum AceStepSampler {
    /// `get_timestep_schedule(shift, infer_steps=steps)`: `1 - i/steps`, shifted by `shift*t / (1 + (shift-1)*t)` when `shift != 1`.
    /// Descending, no trailing 0 (the last step goes straight to x0).
    static func schedule(shift: Double, steps: Int) -> [Double] {
        let raw = (0 ..< steps).map { 1.0 - Double($0) / Double(steps) }
        return shift == 1.0 ? raw : raw.map { shift * $0 / (1.0 + (shift - 1.0) * $0) }
    }

    /// Runs the loop from `noise` `[B, T, 64]`. `velocity(x, t)` is one DiT pass (`timestep == timestep_r == t`). The last step is
    /// `x0 = x - v * t`; DCW runs after every step including the last. Cancellation is checked before each step; `progress(done, total)`
    /// is called after each.
    static func run(noise: MLXArray, config: AceStepSamplerConfig,
                    velocity: (MLXArray, Float) throws -> MLXArray,
                    progress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        let times = schedule(shift: config.shift, steps: config.steps)
        let dcw = config.dcw.flatMap { $0.isActive ? $0 : nil }
        var xt = noise
        for (i, t) in times.enumerated() {
            try Task.checkCancellation()
            let v = try velocity(xt, Float(t))
            let before = xt
            let dt = i == times.count - 1 ? t : t - times[i + 1]      // the last step lands on x0 (t -> 0)
            xt = xt - v * Float(dt)
            if let dcw { xt = dcw.apply(xt, denoised: before - v * Float(t), t: t) }
            eval(xt)
            progress?(i + 1, times.count)
        }
        return xt
    }
}
