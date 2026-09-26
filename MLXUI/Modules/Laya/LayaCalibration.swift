import Foundation

/// Calibration math, ported from `laya_mlx/common.py`: `temp_bucket`, `clamp_temperature`,
/// `confidence_from_probs`.
nonisolated enum LayaCalibration {
    /// A fitted temperature below 1 sharpens the logits instead of softening them. The
    /// shipped `choice:11+` bucket is 0.1006, which multiplies logits ~10x: a 0.24 top
    /// probability is published as 0.99, so a caller gating on confidence is told a coin flip
    /// is a certainty. No honest calibration needs to sharpen this hard, so refuse to apply
    /// one that does (`common.py`'s own comment, carried over verbatim).
    static let temperatureMin: Float = 0.5
    static let temperatureMax: Float = 5.0

    /// `common.py::temp_bucket`.
    static func temperatureBucket(type: LayaQuestionType, optionCount k: Int) -> String {
        let size: String
        switch k {
        case ...2: size = "2"
        case 3 ... 5: size = "3-5"
        case 6 ... 10: size = "6-10"
        default: size = "11+"
        }
        return "\(type.rawValue):\(size)"
    }

    /// `common.py::clamp_temperature`. A usable temperature: `t` confined to `[lo, hi]`,
    /// falling back to `1.0` if it is not a finite number.
    static func clampTemperature(_ t: Float, lo: Float = temperatureMin, hi: Float = temperatureMax) -> Float {
        guard t.isFinite else { return 1.0 }
        return min(hi, max(lo, t))
    }

    /// `common.py::confidence_from_probs`. Normalized Shannon entropy confidence:
    /// `1 - H(p) / log(k)`.
    static func confidence(fromProbabilities p: [Double], optionCount k: Int) -> Double {
        guard k >= 2 else { return 1.0 }
        let top = Array(p.prefix(k))
        let entropy = -top.reduce(0.0) { sum, value in sum + value * log(max(value, 1e-12)) }
        let confidence = 1.0 - entropy / log(Double(k))
        return min(1.0, max(0.0, confidence))
    }
}
