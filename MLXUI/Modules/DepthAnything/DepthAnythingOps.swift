import Foundation
import MLX

/// Errors for the Depth Anything port. Every sentence names what was wrong; the Run screen shows it as-is.
nonisolated enum DepthAnythingError: Error, LocalizedError, Equatable {
    case missingFile(String)
    case unsupportedConfig(String)
    case badWeights(String)
    case badImage(String)
    case badInput(String)

    var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "Depth Anything: required file not found: \(name). Install the model again."
        case .unsupportedConfig(let detail):
            return "Depth Anything: this checkpoint isn't the ported Small network (\(detail))."
        case .badWeights(let detail):
            return "Depth Anything: the checkpoint doesn't match this port (\(detail))."
        case .badImage(let detail):
            return "Depth Anything: \(detail)"
        case .badInput(let detail):
            return "Depth Anything: \(detail)"
        }
    }
}

/// Tensor operations the port needs that MLXNN does not expose with an **exact target size**: PyTorch's
/// `F.interpolate` takes a size, while `MLXNN.Upsample` takes a scale factor, so a fractional scale can
/// round to the wrong output shape. These helpers compute the source coordinates once and gather, matching
/// PyTorch's `area_pixel_compute_source_index` semantics (including the bicubic a = -0.75 kernel and the
/// separate negative-index clamp for the linear modes).
nonisolated enum DepthAnythingOps {

    /// `F.interpolate(mode: "nearest"/"bilinear", align_corners:)` source coordinates along one axis.
    static func sourceCoordinates(out: Int, in inSize: Int, alignCorners: Bool) -> MLXArray {
        let destination = MLXArray(0 ..< out).asType(.float32)
        if alignCorners {
            if out == 1 { return MLXArray.zeros([1]) }
            let scale = Float(inSize - 1) / Float(out - 1)
            return destination * scale
        }
        let scale = Float(inSize) / Float(out)
        return (destination + 0.5) * scale - 0.5
    }

    /// `F.interpolate(mode: "bilinear", align_corners:)` on `[N, H, W, C]` with an exact output size.
    static func resizeBilinear(_ x: MLXArray, outH: Int, outW: Int, alignCorners: Bool) -> MLXArray {
        let y = resizeLinear1D(x, axis: 1, out: outH, alignCorners: alignCorners)
        return resizeLinear1D(y, axis: 2, out: outW, alignCorners: alignCorners)
    }

    private static func resizeLinear1D(_ x: MLXArray, axis: Int, out: Int, alignCorners: Bool) -> MLXArray {
        let inSize = x.dim(axis)
        if inSize == out { return x }
        var source = sourceCoordinates(out: out, in: inSize, alignCorners: alignCorners)
        // PyTorch clamps a negative source to 0 for the non-cubic modes (it does not for cubic: the
        // four clamped taps and the unclamped fractional position give the border behaviour there).
        if !alignCorners { source = maximum(source, 0) }
        let lower = floor(source)
        let upper = lower + 1
        let w1 = source - lower
        let w0 = 1.0 - w1
        let i0 = clip(lower, min: 0, max: inSize - 1).asType(.int32)
        let i1 = clip(upper, min: 0, max: inSize - 1).asType(.int32)
        var shape = [Int](repeating: 1, count: x.ndim)
        shape[axis] = out
        return take(x, i0, axis: axis) * w0.reshaped(shape)
            + take(x, i1, axis: axis) * w1.reshaped(shape)
    }

    /// `F.interpolate(mode: "bicubic", align_corners:)` on `[N, H, W, C]` with an exact output size.
    /// PyTorch's kernel is the separable `a = -0.75` cubic; both axes are interpolated independently.
    static func resizeBicubic(_ x: MLXArray, outH: Int, outW: Int, alignCorners: Bool) -> MLXArray {
        let y = resizeCubic1D(x, axis: 1, out: outH, alignCorners: alignCorners)
        return resizeCubic1D(y, axis: 2, out: outW, alignCorners: alignCorners)
    }

    /// PyTorch's cubic convolution kernel weights (`A = -0.75`, the antialias-free kernel).
    static func cubicWeight(_ x: MLXArray) -> MLXArray {
        let a: Float = -0.75
        let ax = abs(x)
        let near = (a + 2) * ax * ax * ax - (a + 3) * ax * ax + 1.0
        let far = a * ax * ax * ax - 5 * a * ax * ax + 8 * a * ax - 4 * a
        return MLX.where(ax .<= 1.0, near, MLX.where(ax .< 2.0, far, MLXArray.zeros(ax.shape)))
    }

    private static func resizeCubic1D(_ x: MLXArray, axis: Int, out: Int, alignCorners: Bool) -> MLXArray {
        let inSize = x.dim(axis)
        if inSize == out { return x }
        let source = sourceCoordinates(out: out, in: inSize, alignCorners: alignCorners)
        let base = floor(source)
        let t = source - base
        var shape = [Int](repeating: 1, count: x.ndim)
        shape[axis] = out
        var accumulated: MLXArray?
        // Taps at base-1, base, base+1, base+2 with weights W(t+1), W(t), W(t-1), W(t-2).
        let offsets: [Float] = [-1, 0, 1, 2]
        let weights = [cubicWeight(t + 1.0), cubicWeight(t), cubicWeight(t - 1.0), cubicWeight(t - 2.0)]
        for (offset, weight) in zip(offsets, weights) {
            let index = clip(base + offset, min: 0, max: inSize - 1).asType(.int32)
            let gathered = take(x, index, axis: axis) * weight.reshaped(shape)
            accumulated = accumulated.map { $0 + gathered } ?? gathered
        }
        return accumulated ?? x
    }

    // MARK: - Checkpoint layout

    /// The camelCase module key for a lowercase snake_case PyTorch tensor name.
    /// `cls_token` → `clsToken`, `layer_scale1` → `layerScale1`, `attention.attention.query` unchanged.
    static func moduleKey(_ name: String) -> String {
        name.split(separator: ".", omittingEmptySubsequences: false).map { segment -> String in
            let parts = segment.split(separator: "_")
            guard parts.count > 1 else { return String(segment) }
            let head = String(parts[0])
            return head + parts.dropFirst().map { $0.prefix(1).uppercased() + $0.dropFirst() }.joined()
        }.joined(separator: ".")
    }

    /// Converts one published safetensors tensor to the layout this port's module expects.
    ///
    /// PyTorch stores `Conv2d` weights `[out, in, kH, kW]` and `ConvTranspose2d` weights `[in, out, kH, kW]`;
    /// MLXNN wants `[out, kH, kW, in]` for both. Linear weights and all 1-D parameters are already in the
    /// layout MLXNN uses. The one tensor that is a transposed convolution is a reassemble layer whose factor
    /// is greater than 1 (factors 4 and 2); the factor-0.5 layer is a strided `Conv2d`.
    static func convertedWeight(name: String, array: MLXArray, config: DepthAnythingConfig) -> MLXArray {
        guard array.ndim == 4 else { return array }
        if name.contains("neck.reassemble_stage.layers."), name.hasSuffix(".resize.weight") {
            let indexText = name.split(separator: ".")[3]
            if let index = Int(indexText), index < config.reassembleFactors.count,
               config.reassembleFactors[index] > 1 {
                return array.transposed(1, 2, 3, 0)
            }
        }
        return array.transposed(0, 2, 3, 1)
    }
}
