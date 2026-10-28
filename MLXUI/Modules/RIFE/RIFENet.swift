import Foundation
import MLX
import MLXNN

// RIFE 4.25 (IFNet), a Swift/MLX port of Practical-RIFE 4.25 (MIT, © hzwer), translated from the MLX reference
// github.com/xocialize/rife-mlx (MIT, declared in its pyproject.toml and README; owner accepted 2026-10-28).
// NHWC throughout. Parameter keys are those of the published mlx-community/RIFE-4.25 safetensors, so the checkpoint
// loads with `update(parameters:verify: .all)` and no remapping:
//   block{0..4}.conv0.{0,1}.conv.{weight,bias}  block{i}.convblock.{0..7}.{conv.weight,conv.bias,beta}
//   block{i}.lastconv.{weight,bias}             encode.cnn{0..3}.{weight,bias}

private let rifeSlope: Float = 0.2

private nonisolated func lrelu(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: rifeSlope) }

// MARK: - Ops

nonisolated enum RIFEOps {

    /// Source coordinates of a bilinear resize along one axis (`align_corners=False`).
    private static func sourceCoordinates(out: Int, in inSize: Int) -> MLXArray {
        let dst = MLXArray(0 ..< out).asType(.float32)
        let scale = Float(inSize) / Float(out)
        return (dst + 0.5) * scale - 0.5
    }

    private static func bilinear1D(_ x: MLXArray, axis: Int, out: Int) -> MLXArray {
        let inSize = x.dim(axis)
        if inSize == out { return x }
        let src = sourceCoordinates(out: out, in: inSize)
        let lower = floor(src)
        let upper = lower + 1
        let w1 = src - lower
        let w0 = 1.0 - w1
        let i0 = clip(lower, min: 0, max: inSize - 1).asType(.int32)
        let i1 = clip(upper, min: 0, max: inSize - 1).asType(.int32)
        let g0 = take(x, i0, axis: axis)
        let g1 = take(x, i1, axis: axis)
        var shape = [Int](repeating: 1, count: x.ndim)
        shape[axis] = out
        return g0 * w0.reshaped(shape) + g1 * w1.reshaped(shape)
    }

    /// `F.interpolate(mode: "bilinear", align_corners: False)` on `[N, H, W, C]`, to `scaleFactor` times the size.
    static func resizeBilinear(_ x: MLXArray, scaleFactor: Float) -> MLXArray {
        let outH = Int((Float(x.dim(1)) * scaleFactor).rounded())
        let outW = Int((Float(x.dim(2)) * scaleFactor).rounded())
        return bilinear1D(bilinear1D(x, axis: 1, out: outH), axis: 2, out: outW)
    }

    /// `[N, H, W, C·r²] → [N, H·r, W·r, C]`, channel split `(C, r, r)` like `torch.nn.PixelShuffle`.
    static func pixelShuffle(_ x: MLXArray, _ r: Int) -> MLXArray {
        let n = x.dim(0), h = x.dim(1), w = x.dim(2)
        let c = x.dim(3) / (r * r)
        return x.reshaped(n, h, w, c, r, r).transposed(0, 1, 4, 2, 5, 3).reshaped(n, h * r, w * r, c)
    }

    /// `grid_sample(mode: bilinear, padding_mode: border, align_corners: true)`. `grid` is `[N, gH, gW, 2]` in `[-1, 1]`
    /// (channel 0 = x). Neighbour indices are clamped, the weights come from the unclamped coordinate (torch's border mode).
    static func gridSampleBorder(_ input: MLXArray, grid: MLXArray) -> MLXArray {
        let n = input.dim(0), h = input.dim(1), w = input.dim(2), c = input.dim(3)
        let gH = grid.dim(1), gW = grid.dim(2)
        let ix = (grid[0..., 0..., 0..., 0] + 1) * 0.5 * Float(w - 1)
        let iy = (grid[0..., 0..., 0..., 1] + 1) * 0.5 * Float(h - 1)
        let x0 = floor(ix), y0 = floor(iy)
        let wx1 = ix - x0, wy1 = iy - y0
        let wx0 = 1.0 - wx1, wy0 = 1.0 - wy1
        let x0c = clip(x0, min: 0, max: w - 1).asType(.int32)
        let x1c = clip(x0 + 1, min: 0, max: w - 1).asType(.int32)
        let y0c = clip(y0, min: 0, max: h - 1).asType(.int32)
        let y1c = clip(y0 + 1, min: 0, max: h - 1).asType(.int32)
        let flat = input.reshaped(n, h * w, c)
        func gather(_ yc: MLXArray, _ xc: MLXArray) -> MLXArray {
            let idx = (yc * w + xc).reshaped(n, gH * gW, 1)
            let wide = broadcast(idx, to: [n, gH * gW, c])
            return takeAlong(flat, wide, axis: 1).reshaped(n, gH, gW, c)
        }
        let v00 = gather(y0c, x0c), v01 = gather(y0c, x1c)
        let v10 = gather(y1c, x0c), v11 = gather(y1c, x1c)
        return v00 * (wy0 * wx0).expandedDimensions(axis: -1)
            + v01 * (wy0 * wx1).expandedDimensions(axis: -1)
            + v10 * (wy1 * wx0).expandedDimensions(axis: -1)
            + v11 * (wy1 * wx1).expandedDimensions(axis: -1)
    }

    /// RIFE's backward warp: the identity grid plus `flow` (pixel units, channel 0 = x, 1 = y) normalised by `(size-1)/2`.
    static func warp(_ input: MLXArray, flow: MLXArray) -> MLXArray {
        let n = flow.dim(0), h = flow.dim(1), w = flow.dim(2)
        let xs = broadcast(linspace(Float(-1), Float(1), count: w).reshaped(1, 1, w, 1), to: [n, h, w, 1])
        let ys = broadcast(linspace(Float(-1), Float(1), count: h).reshaped(1, h, 1, 1), to: [n, h, w, 1])
        let fx = flow[0..., 0..., 0..., 0 ..< 1] / (Float(w - 1) / 2)
        let fy = flow[0..., 0..., 0..., 1 ..< 2] / (Float(h - 1) / 2)
        let grid = concatenated([xs, ys], axis: -1) + concatenated([fx, fy], axis: -1)
        return gridSampleBorder(input, grid: grid)
    }
}

// MARK: - Layers

/// `ConvTranspose2d` (mlx-swift 0.32 has the op but no layer). Weight `[out, kH, kW, in]`, as the checkpoint stores it.
nonisolated final class RIFEConvTranspose2d: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray
    private let stride: Int
    private let padding: Int

    init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int, padding: Int) {
        self.weight = MLXArray.zeros([outputChannels, kernelSize, kernelSize, inputChannels])
        self.bias = MLXArray.zeros([outputChannels])
        self.stride = stride
        self.padding = padding
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        convTransposed2d(x, weight, stride: IntOrPair((stride, stride)), padding: IntOrPair((padding, padding))) + bias
    }
}

/// Conv2d + LeakyReLU(0.2). Key: `<p>.conv.{weight,bias}`.
nonisolated final class RIFEConv: Module, UnaryLayer {
    let conv: Conv2d

    init(_ i: Int, _ o: Int, stride: Int) {
        self.conv = Conv2d(inputChannels: i, outputChannels: o, kernelSize: 3, stride: IntOrPair((stride, stride)), padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { lrelu(conv(x)) }
}

/// `conv(x) · beta + x → LeakyReLU`. Keys: `.conv.{weight,bias}`, `.beta` (`[1, 1, 1, C]`).
nonisolated final class RIFEResConv: Module, UnaryLayer {
    let conv: Conv2d
    let beta: MLXArray

    init(_ c: Int) {
        self.conv = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 3, stride: 1, padding: 1)
        self.beta = MLXArray.ones([1, 1, 1, c])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { lrelu(conv(x) * beta + x) }
}

/// Feature encoder, 3 channels → 4. Keys `cnn0…cnn3`.
nonisolated final class RIFEHead: Module, UnaryLayer {
    let cnn0 = Conv2d(inputChannels: 3, outputChannels: 16, kernelSize: 3, stride: 2, padding: 1)
    let cnn1 = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: 3, stride: 1, padding: 1)
    let cnn2 = Conv2d(inputChannels: 16, outputChannels: 16, kernelSize: 3, stride: 1, padding: 1)
    let cnn3 = RIFEConvTranspose2d(inputChannels: 16, outputChannels: 4, kernelSize: 4, stride: 2, padding: 1)

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cnn3(lrelu(cnn2(lrelu(cnn1(lrelu(cnn0(x)))))))
    }
}

nonisolated final class RIFEIFBlock: Module {
    let conv0: [RIFEConv]
    let convblock: [RIFEResConv]
    let lastconv: RIFEConvTranspose2d

    init(inPlanes: Int, c: Int) {
        self.conv0 = [RIFEConv(inPlanes, c / 2, stride: 2), RIFEConv(c / 2, c, stride: 2)]
        self.convblock = (0 ..< 8).map { _ in RIFEResConv(c) }
        self.lastconv = RIFEConvTranspose2d(inputChannels: c, outputChannels: 4 * 13, kernelSize: 4, stride: 2, padding: 1)
        super.init()
    }

    /// Returns (flow `[.., 4]`, mask `[.., 1]`, feat `[.., 8]`).
    func callAsFunction(_ input: MLXArray, flow flowIn: MLXArray?, scale: Float) -> (MLXArray, MLXArray, MLXArray) {
        var x = RIFEOps.resizeBilinear(input, scaleFactor: 1 / scale)
        if let flowIn {
            let f = RIFEOps.resizeBilinear(flowIn, scaleFactor: 1 / scale) * (1 / scale)
            x = concatenated([x, f], axis: -1)
        }
        var feat = conv0[1](conv0[0](x))
        for block in convblock { feat = block(feat) }
        var tmp = RIFEOps.pixelShuffle(lastconv(feat), 2)
        tmp = RIFEOps.resizeBilinear(tmp, scaleFactor: scale)
        return (tmp[0..., 0..., 0..., 0 ..< 4] * scale, tmp[0..., 0..., 0..., 4 ..< 5], tmp[0..., 0..., 0..., 5...])
    }
}

// MARK: - IFNet

/// The 4.25 flow network: five coarse-to-fine blocks, channels 192/128/96/64/32.
nonisolated final class RIFEIFNet: Module {
    static let blockChannels = [192, 128, 96, 64, 32]
    static let blockInPlanes = [15, 28, 28, 28, 28]

    let block0 = RIFEIFBlock(inPlanes: 7 + 8, c: 192)
    let block1 = RIFEIFBlock(inPlanes: 8 + 4 + 8 + 8, c: 128)
    let block2 = RIFEIFBlock(inPlanes: 8 + 4 + 8 + 8, c: 96)
    let block3 = RIFEIFBlock(inPlanes: 8 + 4 + 8 + 8, c: 64)
    let block4 = RIFEIFBlock(inPlanes: 8 + 4 + 8 + 8, c: 32)
    let encode = RIFEHead()

    /// `x`: `[N, H, W, 6]` (img0 | img1), H and W multiples of the coarsest scale. Returns the blended middle frame at `timestep`.
    func callAsFunction(_ x: MLXArray, timestep: Float, scaleList: [Float]) -> MLXArray {
        let img0 = x[0..., 0..., 0..., 0 ..< 3]
        let img1 = x[0..., 0..., 0..., 3 ..< 6]
        let n = img0.dim(0), h = img0.dim(1), w = img0.dim(2)
        let t = MLXArray.full([n, h, w, 1], values: MLXArray(timestep))
        let f0 = encode(img0), f1 = encode(img1)
        let blocks = [block0, block1, block2, block3, block4]
        var flow: MLXArray?
        var mask: MLXArray?
        var feat: MLXArray?
        var warped0 = img0, warped1 = img1
        for i in 0 ..< 5 {
            if let flowNow = flow, let maskNow = mask, let featNow = feat {
                let wf0 = RIFEOps.warp(f0, flow: flowNow[0..., 0..., 0..., 0 ..< 2])
                let wf1 = RIFEOps.warp(f1, flow: flowNow[0..., 0..., 0..., 2 ..< 4])
                let input = concatenated([warped0, warped1, wf0, wf1, t, maskNow, featNow], axis: -1)
                let (delta, m, ft) = blocks[i](input, flow: flowNow, scale: scaleList[i])
                mask = m
                feat = ft
                flow = flowNow + delta
            } else {
                let input = concatenated([img0, img1, f0, f1, t], axis: -1)
                let (fl, m, ft) = blocks[i](input, flow: nil, scale: scaleList[i])
                flow = fl
                mask = m
                feat = ft
            }
            guard let current = flow else { break }
            warped0 = RIFEOps.warp(img0, flow: current[0..., 0..., 0..., 0 ..< 2])
            warped1 = RIFEOps.warp(img1, flow: current[0..., 0..., 0..., 2 ..< 4])
        }
        let blend = sigmoid(mask ?? MLXArray(0))
        return warped0 * blend + warped1 * (1 - blend)
    }
}
