import Foundation
import MLX
import MLXNN

// MC5-3h: the ACE-Step Oobleck VAE decoder (diffusers `AutoencoderOobleck`, decoder only; upstream's native MLX port `mlx/vae_model.py` and the
// tiled decode in `core/generation/handler/mlx_vae_decode_native.py`). Weights: the 3c `AceStepWeights.loadVAEDecoder` result (weight-norm fused,
// MLX layouts, f32, names under `decoder.`); the real path casts them to f16 (upstream ACESTEP_MLX_VAE_FP16=1), f32 is selectable.
// LAYOUT: internally channels-last, latents `[B, T, 64]` -> audio `[B, samples, 2]` (MLX Conv1d). `decodeChannelsFirst` takes the dumps'
// `[B, 64, T]` and returns f32 `[B, 2, samples]` (hop 1920 -> 48 kHz stereo).

nonisolated struct AceStepVAEConfig: Sendable, Equatable {
    var decoderChannels = 128
    var inputChannels = 64
    var audioChannels = 2
    /// Encoder-order strides (`downsampling_ratios`); the decoder upsamples in the reverse order.
    var downsamplingRatios = [2, 4, 4, 6, 10]
    var channelMultiples = [1, 2, 4, 8, 16]

    init() {}

    init(json: Data) throws {
        guard let o = try JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw AceStepWeightsError.badHeader("vae config", "not a JSON object")
        }
        func int(_ k: String) throws -> Int {
            guard let v = (o[k] as? NSNumber)?.intValue else { throw AceStepWeightsError.badHeader("vae config", "missing '\(k)'") }
            return v
        }
        func ints(_ k: String) throws -> [Int] {
            guard let v = o[k] as? [NSNumber] else { throw AceStepWeightsError.badHeader("vae config", "missing '\(k)'") }
            return v.map(\.intValue)
        }
        decoderChannels = try int("decoder_channels")
        inputChannels = try int("decoder_input_channels")
        audioChannels = try int("audio_channels")
        downsamplingRatios = try ints("downsampling_ratios")
        channelMultiples = try ints("channel_multiples")
    }

    /// Samples per latent frame (1920 for the real VAE).
    var hop: Int { downsamplingRatios.reduce(1, *) }
}

// MARK: - Layers

/// `x + (1/(exp(beta) + 1e-9)) * sin(exp(alpha) * x)^2`, as upstream (no f32 upcast: this is what its f16 path does).
nonisolated final class AceStepSnake1d: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "beta") var beta: MLXArray
    init(_ channels: Int) {
        _alpha.wrappedValue = MLXArray.zeros([channels])
        _beta.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + reciprocal(exp(beta) + 1e-9) * sin(exp(alpha) * x).square()
    }
}

nonisolated final class AceStepVAEResidualUnit: Module {
    let snake1: AceStepSnake1d
    let conv1: Conv1d
    let snake2: AceStepSnake1d
    let conv2: Conv1d
    init(_ dim: Int, dilation: Int) {
        snake1 = AceStepSnake1d(dim)
        conv1 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 7, padding: 3 * dilation, dilation: dilation)
        snake2 = AceStepSnake1d(dim)
        conv2 = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + conv2(snake2(conv1(snake1(x)))) }
}

nonisolated final class AceStepVAEDecoderBlock: Module {
    let snake1: AceStepSnake1d
    @ModuleInfo(key: "conv_t1") var convT1: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") var res1: AceStepVAEResidualUnit
    @ModuleInfo(key: "res_unit2") var res2: AceStepVAEResidualUnit
    @ModuleInfo(key: "res_unit3") var res3: AceStepVAEResidualUnit
    init(input: Int, output: Int, stride: Int) {
        snake1 = AceStepSnake1d(input)
        _convT1.wrappedValue = ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride,
                                                padding: (stride + 1) / 2)
        _res1.wrappedValue = AceStepVAEResidualUnit(output, dilation: 1)
        _res2.wrappedValue = AceStepVAEResidualUnit(output, dilation: 3)
        _res3.wrappedValue = AceStepVAEResidualUnit(output, dilation: 9)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { res3(res2(res1(convT1(snake1(x))))) }
}

nonisolated final class AceStepOobleckDecoder: Module {
    let conv1: Conv1d
    let block: [AceStepVAEDecoderBlock]
    let snake1: AceStepSnake1d
    let conv2: Conv1d
    init(_ c: AceStepVAEConfig) {
        let strides = Array(c.downsamplingRatios.reversed())
        let cm = [1] + c.channelMultiples
        conv1 = Conv1d(inputChannels: c.inputChannels, outputChannels: c.decoderChannels * cm[cm.count - 1], kernelSize: 7, padding: 3)
        block = strides.enumerated().map { i, stride in
            AceStepVAEDecoderBlock(input: c.decoderChannels * cm[strides.count - i], output: c.decoderChannels * cm[strides.count - i - 1],
                                   stride: stride)
        }
        snake1 = AceStepSnake1d(c.decoderChannels)
        conv2 = Conv1d(inputChannels: c.decoderChannels, outputChannels: c.audioChannels, kernelSize: 7, padding: 3, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block { h = b(h) }
        return conv2(snake1(h))
    }
}

nonisolated final class AceStepVAENetwork: Module {
    let decoder: AceStepOobleckDecoder
    init(_ c: AceStepVAEConfig) { decoder = AceStepOobleckDecoder(c); super.init() }
}

// MARK: - Decoder

nonisolated final class AceStepVAE: @unchecked Sendable {
    let config: AceStepVAEConfig
    let dtype: DType
    private let network: AceStepVAENetwork

    /// Upstream's tiling: the requested chunk is raised to at least 192 frames, the overlap is 64 each side.
    static let minChunk = 192, overlap = 64, defaultChunk = 256

    /// Names `AceStepWeights.loadVAEDecoder` returns (under `decoder.`), and nothing else.
    static func expectedParameterNames(_ c: AceStepVAEConfig) -> [String] {
        var names = ["decoder.conv1.weight", "decoder.conv1.bias", "decoder.snake1.alpha", "decoder.snake1.beta", "decoder.conv2.weight"]
        for i in 0 ..< c.downsamplingRatios.count {
            let b = "decoder.block.\(i)"
            names += ["\(b).snake1.alpha", "\(b).snake1.beta", "\(b).conv_t1.weight", "\(b).conv_t1.bias"]
            for u in 1 ... 3 {
                let r = "\(b).res_unit\(u)"
                names += ["\(r).snake1.alpha", "\(r).snake1.beta", "\(r).snake2.alpha", "\(r).snake2.beta"]
                names += ["\(r).conv1.weight", "\(r).conv1.bias", "\(r).conv2.weight", "\(r).conv2.bias"]
            }
        }
        return names
    }

    /// Strict: every expected tensor present, none left over. `dtype` is the compute dtype (parameters are cast to it): `.float16` for the
    /// real path, `.float32` for exact parity.
    init(config: AceStepVAEConfig, weights: [String: MLXArray], dtype: DType = .float16) throws {
        self.config = config
        self.dtype = dtype
        try AceStepWeights.checkStrict(actual: weights.keys, expected: Self.expectedParameterNames(config))
        let net = AceStepVAENetwork(config)
        try net.update(parameters: ModuleParameters.unflattened(weights.mapValues { $0.asType(dtype) }), verify: [.all])
        eval(net)
        network = net
    }

    /// One direct decode, `[B, T, 64]` -> `[B, T x hop, 2]` (compute dtype).
    func decode(_ latents: MLXArray) -> MLXArray { network.decoder(latents.asType(dtype)) }

    /// The tiling plan (`_mlx_decode_single`): `(coreStart, coreEnd, windowStart, windowEnd)` in latent frames, empty when the input fits one chunk.
    static func plan(frames: Int, chunk: Int = defaultChunk) -> [(coreStart: Int, coreEnd: Int, winStart: Int, winEnd: Int)] {
        let chunk = max(minChunk, chunk)
        if frames <= chunk { return [] }
        let stride = chunk - 2 * overlap
        return (0 ..< (frames + stride - 1) / stride).map { i in
            let coreStart = i * stride, coreEnd = min(coreStart + stride, frames)
            return (coreStart, coreEnd, max(0, coreStart - overlap), min(frames, coreEnd + overlap))
        }
    }

    /// Overlap-discard tiled decode, `[1, T, 64]` -> `[1, samples, 2]`: each window is decoded, the overlap margins are trimmed, the cores are
    /// concatenated (no crossfade, as upstream). Cancellation is checked before every chunk; the buffer cache is cleared every 4 chunks.
    func decodeTiled(_ latents: MLXArray, chunk: Int = AceStepVAE.defaultChunk, progress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        let frames = latents.dim(1)
        let windows = Self.plan(frames: frames, chunk: chunk)
        if windows.isEmpty { return decode(latents) }
        var parts: [MLXArray] = []
        var upsample: Double?
        for (idx, w) in windows.enumerated() {
            try Task.checkCancellation()
            let piece = latents[0..., w.winStart ..< w.winEnd, 0...]
            let audio = decode(piece)
            eval(audio)
            let factor = upsample ?? Double(audio.dim(1)) / Double(piece.dim(1))
            upsample = factor
            let trimStart = Int((Double(w.coreStart - w.winStart) * factor).rounded())
            let trimEnd = Int((Double(w.winEnd - w.coreEnd) * factor).rounded())
            let endIdx = trimEnd > 0 ? audio.dim(1) - trimEnd : audio.dim(1)
            let trimmed = audio[0..., trimStart ..< endIdx, 0...]
            eval(trimmed)
            parts.append(trimmed)
            if (idx + 1) % 4 == 0 { Memory.clearCache() }
            progress?(idx + 1, windows.count)
        }
        let out = concatenated(parts, axis: 1)
        eval(out)
        return out
    }

    /// The dumps' layout: latents `[B, 64, T]` -> f32 audio `[B, 2, samples]`.
    func decodeChannelsFirst(_ latents: MLXArray, chunk: Int = AceStepVAE.defaultChunk,
                             progress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        let audio = try decodeTiled(latents.transposed(0, 2, 1), chunk: chunk, progress: progress)
        return audio.asType(.float32).transposed(0, 2, 1)
    }
}
