import Foundation
import MLX
import MLXNN

/// EnCodec **decoder** for MusicGen (MG-ENG2) — a bespoke Swift/MLX port of the decoder half of
/// `ml-explore/mlx-examples/encodec/encodec.py` (`EncodecDecoder`), using GPU primitives.
///
/// **Why bespoke, not `MLXAudioCodecs.Encodec`:** the mlx-audio-swift implementation's
/// `EncodecBaseConvTranspose1d` is a pure-Swift scatter loop (`EncodecLayers.swift:406`), which
/// measured ~48 s for a 10-frame decode (≈20 min for MusicGen's 250-frame/5 s output) — unusable
/// for the engine. The decoder port below uses `MLX.convTransposed1d` (GPU), `Conv1d`, and
/// vectorised gates, matching the reference's architecture and weight layout exactly.
///
/// Loads the **bundled** 32 kHz weights (`mlx-community/encodec-32khz-float32`, shipped into
/// `encodec/` by the MG-DL1 companion-repo install): `config.json` + `model.safetensors`
/// (channels-last conv weights, verified against the safetensors header).
nonisolated final class MusicGenEncodec: @unchecked Sendable {
    private let config: MusicGenEncodecConfig
    private let quantizer: EncodecRVQ
    private let decoder: EncodecDecoderPort

    /// Load from the installed model dir (`encodec/` subdir).
    init(modelDirectory: URL) throws {
        let encodecDir = modelDirectory.appendingPathComponent("encodec", isDirectory: true)
        guard FileManager.default.fileExists(atPath: encodecDir.appendingPathComponent("model.safetensors").path) else {
            throw StageError.engineFailure(
                stage: "MusicGen", underlying: MusicGenEncodecError.missingEncodec(encodecDir.path))
        }
        let config = try MusicGenEncodecConfig(url: encodecDir.appendingPathComponent("config.json"))
        self.config = config
        self.quantizer = EncodecRVQ(config: config)
        self.decoder = EncodecDecoderPort(config: config)
        let raw = try MLX.loadArrays(url: encodecDir.appendingPathComponent("model.safetensors"))
        // Each submodule owns only its own subtree of the flat checkpoint keys. Feed it just
        // those (prefix stripped) so `.noUnusedKeys` still catches real mismatches instead of
        // reporting every foreign `decoder.*`/`encoder.*`/`quantizer.*` key as unhandled.
        try quantizer.update(parameters: ModuleParameters.unflattened(Self.subtree(raw, prefix: "quantizer.")), verify: .noUnusedKeys)
        try decoder.update(parameters: ModuleParameters.unflattened(Self.subtree(raw, prefix: "decoder.")), verify: .noUnusedKeys)
        eval(quantizer, decoder)
    }

    /// The keys of `raw` that fall under `prefix`, with the prefix stripped (e.g.
    /// `decoder.layers.0.conv.weight` → `layers.0.conv.weight`).
    nonisolated static func subtree(_ raw: [String: MLXArray], prefix: String) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in raw where key.hasPrefix(prefix) {
            out[String(key.dropFirst(prefix.count))] = value
        }
        return out
    }

    /// Test seam: build from a config + zeroed weights (shape tests).
    init(config: MusicGenEncodecConfig) {
        self.config = config
        self.quantizer = EncodecRVQ(config: config)
        self.decoder = EncodecDecoderPort(config: config)
    }

    /// Decode `codes` `[T, 4]` int32 codebook indices → mono float32 waveform `[T × 640]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        // quantizer.decode: sum the 4 codebook embeddings at each frame → [1, T, 128].
        let embeddings = quantizer.decode(codes.transposed(1, 0))   // [4, T] → [1, T, 128]
        let audio = decoder(embeddings)                              // [1, T×640, 1]
        eval(audio)
        let mono = audio.reshaped([-1])
        eval(mono)
        return mono
    }
}

nonisolated enum MusicGenEncodecError: LocalizedError {
    case missingEncodec(String)

    var errorDescription: String? {
        switch self {
        case .missingEncodec(let path): return "EnCodec weights not found at \(path) — reinstall the model (the 32 kHz codec is bundled into the install)."
        }
    }
}

// MARK: - Config

/// Config for the 32 kHz / 4-codebook EnCodec (from `mlx-community/encodec-32khz-float32`'s
/// `config.json`). Only the fields the decoder + quantizer need.
nonisolated struct MusicGenEncodecConfig: Sendable {
    let audioChannels: Int
    let numFilters: Int
    let kernelSize: Int
    let numResidualLayers: Int
    let dilationGrowthRate: Int
    let codebookSize: Int
    let codebookDim: Int
    let hiddenSize: Int
    let numLSTMLayers: Int
    let residualKernelSize: Int
    let useCausalConv: Bool
    let padMode: String
    let lastKernelSize: Int
    let compress: Int
    let upsamplingRatios: [Int]
    let targetBandwidths: [Float]
    let samplingRate: Int
    let trimRightRatio: Float

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let dict = try JSONDecoder().decode(EncodecConfigDTO.self, from: data)
        self.init(dto: dict)
    }

    /// Shape-test seam: real 32 kHz config.
    static var encodec32k: MusicGenEncodecConfig {
        MusicGenEncodecConfig(dto: EncodecConfigDTO(
            audioChannels: 1, numFilters: 64, kernelSize: 7, numResidualLayers: 1,
            dilationGrowthRate: 2, codebookSize: 2048, codebookDim: 128, hiddenSize: 128,
            numLSTMLayers: 2, residualKernelSize: 3, useCausalConv: false, padMode: "reflect",
            lastKernelSize: 7, compress: 2, upsamplingRatios: [8, 5, 4, 4],
            targetBandwidths: [2.2], samplingRate: 32000, trimRightRatio: 1.0))
    }

    private init(dto: EncodecConfigDTO) {
        self.audioChannels = dto.audioChannels
        self.numFilters = dto.numFilters
        self.kernelSize = dto.kernelSize
        self.numResidualLayers = dto.numResidualLayers
        self.dilationGrowthRate = dto.dilationGrowthRate
        self.codebookSize = dto.codebookSize
        self.codebookDim = dto.codebookDim
        self.hiddenSize = dto.hiddenSize
        self.numLSTMLayers = dto.numLSTMLayers
        self.residualKernelSize = dto.residualKernelSize
        self.useCausalConv = dto.useCausalConv
        self.padMode = dto.padMode
        self.lastKernelSize = dto.lastKernelSize
        self.compress = dto.compress
        self.upsamplingRatios = dto.upsamplingRatios
        self.targetBandwidths = dto.targetBandwidths
        self.samplingRate = dto.samplingRate
        self.trimRightRatio = dto.trimRightRatio
    }

    /// Hop length = product of upsampling ratios (8·5·4·4 = 640). 32 kHz / 640 = 50 frames/s.
    var hopLength: Int { upsamplingRatios.reduce(1, *) }
}

private struct EncodecConfigDTO: Codable {
    let audioChannels: Int
    let numFilters: Int
    let kernelSize: Int
    let numResidualLayers: Int
    let dilationGrowthRate: Int
    let codebookSize: Int
    let codebookDim: Int
    let hiddenSize: Int
    let numLSTMLayers: Int
    let residualKernelSize: Int
    let useCausalConv: Bool
    let padMode: String
    let lastKernelSize: Int
    let compress: Int
    let upsamplingRatios: [Int]
    let targetBandwidths: [Float]
    let samplingRate: Int
    let trimRightRatio: Float

    enum CodingKeys: String, CodingKey {
        case audioChannels = "audio_channels"
        case numFilters = "num_filters"
        case kernelSize = "kernel_size"
        case numResidualLayers = "num_residual_layers"
        case dilationGrowthRate = "dilation_growth_rate"
        case codebookSize = "codebook_size"
        case codebookDim = "codebook_dim"
        case hiddenSize = "hidden_size"
        case numLSTMLayers = "num_lstm_layers"
        case residualKernelSize = "residual_kernel_size"
        case useCausalConv = "use_causal_conv"
        case padMode = "pad_mode"
        case lastKernelSize = "last_kernel_size"
        case compress
        case upsamplingRatios = "upsampling_ratios"
        case targetBandwidths = "target_bandwidths"
        case samplingRate = "sampling_rate"
        case trimRightRatio = "trim_right_ratio"
    }
}

// MARK: - Quantizer

/// Residual vector quantizer (decode half). `layers.N.codebook.embed` [2048, 128].
nonisolated final class EncodecRVQ: Module {
    @ModuleInfo(key: "layers") var layers: [EncodecVQ]

    init(config: MusicGenEncodecConfig) {
        let frameRate = Int(ceil(Double(config.samplingRate) / Double(config.hopLength)))
        let bandwidth: Double = Double(config.targetBandwidths.max() ?? 2.2)
        let numerator: Double = bandwidth * 1000.0
        let denominator: Double = Double(frameRate * 10)
        let numQuantizers = Int(numerator / denominator)
        self._layers.wrappedValue = (0 ..< numQuantizers).map { _ in EncodecVQ(config: config) }
        super.init()
    }

    /// Sum the codebook embeddings across the 4 codebooks at each frame: `[4, T]` → `[1, T, 128]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        var out: MLXArray?
        for i in 0 ..< codes.dim(0) {
            let indices = codes[i]                                    // [T]
            let emb = layers[i](indices)                              // [T, 128]
            out = out.map { $0 + emb } ?? emb
        }
        return out!.expandedDimensions(axis: 0)                       // [1, T, 128]
    }
}

nonisolated final class EncodecVQ: Module {
    @ModuleInfo(key: "codebook") var codebook: EncodecCodebook

    init(config: MusicGenEncodecConfig) {
        self._codebook.wrappedValue = EncodecCodebook(config: config)
        super.init()
    }

    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        codebook(indices)
    }
}

nonisolated final class EncodecCodebook: Module {
    @ModuleInfo(key: "embed") var embed: MLXArray

    init(config: MusicGenEncodecConfig) {
        self._embed.wrappedValue = MLXArray.zeros([config.codebookSize, config.codebookDim])
        super.init()
    }

    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        embed[indices]
    }
}

// MARK: - Decoder

/// The SEANet decoder (decoder half of `encodec.py`), matching the checkpoint's `decoder.*` keys.
nonisolated final class EncodecDecoderPort: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(config: MusicGenEncodecConfig) {
        var scaling = Int(pow(2.0, Double(config.upsamplingRatios.count)))
        var model: [Module] = []
        model.append(EncodecConv1dPort(config: config, inChannels: config.hiddenSize, outChannels: scaling * config.numFilters, kernelSize: config.kernelSize))
        model.append(EncodecLSTMBlockPort(config: config, dimension: scaling * config.numFilters))
        for ratio in config.upsamplingRatios {
            let currentScale = scaling * config.numFilters
            model.append(ELUPort())
            model.append(EncodecConvTranspose1dPort(config: config, inChannels: currentScale, outChannels: currentScale / 2, kernelSize: ratio * 2, stride: ratio))
            for j in 0 ..< config.numResidualLayers {
                let dilation = Int(pow(Double(config.dilationGrowthRate), Double(j)))
                model.append(EncodecResnetBlockPort(config: config, dim: currentScale / 2, dilations: [dilation, 1]))
            }
            scaling /= 2
        }
        model.append(ELUPort())
        model.append(EncodecConv1dPort(config: config, inChannels: config.numFilters, outChannels: config.audioChannels, kernelSize: config.lastKernelSize))
        self._layers.wrappedValue = model
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let conv = layer as? EncodecConv1dPort { h = conv(h) }
            else if let convT = layer as? EncodecConvTranspose1dPort { h = convT(h) }
            else if let resnet = layer as? EncodecResnetBlockPort { h = resnet(h) }
            else if let lstm = layer as? EncodecLSTMBlockPort { h = lstm(h) }
            else if let elu = layer as? ELUPort { h = elu(h) }
        }
        return h
    }
}

/// Conv1d with asymmetric reflect/zero padding (channels-last NLC).
nonisolated final class EncodecConv1dPort: Module {
    let causal: Bool
    let padMode: String
    let stride: Int
    let paddingTotal: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(config: MusicGenEncodecConfig, inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1) {
        self.causal = config.useCausalConv
        self.padMode = config.padMode
        self.stride = stride
        self.paddingTotal = kernelSize - stride
        self._conv.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: kernelSize, stride: stride, dilation: dilation)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let extra = extraPadding(x)
        let pads = causal
            ? (paddingTotal, extra)
            : (paddingTotal - paddingTotal / 2, paddingTotal / 2 + extra)
        let padded = Self.pad1d(x, paddings: pads, mode: padMode)
        return conv(padded)
    }

    private func extraPadding(_ x: MLXArray) -> Int {
        let length = x.dim(1)
        let nFrames = Float(length - paddingTotal) / Float(stride) + 1
        let ideal = Int(ceil(nFrames)) - 1
        return max(0, ideal * stride + paddingTotal - length)
    }

    static func pad1d(_ x: MLXArray, paddings: (Int, Int), mode: String) -> MLXArray {
        if mode != "reflect" {
            return MLX.padded(x, widths: [.init(0), .init((paddings.0, paddings.1)), .init(0)])
        }
        var parts: [MLXArray] = []
        if paddings.0 > 0 {
            let indices = (0 ..< paddings.0).map { i -> Int in
                let idx = paddings.0 - i
                return min(idx, x.dim(1) - 1)
            }
            let slices = indices.map { x[0..., $0..<($0 + 1), 0...] }
            parts.append(MLX.concatenated(slices, axis: 1))
        }
        parts.append(x)
        if paddings.1 > 0 {
            let indices = (0 ..< paddings.1).map { i -> Int in
                max(x.dim(1) - 2 - i, 0)
            }
            let slices = indices.map { x[0..., $0..<($0 + 1), 0...] }
            parts.append(MLX.concatenated(slices, axis: 1))
        }
        return MLX.concatenated(parts, axis: 1)
    }
}

/// ConvTranspose1d using the GPU primitive `MLX.convTransposed1d` (weight `[out, k, in]`, NLC).
nonisolated final class EncodecConvTranspose1dPort: Module {
    let causal: Bool
    let trimRightRatio: Float
    let paddingTotal: Int

    @ModuleInfo(key: "conv") var conv: EncodecConvTransposed

    init(config: MusicGenEncodecConfig, inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self.causal = config.useCausalConv
        self.trimRightRatio = config.trimRightRatio
        self.paddingTotal = kernelSize - stride
        self._conv.wrappedValue = EncodecConvTransposed(inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, stride: stride)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv(x)
        let paddingRight = causal ? Int(ceil(Float(paddingTotal) * trimRightRatio)) : paddingTotal / 2
        let paddingLeft = paddingTotal - paddingRight
        let end = h.dim(1) - paddingRight
        if end > paddingLeft {
            h = h[0..., paddingLeft..<end, 0...]
        }
        return h
    }
}

/// Thin wrapper exposing `MLX.convTransposed1d` as a `Module` with `weight`/`bias` keys.
nonisolated final class EncodecConvTransposed: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    let inChannels: Int
    let outChannels: Int
    let kernelSize: Int
    let stride: Int

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int) {
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        let scale = sqrt(1.0 / Float(inChannels * kernelSize))
        self._weight.wrappedValue = MLXRandom.uniform(low: -scale, high: scale, [outChannels, kernelSize, inChannels])
        self._bias.wrappedValue = MLXArray.zeros([outChannels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = MLX.convTransposed1d(x, weight, stride: stride)
        y = y + bias.reshaped([1, 1, outChannels])
        return y
    }
}

/// 2-layer LSTM block with residual connection.
nonisolated final class EncodecLSTMBlockPort: Module {
    @ModuleInfo(key: "lstm") var lstm: [EncodecLSTMPort]

    init(config: MusicGenEncodecConfig, dimension: Int) {
        self._lstm.wrappedValue = (0 ..< config.numLSTMLayers).map { _ in EncodecLSTMPort(dimension: dimension) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in lstm { h = layer(h) }
        return h + x
    }
}

/// One LSTM layer. Runs the gate math vectorised per timestep (matmul on GPU).
nonisolated final class EncodecLSTMPort: Module {
    let hiddenSize: Int

    @ModuleInfo(key: "Wx") var Wx: MLXArray
    @ModuleInfo(key: "Wh") var Wh: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray

    init(dimension: Int) {
        self.hiddenSize = dimension
        self._Wx.wrappedValue = MLXArray.zeros([4 * dimension, dimension])
        self._Wh.wrappedValue = MLXArray.zeros([4 * dimension, dimension])
        self._bias.wrappedValue = MLXArray.zeros([4 * dimension])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [1, T, dim]; gates = x·Wxᵀ + b + h·Whᵀ, then per-timestep i/f/g/o.
        // `h` is the previous hidden output [1, dim]; it is projected to the 4·dim gate space
        // via `h @ Whᵀ` each step (matches the reference LSTM's state/cell shapes).
        let xProj = matmul(x, Wx.transposed()) + bias   // [1, T, 4·dim]
        let T = x.dim(1)
        let dim = hiddenSize
        var h: MLXArray = MLXArray.zeros([1, dim])
        var c: MLXArray = MLXArray.zeros([1, dim])
        var outs: [MLXArray] = []
        for t in 0 ..< T {
            let gates = xProj[0..., t, 0...] + matmul(h, Wh.transposed())   // [1, 4·dim]
            let i = sigmoid(gates[0..., 0..<dim])
            let f = sigmoid(gates[0..., dim..<(2 * dim)])
            let g = tanh(gates[0..., (2 * dim)..<(3 * dim)])
            let o = sigmoid(gates[0..., (3 * dim)...])
            c = f * c + i * g
            h = o * tanh(c)
            outs.append(h)
        }
        return MLX.stacked(outs, axis: 1)
    }
}

/// SEANet resnet block: ELU → conv(dilation) → ELU → conv(1), + identity residual.
/// `use_conv_shortcut: false` in the 32 kHz config → the checkpoint ships NO `shortcut` keys,
/// so the residual is a plain add (verified against the safetensors header).
nonisolated final class EncodecResnetBlockPort: Module {
    @ModuleInfo(key: "block") var block: [Module]

    init(config: MusicGenEncodecConfig, dim: Int, dilations: [Int]) {
        let kernelSizes = [config.residualKernelSize, 1]
        let hidden = dim / config.compress
        var layers: [Module] = []
        for (i, k) in kernelSizes.enumerated() {
            let inC = i == 0 ? dim : hidden
            let outC = i == kernelSizes.count - 1 ? dim : hidden
            layers.append(ELUPort())
            layers.append(EncodecConv1dPort(config: config, inChannels: inC, outChannels: outC, kernelSize: k, dilation: dilations[i]))
        }
        self._block.wrappedValue = layers
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let elu = layer as? ELUPort { h = elu(h) }
            else if let conv = layer as? EncodecConv1dPort { h = conv(h) }
        }
        return x + h
    }
}

nonisolated final class ELUPort: Module {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLX.where(x .> 0, x, 1.0 * (MLX.exp(x) - 1))
    }
}
