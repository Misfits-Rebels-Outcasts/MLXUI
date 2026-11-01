import Foundation
import MLX
import MLXNN

// FLUX.2 Klein VAE decoder (MP7-3g). `AutoencoderKLFlux2` of diffusers 0.37.1, decode half only, in the checkpoint's diffusers key names:
//   post_quant_conv, decoder.{conv_in, mid_block.{resnets.0, attentions.0, resnets.1}, up_blocks.N.{resnets.M, upsamplers.0.conv}, conv_norm_out, conv_out}
// plus the two BatchNorm vectors bn.running_mean / bn.running_var ([128]) that denormalise the packed latents. The encoder, quant_conv and
// bn.num_batches_tracked in the checkpoint are not built and are ignored by name when loading; any other unexpected tensor throws.
//
// Structure follows Modules/Flux/FluxVAE.swift (not edited): GroupNorm(eps 1e-6, pytorch-compatible), resnets, nearest-2x upsample + 3x3 conv, NHWC inside.
// Differences from FluxVAE, each checked against diffusers 0.37.1 and the installed header: conv_in takes `latent_channels` (32) not 16; there is a 1x1
// post_quant_conv; there is NO scale/shift division (the BatchNorm denormalise on the 128-channel latent replaces it); the mid-block attention Linears are plain
// bf16 Linear WITH bias (keys to_q / to_k / to_v / to_out.0); keys are diffusers' (conv_in not conv_in.conv2d); config-driven so a tiny config builds the same class.
// The 3a glue (Flux2KleinLatents.denormalize / unpatchify) is reused, not duplicated.

nonisolated enum Flux2KleinVAEError: Error, Equatable, LocalizedError {
    case noWeightFiles(String)
    case missingWeight(String)
    case unexpectedWeight(String)
    case badInput(String)

    var errorDescription: String? {
        switch self {
        case let .noWeightFiles(path): return "No .safetensors file in \(path)."
        case let .missingWeight(names): return "The FLUX.2 Klein VAE is missing: \(names)."
        case let .unexpectedWeight(name): return "The FLUX.2 Klein VAE has a tensor this decoder does not know: \(name)."
        case let .badInput(why): return "FLUX.2 Klein VAE decode: \(why)."
        }
    }
}

/// The dimensions the decoder is built from. `klein4B` is the checkpoint; any other value is for tests (the tiny golden).
nonisolated struct Flux2KleinVAEShape: Equatable, Sendable {
    var blockOutChannels: [Int]
    var layersPerBlock: Int
    var groups: Int
    var latentChannels: Int
    var outChannels: Int = 3
    var batchNormEps: Double = Flux2KleinDims.vaeBatchNormEps
    var patch: Int = Flux2KleinDims.vaePatch

    var packedChannels: Int { latentChannels * patch * patch }

    static let klein4B = Flux2KleinVAEShape(blockOutChannels: [128, 256, 512, 512], layersPerBlock: 2, groups: 32, latentChannels: Flux2KleinDims.latentChannels)

    init(blockOutChannels: [Int], layersPerBlock: Int, groups: Int, latentChannels: Int, outChannels: Int = 3, batchNormEps: Double = Flux2KleinDims.vaeBatchNormEps,
         patch: Int = Flux2KleinDims.vaePatch) {
        self.blockOutChannels = blockOutChannels; self.layersPerBlock = layersPerBlock; self.groups = groups; self.latentChannels = latentChannels
        self.outChannels = outChannels; self.batchNormEps = batchNormEps; self.patch = patch
    }

    /// From a decoded vae/config.json. Production configs are validated by `Flux2KleinVAEConfig.decode` first; the tiny test configs are not.
    init(_ c: Flux2KleinVAEConfig) {
        self.init(blockOutChannels: c.blockOutChannels, layersPerBlock: c.layersPerBlock, groups: c.normNumGroups, latentChannels: c.latentChannels,
                  outChannels: c.outChannels, batchNormEps: c.batchNormEps, patch: c.patchSize.first ?? 2)
    }
}

// MARK: - Modules

nonisolated final class Flux2KleinVAEResnet: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    init(inChannels: Int, outChannels: Int, groups: Int) {
        self._norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inChannels, eps: 1e-6, affine: true, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outChannels, eps: 1e-6, affine: true, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        self._convShortcut.wrappedValue = inChannels != outChannels ? Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(silu(norm1(x)))
        h = conv2(silu(norm2(h)))
        if let convShortcut { return convShortcut(x) + h }
        return x + h
    }
}

/// diffusers `Attention` as used in the VAE mid block: GroupNorm -> q/k/v (plain Linear with bias) -> single-head SDPA over all pixels -> to_out.0 -> residual.
nonisolated final class Flux2KleinVAEAttention: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]

    let channels: Int

    init(channels: Int, groups: Int) {
        self.channels = channels
        self._groupNorm.wrappedValue = GroupNorm(groupCount: groups, dimensions: channels, eps: 1e-6, affine: true, pytorchCompatible: true)
        self._toQ.wrappedValue = Linear(channels, channels, bias: true)
        self._toK.wrappedValue = Linear(channels, channels, bias: true)
        self._toV.wrappedValue = Linear(channels, channels, bias: true)
        self._toOut.wrappedValue = [Linear(channels, channels, bias: true)]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w) = (x.dim(0), x.dim(1), x.dim(2))
        let y = groupNorm(x)
        let q = toQ(y).reshaped([b, 1, h * w, channels])
        let k = toK(y).reshaped([b, 1, h * w, channels])
        let v = toV(y).reshaped([b, 1, h * w, channels])
        let scale = 1.0 / Float(channels).squareRoot()
        let out = scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let projected = toOut[0](out.reshaped([b, h, w, channels]))
        return x + projected
    }
}

nonisolated final class Flux2KleinVAEMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2KleinVAEResnet]
    @ModuleInfo(key: "attentions") var attentions: [Flux2KleinVAEAttention]

    init(channels: Int, groups: Int) {
        self._resnets.wrappedValue = [Flux2KleinVAEResnet(inChannels: channels, outChannels: channels, groups: groups),
                                      Flux2KleinVAEResnet(inChannels: channels, outChannels: channels, groups: groups)]
        self._attentions.wrappedValue = [Flux2KleinVAEAttention(channels: channels, groups: groups)]
        super.init()
    }

    var evalBetweenLayers = true

    /// diffusers `UNetMidBlock2D` order: resnets[0], then attention, then resnets[1].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = resnets[0](x)
        if evalBetweenLayers { eval(h) }
        h = attentions[0](h)
        if evalBetweenLayers { eval(h) }
        return resnets[1](h)
    }
}

nonisolated final class Flux2KleinVAEUpsampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        self._conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(MLXNN.Upsample(scaleFactor: 2.0, mode: .nearest)(x)) }
}

nonisolated final class Flux2KleinVAEUpBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2KleinVAEResnet]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Flux2KleinVAEUpsampler]

    init(inChannels: Int, outChannels: Int, resnetCount: Int, groups: Int, hasUpsampler: Bool) {
        self._resnets.wrappedValue = (0 ..< resnetCount).map {
            Flux2KleinVAEResnet(inChannels: $0 == 0 ? inChannels : outChannels, outChannels: outChannels, groups: groups)
        }
        self._upsamplers.wrappedValue = hasUpsampler ? [Flux2KleinVAEUpsampler(channels: outChannels)] : []
        super.init()
    }

    var evalBetweenLayers = true

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for r in resnets {
            h = r(h)
            if evalBetweenLayers { eval(h) }
        }
        if let up = upsamplers.first {
            h = up(h)
            if evalBetweenLayers { eval(h) }
        }
        return h
    }
}

nonisolated final class Flux2KleinVAEDecoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: Flux2KleinVAEMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [Flux2KleinVAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    init(_ s: Flux2KleinVAEShape) {
        let reversed = Array(s.blockOutChannels.reversed())
        let top = reversed[0]
        self._convIn.wrappedValue = Conv2d(inputChannels: s.latentChannels, outputChannels: top, kernelSize: 3, padding: 1)
        self._midBlock.wrappedValue = Flux2KleinVAEMidBlock(channels: top, groups: s.groups)
        var blocks: [Flux2KleinVAEUpBlock] = []
        var previous = top
        for (i, out) in reversed.enumerated() {
            blocks.append(Flux2KleinVAEUpBlock(inChannels: previous, outChannels: out, resnetCount: s.layersPerBlock + 1, groups: s.groups, hasUpsampler: i < reversed.count - 1))
            previous = out
        }
        self._upBlocks.wrappedValue = blocks
        self._convNormOut.wrappedValue = GroupNorm(groupCount: s.groups, dimensions: s.blockOutChannels[0], eps: 1e-6, affine: true, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(inputChannels: s.blockOutChannels[0], outputChannels: s.outChannels, kernelSize: 3, padding: 1)
        super.init()
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = midBlock(convIn(z))
        for block in upBlocks { h = block(h) }
        return convOut(silu(convNormOut(h)))
    }
}

/// `bn.running_mean` / `bn.running_var`: the 128-channel statistics `AutoencoderKLFlux2` keeps for the packed latents.
nonisolated final class Flux2KleinVAEBatchNormStats: Module {
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    init(channels: Int) {
        self._runningMean.wrappedValue = MLXArray.zeros([channels])
        self._runningVar.wrappedValue = MLXArray.ones([channels])
        super.init()
    }
}

// MARK: - The decoder

nonisolated final class Flux2KleinVAE: Module {
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "decoder") var decoder: Flux2KleinVAEDecoderNet
    @ModuleInfo(key: "bn") var bn: Flux2KleinVAEBatchNormStats

    let shape: Flux2KleinVAEShape

    init(_ shape: Flux2KleinVAEShape) {
        self.shape = shape
        self._postQuantConv.wrappedValue = Conv2d(inputChannels: shape.latentChannels, outputChannels: shape.latentChannels, kernelSize: 1)
        self._decoder.wrappedValue = Flux2KleinVAEDecoderNet(shape)
        self._bn.wrappedValue = Flux2KleinVAEBatchNormStats(channels: shape.packedChannels)
        super.init()
    }

    convenience init(_ config: Flux2KleinVAEConfig) { self.init(Flux2KleinVAEShape(config)) }

    /// Decode already-unpatchified latents `[B, latent, H/8, W/8]` (NCHW, no denormalisation) to an image `[B, 3, H, W]`, float32, roughly [-1, 1].
    /// Computes in the weights' dtype (see `load(..., computeDType:)`); evaluates the result before returning (rule 17b).
    func decode(latents z: MLXArray) throws -> MLXArray {
        guard z.ndim == 4, z.dim(1) == shape.latentChannels else {
            throw Flux2KleinVAEError.badInput("expected latents [B, \(shape.latentChannels), h, w], got \(z.shape)")
        }
        let nhwc = z.asType(computeDType).transposed(0, 2, 3, 1)
        let image = decoder(postQuantConv(nhwc)).transposed(0, 3, 1, 2).asType(.float32)
        try MLXErrorGuard.evaluate(image)
        return image
    }

    /// The pipeline's tail: packed, still-normalised latents `[B, 4*latent, H/16, W/16]` (what the denoise loop leaves after unpacking the sequence) ->
    /// BatchNorm denormalise (`x * sqrt(var + eps) + mean`, in float32; implementer's call: diffusers does it in the latents' dtype) -> unpatchify -> decode.
    func decode(packedLatents x: MLXArray) throws -> MLXArray {
        guard x.ndim == 4, x.dim(1) == shape.packedChannels else {
            throw Flux2KleinVAEError.badInput("expected packed latents [B, \(shape.packedChannels), h, w], got \(x.shape)")
        }
        let denorm = Flux2KleinLatents.denormalize(x.asType(.float32), mean: bn.runningMean.asType(.float32), variance: bn.runningVar.asType(.float32),
                                                   eps: shape.batchNormEps)
        return try decode(latents: Flux2KleinLatents.unpatchify(denorm))
    }

    /// The dtype the convolutions run in (the weights' dtype).
    var computeDType: DType { postQuantConv.weight.dtype }

    /// Evaluate after every resnet, attention and upsampler instead of once at the end, so MLX frees each layer's temporaries sooner. ON by default (implementer's call, pending
    /// owner confirmation): measured at 1024 x 1024 bf16, MLX peak 7.24 GB -> 5.74 GB at the same speed within noise (MP7-3g-report.md). Same result.
    var evaluatesBetweenLayers: Bool { decoder.midBlock.evalBetweenLayers }

    func setEvaluatesBetweenLayers(_ on: Bool) {
        decoder.midBlock.evalBetweenLayers = on
        for b in decoder.upBlocks { b.evalBetweenLayers = on }
    }

    // MARK: Parameter accounting

    func parameterBytes() -> Int { parameters().flattened().reduce(0) { $0 + $1.1.nbytes } }

    // MARK: Loading

    /// Names the loader ignores (they are in the checkpoint, the decoder does not need them): the whole encoder, quant_conv, and the int64 BatchNorm step counter.
    static func isIgnoredName(_ key: String) -> Bool {
        key.hasPrefix("encoder.") || key.hasPrefix("quant_conv.") || key == "bn.num_batches_tracked"
    }

    /// Production entry: `vae/` of an installed Klein model. `computeDType` nil (the default) keeps the checkpoint dtype (bf16); `.float32` upcasts every weight at load.
    /// Default bf16 is the implementer's call, pending owner confirmation, from MP7-3g-report.md: on the real weights bf16 compute differs from fp32 compute by about
    /// 49-50 dB PSNR (random latents) while fp32 doubles the decode's peak memory (1024 x 1024: 5.7 GB bf16 vs about 11 GB fp32, extrapolated) and takes about 1.8x as long.
    /// diffusers' `vae/config.json` says `force_upcast: true` (an SDXL-pipeline flag; Flux2KleinPipeline decodes in the latents' dtype).
    static func load(vaeDirectory dir: URL, computeDType: DType? = nil) throws -> Flux2KleinVAE {
        try load(vaeDirectory: dir, computeDType: computeDType, config: Flux2KleinVAEConfig.decode(from: Data(contentsOf: dir.appendingPathComponent("config.json"))))
    }

    /// Test-only: the same directory loader, but the config is decoded without the Klein 4B dimension check (so a tiny config can be used).
    static func loadForTesting(vaeDirectory dir: URL, computeDType: DType?) throws -> Flux2KleinVAE {
        let config = try JSONDecoder().decode(Flux2KleinVAEConfig.self, from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        return try load(vaeDirectory: dir, computeDType: computeDType, config: config)
    }

    private static func load(vaeDirectory dir: URL, computeDType: DType?, config: Flux2KleinVAEConfig) throws -> Flux2KleinVAE {
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Flux2KleinVAEError.noWeightFiles(dir.path) }
        var tensors: [String: MLXArray] = [:]
        for file in files {
            for (k, v) in try loadArrays(url: file) {
                guard tensors.updateValue(v, forKey: k) == nil else { throw Flux2KleinVAEError.unexpectedWeight("\(k) (duplicate)") }
            }
        }
        return try load(shape: Flux2KleinVAEShape(config), tensors: tensors, computeDType: computeDType)
    }

    /// Loader core (tests call it with tiny tensors). Tensors are in diffusers layout; every 4-D `.weight` is a convolution kernel `[O, I, kH, kW]` and is
    /// transposed to MLX's `[O, kH, kW, I]`. Unexpected tensors throw by name (the encoder / quant_conv / step counter are skipped); missing ones throw by name.
    /// Floating tensors are cast to `computeDType` when it is set; nothing is quantized.
    static func load(shape: Flux2KleinVAEShape, tensors: [String: MLXArray], computeDType: DType?) throws -> Flux2KleinVAE {
        let model = Flux2KleinVAE(shape)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        var used: [String: MLXArray] = [:]
        for key in tensors.keys.sorted() where !isIgnoredName(key) {
            guard let want = expected[key], let tensor = tensors[key] else { throw Flux2KleinVAEError.unexpectedWeight(key) }
            var t = tensor
            if key.hasSuffix(".weight"), t.ndim == 4 { t = t.transposed(0, 2, 3, 1) }
            if let computeDType, t.dtype.isFloatingPoint, t.dtype != computeDType { t = t.asType(computeDType) }
            guard t.shape == want else {
                throw Flux2KleinVAEError.unexpectedWeight("\(key) has shape \(tensor.shape) (as loaded \(t.shape)), the decoder expects \(want)")
            }
            used[key] = t
        }
        let missing = expected.keys.filter { used[$0] == nil }.sorted()
        if !missing.isEmpty { throw Flux2KleinVAEError.missingWeight(missing.joined(separator: ", ")) }
        try model.update(parameters: ModuleParameters.unflattened(used), verify: [.noUnusedKeys, .shapeMismatch])
        try MLXErrorGuard.evaluate(model.parameters().flattened().map { $0.1 })
        Memory.clearCache()
        return model
    }
}
