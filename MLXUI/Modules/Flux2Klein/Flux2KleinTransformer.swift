import Foundation
import MLX
import MLXNN

// FLUX.2 Klein MMDiT: the full forward and the streamed, quantize-at-load weight loader (MP7-3f).
// Structure follows VincentGourbin/flux-2-swift-mlx @ 315c27e (MIT, © 2025 Vincent Gourbin) `Flux2Transformer`; the numbers are checked against diffusers 0.37.1
// `Flux2Transformer2DModel` (tiny-config fp32 golden g3f_dit_stack.safetensors: 3 double + 3 single blocks, every block output, the final prediction).
// diffusers is the arbiter wherever the two disagree.
//
// Keys are the checkpoint's own diffusers names at the TOP LEVEL (no remap): the nine front-end tensors of `Flux2KleinEmbedders` (same module types, same names),
// `transformer_blocks.N.*` (5 x 16 tensors) and `single_transformer_blocks.N.*` (20 x 4 tensors). 169 tensors in all.
//
// Forward (diffusers `Flux2Transformer2DModel.forward`): embedders -> conditioning (temb, 2 + 2 + 1 modulation sets, shared by every block) -> x_embedder / context_embedder ->
// one joint [text, image] rotary table -> all double blocks (two streams) -> concat [text, image] -> all single blocks -> drop the text rows -> norm_out -> proj_out.
//
// Quantization (implementer's call, pending owner confirmation): see `Flux2KleinTransformer.quantizesFront(_:)`.

nonisolated enum Flux2KleinTransformerError: Error, Equatable, LocalizedError {
    case unsupportedBits(Int)
    case missingWeight(String)
    case unexpectedWeight(String)
    case quantizationFailed(String)
    case noWeightFiles(String)
    case badInput(String)

    var errorDescription: String? {
        switch self {
        case let .unsupportedBits(bits):
            return "FLUX.2 Klein can quantize its image model to 3, 4, 5, 6 or 8 bits; \(bits) is not one of them."
        case let .missingWeight(name):
            return "The FLUX.2 Klein image model is missing the weight \(name). Reinstall the model."
        case let .unexpectedWeight(name):
            return "The FLUX.2 Klein image model has a weight it does not know: \(name). The checkpoint is not the one this build expects."
        case let .quantizationFailed(what):
            return "FLUX.2 Klein could not quantize its image model (\(what))."
        case let .noWeightFiles(dir):
            return "No FLUX.2 Klein image-model weights (*.safetensors) were found in \(dir). Reinstall the model."
        case let .badInput(what):
            return "FLUX.2 Klein image model input: \(what)."
        }
    }
}

extension Flux2KleinQuantization {
    /// Widths MLX can quantize to (affine, group 64). 7 is not an MLX width.
    static let ditSupportedBits: Set<Int> = [3, 4, 5, 6, 8]
}

extension Flux2KleinDoubleBlock {
    /// The 16 tensors a double block owns, by their name inside the block.
    static let expectedTensorNames: Set<String> = [
        "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight",
        "attn.to_out.0.weight", "attn.to_add_out.weight",
        "attn.norm_q.weight", "attn.norm_k.weight", "attn.norm_added_q.weight", "attn.norm_added_k.weight",
        "ff.linear_in.weight", "ff.linear_out.weight", "ff_context.linear_in.weight", "ff_context.linear_out.weight",
    ]
}

/// Every dimension the transformer is built from. `init(_:)` takes the validated real config; the memberwise init exists for the tiny test configs.
nonisolated struct Flux2KleinTransformerShape: Equatable, Sendable {
    var embedder: Flux2KleinEmbedderShape
    var heads: Int
    var headDim: Int
    var mlpRatio: Double
    var doubleBlocks: Int
    var singleBlocks: Int

    var inner: Int { embedder.inner }

    init(_ c: Flux2KleinTransformerConfig) {
        self.init(embedder: Flux2KleinEmbedderShape(c), heads: c.numAttentionHeads, headDim: c.attentionHeadDim, mlpRatio: c.mlpRatio,
                  doubleBlocks: c.numLayers, singleBlocks: c.numSingleLayers)
    }

    init(embedder: Flux2KleinEmbedderShape, heads: Int, headDim: Int, mlpRatio: Double, doubleBlocks: Int, singleBlocks: Int) {
        self.embedder = embedder; self.heads = heads; self.headDim = headDim; self.mlpRatio = mlpRatio
        self.doubleBlocks = doubleBlocks; self.singleBlocks = singleBlocks
    }
}

nonisolated final class Flux2KleinTransformer: Module {
    let shape: Flux2KleinTransformerShape
    @ModuleInfo(key: "x_embedder") var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "time_guidance_embed") var timeGuidanceEmbed: Flux2KleinTimeEmbed
    @ModuleInfo(key: "double_stream_modulation_img") var doubleModulationImg: Flux2KleinModulation
    @ModuleInfo(key: "double_stream_modulation_txt") var doubleModulationTxt: Flux2KleinModulation
    @ModuleInfo(key: "single_stream_modulation") var singleModulation: Flux2KleinModulation
    @ModuleInfo(key: "norm_out") var normOut: Flux2KleinAdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "transformer_blocks") var doubleBlocks: [Flux2KleinDoubleBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleBlocks: [Flux2KleinSingleBlock]

    /// A model with default (lazy random) weights, or with pre-built blocks (the streaming loader passes its own, already quantized).
    init(shape s: Flux2KleinTransformerShape, doubleBlocks: [Flux2KleinDoubleBlock]? = nil, singleBlocks: [Flux2KleinSingleBlock]? = nil) {
        shape = s
        let e = s.embedder
        _xEmbedder.wrappedValue = Linear(e.inChannels, e.inner, bias: false)
        _contextEmbedder.wrappedValue = Linear(e.jointAttentionDim, e.inner, bias: false)
        _timeGuidanceEmbed.wrappedValue = Flux2KleinTimeEmbed(channels: e.timestepChannels, inner: e.inner)
        _doubleModulationImg.wrappedValue = Flux2KleinModulation(inner: e.inner, sets: 2)
        _doubleModulationTxt.wrappedValue = Flux2KleinModulation(inner: e.inner, sets: 2)
        _singleModulation.wrappedValue = Flux2KleinModulation(inner: e.inner, sets: 1)
        _normOut.wrappedValue = Flux2KleinAdaLayerNormContinuous(inner: e.inner, eps: e.eps)
        _projOut.wrappedValue = Linear(e.inner, e.inChannels, bias: false)
        _doubleBlocks.wrappedValue = doubleBlocks ?? (0 ..< s.doubleBlocks).map { _ in
            Flux2KleinDoubleBlock(dim: e.inner, heads: s.heads, headDim: s.headDim, mlpRatio: s.mlpRatio, eps: e.eps)
        }
        _singleBlocks.wrappedValue = singleBlocks ?? (0 ..< s.singleBlocks).map { _ in
            Flux2KleinSingleBlock(dim: e.inner, heads: s.heads, headDim: s.headDim, mlpRatio: s.mlpRatio, eps: e.eps)
        }
        super.init()
    }

    convenience init(_ c: Flux2KleinTransformerConfig) { self.init(shape: Flux2KleinTransformerShape(c)) }

    /// The nine tensors outside the blocks (same names as `Flux2KleinEmbedders`).
    static let frontTensorNames: Set<String> = [
        "x_embedder.weight", "context_embedder.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight", "time_guidance_embed.timestep_embedder.linear_2.weight",
        "double_stream_modulation_img.linear.weight", "double_stream_modulation_txt.linear.weight", "single_stream_modulation.linear.weight",
        "norm_out.linear.weight", "proj_out.weight",
    ]

    // MARK: Forward

    /// Called with a stage name and the (lazy) hidden state after each block: `double_N_txt`, `double_N_img`, `single_N` (the joint [text, image] stream) and `out`.
    /// nil in production. The probe may evaluate what it is given.
    typealias Probe = (String, MLXArray) -> Void

    /// `hidden` `[B, Si, in_channels]`, `encoder` `[B, St, joint_attention_dim]`, `sigma` `[B]` in [0, 1] (multiplied by 1000 inside), `imgIDs` `[Si, 4]` (or `[1, Si, 4]`),
    /// `txtIDs` `[St, 4]`. Returns the prediction `[B, Si, in_channels]`, EVALUATED (so it may cross an actor hop, rule 17b). The model computes in the dtype of `hidden`.
    func callAsFunction(hidden: MLXArray, encoder: MLXArray, sigma: MLXArray, imgIDs: MLXArray, txtIDs: MLXArray, probe: Probe? = nil) throws -> MLXArray {
        let e = shape.embedder
        guard hidden.ndim == 3, encoder.ndim == 3, hidden.dim(0) == encoder.dim(0), hidden.dim(2) == e.inChannels, encoder.dim(2) == e.jointAttentionDim else {
            throw Flux2KleinTransformerError.badInput("hidden must be [batch, tokens, \(e.inChannels)] and encoder [batch, rows, \(e.jointAttentionDim)] with the same batch")
        }
        let si = hidden.dim(1), st = encoder.dim(1)
        let imgRows = imgIDs.ndim == 3 ? imgIDs.dim(1) : imgIDs.dim(0), txtRows = txtIDs.ndim == 3 ? txtIDs.dim(1) : txtIDs.dim(0)
        guard imgRows == si, txtRows == st else {
            throw Flux2KleinTransformerError.badInput("ids must have one row per token (image \(imgRows) vs \(si), text \(txtRows) vs \(st))")
        }
        let dtype = hidden.dtype
        // Same pieces as Flux2KleinEmbedders.conditioning(sigma:dtype:), against this model's own front-end modules.
        let temb = timeGuidanceEmbed(sigma: sigma, dtype: dtype)
        let imgMod = doubleModulationImg.chunks(temb), txtMod = doubleModulationTxt.chunks(temb), singleMod = singleModulation.chunks(temb)
        var img = xEmbedder(hidden)
        var txt = contextEmbedder(encoder)
        let (cos, sin) = Flux2KleinRoPE.jointCosSin(textIDs: txtIDs, imageIDs: imgIDs, axesDims: e.axesDims, theta: e.ropeTheta)

        for (index, block) in doubleBlocks.enumerated() {
            try Task.checkCancellation()
            (txt, img) = block(img: img, txt: txt, imgMod: imgMod, txtMod: txtMod, cos: cos, sin: sin)
            probe?("double_\(index)_txt", txt)
            probe?("double_\(index)_img", img)
        }
        var joint = concatenated([txt, img], axis: 1)
        for (index, block) in singleBlocks.enumerated() {
            try Task.checkCancellation()
            joint = block(joint, mod: singleMod[0], cos: cos, sin: sin)
            probe?("single_\(index)", joint)
        }
        let imageRows = MLX.split(joint, indices: [st], axis: 1)[1]
        let out = projOut(normOut(imageRows, temb: temb))
        probe?("out", out)
        eval(out)
        return out
    }

    // MARK: Memory accounting

    /// Sum of the bytes of every parameter whose flattened name passes `where` (a quantized layer counts its packed weight, scales and biases).
    func parameterBytes(where include: (String) -> Bool = { _ in true }) -> Int {
        parameters().flattened().reduce(0) { include($1.0) ? $0 + $1.1.nbytes : $0 }
    }

    // MARK: Quantization policy

    /// Which front-end Linears are quantized (implementer's call, pending owner confirmation). `path` is the layer's module path, e.g. `norm_out.linear`.
    /// Quantized: `context_embedder` (7680 -> 3072), the three modulation Linears (142 M parameters, one matmul per forward) and `norm_out.linear`; all of the
    /// 25 blocks' Linears are quantized too. Kept in the checkpoint dtype: `x_embedder`, the two timestep-MLP Linears and `proj_out` (together 11 M parameters, 22 MB in
    /// bf16), which sit at the input, the time conditioning and the output and are the layers most likely to move the result for the fewest bytes saved.
    static func quantizesFront(_ path: String) -> Bool {
        ["context_embedder", "double_stream_modulation_img.linear", "double_stream_modulation_txt.linear",
         "single_stream_modulation.linear", "norm_out.linear"].contains(path)
    }

    // MARK: Loading

    /// Where the weights come from: one closure per shard, called in order. Each returns the (lazy) tensors of that file.
    typealias Shard = () throws -> [String: MLXArray]

    /// Production entry: loads `transformer/` of an installed Klein model at `bits` (3, 4, 5, 6 or 8; affine, group size 64). Needs `config.json` and the `*.safetensors`
    /// file(s); it does NOT need an index file, so block membership is read from the tensors' own names. There is no unquantized production path.
    static func load(transformerDirectory dir: URL, bits: Int, groupSize: Int = 64, stage: (String) -> Void = { _ in }) throws -> Flux2KleinTransformer {
        try load(transformerDirectory: dir, optionalBits: bits, groupSize: groupSize, stage: stage)
    }

    /// Test/measurement-only: the checkpoint dtype (bf16), nothing quantized. Not for the engine.
    static func loadUnquantizedForTesting(transformerDirectory dir: URL, stage: (String) -> Void = { _ in }) throws -> Flux2KleinTransformer {
        try load(transformerDirectory: dir, optionalBits: nil, groupSize: 64, stage: stage)
    }

    private static func load(transformerDirectory dir: URL, optionalBits: Int?, groupSize: Int, stage: (String) -> Void) throws -> Flux2KleinTransformer {
        let config = try Flux2KleinTransformerConfig.decode(from: Data(contentsOf: dir.appendingPathComponent("config.json")))
        return try load(shape: Flux2KleinTransformerShape(config), shards: try weightShards(in: dir), bits: optionalBits, groupSize: groupSize, stage: stage)
    }

    /// One lazy shard per `*.safetensors` file of the directory, in file-name order (single file or sharded; no index file is read).
    static func weightShards(in dir: URL) throws -> [Shard] {
        let files = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw Flux2KleinTransformerError.noWeightFiles(dir.path) }
        return files.map { url in { try loadArrays(url: url) } }
    }

    private enum Slot { case front(String), double(Int, String), single(Int, String) }

    private static func classify(_ key: String, in shape: Flux2KleinTransformerShape) -> Slot? {
        if frontTensorNames.contains(key) { return .front(key) }
        for (prefix, isDouble) in [("transformer_blocks.", true), ("single_transformer_blocks.", false)] where key.hasPrefix(prefix) {
            let rest = key.dropFirst(prefix.count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else { return nil }
            let local = String(rest[rest.index(after: dot)...])
            return isDouble ? .double(index, local) : .single(index, local)
        }
        return nil
    }

    /// The streaming loader. Tensors are matched by name. A block is built the moment all of its tensors have arrived (so a block that straddles two shards waits for its
    /// second half), quantized right away if `bits` is set, evaluated, and its bf16 originals dropped (`clearCache`); the bf16 model is never fully resident.
    /// Missing tensors throw by name at the end; an unknown, renamed, duplicated or out-of-range tensor throws by name as soon as it arrives. Cancellation is checked per shard
    /// and per block. A layer that cannot be quantized throws (no silent bf16 fallback). The test-facing entry takes `bits: nil` (unquantized).
    static func load(shape: Flux2KleinTransformerShape, shards: [Shard], bits: Int?, groupSize: Int = 64,
                     stage: (String) -> Void = { _ in }) throws -> Flux2KleinTransformer {
        if let bits, !Flux2KleinQuantization.ditSupportedBits.contains(bits) { throw Flux2KleinTransformerError.unsupportedBits(bits) }
        let e = shape.embedder
        var front: [String: MLXArray] = [:]
        var pendingDouble: [Int: [String: MLXArray]] = [:], pendingSingle: [Int: [String: MLXArray]] = [:]
        var doubles: [Flux2KleinDoubleBlock?] = Array(repeating: nil, count: shape.doubleBlocks)
        var singles: [Flux2KleinSingleBlock?] = Array(repeating: nil, count: shape.singleBlocks)

        /// Quantize a module's Linears whose (module-local) path passes `filter`; throws for any that cannot be (indivisible width) or that stay plain.
        func quantizeChecked(_ module: Module, label: String, filter: (String) -> Bool) throws {
            guard let bits else { return }
            for (path, leaf) in module.leafModules().flattened() where filter(path) {
                guard let linear = leaf as? Linear, !(leaf is QuantizedLinear) else { continue }
                guard linear.weight.dim(-1) % groupSize == 0 else {
                    throw Flux2KleinTransformerError.quantizationFailed("\(label)\(path): input width \(linear.weight.dim(-1)) is not a multiple of the group size \(groupSize)")
                }
            }
            quantize(model: module, groupSize: groupSize, bits: bits, filter: { path, _ in filter(path) })
            for (path, leaf) in module.leafModules().flattened() where filter(path) && leaf is Linear && !(leaf is QuantizedLinear) {
                throw Flux2KleinTransformerError.quantizationFailed("\(label)\(path), \(bits) bits, group \(groupSize)")
            }
        }
        func finish(_ module: Module) {
            eval(module.parameters().flattened().map { $0.1 })
            Memory.clearCache()
        }

        for (shardIndex, shard) in shards.enumerated() {
            try Task.checkCancellation()
            var tensors = try shard()
            stage("shard \(shardIndex) opened (\(tensors.count) tensors)")
            for key in tensors.keys.sorted() {
                guard let slot = classify(key, in: shape), let tensor = tensors.removeValue(forKey: key) else {
                    throw Flux2KleinTransformerError.unexpectedWeight(key)
                }
                switch slot {
                case let .front(name):
                    guard front[name] == nil else { throw Flux2KleinTransformerError.unexpectedWeight("\(key) (duplicate)") }
                    front[name] = tensor
                case let .double(index, local):
                    guard index < shape.doubleBlocks, Flux2KleinDoubleBlock.expectedTensorNames.contains(local), doubles[index] == nil,
                          pendingDouble[index]?[local] == nil else { throw Flux2KleinTransformerError.unexpectedWeight(key) }
                    pendingDouble[index, default: [:]][local] = tensor
                case let .single(index, local):
                    guard index < shape.singleBlocks, Flux2KleinSingleBlock.expectedTensorNames.contains(local), singles[index] == nil,
                          pendingSingle[index]?[local] == nil else { throw Flux2KleinTransformerError.unexpectedWeight(key) }
                    pendingSingle[index, default: [:]][local] = tensor
                }
            }
            tensors.removeAll()

            for index in pendingDouble.keys.sorted() where pendingDouble[index]?.count == Flux2KleinDoubleBlock.expectedTensorNames.count {
                try Task.checkCancellation()
                let block = Flux2KleinDoubleBlock(dim: e.inner, heads: shape.heads, headDim: shape.headDim, mlpRatio: shape.mlpRatio, eps: e.eps)
                try block.update(parameters: ModuleParameters.unflattened(pendingDouble[index] ?? [:]), verify: .all)
                pendingDouble.removeValue(forKey: index)
                try quantizeChecked(block, label: "transformer_blocks.\(index).") { _ in true }
                finish(block)
                doubles[index] = block
                stage("double block \(index) ready")
            }
            for index in pendingSingle.keys.sorted() where pendingSingle[index]?.count == Flux2KleinSingleBlock.expectedTensorNames.count {
                try Task.checkCancellation()
                let block = Flux2KleinSingleBlock(dim: e.inner, heads: shape.heads, headDim: shape.headDim, mlpRatio: shape.mlpRatio, eps: e.eps)
                try block.update(parameters: ModuleParameters.unflattened(pendingSingle[index] ?? [:]), verify: .all)
                pendingSingle.removeValue(forKey: index)
                try quantizeChecked(block, label: "single_transformer_blocks.\(index).") { _ in true }
                finish(block)
                singles[index] = block
                stage("single block \(index) ready")
            }
        }

        // Completeness, by name.
        var missing = frontTensorNames.subtracting(front.keys).sorted()
        for (index, block) in doubles.enumerated() where block == nil {
            let have = Set(pendingDouble[index]?.keys.map { $0 } ?? [])
            missing += Flux2KleinDoubleBlock.expectedTensorNames.subtracting(have).sorted().map { "transformer_blocks.\(index).\($0)" }
        }
        for (index, block) in singles.enumerated() where block == nil {
            let have = Set(pendingSingle[index]?.keys.map { $0 } ?? [])
            missing += Flux2KleinSingleBlock.expectedTensorNames.subtracting(have).sorted().map { "single_transformer_blocks.\(index).\($0)" }
        }
        if !missing.isEmpty { throw Flux2KleinTransformerError.missingWeight(missing.joined(separator: ", ")) }

        try Task.checkCancellation()
        let model = Flux2KleinTransformer(shape: shape, doubleBlocks: doubles.compactMap { $0 }, singleBlocks: singles.compactMap { $0 })
        try model.update(parameters: ModuleParameters.unflattened(front), verify: [.noUnusedKeys, .shapeMismatch])
        try quantizeChecked(model, label: "") { Flux2KleinTransformer.quantizesFront($0) }
        finish(model)
        stage("front end ready")
        return model
    }
}
