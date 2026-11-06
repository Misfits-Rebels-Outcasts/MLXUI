// Qwen3-VL forward pass up to the final norm — a trimmed copy of mlx-swift-lm's
// `Libraries/MLXVLM/Models/Qwen3VL.swift`.
//
//   Source: ml-explore/mlx-swift-lm, version 3.32.3, commit 3b339ad
//   Licence: MIT (Copyright © 2025 Apple Inc.)
//   Copied for MC4: Qwen3-VL Embedding 2B (mlx-community/Qwen3-VL-Embedding-2B-4bit).
//
// The package's Qwen3VLVision and Qwen3VLLanguage are internal, and the model returns logits;
// this file keeps the vision tower, the language model and the multimodal fusion, and returns the
// **final-norm hidden states**. Trimmed from upstream:
//   - no generation: no `lm_head`, no logits, no KV cache, no `LanguageModel` generation path;
//   - no video path, no windowed/prefill continuation;
//   - `QwenVL.rotateHalf` (internal to MLXVLM) is spelled `Qwen3VLVision.rotateHalf`, the same
//     function, which lives in this file;
//   - `getRopeIndex` throws instead of reading arrays bare (`MLXErrorGuard`, WW2-3-8).
// Everything else — the vision tower, DeepStack (blocks 5/11/17), interleaved M-RoPE — is
// upstream's code unchanged (issue Trap 3). Keep upstream names so a future diff against a new
// package version stays readable.

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXVLM

// MARK: - Vision

nonisolated enum Qwen3VLVision {

    private static func geluApproximation(for hiddenAct: String) -> GELU.Approximation {
        switch hiddenAct {
        case "gelu_pytorch_tanh", "gelu_new":
            .tanh
        case "gelu_fast":
            .fast
        default:
            .none
        }
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0 ..< half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    /// The fused SDPA Metal kernel supports head dims 64/80/128; the vision tower's is 72
    /// (1152/16), which otherwise falls back to composed ops that materialize the
    /// [L, L] score matrix per head. Zero-padding Q/K/V to 80 preserves the math exactly
    /// (padded dims contribute nothing to the dot products; `scale` is passed explicitly)
    /// and dispatches the O(L)-memory fused kernel. Same trick as `gemma4EnsureFusedSDPA`
    /// and mlx-vlm's Python `ensure_fused_sdpa`.
    fileprivate static func ensureFusedSDPA(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> MLXArray {
        let fusedDims = [64, 80, 128]
        let headDim = queries.dim(queries.ndim - 1)
        let target = fusedDims.first(where: { headDim <= $0 }) ?? headDim

        if target == headDim {
            return MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none)
        }

        let widths: [IntOrPair] = [0, 0, 0, .init((0, target - headDim))]
        return MLXFast.scaledDotProductAttention(
            queries: MLX.padded(queries, widths: widths),
            keys: MLX.padded(keys, widths: widths),
            values: MLX.padded(values, widths: widths),
            scale: scale, mask: .none
        )[.ellipsis, ..<headDim]
    }

    nonisolated final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension)
                )
            let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    nonisolated final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    nonisolated final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: Qwen3VLConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            self.hiddenSize =
                config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    nonisolated final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) throws -> MLXArray {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            // Attention is block-diagonal over the images in `cuSeqlens`, so each image is
            // attended independently instead of materializing a dense [L, L] additive mask
            // over the joint sequence (memory O((sum L_i)^2) for math that never crosses an
            // image boundary). Mirrors mlx-vlm's Python implementation.
            let seqlens = try MLXErrorGuard.read(cuSeqlens, as: Int.self)
            var attendedParts: [MLXArray] = []
            for idx in 1 ..< seqlens.count {
                let start = seqlens[idx - 1]
                let end = seqlens[idx]
                guard end > start else { continue }
                attendedParts.append(
                    Qwen3VLVision.ensureFusedSDPA(
                        queries: queries[0..., 0..., start ..< end, 0...],
                        keys: keys[0..., 0..., start ..< end, 0...],
                        values: values[0..., 0..., start ..< end, 0...],
                        scale: scale))
            }
            let attended = concatenated(attendedParts, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(sequenceLength, -1)

            return proj(attended)
        }
    }

    nonisolated final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int, hiddenAct: String) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            _activation.wrappedValue = GELU(
                approximation: Qwen3VLVision.geluApproximation(for: hiddenAct))
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    nonisolated final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(
                dim: config.hiddenSize,
                hiddenDim: config.intermediateSize,
                hiddenAct: config.hiddenAct)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) throws -> MLXArray {
            var states = hiddenStates
            states =
                states + (try attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb))
            states = states + mlp(norm2(states))
            return states
        }
    }

    nonisolated final class VisionModel: Module {

        let config: Qwen3VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: Qwen3VLConfiguration.VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            self.deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize)

            _blocks.wrappedValue = (0 ..< config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)

            let merge = spatialMergeSize
            var allCoords: [MLXArray] = []
            let mergeScalar = MLXArray(Int32(merge))

            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge

                guard mergedH > 0, mergedW > 0 else { continue }

                // Generate block and intra-block indices fully in MLX
                var blockRows = MLXArray(0 ..< mergedH).asType(.int32)
                blockRows = blockRows.reshaped([mergedH, 1, 1, 1])

                var blockCols = MLXArray(0 ..< mergedW).asType(.int32)
                blockCols = blockCols.reshaped([1, mergedW, 1, 1])

                let intra = MLXArray(0 ..< merge).asType(.int32)
                let intraRow = intra.reshaped([1, 1, merge, 1])
                let intraCol = intra.reshaped([1, 1, 1, merge])

                // Broadcast arithmetic mirrors the Python implementation
                var hIndex = blockRows * mergeScalar + intraRow
                var wIndex = blockCols * mergeScalar + intraCol

                hIndex = broadcast(hIndex, to: [mergedH, mergedW, merge, merge])
                wIndex = broadcast(wIndex, to: [mergedH, mergedW, merge, merge])

                // Flatten and stack coordinate pairs
                let hFlattened = hIndex.flattened()
                let wFlattened = wIndex.flattened()
                var coords = stacked([hFlattened, wFlattened], axis: -1)

                // Repeat for temporal frames
                if grid.t > 1 {
                    coords = tiled(coords, repetitions: [grid.t, 1])
                }

                allCoords.append(coords)
            }

            guard !allCoords.isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }

            // Concatenate all coordinate pairs
            let allCoordsConcat = concatenated(allCoords, axis: 0)  // (total_tokens, 2)

            // Extract h and w indices and lookup embeddings
            let hIndices = allCoordsConcat[0..., 0].asType(.int32)
            let wIndices = allCoordsConcat[0..., 1].asType(.int32)

            let hEmbeds = freqTable[hIndices, 0...]
            let wEmbeds = freqTable[wIndices, 0...]

            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        private func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1

            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []

            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)

                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0 ..< h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))

                var wLinspace = MLXArray(0 ..< w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))

                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)

                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)

                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1)  // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0)  // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)

                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide

                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())

                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)

                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }

            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: posEmbed.weight.dtype)
            }

            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map {
                concatenated($0, axis: 0).asType(posEmbed.weight.dtype)
            }

            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros(
                [totalPatches, hiddenSize], dtype: posEmbed.weight.dtype)

            for cornerIdx in 0 ..< 4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted =
                    cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }

            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0

            for size in gridSizes {
                let slice = patchPosEmbeds[offset ..< (offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }

            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize

            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t

                let featureDim = posEmbed.dim(-1)

                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])

                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)

                resultEmbeds.append(temporalEmbeds)
            }

            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var seqLengths: [MLXArray] = []

            for grid in grids {
                let perFrame = grid.h * grid.w
                let repeated = tiled(MLXArray(perFrame), repetitions: [grid.t])
                seqLengths.append(repeated)
            }

            guard !seqLengths.isEmpty else {
                return MLXArray(0, dtype: .int32)
            }

            let concatSeqLengths = concatenated(seqLengths).asType(.int32)

            let cumSum = concatSeqLengths.cumsum()

            return padded(
                cumSum, widths: [IntOrPair((1, 0))], mode: .constant,
                value: MLXArray(0, dtype: cumSum.dtype))
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) throws -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)

            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                hiddenStates = try block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5 && value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}
// MARK: - Language

nonisolated enum Qwen3VLLanguage {

    nonisolated final class RotaryEmbedding {

        private let invFreq: MLXArray
        private let mropeIndices: MLXArray

        init(headDim: Int, base: Double, ropeScaling: Qwen3VLConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            self.invFreq = 1.0 / pow(baseArray, freq)
            let sections = ropeScaling?.mropeSection ?? [24, 20, 20]
            var indices = [Int32](repeating: 0, count: freq.dim(0))
            for (dimension, offset) in [(1, 1), (2, 2)] {
                let end = min(sections[dimension] * 3, indices.count)
                for index in stride(from: offset, to: end, by: 3) {
                    indices[index] = Int32(dimension)
                }
            }
            self.mropeIndices = MLXArray(indices).reshaped(1, 1, 1, -1)
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            takeAlong(freqs, mropeIndices, axis: 0).squeezed(axis: 0)
        }

        func callAsFunction(positionIds: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = positionIds[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds, repetitions: [3, 1, 1])
            }

            let pos = positionIds.asType(.float32)
            var invFreq = self.invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (Qwen3VLVision.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (Qwen3VLVision.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }

    nonisolated final class Attention: Module {

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            self.heads = config.numAttentionHeads
            self.kvHeads = config.numKeyValueHeads
            self.headDim = config.headDim
            self.scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIds: positionIds!, dtype: x.dtype)

            (queries, keys) = Qwen3VLLanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            let result = wo(output)

            return result
        }
    }

    nonisolated final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    nonisolated final class DecoderLayer: Module {

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIds: positionIds)
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }
    nonisolated final class Model: Module {

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3VLConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
        }

        func callAsFunction(
            _ inputIds: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIds: MLXArray?,
            visualIndices: [Int]?,
            deepstackEmbeds: [MLXArray]?
        ) -> MLXArray {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIds {
                hidden = embedTokens(inputIds)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            // Loop-invariant: the scatter targets the same token positions in
            // every deepstack layer, so build the index array once rather than
            // re-uploading it per layer.
            var deepstackIndices: MLXArray?
            if deepstackEmbeds != nil, let visualIndices, !visualIndices.isEmpty {
                deepstackIndices = MLXArray(visualIndices.map { UInt32($0) })
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIds: positionIds)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let deepstackIndices
                {
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualIndices: deepstackIndices,
                        visualEmbeds: embeds[index])
                }
            }

            return norm(hidden)
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualIndices: MLXArray,
            visualEmbeds: MLXArray
        ) -> MLXArray {
            let result = hiddenStates
            // `take` (not a slice-then-array-index gather: that records on the building thread's
            // stream and is the MLX-swift 0.32 hazard, MixedSubscriptScanTests). The assignment
            // target itself is a scatter, which measured safe.
            let gathered = take(result, visualIndices, axis: 1)
            result[0..., visualIndices, 0...] = gathered + visualEmbeds

            return result
        }
    }

    /// The checkpoint's `language_model` name. Trimmed from upstream `LanguageModel`: no
    /// `lm_head`, no KV cache, no generation — the forward stops at the final norm.
    nonisolated final class LanguageModel: Module {

        @ModuleInfo var model: Model

        let config: Qwen3VLConfiguration

        init(_ config: Qwen3VLConfiguration) {
            self.config = config
            self.model = Model(config.textConfiguration)
        }

        /// Final-norm hidden states `[1, L, hidden]` for one prompt. `positionIds` are the
        /// interleaved M-RoPE positions `getRopeIndex` computes from the token ids and grids.
        func hiddenStates(
            _ inputIds: MLXArray,
            inputEmbeddings: MLXArray?,
            visualIndices: [Int]?,
            deepstackEmbeds: [MLXArray]?,
            imageGridTHW: [THW]?
        ) throws -> MLXArray {
            let (positionIds, _) = try Qwen3VLLanguage.getRopeIndex(
                inputIds: inputIds,
                imageGridTHW: imageGridTHW,
                videoGridTHW: nil,
                spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                imageTokenId: config.imageTokenIndex,
                videoTokenId: config.videoTokenIndex,
                visionStartTokenId: config.visionStartTokenId,
                attentionMask: nil)
            return model(
                inputIds, cache: nil, inputEmbeddings: inputEmbeddings, mask: nil,
                positionIds: positionIds, visualIndices: visualIndices,
                deepstackEmbeds: deepstackEmbeds)
        }
    }
}

nonisolated extension Qwen3VLLanguage {

    /// - Parameter positionOffset: the absolute M-RoPE position the first
    ///   token should occupy. Zero (the default) starts the index at absolute
    ///   zero — correct for a cold full prefill. A warm continuation that
    ///   resumes a cached prefix passes the **Position Anchor** (cache offset +
    ///   the rope delta accumulated by the cached images), so a *new* image in
    ///   the remainder gets its diverging temporal/height/width positions
    ///   computed *from* the anchor instead of resetting to zero. The returned
    ///   delta is shifted with the positions, so it remains
    ///   `maxPositionId + 1 − tokenCount` in the offset frame.
    static func getRopeIndex(
        inputIds: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenId: Int,
        videoTokenId: Int,
        visionStartTokenId: Int,
        attentionMask: MLXArray? = nil,
        positionOffset: Int = 0
    ) throws -> (MLXArray, MLXArray) {

        let (batchSize, seqLength) = (inputIds.dim(0), inputIds.dim(1))

        var positionIds = MLXArray(0 ..< seqLength).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0...], to: [batchSize, seqLength])

        guard inputIds.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            var positionIds3D = broadcast(
                positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
            if positionOffset != 0 {
                positionIds3D = positionIds3D + MLXArray(Int32(positionOffset))
            }
            // Text-only positions are `arange + positionOffset`, so the delta
            // (`maxPos + 1 − tokenCount`) collapses to `positionOffset`.
            let deltas = MLXArray(Array(repeating: Int32(positionOffset), count: batchSize))
            return (positionIds3D, deltas)
        }

        positionIds = ones(like: inputIds).asType(.int32)
        positionIds = broadcast(positionIds[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIds)

        // Process each batch item (assume batch=1 for now)
        for batchIdx in 0 ..< batchSize {
            var batchInputIds = inputIds[batchIdx, 0...]

            // Mask out padding - use where from MLX module
            batchInputIds = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIds, zeros(like: batchInputIds))

            // Count images and videos in this sequence
            let visionStartMask = (batchInputIds .== MLXArray(visionStartTokenId))
            let visionStartWeighted = `where`(
                visionStartMask, MLXArray(0 ..< seqLength), zeros(like: batchInputIds))
            let visionStartIdx = try MLXErrorGuard.item(argMax(visionStartWeighted), as: Int.self)

            guard visionStartIdx < seqLength - 1 else {
                continue  // No vision tokens
            }

            let imageNums = try MLXErrorGuard.item(
                ((batchInputIds .== MLXArray(imageTokenId)).asType(.int32).sum()), as: Int.self)
            let videoNums = try MLXErrorGuard.item(
                ((batchInputIds .== MLXArray(videoTokenId)).asType(.int32).sum()), as: Int.self)

            let inputTokens = try MLXErrorGuard.read(batchInputIds, as: Int32.self).map { Int($0) }
            var llmPosIdsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            // Process each image/video in sequence
            for _ in 0 ..< (imageNums + videoNums) {
                // Find next image/video token position
                let edImage: Int
                if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenId) {
                    edImage = idx
                } else {
                    edImage = inputTokens.count + 1
                }

                let edVideo: Int
                if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenId) {
                    edVideo = idx
                } else {
                    edVideo = inputTokens.count + 1
                }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    // Process image
                    guard let grid = imageGridTHW, imageIndex < grid.count else { break }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    // Process video
                    guard let grid = videoGridTHW, videoIndex < grid.count else { break }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                // Calculate starting index
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = try MLXErrorGuard.item(lastArray.max(), as: Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                // Add text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIdsList.append(index)
                }

                // Add 3D position IDs for visual tokens
                // Python: mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                // Adds offset to ALL three dimensions!
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIds = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIdsList.append(visualPosIds)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Add remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIdsList.last {
                    let maxVal = try MLXErrorGuard.item(lastArray.max(), as: Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIdsList.append(tIndex + MLXArray(stIdx))
            }

            // Concatenate all position IDs for this batch item
            if !llmPosIdsList.isEmpty {
                var llmPositions = concatenated(llmPosIdsList, axis: 1)  // [3, seq]
                // Shift the whole batch item into the anchor frame. Applied to
                // `llmPositions` (not just the final array) so `maxPosId` below
                // — and therefore the returned delta — is computed in the same
                // offset frame the positions live in.
                if positionOffset != 0 {
                    llmPositions = llmPositions + MLXArray(Int32(positionOffset))
                }

                // Update position_ids for this batch
                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength])
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIds[0..., batchIdx ..< batchIdx + 1, 0...])

                // Replace this batch's position IDs (assumes batch size = 1)
                positionIds = newPositions

                let maxPosId = try MLXErrorGuard.item(llmPositions.max(), as: Int.self)
                mropePositionDeltas.append(maxPosId + 1 - inputTokens.count)
            }
        }

        // Python always returns deltas array (zeros for text-only, computed values for multimodal)
        let deltas: MLXArray
        if mropePositionDeltas.isEmpty {
            deltas = MLXArray.zeros([batchSize], dtype: .int32)
        } else {
            deltas = MLXArray(mropePositionDeltas.map { Int32($0) })
        }
        return (positionIds, deltas)
    }
}

// MARK: - The app's embedding wrapper

/// The checkpoint's own module layout (`vision_tower.*`, `language_model.model.*`) with a
/// `hiddenStates` forward pass. Built and loaded by ``Qwen3VLEmbedder``.
nonisolated final class Qwen3VLEmbedModel: Module {

    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen3VLLanguage.LanguageModel

    let config: Qwen3VLConfiguration

    init(_ config: Qwen3VLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3VLLanguage.LanguageModel(config)
        super.init()
    }

    /// Upstream `Qwen3VL.sanitize(weights:)` plus the vision tower's own patch-embed transpose.
    /// The mlx-community conversion already writes `vision_tower.*` / `language_model.model.*`,
    /// but the remap keeps an official-layout checkpoint loadable too. The head is dropped: this
    /// model has none.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)
        for (key, value) in weights {
            var newKey = key
            if newKey.contains("model") {
                if newKey.contains("model.visual") {
                    newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if newKey.contains("model.language_model") {
                    newKey = newKey.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                }
            } else if newKey.contains("lm_head") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head", with: "language_model.lm_head")
            }
            if newKey.split(separator: ".").contains("lm_head") { continue }
            adjusted[newKey] = value
        }
        return visionModel.sanitize(weights: adjusted)
    }

    /// Final-norm hidden states `[1, L, hidden]` for one prepared prompt.
    ///
    /// Text-only runs the language model over the token embeddings; with an image, the vision
    /// tower's merged features (and its DeepStack features) are scattered into the
    /// `<|image_pad|>` positions first, exactly like upstream's `visionInputs`. M-RoPE positions
    /// are computed from the token ids and grids by `getRopeIndex`.
    func hiddenStates(
        inputIds: MLXArray, imagePixels: MLXArray?, imageFrames: [THW]?
    ) throws -> MLXArray {
        var inputEmbeddings: MLXArray?
        var visualIndices: [Int]?
        var deepstackEmbeds: [MLXArray]?
        var frames: [THW]? = nil

        if let imagePixels, let imageFrames, !imageFrames.isEmpty {
            frames = imageFrames
            let dtype = visionModel.patchEmbed.proj.weight.dtype
            let (visionHidden, deepstackOutputs) = try visionModel(
                imagePixels.asType(dtype), gridTHW: imageFrames)
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = imageFrames.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let flattenedFeatures = concatenated(visionHidden.split(indices: splitIndices))
                .asType(textEmbeds.dtype)
            let (merged, indices) = try mergeInputIdsWithImageFeatures(
                imageFeatures: flattenedFeatures, inputEmbeds: textEmbeds, inputIds: inputIds)
            inputEmbeddings = merged
            visualIndices = indices
            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    concatenated(layerFeatures.split(indices: splitIndices))
                        .asType(textEmbeds.dtype)
                }
            }
        }

        return try languageModel.hiddenStates(
            inputIds,
            inputEmbeddings: inputEmbeddings,
            visualIndices: visualIndices,
            deepstackEmbeds: deepstackEmbeds,
            imageGridTHW: frames)
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    /// Scatter the image features into the `<|image|>` token positions and return those flat
    /// positions (DeepStack scatters into the same rows). Upstream's version; the mask read goes
    /// through `MLXErrorGuard` so a failed array cannot segfault.
    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray, inputEmbeds: MLXArray, inputIds: MLXArray
    ) throws -> (MLXArray, [Int]) {
        let imageMask = (inputIds .== MLXArray(config.imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(config.videoTokenIndex))
        let specialMask = (imageMask .|| videoMask)
        let visualIndices = try nonZero(specialMask)

        let hiddenSize = inputEmbeds.dim(-1)
        guard visualIndices.count * hiddenSize == imageFeatures.size else {
            throw Qwen3VLEmbedError.featureTokenMismatch(
                expected: visualIndices.count, actual: imageFeatures.dim(0))
        }
        guard !visualIndices.isEmpty else { return (inputEmbeds, visualIndices) }

        let result = inputEmbeds.reshaped([-1, hiddenSize])
        result[MLXArray(visualIndices.map { Int32($0) }), 0...] =
            imageFeatures.reshaped([visualIndices.count, hiddenSize])
        return (result.reshaped(inputEmbeds.shape), visualIndices)
    }

    private func nonZero(_ mask: MLXArray) throws -> [Int] {
        let values = try MLXErrorGuard.read(mask, as: Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }
}
