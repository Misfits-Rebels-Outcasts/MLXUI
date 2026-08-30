import CoreGraphics
import Foundation
import MLX
import MLXNN

// MARK: - Errors

enum SegmentAnythingError: Error, LocalizedError {
    case missingWeights(String)
    case imageConversionFailed
    case maskRenderFailed

    var errorDescription: String? {
        switch self {
        case .missingWeights(let f): "SAM3 weights not found: \(f)"
        case .imageConversionFailed: "Failed to convert image to MLX tensor"
        case .maskRenderFailed: "Failed to render the segmentation mask"
        }
    }
}

// MARK: - Model container

/// Top-level module holding the SAM3 image-segmentation components.
nonisolated final class SAM3ModelContainer: Module {
    @ModuleInfo var backbone: SAM3ViTBackbone
    @ModuleInfo var neck: SAM3Neck
    @ModuleInfo var promptEncoder: SAM3PromptEncoder
    @ModuleInfo var maskDecoder: SAM3MaskDecoder
    var noMemEmbed: MLXArray   // [1, 1, 256], loaded manually (a buffer)

    override init() {
        _backbone.wrappedValue      = SAM3ViTBackbone()
        _neck.wrappedValue          = SAM3Neck()
        _promptEncoder.wrappedValue = SAM3PromptEncoder()
        _maskDecoder.wrappedValue   = SAM3MaskDecoder()
        noMemEmbed = MLXArray.zeros([1, 1, 256])
        super.init()
    }
}

// MARK: - Encoded image features

/// The cached image-side tensors produced once per image (`encodeImage`), re-used on every
/// prompt decode. `imagePe` is the dense positional encoding of the 72×72 grid.
struct SAM3ImageFeatures {
    let imageEmbedding: MLXArray     // [1, 72, 72, 256]  (with `no_mem_embed` added)
    let highResFeatures: [MLXArray]  // [[1,288,288,32] (conv_s0), [1,144,144,64] (conv_s1)]
    let imagePe: MLXArray            // [1, 72, 72, 256]
}

// MARK: - Engine

/// Public interface for SAM3 image segmentation.
nonisolated enum SegmentAnythingEngine {

    // SAM3 image encoder input: 1008² (square resize, aspect-distorting), patch 14 → 72×72.
    static let imageSize = 1008
    static let gridSize  = imageSize / 14   // 72

    // MARK: Weight loading

    private static func loadModel(from dir: URL) async throws -> SAM3ModelContainer {
        let url = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SegmentAnythingError.missingWeights("model.safetensors")
        }
        let model = SAM3ModelContainer()
        let raw = try MLX.loadArrays(url: url)
        let weights = remapKeys(raw)

        print("[SAM3] weight coverage: \(weights.count)/\(raw.count) checkpoint tensors remapped into module paths")

        // Raw buffers are assigned by hand below (they are not `@ModuleInfo` params).
        let positionEmbeddings = weights["backbone.embeddings.positionEmbeddings"]
        let gaussianMatrix = weights["promptEncoder.gaussianMatrix"]
        let noMemEmbed = weights["noMemEmbed"]
        var moduleWeights = weights
        for k in ["backbone.embeddings.positionEmbeddings", "promptEncoder.gaussianMatrix", "noMemEmbed"] {
            moduleWeights.removeValue(forKey: k)
        }

        // Convert Linear → QuantizedLinear for any layer present in the 4-bit checkpoint.
        quantize(model: model) { path, _ in
            moduleWeights["\(path).scales"] != nil ? (64, 4, QuantizationMode.affine) : nil
        }

        try model.update(parameters: ModuleParameters.unflattened(moduleWeights), verify: .none)
        if let pe = positionEmbeddings { model.backbone.embeddings.positionEmbeddings = pe }
        if let gm = gaussianMatrix { model.promptEncoder.gaussianMatrix = gm }
        if let nm = noMemEmbed { model.noMemEmbed = nm }
        eval(model)
        return model
    }

    /// Remap mlx-community/sam3-4bit keys into the Swift module hierarchy.
    private static func remapKeys(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, val) in raw {
            switch true {
            case key.hasPrefix("detector_model.vision_encoder.backbone."):
                let suffix = key.dropFirst("detector_model.vision_encoder.backbone.".count)
                out["backbone." + camelPath(suffix)] = val
            case key.hasPrefix("tracker_model.prompt_encoder.shared_embedding.positional_embedding"):
                out["promptEncoder.gaussianMatrix"] = val
            case key.hasPrefix("tracker_model.prompt_encoder."):
                let suffix = key.dropFirst("tracker_model.prompt_encoder.".count)
                out["promptEncoder." + camelPath(suffix)] = val
            case key.hasPrefix("tracker_model.mask_decoder."):
                let suffix = key.dropFirst("tracker_model.mask_decoder.".count)
                out["maskDecoder." + camelPath(suffix)] = val
            case key.hasPrefix("tracker_neck."):
                if let mapped = neckRemap(key) { out[mapped] = val }
            case key == "tracker_model.no_memory_embedding":
                out["noMemEmbed"] = val
            default:
                continue
            }
        }
        return out
    }

    /// `tracker_neck.fpn_layers.{i}.{proj1|proj2|scale_layers.N}.*` → `neck.levels.{i}.{...}`.
    private static func neckRemap(_ key: String) -> String? {
        let body = key.dropFirst("tracker_neck.fpn_layers.".count)
        let parts = body.split(separator: ".", maxSplits: 1)
        guard parts.count == 2 else { return nil }
        let level = parts[0]
        let rest = String(parts[1])
        let mapped: String
        switch true {
        case rest.hasPrefix("proj1."): mapped = "proj1." + String(rest.dropFirst("proj1.".count))
        case rest.hasPrefix("proj2."): mapped = "proj2." + String(rest.dropFirst("proj2.".count))
        case rest.hasPrefix("scale_layers.0."): mapped = "upsample1." + String(rest.dropFirst("scale_layers.0.".count))
        case rest.hasPrefix("scale_layers.2."): mapped = "upsample2." + String(rest.dropFirst("scale_layers.2.".count))
        default: return nil
        }
        return "neck.levels.\(level).\(mapped)"
    }

    /// Convert a dot-separated path of snake_case segments to camelCase Swift property names.
    private static func camelPath(_ s: Substring) -> String {
        s.split(separator: ".", omittingEmptySubsequences: false)
            .map { snakeToCamel(String($0)) }
            .joined(separator: ".")
    }

    private static func snakeToCamel(_ s: String) -> String {
        guard s.contains("_") else { return s }
        var result = ""
        var capitalize = false
        for ch in s {
            if ch == "_" {
                capitalize = true
            } else if capitalize {
                result.append(contentsOf: ch.uppercased())
                capitalize = false
            } else {
                result.append(ch)
            }
        }
        return result
    }

    // MARK: Image preprocessing

    /// Square-resize to 1008² (aspect-distorting, matching SAM2Transforms) and normalize
    /// with mean/std 0.5 → values in [-1, 1], RGB.
    private static func preprocessImage(_ cgImage: CGImage) -> MLXArray? {
        let side = imageSize
        guard let ctx = CGContext(
            data: nil, width: side, height: side,
            bitsPerComponent: 8, bytesPerRow: side * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: side, height: side))
        guard let data = ctx.data else { return nil }

        let byteCount = side * side * 4
        let bytes = Array(UnsafeBufferPointer(
            start: data.assumingMemoryBound(to: UInt8.self), count: byteCount))
        var rgb = [Float](repeating: 0, count: side * side * 3)
        for i in 0 ..< side * side {
            rgb[i * 3 + 0] = Float(bytes[i * 4 + 0]) / 255.0 * 2.0 - 1.0
            rgb[i * 3 + 1] = Float(bytes[i * 4 + 1]) / 255.0 * 2.0 - 1.0
            rgb[i * 3 + 2] = Float(bytes[i * 4 + 2]) / 255.0 * 2.0 - 1.0
        }
        return MLXArray(rgb, [1, side, side, 3])
    }

    // MARK: Public API

    /// Encodes the image once: backbone → FPN neck → image embedding (+no_mem_embed) and
    /// the two pre-projected high-res feature maps.
    static func encodeImage(_ image: CGImage, modelID: String) async throws -> SAM3ImageFeatures {
        let model = try await loadModel(from: ModelStore.shared.directory(forModelID: modelID))
        guard let tensor = preprocessImage(image) else {
            throw SegmentAnythingError.imageConversionFailed
        }
        let backboneOut = model.backbone(tensor)          // [1, 72, 72, 1024]
        let levels = model.neck(backboneOut)              // [288², 144², 72²] @256
        let featS0 = model.maskDecoder.convS0(levels[0])  // [1, 288, 288, 32]
        let featS1 = model.maskDecoder.convS1(levels[1])  // [1, 144, 144, 64]
        let imageEmbed = levels[2] + model.noMemEmbed.reshaped([1, 1, 1, 256])  // [1, 72, 72, 256]
        let imagePe = model.promptEncoder.densePositionalEncoding()
        eval(featS0, featS1, imageEmbed, imagePe)
        return SAM3ImageFeatures(imageEmbedding: imageEmbed, highResFeatures: [featS0, featS1], imagePe: imagePe)
    }

    /// Decodes point prompts into masks. `points` are normalized [0,1]; `labels` 1=foreground, 0=background.
    /// Returns (masks: [1, 288, 288, 3], iouScores: [1, 3]) — the 3 multimask candidates.
    static func decodeMasks(
        features: SAM3ImageFeatures,
        points: [[Float]],
        labels: [Int],
        modelID: String
    ) async throws -> (masks: MLXArray, iouScores: MLXArray) {
        let model = try await loadModel(from: ModelStore.shared.directory(forModelID: modelID))
        let pts = MLXArray(points.flatMap { $0 }.map { $0 * Float(imageSize) }, [points.count, 2])
        let lbl = MLXArray(labels.map { Int32($0) })
        let sparse = model.promptEncoder(points: pts, labels: lbl)            // [1, N+1, 256]
        let dense = model.promptEncoder.noMaskEmbed(MLXArray([Int32(0)]))     // [1, 256]
            .reshaped([1, 1, 1, 256])
        let (masks, iou) = model.maskDecoder(
            imageEmbedding: features.imageEmbedding,
            imagePe: features.imagePe,
            sparsePrompts: sparse,
            densePrompts: dense,
            highResFeatures: features.highResFeatures)                        // masks [1,4,288,288], iou [1,4]
        // multimask_output: drop the single-mask token (index 0), keep candidates 1..3.
        let multiMasks = stacked((1 ..< 4).map { masks[0..., $0, 0..., 0...] }, axis: 1)  // [1,3,288,288]
        let multiIou = stacked((1 ..< 4).map { iou[0..., $0] }, axis: 1)                  // [1,3]
        let masksOut = multiMasks.transposed(0, 2, 3, 1)  // [1, 288, 288, 3]
        eval(masksOut, multiIou)
        return (masksOut, multiIou)
    }

    // MARK: Mask rendering

    /// The highest-IoU mask of a `decodeMasks` result, as raw logits (float, un-thresholded).
    private static func topMaskLogits(masks: MLXArray, iouScores: MLXArray) -> (logits: [Float], w: Int, h: Int) {
        let scores = iouScores[0].asArray(Float.self)
        let bestIdx = scores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let (h, w) = (masks.dim(1), masks.dim(2))
        let logits = masks[0, 0..., 0..., bestIdx]
        return (logits.asArray(Float.self), w, h)
    }

    /// Bilinear resize (align_corners=false, matching `F.interpolate(mode="bilinear")`).
    private static func bilinearResize(_ src: [Float], srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> [Float] {
        guard srcW > 0, srcH > 0, dstW > 0, dstH > 0, src.count == srcW * srcH else { return src }
        if srcW == dstW && srcH == dstH { return src }
        let scaleX = Float(srcW) / Float(dstW)
        let scaleY = Float(srcH) / Float(dstH)
        var out = [Float](repeating: 0, count: dstW * dstH)
        for py in 0 ..< dstH {
            let sy = min(max((Float(py) + 0.5) * scaleY - 0.5, 0), Float(srcH - 1))
            let y0 = Int(sy)
            let y1 = min(y0 + 1, srcH - 1)
            let fy = sy - Float(y0)
            let row0 = y0 * srcW, row1 = y1 * srcW
            let outRow = py * dstW
            for px in 0 ..< dstW {
                let sx = min(max((Float(px) + 0.5) * scaleX - 0.5, 0), Float(srcW - 1))
                let x0 = Int(sx)
                let x1 = min(x0 + 1, srcW - 1)
                let fx = sx - Float(x0)
                let top = src[row0 + x0] * (1 - fx) + src[row0 + x1] * fx
                let bottom = src[row1 + x0] * (1 - fx) + src[row1 + x1] * fx
                out[outRow + px] = top * (1 - fy) + bottom * fy
            }
        }
        return out
    }

    static func renderBinaryMask(
        masks: MLXArray, iouScores: MLXArray, width: Int, height: Int
    ) -> CGImage? {
        let (logits, w, h) = topMaskLogits(masks: masks, iouScores: iouScores)
        let resized = bilinearResize(logits, srcW: w, srcH: h, dstW: width, dstH: height)
        let binary = resized.map { $0 > 0 ? Float(1.0) : Float(0.0) }
        return renderBinaryMask(binary: binary, maskWidth: width, maskHeight: height, width: width, height: height)
    }

    static func renderBinaryMask(
        binary: [Float], maskWidth: Int, maskHeight: Int, width: Int, height: Int
    ) -> CGImage? {
        guard width > 0, height > 0, maskWidth > 0, maskHeight > 0,
              binary.count == maskWidth * maskHeight,
              let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let data = ctx.data else { return nil }
        let pixels = data.assumingMemoryBound(to: UInt8.self)
        for py in 0 ..< height {
            let my = min(py * maskHeight / height, maskHeight - 1)
            for px in 0 ..< width {
                let mx = min(px * maskWidth / width, maskWidth - 1)
                pixels[py * width + px] = binary[my * maskWidth + mx] > 0.5 ? 255 : 0
            }
        }
        return ctx.makeImage()
    }

    static func renderTopMask(
        masks: MLXArray, iouScores: MLXArray, sourceImage: CGImage
    ) -> CGImage {
        let W = sourceImage.width, H = sourceImage.height
        let (logits, w, h) = topMaskLogits(masks: masks, iouScores: iouScores)
        let resized = bilinearResize(logits, srcW: w, srcH: h, dstW: W, dstH: H)

        guard let ctx = CGContext(
            data: nil, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let data = ctx.data else { return sourceImage }

        ctx.draw(sourceImage, in: CGRect(x: 0, y: 0, width: W, height: H))
        let pixels = data.assumingMemoryBound(to: UInt8.self)
        for py in 0 ..< H {
            let row = py * W
            for px in 0 ..< W {
                guard resized[row + px] > 0 else { continue }
                let i = row * 4 + px * 4
                pixels[i + 0] = UInt8(min(255, Int(Double(pixels[i + 0]) * 0.6) + 102))
                pixels[i + 1] = UInt8(Double(pixels[i + 1]) * 0.6)
                pixels[i + 2] = UInt8(Double(pixels[i + 2]) * 0.6)
            }
        }
        return ctx.makeImage() ?? sourceImage
    }
}
