import CoreGraphics
import Foundation
import MLX
import MLXNN

// MARK: - Errors

enum SegmentAnythingError: Error, LocalizedError {
    case missingWeights(String)
    case imageConversionFailed

    var errorDescription: String? {
        switch self {
        case .missingWeights(let f): "SAM3 weights not found: \(f)"
        case .imageConversionFailed: "Failed to convert image to MLX tensor"
        }
    }
}

// MARK: - Model container

/// Top-level module holding the three SAM3 inference components.
/// Weight key mapping from mlx-community/sam3-4bit:
///   `detector_model.vision_encoder.backbone.*` → backbone
///   `detector_model.mask_decoder.*`            → maskDecoder / iouPredictor
///   `tracker_model.prompt_encoder.*`           → promptEncoder
nonisolated final class SAM3ModelContainer: Module {
    @ModuleInfo var backbone: SAM3ViTBackbone
    @ModuleInfo var maskDecoder: SAM3PixelDecoder
    @ModuleInfo var iouPredictor: SAM3IoUPredictor
    @ModuleInfo var promptEncoder: SAM3PromptEncoder

    override init() {
        _backbone.wrappedValue      = SAM3ViTBackbone()
        _maskDecoder.wrappedValue   = SAM3PixelDecoder()
        _iouPredictor.wrappedValue  = SAM3IoUPredictor()
        _promptEncoder.wrappedValue = SAM3PromptEncoder()
        super.init()
    }
}

// MARK: - Engine

/// Public interface for SAM3 image segmentation (SA-AM4).
nonisolated enum SegmentAnythingEngine {

    // Positional embeddings in mlx-community/sam3-4bit have 576 = 24×24 tokens,
    // so the patch grid must be 24×24: imageSize = 24 × patchSize(14) = 336.
    static let imageSize = 336
    static let gridSize  = imageSize / 14   // 24

    // MARK: Weight loading

    private static func loadModel(from dir: URL) async throws -> SAM3ModelContainer {
        let url = dir.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw SegmentAnythingError.missingWeights("model.safetensors")
        }
        let model = SAM3ModelContainer()
        let raw = try MLX.loadArrays(url: url)
        let weights = remapKeys(raw)

        // Convert Linear → QuantizedLinear for any layer present in the 4-bit checkpoint.
        // The sam3-4bit repo uses standard 4-bit affine quantization (group_size=64).
        // Layers without a corresponding .scales entry (Conv2d, LayerNorm) are left as-is.
        quantize(model: model) { path, _ in
            weights["\(path).scales"] != nil ? (64, 4, QuantizationMode.affine) : nil
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
        eval(model)
        return model
    }

    /// Remap mlx-community/sam3-4bit weight keys to the Swift module path hierarchy.
    /// Also converts snake_case path components to camelCase to match Swift property names
    /// (e.g. `q_proj` → `qProj`, `layer_norm1` → `layerNorm1`, `neck_conv` → `neckConv`).
    private static func remapKeys(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, val) in raw {
            let prefix: String
            let suffix: Substring
            switch true {
            case key.hasPrefix("detector_model.vision_encoder.backbone."):
                prefix = "backbone."
                suffix = key.dropFirst("detector_model.vision_encoder.backbone.".count)
            case key.hasPrefix("detector_model.mask_decoder.pixel_decoder."):
                prefix = "maskDecoder."
                suffix = key.dropFirst("detector_model.mask_decoder.pixel_decoder.".count)
            case key.hasPrefix("detector_model.mask_decoder.mask_embedder."):
                prefix = "maskDecoder.maskEmbedder."
                suffix = key.dropFirst("detector_model.mask_decoder.mask_embedder.".count)
            case key.hasPrefix("detector_model.mask_decoder.instance_projection"):
                prefix = "maskDecoder.instanceProjection"
                suffix = key.dropFirst("detector_model.mask_decoder.instance_projection".count)
            case key.hasPrefix("detector_model.mask_decoder.iou_predictor."):
                prefix = "iouPredictor."
                suffix = key.dropFirst("detector_model.mask_decoder.iou_predictor.".count)
            case key.hasPrefix("tracker_model.prompt_encoder.point_embed"):
                prefix = "promptEncoder.pointEmbed"
                suffix = key.dropFirst("tracker_model.prompt_encoder.point_embed".count)
            case key.hasPrefix("tracker_model.prompt_encoder.no_mask_embed"):
                prefix = "promptEncoder.noMaskEmbed"
                suffix = key.dropFirst("tracker_model.prompt_encoder.no_mask_embed".count)
            default:
                continue   // skip tracker/FPN weights not needed for still-image inference
            }
            out[prefix + camelPath(suffix)] = val
        }
        return out
    }

    /// Convert a dot-separated path of snake_case segments to camelCase Swift property names.
    /// "layers.0.q_proj.weight" → "layers.0.qProj.weight"
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
        let mean: [Float] = [0.485, 0.456, 0.406]
        let std:  [Float] = [0.229, 0.224, 0.225]
        var rgb = [Float](repeating: 0, count: side * side * 3)
        for i in 0 ..< side * side {
            rgb[i * 3 + 0] = (Float(bytes[i * 4 + 0]) / 255.0 - mean[0]) / std[0]
            rgb[i * 3 + 1] = (Float(bytes[i * 4 + 1]) / 255.0 - mean[1]) / std[1]
            rgb[i * 3 + 2] = (Float(bytes[i * 4 + 2]) / 255.0 - mean[2]) / std[2]
        }
        return MLXArray(rgb, [1, side, side, 3])
    }

    // MARK: Public API

    /// Returns [1, gridSize, gridSize, 256]
    static func encodeImage(_ image: CGImage, modelID: String) async throws -> MLXArray {
        let model = try await loadModel(from: ModelStore.shared.directory(forModelID: modelID))
        guard let tensor = preprocessImage(image) else {
            throw SegmentAnythingError.imageConversionFailed
        }
        let embedding = model.backbone(tensor)
        eval(embedding)
        return embedding
    }

    /// Returns (masks: [1, H', W', 3], iouScores: [1, 3])
    static func decodeMasks(
        embedding: MLXArray,
        points: [[Float]],
        labels: [Int],
        modelID: String
    ) async throws -> (masks: MLXArray, iouScores: MLXArray) {
        let model = try await loadModel(from: ModelStore.shared.directory(forModelID: modelID))
        let pts = MLXArray(points.flatMap { $0 }, [points.count, 2])
        let lbl = MLXArray(labels.map { Int32($0) })
        let promptTokens = model.promptEncoder(points: pts, labels: lbl)
        let masks        = model.maskDecoder(embedding, promptTokens: promptTokens)
        let iouScores    = model.iouPredictor(embedding)
        eval(masks, iouScores)
        return (masks, iouScores)
    }

    // MARK: Mask rendering

    /// Render the highest-IoU mask as a red-tinted overlay on the source image.
    /// Kept here (nonisolated enum) so callers on any actor can invoke it without inference issues.
    static func renderTopMask(
        masks: MLXArray, iouScores: MLXArray, sourceImage: CGImage
    ) -> CGImage {
        let scores = iouScores[0].asArray(Float.self)
        let bestIdx = scores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0

        let (h, w) = (masks.dim(1), masks.dim(2))
        let logits = masks[0, 0..., 0..., bestIdx]
        let binary  = (logits .> 0).asArray(Float.self)

        let W = sourceImage.width, H = sourceImage.height
        guard let ctx = CGContext(
            data: nil, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return sourceImage }

        ctx.draw(sourceImage, in: CGRect(x: 0, y: 0, width: W, height: H))
        ctx.setBlendMode(.normal)
        for py in 0 ..< H {
            for px in 0 ..< W {
                let mx = min(Int(Float(px) / Float(W) * Float(w)), w - 1)
                let my = min(Int(Float(py) / Float(H) * Float(h)), h - 1)
                if binary[my * w + mx] > 0.5 {
                    ctx.setFillColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 0.4)
                    ctx.fill(CGRect(x: px, y: H - 1 - py, width: 1, height: 1))
                }
            }
        }
        return ctx.makeImage() ?? sourceImage
    }
}
