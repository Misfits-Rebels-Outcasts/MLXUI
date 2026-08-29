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

    /// The highest-IoU mask of a `decodeMasks` result, flattened and thresholded to 0/1.
    private static func topBinaryMask(masks: MLXArray, iouScores: MLXArray) -> (binary: [Float], w: Int, h: Int) {
        let scores = iouScores[0].asArray(Float.self)
        let bestIdx = scores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        let (h, w) = (masks.dim(1), masks.dim(2))
        let logits = masks[0, 0..., 0..., bestIdx]
        return ((logits .> 0).asArray(Float.self), w, h)
    }

    /// Render the highest-IoU mask as a **binary** mask: white = mask, black = background, and
    /// nothing of the source photo survives — the `tools/media.py::save_mask` L-mode boundary
    /// the `.cat` contract expects (`63-CutOutSubject`'s row 3 is `Save Image photo-mask.png`).
    /// This is the stage's renderer (CFM-R15-1); the red-tinted overlay is the interactive
    /// display rendering and belongs to `SegmentAnythingRunView` alone. Written straight into an
    /// 8-bit grayscale pixel buffer, one store per pixel.
    static func renderBinaryMask(
        masks: MLXArray, iouScores: MLXArray, width: Int, height: Int
    ) -> CGImage? {
        let (binary, w, h) = topBinaryMask(masks: masks, iouScores: iouScores)
        return renderBinaryMask(binary: binary, maskWidth: w, maskHeight: h, width: width, height: height)
    }

    /// The pure pixel-buffer writer behind `renderBinaryMask` — testable without weights.
    /// `binary` is the flattened highest-IoU mask at `maskWidth`×`maskHeight`; the output is a
    /// two-valued grayscale image at `width`×`height`, nearest-neighbour upsampled from the mask
    /// grid. The source photo is never drawn, so no source pixel can survive.
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

    /// Render the highest-IoU mask as a red-tinted overlay on the source image — the interactive
    /// *display* rendering for `SegmentAnythingRunView` (SPEC-Q210: the flow boundary wants the
    /// binary mask above; the overlay belongs here alone). Written straight into the context's
    /// premultiplied pixel buffer — one blend per mask pixel, not one `CGContext.fill` per pixel
    /// (the previous loop was ~12M CoreGraphics calls on a phone-camera photo).
    static func renderTopMask(
        masks: MLXArray, iouScores: MLXArray, sourceImage: CGImage
    ) -> CGImage {
        let (binary, w, h) = topBinaryMask(masks: masks, iouScores: iouScores)

        let W = sourceImage.width, H = sourceImage.height
        guard let ctx = CGContext(
            data: nil, width: W, height: H,
            bitsPerComponent: 8, bytesPerRow: W * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let data = ctx.data else { return sourceImage }

        ctx.draw(sourceImage, in: CGRect(x: 0, y: 0, width: W, height: H))
        let pixels = data.assumingMemoryBound(to: UInt8.self)
        for py in 0 ..< H {
            let my = min(py * h / H, h - 1)
            for px in 0 ..< W {
                let mx = min(px * w / W, w - 1)
                guard binary[my * w + mx] > 0.5 else { continue }
                let i = py * W * 4 + px * 4
                // Straight 40%-opaque red over the source pixel, in one buffer pass — the
                // equivalent of the old per-pixel `fill`, without the per-pixel CG call.
                pixels[i + 0] = UInt8(min(255, Int(Double(pixels[i + 0]) * 0.6) + 102))
                pixels[i + 1] = UInt8(Double(pixels[i + 1]) * 0.6)
                pixels[i + 2] = UInt8(Double(pixels[i + 2]) * 0.6)
            }
        }
        return ctx.makeImage() ?? sourceImage
    }
}
