import Foundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM

/// `UserInputProcessor` for DeepSeek-OCR-2 (registered as `DeepseekVLV2Processor`). Prepares the
/// image + `<image>`-token sequence the `DeepSeekOCR` model expects.
///
/// **Dynamic tiling** (port of `processing_deepseekocr.py`'s `dynamic_preprocess` +
/// `tokenize_with_images`): `N` local 768² patches on a `dynamic_preprocess` grid (1–6 tiles picked by
/// aspect ratio) plus one padded 1024² global view. Pixels are packed into one flat HWC buffer in the
/// order `[local patches…, global]`, with a matching `frames: [THW]`; the model reconstructs each tile,
/// runs SAM→Qwen2→projector, and appends the learned `view_separator` last — so the feature order is
/// `[local patches…, global, view_separator]`. The `<image>` token block is contiguous and its length is
/// `num_patches·144 + 256 + 1` (144 = 12×12 SAM tokens per 768² patch, 256 = 16×16 for the 1024² global,
/// +1 separator), matching that feature count exactly.
///
/// Don't `import Tokenizers` (keeps `Tokenizer` unambiguous — resolves to the registry's protocol).
struct DeepSeekOCRProcessor: UserInputProcessor {
    /// `image_token_index` from the model config (fixed for this architecture).
    static let imageTokenId = 128815
    static let globalSize = 1024
    static let localSize = 768
    static let visionMean: CGFloat = 0.5

    /// Global 1024² → 16×16 SAM grid → `query_1024` = 256 Qwen2 tokens; each local 768² → 12×12 → 144
    /// (`query_768`); + 1 learned view-separator appended by the model.
    static let globalTokenCount = 256
    static let localTokenCount = 144
    static let viewSeparatorCount = 1

    /// `dynamic_preprocess` bounds (reference defaults: 1–6 local patches).
    static let minPatches = 1
    static let maxPatches = 6

    private let tokenizer: any Tokenizer

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard let image = input.images.first else {
            let tokens = try tokenizer.encode(text: "Free OCR.")
            return LMInput(tokens: MLXArray(tokens.map { Int32($0) }))
        }

        let ci = try image.asCIImage()

        // Tile order MUST be [local patches…, global]; the model appends the view_separator, giving the
        // [local, global, separator] feature order the `<image>` block is spliced into.
        let (cols, rows) = Self.tileGrid(for: ci.extent.size)
        // A 1×1 grid's single "local patch" is just the whole page stretched to 768² — a lower-res
        // duplicate of the 1024² global view. It adds no detail, and feeding the page twice biases the
        // decoder toward re-emitting it (observed: whole-output repetition on near-square inputs). Only
        // tile when the grid actually subdivides the image (cols·rows > 1).
        let numPatches = (cols * rows > 1) ? cols * rows : 0

        var chunks: [MLXArray] = []
        var frames: [THW] = []
        if numPatches > 0 {
            for tile in Self.localPatchPixels(ci, cols: cols, rows: rows) {
                chunks.append(tile)
                frames.append(THW(1, Self.localSize, Self.localSize))
            }
        }
        chunks.append(Self.globalViewPixels(ci))
        frames.append(THW(1, Self.globalSize, Self.globalSize))
        let pixels = concatenated(chunks, axis: 0)

        // Order: [local_patches, global_view, view_separator] → num_patches·144 + 256 + 1.
        let numImageTokens =
            numPatches * Self.localTokenCount + Self.globalTokenCount + Self.viewSeparatorCount

        // Token layout: [BOS] <image>×numImageTokens <instruction>. The model splices vision features
        // into the <image> positions in order; the instruction follows so generation continues from it.
        var ids: [Int] = [0]                                      // BOS (id 0 for this tokenizer)
        ids += Array(repeating: Self.imageTokenId, count: numImageTokens)
        // "Free OCR." → plain reading-order text. The `<|grounding|>OCR this image.` prompt instead
        // emits layout markup (`<|ref|>text<|/ref|><|det|>[[bbox]]<|/det|>`); use that only when the
        // caller wants bounding boxes. Matches the no-image fallback above.
        ids += try tokenizer.encode(text: "\nFree OCR.")

        return LMInput(
            text: .init(tokens: MLXArray(ids.map { Int32($0) })),
            image: .init(pixels: pixels, frames: frames))
    }

    /// Reference `dynamic_preprocess` grid selection: among all `(cols, rows)` with
    /// `minPatches ≤ cols·rows ≤ maxPatches`, pick the one whose aspect ratio is closest to the image's.
    /// On a tie, prefer the finer grid only when the image is large enough to fill it. Returns `(cols, rows)`.
    static func tileGrid(for size: CGSize) -> (cols: Int, rows: Int) {
        let w = Double(size.width), h = Double(size.height)
        guard w > 0, h > 0 else { return (1, 1) }
        let aspect = w / h
        let area = w * h
        let px = Double(localSize * localSize)

        var ratios: [(Int, Int)] = []
        for n in minPatches...maxPatches {
            for i in 1...n {
                for j in 1...n where (i * j) >= minPatches && (i * j) <= maxPatches {
                    if !ratios.contains(where: { $0 == (i, j) }) { ratios.append((i, j)) }
                }
            }
        }
        ratios.sort { $0.0 * $0.1 < $1.0 * $1.1 }

        var best = (cols: 1, rows: 1)
        var bestDiff = Double.greatestFiniteMagnitude
        for r in ratios {
            let diff = abs(aspect - Double(r.0) / Double(r.1))
            if diff < bestDiff {
                bestDiff = diff
                best = (r.0, r.1)
            } else if diff == bestDiff, area > 0.5 * px * Double(r.0 * r.1) {
                best = (r.0, r.1)
            }
        }
        return best
    }

    /// Stretch-resize to `(cols·768, rows·768)` (independent x/y scale, matching PIL `image.resize`),
    /// then crop row-major 768² tiles (top→bottom, left→right) and normalize each to a flat HWC buffer.
    private static func localPatchPixels(_ ci: CIImage, cols: Int, rows: Int) -> [MLXArray] {
        let side = CGFloat(localSize)
        let srgb = MediaProcessing.inSRGBToneCurveSpace(ci)
        // `resampleBicubic` scales x/y independently and crops to origin (0,0) at the exact size.
        let resized = MediaProcessing.resampleBicubic(
            srgb, to: CGSize(width: side * CGFloat(cols), height: side * CGFloat(rows)))

        var tiles: [MLXArray] = []
        for r in 0..<rows {
            for c in 0..<cols {
                // CIImage is y-up, so the top PIL row (r == 0) sits at the highest y.
                let originX = CGFloat(c) * side
                let originY = CGFloat(rows - 1 - r) * side
                let tile = resized
                    .cropped(to: CGRect(x: originX, y: originY, width: side, height: side))
                    .transformed(by: CGAffineTransform(translationX: -originX, y: -originY))
                    .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
                tiles.append(flatHWC(tile, side: localSize))
            }
        }
        return tiles
    }

    /// Aspect-preserving pad to `globalSize`² (mean-gray fill), normalize, → flat HWC.
    private static func globalViewPixels(_ ci: CIImage) -> MLXArray {
        let side = CGFloat(globalSize)
        let extent = ci.extent
        let scale = side / max(extent.width, extent.height)

        let srgb = MediaProcessing.inSRGBToneCurveSpace(ci)
        let resized = MediaProcessing.resampleBicubic(
            srgb, to: CGSize(width: extent.width * scale, height: extent.height * scale))

        // Center on a mean-gray 1024² canvas (ImageOps.pad equivalent).
        let e = resized.extent
        let dx = ((side - e.width) / 2).rounded()
        let dy = ((side - e.height) / 2).rounded()
        let moved = resized.transformed(by: CGAffineTransform(translationX: dx - e.origin.x, y: dy - e.origin.y))
        let background = CIImage(color: CIColor(red: visionMean, green: visionMean, blue: visionMean))
            .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        let padded = moved.composited(over: background)
            .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))

        return flatHWC(padded, side: globalSize)
    }

    /// Normalize (mean/std 0.5) a `side²` CIImage anchored at the origin, drop any alpha, → flat HWC
    /// `(side·side·3,)` matching the model's `(1, h, w, 3)` reshape.
    private static func flatHWC(_ image: CIImage, side: Int) -> MLXArray {
        let normalized = MediaProcessing.normalize(
            image, mean: (visionMean, visionMean, visionMean), std: (0.5, 0.5, 0.5))
        return MediaProcessing.asMLXArray(normalized)
            .reshaped(side, side, -1)[0..., 0..., 0 ..< 3]
            .reshaped(-1)
    }
}
