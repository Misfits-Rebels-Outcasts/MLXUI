import Foundation
import MLX
import MLXNN

/// Vision→language projector for PaddleOCR-VL 1.5. Matches the checkpoint's `visual.projector.*`:
/// a `pre_norm` LayerNorm over the vision hidden size, a 2×2 spatial pixel-unshuffle that merges
/// each 2×2 patch block into one token (multiplying the feature dim by 4), then two linear layers
/// with GELU. See the reference `Projector` (merge_kernel_size = (2, 2)) in mlx-vlm's
/// modeling_paddleocr_vl.py.
public class MultiModalProjector: Module {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    let visionHidden: Int
    let mergedHidden: Int   // visionHidden * mergeH * mergeW
    let textHidden: Int
    static let mergeH = 2
    static let mergeW = 2

    public init(config: PaddleOCRVLConfig) {
        self.visionHidden = config.visionConfig.hiddenSize
        self.mergedHidden = config.visionConfig.hiddenSize * Self.mergeH * Self.mergeW
        self.textHidden = config.textConfig.hiddenSize

        self._preNorm.wrappedValue = LayerNorm(dimensions: visionHidden, eps: 1e-5)
        self._linear1.wrappedValue = Linear(mergedHidden, mergedHidden, bias: true)
        self._linear2.wrappedValue = Linear(mergedHidden, textHidden, bias: true)

        super.init()
    }

    /// - features: `[B, gridH*gridW, visionHidden]` (row-major token order).
    /// - gridH/gridW: patch grid dimensions; both must be even (2×2 merge).
    /// Returns `[B, (gridH/2)*(gridW/2), textHidden]`.
    public func callAsFunction(_ features: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let b = features.dim(0)
        var x = preNorm(features)

        // Pixel-unshuffle 2×2, matching einops "(h p1 w p2) d -> (h w) (p1 p2 d)" on a row-major
        // (h, w) grid: token index = row*gridW + col, row = h*2+p1, col = w*2+p2.
        let h = gridH / Self.mergeH
        let w = gridW / Self.mergeW
        x = x.reshaped(b, h, Self.mergeH, w, Self.mergeW, visionHidden)  // [B,h,p1,w,p2,C]
        x = x.transposed(0, 1, 3, 2, 4, 5)                              // [B,h,w,p1,p2,C]
        x = x.reshaped(b, h * w, mergedHidden)                          // [B,h*w,4C]

        x = linear1(x)
        x = gelu(x)
        x = linear2(x)
        return x
    }
}
