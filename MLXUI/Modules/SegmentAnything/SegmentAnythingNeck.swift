import Foundation
import MLX
import MLXNN

// MARK: - FPN neck (tracker_neck)

/// `tracker_neck.fpn_layers.N` — a 4-level SimpleFPN neck (ViTDet style) applied to the
/// single 72×72×1024 backbone output. Each level independently upsamples/downsamples and
/// projects to 256 channels. The tracker drops the 36×36 level (`scalp=1`), so we model the
/// three kept levels: 4.0 → 288², 2.0 → 144², 1.0 → 72².
///
/// Checkpoint key → Swift property mapping (handled in `SegmentAnythingEngine.remapKeys`):
///   `scale_layers.0` → `upsample1` (ConvTransposed2d 1024→512)
///   `scale_layers.2` → `upsample2` (ConvTransposed2d 512→256)
///   `proj1`         → `proj1` (Conv2d 1×1)
///   `proj2`         → `proj2` (Conv2d 3×3)
nonisolated final class SAM3NeckLevel: Module {
    @ModuleInfo var upsample1: ConvTransposed2d   // 1024 → 512, 2× (scale 4.0 and 2.0)
    @ModuleInfo var upsample2: ConvTransposed2d   // 512  → 256, 2× (scale 4.0 only)
    @ModuleInfo var proj1: Conv2d                // 1×1
    @ModuleInfo var proj2: Conv2d                // 3×3, pad 1
    let scale: Float

    init(scale: Float, proj1In: Int) {
        _upsample1.wrappedValue = ConvTransposed2d(inputChannels: 1024, outputChannels: 512,
                                                  kernelSize: .init(2), stride: .init(2), bias: true)
        _upsample2.wrappedValue = ConvTransposed2d(inputChannels: 512, outputChannels: 256,
                                                  kernelSize: .init(2), stride: .init(2), bias: true)
        _proj1.wrappedValue = Conv2d(inputChannels: proj1In, outputChannels: 256,
                                     kernelSize: .init(1), bias: true)
        _proj2.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256,
                                     kernelSize: .init(3), padding: .init(1), bias: true)
        self.scale = scale
        super.init()
    }

    /// x: [B, 72, 72, 1024] → [B, outH, outW, 256]
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if scale == 4.0 {
            h = upsample1(h)
            h = MLXNN.gelu(h)
            h = upsample2(h)
        } else if scale == 2.0 {
            h = upsample1(h)
        }
        h = proj1(h)
        h = proj2(h)
        return h
    }
}

/// `tracker_neck` — the SAM2-side neck (deepcopy of the detector neck with its own weights).
nonisolated final class SAM3Neck: Module {
    @ModuleInfo var levels: [SAM3NeckLevel]

    override init() {
        _levels.wrappedValue = [
            SAM3NeckLevel(scale: 4.0, proj1In: 256),   // 288×288
            SAM3NeckLevel(scale: 2.0, proj1In: 512),   // 144×144
            SAM3NeckLevel(scale: 1.0, proj1In: 1024),  // 72×72
        ]
        super.init()
    }

    /// x: [B, 72, 72, 1024] → [288², 144², 72²] each at 256 channels.
    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        levels.map { $0(x) }
    }
}
