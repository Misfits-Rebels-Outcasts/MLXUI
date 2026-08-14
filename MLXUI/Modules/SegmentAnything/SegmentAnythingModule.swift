import Foundation

/// Bundles the SAM3 image segmentation SDK and registers it into the `ModelRegistry`.
/// A from-reference Swift/MLX port of Meta's SAM3 (`sam3_video` / `Sam3VideoModel`),
/// loading the installed `mlx-community/sam3-4bit` weights. SA-MOD1.
enum SegmentAnythingModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "segment-anything",
        displayName: "Segment Anything Model 3 (SAM3, MLX)",
        modalities: ["image"],
        modelTypes: [.segmentation],
        backingPackage: "facebookresearch/sam2 (Swift port)",
        packageLicense: "apache-2.0",
        maintainers: ["Meta", "mlx-community"],
        notes: """
        Image segmentation via a Swift/MLX port of Meta's SAM3 (sam3_video architecture): \
        32-layer ViT backbone with windowed + global attention and RoPE positional encodings, \
        256-channel neck projection, pixel decoder with 3-stage conv upsampling, and a \
        center-point default prompt. Interactive point/box prompting is wired up in \
        SegmentAnythingRunView (SA-AM5). Runs the installed mlx-community/sam3-4bit 4-bit \
        quantized weights.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: SegmentAnythingSDK(), ui: SegmentAnythingUI(), descriptor: descriptor)
    }
}
