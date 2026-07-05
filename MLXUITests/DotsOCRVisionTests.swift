import Testing
import MLX
@testable import MLXUI

/// SUP-3 slice 3 (pure): the `dots_vit` conv-weight `sanitize`. No MLX GPU eval here, so it's safe
/// to run in parallel — the MLX *compute* smoke lives in `DotsOCRMLXTests` (serialized).
struct DotsOCRVisionTests {

    @Test func convSanitizeTransposesPyTorchWeightOnly() {
        // PyTorch conv weight (out,in,kH,kW) → MLX (out,kH,kW,in); already-MLX weights are untouched.
        let pytorch = MLXArray.zeros([32, 3, 2, 2])   // (out,in,kH,kW): kH==kW but layout is PyTorch
        #expect(DotsVisionModel.isMLXConvShape(pytorch) == false)

        let mlxShaped = MLXArray.zeros([32, 2, 2, 3])  // (out,kH,kW,in)
        #expect(DotsVisionModel.isMLXConvShape(mlxShaped) == true)

        let key = "vision_tower.patch_embed.patchifier.proj.weight"
        let cleaned = DotsVisionModel.sanitize([
            key: pytorch,
            "vision_tower.blocks.0.something.position_ids": MLXArray.zeros([4]),
        ])
        #expect(cleaned[key]?.shape == [32, 2, 2, 3])                            // transposed
        #expect(cleaned["vision_tower.blocks.0.something.position_ids"] == nil)  // dropped
    }
}
