import Foundation
import MLX
import MLXNN
import MLXFast

/// OpenCLIP ViT-G/14 text encoder (`text_encoder_2/`) → `[1, 77, 1280]`. Same architecture and
/// key layout as `SDCLIPEncoder` (reuses `SDCLIPCoreTextModel`), but the larger config:
/// 1280-dim, **32 layers**, 20 heads, 5120 FFN, **plain GELU** (not quick-GELU), no projection.
///
/// Checkpoint facts: the `stabilityai/sdxl-turbo` `text_encoder_2/model.fp16.safetensors`
/// ships `text_projection.weight [1280,1280]` (CLIPTextModelWithProjection). The UNet's
/// `text_time` added conditioning consumes the **pooled** output — the EOS last-hidden state
/// passed through `text_projection` — so this encoder declares the projection and exposes
/// `pooled(_:)` (SD-ENG4). The UNet itself cross-attends to the *unprojected* `last_hidden_state`.
nonisolated final class SDOpenCLIPEncoder: Module {
    static let config = SDCLIPConfig(
        dModel: 1280, dFF: 5120, heads: 20, headDim: 64,
        vocab: 49408, maxLength: 77, layers: 32, activation: .gelu)

    @ModuleInfo(key: "text_model") var textModel: SDCLIPCoreTextModel
    @ModuleInfo(key: "text_projection") var textProjection: Linear

    override init() {
        self._textModel.wrappedValue = SDCLIPCoreTextModel(config: Self.config)
        self._textProjection.wrappedValue = Linear(Self.config.dModel, Self.config.dModel, bias: false)
        super.init()
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        textModel(tokens)
    }

    /// `CLIPTextModelWithProjection` pooled output: the last hidden state at the EOS (argmax)
    /// token, passed through `text_projection` → `[1, 1280]`. Used for the UNet's `text_embeds`.
    func pooled(_ tokens: MLXArray) -> MLXArray {
        let hidden = textModel(tokens)
        let eosIndex = argMax(tokens, axis: -1)[0]       // scalar — EOS position
        let vec = hidden[0 ..< 1, eosIndex]              // [1, 1280] (keep the batch axis)
        return textProjection(vec)
    }
}
