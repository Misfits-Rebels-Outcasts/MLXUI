import Testing
import Foundation
@testable import MLXUI

/// SUP-3 slice 2: `DotsOCRConfig` decoding + `DotsOCRWeights.remapKey`. The fixture is the real
/// `mlx-community/dots.mocr-4bit` `config.json` (trimmed to the fields we decode), so a schema drift
/// in the checkpoint surfaces here rather than at load time.
struct DotsOCRConfigTests {

    private static let configJSON = """
    {
        "architectures": ["DotsOCRForCausalLM"],
        "attention_bias": true,
        "eos_token_id": [151643, 151672, 151673],
        "hidden_act": "silu",
        "hidden_size": 1536,
        "image_token_id": 151665,
        "intermediate_size": 8960,
        "max_position_embeddings": 131072,
        "model_type": "dots_ocr",
        "num_attention_heads": 12,
        "num_hidden_layers": 28,
        "num_key_value_heads": 2,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000,
        "tie_word_embeddings": false,
        "video_token_id": 151656,
        "vision_config": {
            "embed_dim": 1536,
            "hidden_size": 1536,
            "intermediate_size": 4224,
            "num_hidden_layers": 42,
            "num_attention_heads": 12,
            "num_channels": 3,
            "patch_size": 14,
            "post_norm": true,
            "rms_norm_eps": 1e-05,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "use_bias": false
        },
        "vocab_size": 151936
    }
    """

    @Test func decodesTextAndVisionConfig() throws {
        let cfg = try JSONDecoder().decode(DotsOCRConfig.self, from: Data(Self.configJSON.utf8))

        // Text (Qwen2, flat at top level)
        #expect(cfg.modelType == "dots_ocr")
        #expect(cfg.hiddenSize == 1536)
        #expect(cfg.intermediateSize == 8960)
        #expect(cfg.numHiddenLayers == 28)
        #expect(cfg.numAttentionHeads == 12)
        #expect(cfg.numKeyValueHeads == 2)
        #expect(cfg.vocabSize == 151936)
        #expect(cfg.ropeTheta == 1_000_000)
        #expect(cfg.attentionBias == true)
        #expect(cfg.tieWordEmbeddings == false)
        #expect(cfg.imageTokenId == 151665)
        #expect(cfg.videoTokenId == 151656)
        #expect(cfg.headDim == 128)  // 1536 / 12

        // Vision (dots_vit, nested)
        #expect(cfg.visionConfig.embedDim == 1536)
        #expect(cfg.visionConfig.intermediateSize == 4224)  // ≠ text intermediate (8960)
        #expect(cfg.visionConfig.numHiddenLayers == 42)
        #expect(cfg.visionConfig.numAttentionHeads == 12)
        #expect(cfg.visionConfig.patchSize == 14)
        #expect(cfg.visionConfig.spatialMergeSize == 2)
        #expect(cfg.visionConfig.postNorm == true)
        #expect(cfg.visionConfig.useBias == false)
        #expect(cfg.visionConfig.rmsNormEps == 1e-5)  // vision eps ≠ text eps (1e-6)

        // Quantization (4-bit)
        #expect(cfg.quantization?.bits == 4)
        #expect(cfg.quantization?.groupSize == 64)
    }

    @Test func remapsWeightKeysLikePythonSanitize() {
        // vision_tower prefix wins over the broader model. prefix (order-sensitive)
        #expect(DotsOCRWeights.remapKey("model.vision_tower.blocks.0.attn.qkv.weight")
                == "vision_tower.blocks.0.attn.qkv.weight")
        // model.* → language_model.model.*
        #expect(DotsOCRWeights.remapKey("model.embed_tokens.weight")
                == "language_model.model.embed_tokens.weight")
        #expect(DotsOCRWeights.remapKey("model.layers.5.self_attn.q_proj.weight")
                == "language_model.model.layers.5.self_attn.q_proj.weight")
        // lm_head.* → language_model.model.lm_head.*
        #expect(DotsOCRWeights.remapKey("lm_head.weight")
                == "language_model.model.lm_head.weight")
        // quantized sub-keys ride along untouched by the prefix rewrite
        #expect(DotsOCRWeights.remapKey("model.layers.0.mlp.gate_proj.scales")
                == "language_model.model.layers.0.mlp.gate_proj.scales")
        // already-namespaced / unknown keys pass through
        #expect(DotsOCRWeights.remapKey("vision_tower.patch_embed.patchifier.proj.weight")
                == "vision_tower.patch_embed.patchifier.proj.weight")
    }
}
