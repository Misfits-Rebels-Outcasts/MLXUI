import Testing
import Foundation
@testable import MLXUI

/// SUP-2 slice 2: `DeepSeekOCRConfig` decoding + `DeepSeekOCRWeights.remapKey`. The fixture is the
/// real `mlx-community/DeepSeek-OCR-2-bf16` `config.json` (trimmed), so a schema drift surfaces here.
struct DeepSeekOCRConfigTests {

    private static let configJSON = """
    {
        "architectures": ["DeepseekOCR2ForCausalLM"],
        "bos_token_id": 0, "eos_token_id": 1,
        "candidate_resolutions": [[1024, 1024]],
        "first_k_dense_replace": 1,
        "global_view_pos": "head",
        "hidden_size": 1280,
        "intermediate_size": 6848,
        "max_position_embeddings": 8192,
        "model_type": "deepseekocr_2",
        "moe_intermediate_size": 896,
        "n_group": 1, "n_routed_experts": 64, "n_shared_experts": 2,
        "num_attention_heads": 10, "num_experts_per_tok": 6,
        "num_hidden_layers": 12, "num_key_value_heads": 10,
        "projector_config": {"input_dim": 896, "model_type": "mlp_projector", "n_embed": 1280, "projector_type": "linear"},
        "qk_nope_head_dim": 0, "qk_rope_head_dim": 0,
        "tile_tag": "2D", "topk_group": 1, "topk_method": "greedy",
        "use_mla": false, "v_head_dim": 0,
        "vision_config": {
            "image_size": 1024, "mlp_ratio": 3.7362, "model_type": "vision",
            "width": {
                "qwen2-0-5b": {"dim": 896},
                "sam_vit_b": {"downsample_channels": [512, 1024], "global_attn_indexes": [2, 5, 8, 11], "heads": 12, "layers": 12, "width": 768}
            }
        },
        "vocab_size": 129280
    }
    """

    @Test func decodesTextVisionProjectorConfig() throws {
        let cfg = try JSONDecoder().decode(DeepSeekOCRConfig.self, from: Data(Self.configJSON.utf8))

        // Text (DeepSeek-V2, Llama/MHA)
        #expect(cfg.hiddenSize == 1280)
        #expect(cfg.numHiddenLayers == 12)
        #expect(cfg.numAttentionHeads == 10)
        #expect(cfg.numKeyValueHeads == 10)   // MHA
        #expect(cfg.headDim == 128)           // 1280 / 10
        #expect(cfg.vocabSize == 129280)
        #expect(cfg.firstKDenseReplace == 1)
        #expect(cfg.nRoutedExperts == 64)
        #expect(cfg.nSharedExperts == 2)
        #expect(cfg.numExpertsPerTok == 6)
        #expect(cfg.topkMethod == "greedy")
        // Absent in JSON → config.py defaults
        #expect(cfg.scoringFunc == "softmax")
        #expect(cfg.routedScalingFactor == 1.0)
        #expect(cfg.ropeTheta == 10_000)
        #expect(cfg.rmsNormEps == 1e-6)
        #expect(cfg.attentionBias == false)
        #expect(cfg.imageTokenIndex == 128815)

        // Multimodal top-level
        #expect(cfg.tileTag == "2D")
        #expect(cfg.globalViewPos == "head")

        // Vision (SAM + Qwen2 encoder)
        #expect(cfg.vision.samWidth == 768)
        #expect(cfg.vision.samLayers == 12)
        #expect(cfg.vision.samHeads == 12)
        #expect(cfg.vision.samGlobalAttnIndexes == [2, 5, 8, 11])
        #expect(cfg.vision.samDownsampleChannels == [512, 1024])
        #expect(cfg.vision.samFinalOutChannels == 896)   // default (OCR-2)
        #expect(cfg.vision.qwen2Dim == 896)
        #expect(cfg.vision.qwen2Layers == 24)            // default
        #expect(cfg.vision.qwen2KVHeads == 2)            // default

        // Projector
        #expect(cfg.projector.inputDim == 896)
        #expect(cfg.projector.nEmbed == 1280)
        #expect(cfg.projector.projectorType == "linear")
    }

    @Test func remapsWeightKeysLikePythonTransformKey() {
        // qwen2 encoder rewrites (checked before the broad LM rewrites)
        #expect(DeepSeekOCRWeights.remapKey("model.qwen2_model.model.model.layers.0.self_attn.q_proj.weight")
                == "vision_model.qwen2_encoder.layers.0.self_attn.q_proj.weight")
        #expect(DeepSeekOCRWeights.remapKey("model.qwen2_model.model.model.norm.weight")
                == "vision_model.qwen2_encoder.norm.weight")
        #expect(DeepSeekOCRWeights.remapKey("model.qwen2_model.query_1024.weight")
                == "vision_model.qwen2_encoder.query_1024")
        #expect(DeepSeekOCRWeights.remapKey("model.qwen2_model.query_768.weight")
                == "vision_model.qwen2_encoder.query_768")
        // LM rewrites (guarded against qwen2/language_model)
        #expect(DeepSeekOCRWeights.remapKey("model.layers.5.mlp.gate.weight")
                == "language_model.model.layers.5.mlp.gate.weight")
        #expect(DeepSeekOCRWeights.remapKey("model.embed_tokens.weight")
                == "language_model.model.embed_tokens.weight")
        #expect(DeepSeekOCRWeights.remapKey("model.norm.weight")
                == "language_model.model.norm.weight")
        // vision / sam / projector / separator-typo / lm_head
        #expect(DeepSeekOCRWeights.remapKey("model.vision_model.foo") == "vision_model.foo")
        #expect(DeepSeekOCRWeights.remapKey("model.sam_model.blocks.0.attn.qkv.weight")
                == "sam_model.blocks.0.attn.qkv.weight")
        #expect(DeepSeekOCRWeights.remapKey("model.projector.layers.weight") == "projector.layers.weight")
        #expect(DeepSeekOCRWeights.remapKey("model.view_seperator") == "view_separator")
        #expect(DeepSeekOCRWeights.remapKey("lm_head.weight") == "language_model.lm_head.weight")
    }
}
