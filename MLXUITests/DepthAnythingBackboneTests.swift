import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

/// MC3-4 — the DINOv2 ViT-S/14 backbone of Depth Anything V2 Small.
///
/// The op/layout tests need no weights. The real-weight suite is opt-in: `MLXUI_TEST_DEPTH=1` and
/// `MLXUI_TEST_DEPTH_DIR` must point at an installed copy (`TEST_RUNNER_` prefixes reach the test host).
struct DepthAnythingBackboneOpsTests {

    @Test func snapshotModuleKeyCamelCasesEverySegment() {
        #expect(DepthAnythingOps.moduleKey("embeddings.cls_token") == "embeddings.clsToken")
        #expect(DepthAnythingOps.moduleKey("embeddings.position_embeddings") == "embeddings.positionEmbeddings")
        #expect(DepthAnythingOps.moduleKey("embeddings.patch_embeddings.projection.weight")
                == "embeddings.patchEmbeddings.projection.weight")
        #expect(DepthAnythingOps.moduleKey("encoder.layer.0.layer_scale1.lambda1")
                == "encoder.layer.0.layerScale1.lambda1")
        #expect(DepthAnythingOps.moduleKey("neck.fusion_stage.layers.2.residual_layer1.convolution2.bias")
                == "neck.fusionStage.layers.2.residualLayer1.convolution2.bias")
        #expect(DepthAnythingOps.moduleKey("attention.attention.query.weight") == "attention.attention.query.weight")
    }

    /// A synthetic `config.json`-shaped JSON with the two fields the layout conversion reads.
    private func config(patch: Int = 14) throws -> DepthAnythingConfig {
        let json = """
        {"patch_size":14,"reassemble_hidden_size":384,"reassemble_factors":[4,2,1,0.5],
         "neck_hidden_sizes":[48,96,192,384],"fusion_hidden_size":64,"head_in_index":-1,
         "head_hidden_size":32,"depth_estimation_type":"relative","max_depth":1,
         "backbone_config":{"hidden_size":384,"num_hidden_layers":12,"num_attention_heads":6,"mlp_ratio":4,
          "hidden_act":"gelu","layer_norm_eps":1e-6,"image_size":518,"patch_size":\(patch),"num_channels":3,
          "qkv_bias":true,"layerscale_value":1,"apply_layernorm":true,"reshape_hidden_states":false,
          "use_mask_token":true,"use_swiglu_ffn":false,"out_indices":[3,6,9,12]}}
        """
        let decoded = try JSONDecoder().decode(DepthAnythingConfig.self, from: Data(json.utf8))
        try decoded.validate()
        return decoded
    }

    @Test func configDecodesAndAcceptsTheSmallCheckpoint() throws {
        let c = try config()
        #expect(c.pretrainedPatchGrid == 37)
        #expect(c.reassembleFactors == [4, 2, 1, 0.5])
        #expect(c.backboneConfig.hiddenSize == 384)
    }

    @Test func configRefusesAnotherPatchSize() throws {
        #expect(throws: DepthAnythingError.self) { _ = try config(patch: 16) }
    }

    @Test func moduleParametersMatchTheCheckpointCounts() throws {
        let c = try config()
        let backbone = DINOv2Backbone(config: c)
        let parameters = backbone.parameters().flattened()
        // 223 tensors: 5 embeddings + 12 layers x 18 + 2 final layer norm. The published checkpoint has 287.
        #expect(parameters.count == 223, "backbone tensors: \(parameters.count)")
        let total = parameters.reduce(0) { $0 + $1.1.size }
        #expect(total == 22_056_576, "backbone parameter count: \(total)")
        let keys = Set(parameters.map(\.0))
        for key in ["embeddings.clsToken", "embeddings.maskToken", "embeddings.positionEmbeddings",
                    "embeddings.patchEmbeddings.projection.weight",
                    "encoder.layer.0.attention.attention.query.weight",
                    "encoder.layer.0.layerScale1.lambda1",
                    "encoder.layer.11.mlp.fc2.weight", "layernorm.weight"] {
            #expect(keys.contains(key), "missing \(key)")
        }
        #expect(Dictionary(uniqueKeysWithValues: parameters)["embeddings.positionEmbeddings"]?.shape == [1, 1370, 384])
        #expect(Dictionary(uniqueKeysWithValues: parameters)["embeddings.clsToken"]?.shape == [1, 1, 384])
    }

    @Test func weightLayoutsConvertToTheMLXNCHWFamily() throws {
        let c = try config()
        // Conv2d: PyTorch [out, in, kH, kW] -> [out, kH, kW, in].
        let conv = MLXArray.zeros([384, 3, 14, 14])
        #expect(DepthAnythingOps.convertedWeight(name: "backbone.embeddings.patch_embeddings.projection.weight",
                                                 array: conv, config: c).shape == [384, 14, 14, 3])
        // ConvTranspose2d (reassemble factor 4): PyTorch [in, out, k, k] -> [out, k, k, in].
        let transposed = MLXArray.zeros([48, 48, 4, 4])
        #expect(DepthAnythingOps.convertedWeight(name: "neck.reassemble_stage.layers.0.resize.weight",
                                                 array: transposed, config: c).shape == [48, 4, 4, 48])
        // The factor-0.5 reassemble layer is a strided Conv2d, not a transposed convolution.
        let downsample = MLXArray.zeros([384, 384, 3, 3])
        #expect(DepthAnythingOps.convertedWeight(name: "neck.reassemble_stage.layers.3.resize.weight",
                                                 array: downsample, config: c).shape == [384, 3, 3, 384])
        // Linear weights are already [out, in].
        let linear = MLXArray.zeros([384, 384])
        #expect(DepthAnythingOps.convertedWeight(name: "backbone.encoder.layer.0.attention.attention.query.weight",
                                                 array: linear, config: c).shape == [384, 384])
    }

    @Test func bilinearResizeHitsTheExactTargetShape() {
        let x = MLXArray(0 ..< 16).asType(.float32).reshaped(1, 4, 4, 1)
        #expect(DepthAnythingOps.resizeBilinear(x, outH: 7, outW: 5, alignCorners: false).shape == [1, 7, 5, 1])
        #expect(DepthAnythingOps.resizeBilinear(x, outH: 4, outW: 4, alignCorners: false).shape == [1, 4, 4, 1])
        let up = DepthAnythingOps.resizeBilinear(x, outH: 8, outW: 8, alignCorners: true)
        eval(up)
        // align_corners=True corners map exactly to the input corners.
        #expect(abs(up[0, 0, 0, 0].item(Float.self) - 0) < 1e-5)
        #expect(abs(up[0, 7, 7, 0].item(Float.self) - 15) < 1e-5)
    }

    @Test func bicubicResizeHitsTheExactTargetShapeAndCornersMatch() {
        let x = MLXArray(0 ..< 16).asType(.float32).reshaped(1, 4, 4, 1)
        let up = DepthAnythingOps.resizeBicubic(x, outH: 7, outW: 9, alignCorners: true)
        eval(up)
        #expect(up.shape == [1, 7, 9, 1])
        #expect(abs(up[0, 0, 0, 0].item(Float.self) - 0) < 1e-5)
        #expect(abs(up[0, 6, 8, 0].item(Float.self) - 15) < 1e-5)
    }
}

@Suite(.enabled(if: DepthAnythingRealLoad.canRun, "needs MLXUI_TEST_DEPTH=1 and MLXUI_TEST_DEPTH_DIR"))
struct DepthAnythingBackboneRealLoadTests {

    /// The published checkpoint's backbone tensors load strictly: every tensor present, none unused.
    @Test func checkpointLoadsStrictly() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let config = try DepthAnythingWeights.loadConfig(directory: directory)
        let backbone = try DINOv2Backbone.load(directory: directory, config: config)
        let count = backbone.parameters().flattened().reduce(0) { $0 + $1.1.size }
        #expect(count == 22_056_576)
        // A missing file is named.
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("depth-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        #expect(throws: DepthAnythingError.self) { _ = try DINOv2Backbone.load(directory: empty, config: config) }
    }

    /// Each of the four `out_indices` feature maps after the backbone layer norm matches the reference on the
    /// non-square 112 x 154 probe (patch grid 8 x 11), which is the case that catches position-embedding
    /// interpolation errors (Trap 4).
    ///
    /// Tolerance: the reference is fp32 PyTorch and the port is fp32 MLX; only accumulation order differs.
    @Test func featuresMatchTheReferenceOnANonSquareProbe() throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let config = try DepthAnythingWeights.loadConfig(directory: directory)
        let backbone = try DINOv2Backbone.load(directory: directory, config: config)
        let input = try DepthAnythingFixtures.preprocessed("probe")
        #expect(input.shape == [1, 112, 154, 3])
        let features = backbone(input)
        eval(features)
        #expect(features.count == 4)
        var worst: Float = 0
        for (index, feature) in features.enumerated() {
            let shape = try DepthAnythingFixtures.shape("probe", ["features", String(index)])
            #expect(feature.shape == shape, "feature \(index) shape \(feature.shape), expected \(shape)")
            let file = try DepthAnythingFixtures.file(for: "probe", ["features", String(index)])
            let difference = try DepthAnythingFixtures.maxDifference(feature, file: file)
            print("DEPTH PARITY backbone feature \(index) max |Δ| = \(difference)")
            worst = max(worst, difference)
        }
        #expect(worst < 5e-4, "backbone features: max |Δ| \(worst)")
    }
}
