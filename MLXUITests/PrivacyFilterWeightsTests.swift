import Foundation
import Testing
import MLX
@testable import MLXUI

/// PF-5 — the gate and paths shared by the real-weights suites (the `SAMAudioRealLoadTests` pattern):
/// they run only with `MLXUI_TEST_PRIVACY=1` **and** a `MLXUI_TEST_PRIVACY_DIR` folder that holds the
/// checkpoint. `xcodebuild` reaches the host through the `TEST_RUNNER_` prefix, so both spellings
/// are accepted. A skipped run is not evidence.
nonisolated enum PrivacyFilterTestEnvironment {
    nonisolated static let environment = ProcessInfo.processInfo.environment

    nonisolated static var optedIn: Bool {
        (environment["MLXUI_TEST_PRIVACY"] ?? environment["TEST_RUNNER_MLXUI_TEST_PRIVACY"]) == "1"
    }

    /// The installed copy — normally `/private/tmp/pf-model`, because the sandboxed test host cannot
    /// read `~/Library/Application Support`.
    nonisolated static var directory: URL? {
        guard var raw = environment["MLXUI_TEST_PRIVACY_DIR"]
            ?? environment["TEST_RUNNER_MLXUI_TEST_PRIVACY_DIR"]
        else { return nil }
        if raw.hasPrefix("/tmp/") { raw = "/private" + raw }
        let url = URL(fileURLWithPath: raw)
        let manager = FileManager.default
        guard manager.fileExists(atPath: url.appendingPathComponent("config.json").path),
              manager.fileExists(atPath: url.appendingPathComponent("model.safetensors").path) else {
            return nil
        }
        return url
    }

    nonisolated static var canRun: Bool { optedIn && directory != nil }

    /// The checked-in reference fixtures (`MLXUITests/Fixtures/PrivacyFilter`). The override exists so
    /// a developer can point at a scratch copy of the same layout while PF-3 is still generating.
    nonisolated static var fixturesDirectory: URL {
        if let raw = environment["MLXUI_TEST_PRIVACY_FIXTURES_DIR"]
            ?? environment["TEST_RUNNER_MLXUI_TEST_PRIVACY_FIXTURES_DIR"]
        {
            return URL(fileURLWithPath: raw)
        }
        return URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()                        // MLXUITests
            .appendingPathComponent("Fixtures/PrivacyFilter")
    }

    /// The tensor keys in a `.safetensors` header, without reading any tensor data.
    nonisolated static func safetensorsKeys(at url: URL) -> [String] {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return [] }
        defer { try? handle.close() }
        guard let lengthData = try? handle.read(upToCount: 8), lengthData.count == 8 else { return [] }
        let length = lengthData.withUnsafeBytes { $0.load(as: UInt64.self) }
        guard length > 0, length < 50_000_000,
              let header = try? handle.read(upToCount: Int(length)),
              let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any]
        else { return [] }
        return json.keys.filter { $0 != "__metadata__" }
    }
}

/// A tiny Privacy Filter checkpoint written by the tests themselves: small enough that the strict
/// loader can be exercised without the 2.8 GB download, but with exactly the checkpoint's key set and
/// dtypes. Values are deterministic small numbers, so the model runs a real (if meaningless) forward.
nonisolated enum TinyPrivacyFilterCheckpoint {
    static let hiddenSize = 8
    static let layerCount = 2
    static let headCount = 2
    static let kvHeadCount = 1
    static let headDim = 4
    static let intermediateSize = 8
    static let expertCount = 4
    static let expertsPerToken = 2
    static let vocabSize = 16
    static let labelNames = ["O", "B-secret", "I-secret", "E-secret"]

    static let configJSON = """
    {
      "model_type": "openai_privacy_filter",
      "hidden_size": \(hiddenSize),
      "num_hidden_layers": \(layerCount),
      "num_attention_heads": \(headCount),
      "num_key_value_heads": \(kvHeadCount),
      "head_dim": \(headDim),
      "intermediate_size": \(intermediateSize),
      "num_local_experts": \(expertCount),
      "num_experts_per_tok": \(expertsPerToken),
      "vocab_size": \(vocabSize),
      "rms_norm_eps": 1e-05,
      "sliding_window": 64,
      "max_position_embeddings": 128,
      "rope_parameters": {
        "rope_type": "yarn",
        "rope_theta": 10000.0,
        "factor": 32.0,
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "original_max_position_embeddings": 64,
        "truncate": false
      },
      "id2label": {
        "0": "O", "1": "B-secret", "2": "I-secret", "3": "E-secret"
      }
    }
    """

    /// Deterministic small values, different per tensor.
    static func values(_ count: Int, seed: Int) -> MLXArray {
        MLXArray((0 ..< count).map { index in
            Float(((index &* 31 &+ seed) % 41) - 20) / 40
        })
    }

    static func normalizedWeights(_ count: Int, seed: Int) -> MLXArray {
        MLXArray((0 ..< count).map { index in
            1 + Float(((index &* 17 &+ seed) % 11) - 5) / 20
        })
    }

    static func tensors() -> [String: MLXArray] {
        var tensors: [String: MLXArray] = [:]
        let hidden = hiddenSize
        let qDim = headCount * headDim
        let kvDim = kvHeadCount * headDim

        tensors["model.embed_tokens.weight"] = values(vocabSize * hidden, seed: 1)
            .asType(.bfloat16).reshaped([vocabSize, hidden])
        for layer in 0 ..< layerCount {
            let prefix = "model.layers.\(layer)."
            tensors[prefix + "input_layernorm.weight"] = normalizedWeights(hidden, seed: 2 + layer)
                .asType(.bfloat16)
            tensors[prefix + "self_attn.q_proj.weight"] = values(qDim * hidden, seed: 3 + layer)
                .asType(.bfloat16).reshaped([qDim, hidden])
            tensors[prefix + "self_attn.q_proj.bias"] = values(qDim, seed: 4 + layer)
                .asType(.bfloat16)
            tensors[prefix + "self_attn.k_proj.weight"] = values(kvDim * hidden, seed: 5 + layer)
                .asType(.bfloat16).reshaped([kvDim, hidden])
            tensors[prefix + "self_attn.k_proj.bias"] = values(kvDim, seed: 6 + layer)
                .asType(.bfloat16)
            tensors[prefix + "self_attn.v_proj.weight"] = values(kvDim * hidden, seed: 7 + layer)
                .asType(.bfloat16).reshaped([kvDim, hidden])
            tensors[prefix + "self_attn.v_proj.bias"] = values(kvDim, seed: 8 + layer)
                .asType(.bfloat16)
            tensors[prefix + "self_attn.o_proj.weight"] = values(hidden * qDim, seed: 9 + layer)
                .asType(.bfloat16).reshaped([hidden, qDim])
            tensors[prefix + "self_attn.o_proj.bias"] = values(hidden, seed: 10 + layer)
                .asType(.bfloat16)
            // The one fp32 tensor.
            tensors[prefix + "self_attn.sinks"] = values(headCount, seed: 11 + layer)
                .asType(.float32)
            tensors[prefix + "post_attention_layernorm.weight"] = normalizedWeights(hidden, seed: 12 + layer)
                .asType(.bfloat16)
            tensors[prefix + "mlp.router.weight"] = values(expertCount * hidden, seed: 13 + layer)
                .asType(.bfloat16).reshaped([expertCount, hidden])
            tensors[prefix + "mlp.router.bias"] = values(expertCount, seed: 14 + layer)
                .asType(.bfloat16)
            tensors[prefix + "mlp.experts.gate_up_proj"] =
                values(expertCount * hidden * 2 * intermediateSize, seed: 15 + layer)
                .asType(.bfloat16).reshaped([expertCount, hidden, 2 * intermediateSize])
            tensors[prefix + "mlp.experts.gate_up_proj_bias"] =
                values(expertCount * 2 * intermediateSize, seed: 16 + layer)
                .asType(.bfloat16).reshaped([expertCount, 2 * intermediateSize])
            tensors[prefix + "mlp.experts.down_proj"] =
                values(expertCount * intermediateSize * hidden, seed: 17 + layer)
                .asType(.bfloat16).reshaped([expertCount, intermediateSize, hidden])
            tensors[prefix + "mlp.experts.down_proj_bias"] =
                values(expertCount * hidden, seed: 18 + layer)
                .asType(.bfloat16).reshaped([expertCount, hidden])
        }
        tensors["model.norm.weight"] = normalizedWeights(hidden, seed: 19).asType(.bfloat16)
        tensors["score.weight"] = values(labelNames.count * hidden, seed: 20)
            .asType(.bfloat16).reshaped([labelNames.count, hidden])
        tensors["score.bias"] = values(labelNames.count, seed: 21).asType(.bfloat16)
        return tensors
    }

    /// Write `config.json` + `model.safetensors`, optionally omitting a key / adding an extra tensor.
    @discardableResult
    static func write(
        to directory: URL,
        omitting: String? = nil,
        adding extra: [String: MLXArray] = [:],
        replacing overrides: [String: MLXArray] = [:],
        config: String = configJSON
    ) throws -> URL {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(config.utf8).write(to: directory.appendingPathComponent("config.json"))
        var tensors = tensors()
        if let omitting { tensors.removeValue(forKey: omitting) }
        for (key, value) in overrides { tensors[key] = value }
        for (key, value) in extra { tensors[key] = value }
        try MLX.save(arrays: tensors, url: directory.appendingPathComponent("model.safetensors"))
        return directory
    }

    /// A fresh temporary directory; the caller removes it.
    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("privacy-filter-tiny-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

/// PF-5, Trap 6 — the strict loader: the full tensor set is required in exactly the expected shapes and
/// dtypes (sinks fp32, everything else bf16), and the first missing/unused/mismatched key is named in
/// one plain sentence. The synthetic checkpoint here exercises every rule without the real download;
/// the gated test at the bottom checks the real file's 140 keys.
@Suite(.serialized)
struct PrivacyFilterWeightsTests {

    @Test
    func theStrictLoaderAcceptsATinyCheckpoint() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try TinyPrivacyFilterCheckpoint.write(to: directory)

        let config = try PrivacyFilterConfig.load(from: directory)
        #expect(config.hiddenSize == TinyPrivacyFilterCheckpoint.hiddenSize)
        #expect(config.numLayers == TinyPrivacyFilterCheckpoint.layerCount)
        #expect(config.numExperts == TinyPrivacyFilterCheckpoint.expertCount)
        #expect(config.labelNames == TinyPrivacyFilterCheckpoint.labelNames, "id2label order is the label order")

        let weights = try PrivacyFilterWeights.load(from: directory, config: config)
        #expect(weights.layers.count == TinyPrivacyFilterCheckpoint.layerCount)
        #expect(weights.embedTokens.shape == [TinyPrivacyFilterCheckpoint.vocabSize, TinyPrivacyFilterCheckpoint.hiddenSize])
        #expect(weights.scoreWeight.shape == [TinyPrivacyFilterCheckpoint.labelNames.count, TinyPrivacyFilterCheckpoint.hiddenSize])
        #expect(weights.layers[0].sinks.dtype == .float32, "sinks are the one fp32 tensor")
        #expect(weights.layers[0].gateUpProj.shape == [
            TinyPrivacyFilterCheckpoint.expertCount,
            TinyPrivacyFilterCheckpoint.hiddenSize,
            2 * TinyPrivacyFilterCheckpoint.intermediateSize,
        ])
    }

    @Test
    func theStrictLoaderRunsATinyForwardPassAndReadsLogProbs() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try TinyPrivacyFilterCheckpoint.write(to: directory)

        let model = try PrivacyFilterModel.load(from: directory)
        let tokenIDs = [1, 2, 3]
        let logits = model.logits(tokenIDs: tokenIDs)
        #expect(logits.shape == [tokenIDs.count, TinyPrivacyFilterCheckpoint.labelNames.count])
        let flat = try MLXErrorGuard.readRows(logits.asType(.float32), as: Float.self)
        #expect(flat.count == tokenIDs.count)
        #expect(flat.flatMap { $0 }.allSatisfy { $0.isFinite })

        let logProbs = try model.logProbs(tokenIDs: tokenIDs)
        #expect(logProbs.count == tokenIDs.count)
        for row in logProbs {
            #expect(row.count == TinyPrivacyFilterCheckpoint.labelNames.count)
            let total = row.map { exp(Double($0)) }.reduce(0, +)
            #expect(abs(total - 1) < 1e-4, Comment(rawValue: "each log-softmax row is normalised (\(total))"))
            #expect(row.allSatisfy { $0.isFinite })
        }

        // RoPE positions are absolute: a shifted call is a different (still finite) computation.
        let offsetLogits = model.logits(tokenIDs: tokenIDs, positionOffset: 1024)
        #expect(offsetLogits.shape == logits.shape)
        let offsetValues = try MLXErrorGuard.read(offsetLogits.asType(.float32), as: Float.self)
        #expect(offsetValues.allSatisfy { $0.isFinite })
    }

    @Test
    func aMissingTensorIsNamedInOneSentence() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let key = "model.layers.1.mlp.router.bias"
        try TinyPrivacyFilterCheckpoint.write(to: directory, omitting: key)

        let config = try PrivacyFilterConfig.load(from: directory)
        do {
            _ = try PrivacyFilterWeights.load(from: directory, config: config)
            Issue.record("a checkpoint missing \(key) must not load")
        } catch {
            let sentence = error.localizedDescription
            #expect(sentence.contains(key), Comment(rawValue: "the sentence names the missing tensor: \(sentence)"))
            #expect(sentence.hasPrefix("the model's weights are missing the tensor"), Comment(rawValue: sentence))
            #expect(!sentence.contains(". ") && sentence.hasSuffix("."), Comment(rawValue: "one sentence: \(sentence)"))
        }
    }

    @Test
    func anUnusedTensorIsNamedInOneSentence() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let key = "model.layers.0.mlp.experts.extra_proj"
        try TinyPrivacyFilterCheckpoint.write(
            to: directory,
            adding: [key: MLXArray.zeros([2], dtype: .bfloat16)])

        let config = try PrivacyFilterConfig.load(from: directory)
        do {
            _ = try PrivacyFilterWeights.load(from: directory, config: config)
            Issue.record("an unused tensor must refuse the load")
        } catch {
            let sentence = error.localizedDescription
            #expect(sentence.contains(key), Comment(rawValue: "the sentence names the unused tensor: \(sentence)"))
            #expect(sentence.contains("unused"), Comment(rawValue: sentence))
            #expect(!sentence.contains(". ") && sentence.hasSuffix("."), Comment(rawValue: "one sentence: \(sentence)"))
        }
    }

    @Test
    func aWrongShapeIsNamedInOneSentence() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let key = "model.layers.0.self_attn.q_proj.weight"
        try TinyPrivacyFilterCheckpoint.write(
            to: directory,
            replacing: [key: MLXArray.zeros([7, TinyPrivacyFilterCheckpoint.hiddenSize], dtype: .bfloat16)])

        let config = try PrivacyFilterConfig.load(from: directory)
        do {
            _ = try PrivacyFilterWeights.load(from: directory, config: config)
            Issue.record("a wrong-shaped tensor must refuse the load")
        } catch {
            let sentence = error.localizedDescription
            #expect(sentence.contains(key), Comment(rawValue: "the sentence names the offending tensor: \(sentence)"))
            #expect(sentence.contains("has shape"), Comment(rawValue: sentence))
            #expect(!sentence.contains(". ") && sentence.hasSuffix("."), Comment(rawValue: "one sentence: \(sentence)"))
        }
    }

    @Test
    func sinksMustBeFloat32() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let key = "model.layers.0.self_attn.sinks"
        try TinyPrivacyFilterCheckpoint.write(
            to: directory,
            replacing: [key: MLXArray.zeros([TinyPrivacyFilterCheckpoint.headCount], dtype: .bfloat16)])

        let config = try PrivacyFilterConfig.load(from: directory)
        do {
            _ = try PrivacyFilterWeights.load(from: directory, config: config)
            Issue.record("bf16 sinks must refuse the load (they are fp32 in the checkpoint)")
        } catch {
            let sentence = error.localizedDescription
            #expect(sentence.contains(key), Comment(rawValue: "the sentence names the tensor: \(sentence)"))
            #expect(sentence.contains("expected float32"), Comment(rawValue: sentence))
            #expect(!sentence.contains(". ") && sentence.hasSuffix("."), Comment(rawValue: "one sentence: \(sentence)"))
        }
    }

    @Test
    func anEmptyFolderIsRefusedByName() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        do {
            _ = try PrivacyFilterConfig.load(from: directory)
            Issue.record("a folder without config.json must be refused")
        } catch {
            #expect(error.localizedDescription.contains("config.json"), Comment(rawValue: error.localizedDescription))
        }
    }

    @Test
    func aConfigWithTheWrongRopeTypeIsRefused() throws {
        let directory = try TinyPrivacyFilterCheckpoint.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try TinyPrivacyFilterCheckpoint.write(
            to: directory,
            config: TinyPrivacyFilterCheckpoint.configJSON.replacingOccurrences(
                of: "\"rope_type\": \"yarn\"", with: "\"rope_type\": \"default\""))
        do {
            _ = try PrivacyFilterConfig.load(from: directory)
            Issue.record("a non-YaRN checkpoint must be refused")
        } catch {
            #expect(error.localizedDescription.contains("rope_type"), Comment(rawValue: error.localizedDescription))
        }
    }

    // MARK: - The real checkpoint (gated, `SAMAudioRealLoadTests` pattern)

    @Test(.enabled(if: PrivacyFilterTestEnvironment.canRun))
    func theRealCheckpointHasExactlyTheExpectedTensorsAndLoads() throws {
        let directory = try #require(PrivacyFilterTestEnvironment.directory)
        let keys = PrivacyFilterTestEnvironment.safetensorsKeys(at: directory.appendingPathComponent("model.safetensors"))
        print("PFWEIGHTS realTensorKeys=\(keys.count)")
        #expect(keys.count == 140, "openai/privacy-filter's model.safetensors carries 140 tensors")
        #expect(keys.contains("score.weight") && keys.contains("score.bias"))
        #expect(keys.filter { $0.hasSuffix("self_attn.sinks") }.count == 8)

        let model = try PrivacyFilterModel.load(from: directory)
        print("PFWEIGHTS layers=\(model.weights.layers.count) labels=\(model.config.numLabels) vocab=\(model.config.vocabSize)")
        #expect(model.weights.layers.count == 8)
        #expect(model.config.numLabels == 33)
        #expect(model.weights.layers[0].sinks.dtype == .float32)
        #expect(model.weights.layers[0].sinks.shape == [14])

        // The one primitive decoding needs: [T, 33] logits, then normalised log-softmax rows.
        let tokenIDs = [15339, 1234, 5678, 42]
        let logits = model.logits(tokenIDs: tokenIDs)
        #expect(logits.shape == [tokenIDs.count, 33])
        let logProbs = try model.logProbs(tokenIDs: tokenIDs)
        #expect(logProbs.count == tokenIDs.count)
        for row in logProbs {
            let total = row.map { exp(Double($0)) }.reduce(0, +)
            #expect(abs(total - 1) < 1e-4)
            #expect(row.allSatisfy { $0.isFinite })
        }
        // A working model does not return one flat row: the argmax is a real label id.
        let argmaxes = logProbs.map { row in row.indices.max { row[$0] < row[$1] } ?? 0 }
        print("PFWEIGHTS argmaxes=\(argmaxes) peakMemoryGB=\(Double(Memory.snapshot().peakMemory) / 1e9)")
    }
}
