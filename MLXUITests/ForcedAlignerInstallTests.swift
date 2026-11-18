import Testing
import Foundation
import MLXAudioSTT
@testable import MLXUI

/// FA-2, trap 1 — the aligner shares Qwen3-ASR's install side effect: on first load `Qwen3ForcedAlignerModel
/// .fromModelDirectory` calls `Qwen3ASRModel.generateTokenizerJSONIfMissing`, which writes `tokenizer.json` beside the
/// weights (the repo ships none), built from `vocab.json` + `merges.txt` + `tokenizer_config.json`. These tests decide,
/// with the real library code and the real `InstallManager`, whether that write is harmless — the same decision
/// `Qwen3ASRInstallTests` made for Qwen3-ASR (readme T5):
///
/// - **Verify.** `InstallManager.verifyFiles` runs on the *download* folder's selected files before the atomic move. It
///   is never run on an installed folder, and `tokenizer.json` is not among the selected files, so a generated file
///   cannot fail it.
/// - **Reconcile.** `.installed` is the whole signal; nothing lists the folder.
/// - **Registry.** `sizeBytes` is informational and grows by the generated file's size.
/// - **Re-install and uninstall.** Both remove the folder, generated file included.
/// - **Sandbox.** Application Support is writable in both editions; this test host writes to its temp folder.
@MainActor
struct ForcedAlignerInstallTests {

    private static let hfModelID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    private static let cardID = "mlx-community--Qwen3-ForcedAligner-0.6B-8bit"

    /// A deliberately tiny config in the real file's shape (`thinker_config` wrapper, `support_languages`, the
    /// `qwen3_forced_aligner` thinker with `classify_num`): the library decodes every field with a default, so the model
    /// it builds here is small and nothing is evaluated.
    private static let tinyConfig = """
    {
      "model_type": "qwen3_asr",
      "support_languages": ["Chinese", "English"],
      "timestamp_token_id": 151705,
      "timestamp_segment_time": 80,
      "thinker_config": {
        "model_type": "qwen3_forced_aligner",
        "classify_num": 5000,
        "audio_config": {"encoder_layers": 1, "encoder_attention_heads": 2, "encoder_ffn_dim": 16, "d_model": 16,
                         "output_dim": 16, "downsample_hidden_size": 8, "num_mel_bins": 128},
        "text_config": {"model_type": "qwen3", "vocab_size": 8, "hidden_size": 16, "intermediate_size": 32,
                        "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8}
      }
    }
    """

    /// The files `ModelFileSelector` picks for this repo's pinned sibling list, as stubs (sizes are not the real ones;
    /// the real selection and its sizes are pinned in `ForcedAlignerCatalogTests` and `Fixtures/ForcedAligner`).
    private static let selected = ["chat_template.json", "config.json", "generation_config.json", "merges.txt",
                                   "model.safetensors", "model.safetensors.index.json", "preprocessor_config.json",
                                   "tokenizer_config.json", "vocab.json"]

    private func makeFolder() throws -> (root: URL, model: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("fa-t1-\(UUID().uuidString)", isDirectory: true)
        let model = root.appendingPathComponent("models/\(Self.cardID)", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        return (root, model)
    }

    private func writeStubInstall(into model: URL) throws {
        for name in Self.selected {
            let content: String
            switch name {
            case "config.json": content = Self.tinyConfig
            case "vocab.json": content = #"{"a": 0, "b": 1, "ab": 2}"#
            case "merges.txt": content = "#version: 0.2\na b\n"
            case "tokenizer_config.json":
                content = #"{"tokenizer_class": "Qwen2Tokenizer", "added_tokens_decoder": {"3": {"content": "<|endoftext|>", "special": true}}}"#
            case "model.safetensors": content = ""   // not a real safetensors file: the load must stop at the weights
            default: content = "{}"
            }
            try content.write(to: model.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        FileManager.default.createFile(atPath: model.appendingPathComponent(".installed").path, contents: nil)
    }

    private func listing(_ url: URL) throws -> Set<String> {
        Set(try FileManager.default.contentsOfDirectory(atPath: url.path))
    }

    /// The library's own write, observed: loading the stub install adds exactly one file, `tokenizer.json`, leaves every
    /// other file byte-identical, and leaves the `.installed` marker alone. (The load itself then fails at the weights,
    /// which are a stub; the tokenizer write happens first.)
    @Test func firstLoadAddsExactlyTokenizerJSONAndTouchesNothingElse() async throws {
        let (root, model) = try makeFolder()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeStubInstall(into: model)
        let before = try listing(model)
        var bytesBefore: [String: Data] = [:]
        for name in before { bytesBefore[name] = try Data(contentsOf: model.appendingPathComponent(name)) }

        var loadError: Error?
        do { _ = try await Qwen3ForcedAlignerModel.fromModelDirectory(model) } catch { loadError = error }
        #expect(loadError != nil, "a stub safetensors file cannot be a model; the point is what was written before it failed")

        let after = try listing(model)
        #expect(after.subtracting(before) == ["tokenizer.json"], "added: \(after.subtracting(before).sorted())")
        #expect(before.subtracting(after).isEmpty, "removed: \(before.subtracting(after).sorted())")
        for name in before {
            #expect(try Data(contentsOf: model.appendingPathComponent(name)) == bytesBefore[name], "\(name) changed")
        }
        // The generated file is real JSON with the vocab and merges it was built from.
        let generated = try JSONSerialization.jsonObject(with: Data(contentsOf: model.appendingPathComponent("tokenizer.json")))
        let tokenizer = try #require(generated as? [String: Any])
        let bpe = try #require(tokenizer["model"] as? [String: Any])
        #expect((bpe["vocab"] as? [String: Int])?["ab"] == 2)
        #expect((bpe["merges"] as? [String])?.first == "a b")
    }

    /// Loading twice is idempotent: an existing `tokenizer.json` is not rewritten.
    @Test func aSecondLoadDoesNotRewriteTheGeneratedFile() async throws {
        let (root, model) = try makeFolder()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeStubInstall(into: model)
        _ = try? await Qwen3ForcedAlignerModel.fromModelDirectory(model)
        let url = model.appendingPathComponent("tokenizer.json")
        let first = try Data(contentsOf: url)
        _ = try? await Qwen3ForcedAlignerModel.fromModelDirectory(model)
        #expect(try Data(contentsOf: url) == first)
    }

    /// The `InstallManager` side: the reconcile, the Browse "installed" answer, the flow-readiness answer and the registry
    /// are all the same before and after the file appears; only the registry's informational size grows; uninstall
    /// removes the folder, generated file included.
    @Test func theGeneratedFileIsInvisibleToReconcileRegistryAndUninstall() throws {
        let (root, model) = try makeFolder()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ModelStore(baseDirectory: root)
        let manager = InstallManager(store: store)
        let entry = makeEntry(id: Self.cardID, family: "Qwen3-ForcedAligner", modelType: .align, source: .mlx,
                              hfRepo: "mlx-community", hfModelId: Self.hfModelID)
        try writeStubInstall(into: model)
        #expect(store.directory(forHFModelID: Self.hfModelID).standardizedFileURL == model.standardizedFileURL)

        func snapshot() -> (reconciled: Set<String>, installed: Bool, ready: Set<String>, size: Int?) {
            manager.saveRegistry(installedIDs: [Self.cardID], browserData: nil)
            let registry = (try? JSONDecoder().decode(InstalledModels.self, from: Data(contentsOf: store.installedRegistryURL)))
            return (manager.loadInstalled(modelIDs: [Self.cardID], catalog: [entry]),
                    manager.isInstalled(entry),
                    manager.readyModelIDs(registered: [], catalog: [entry]),
                    registry?.models[Self.cardID]?.sizeBytes)
        }
        let before = snapshot()
        #expect(before.reconciled == [Self.cardID] && before.installed && before.ready == [Self.cardID])
        let sizeBefore = try #require(before.size)

        let generated = String(repeating: "x", count: 4096)
        try generated.write(to: model.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        #expect(FileManager.default.fileExists(atPath: store.installedMarker(forHFModelID: Self.hfModelID).path))

        let after = snapshot()
        #expect(after.reconciled == before.reconciled && after.installed == before.installed && after.ready == before.ready)
        #expect(after.size == sizeBefore + generated.utf8.count, "sizeBytes is informational and follows the folder")

        manager.uninstall(entry, catalog: [entry], installedModelIDs: [Self.cardID])
        #expect(!FileManager.default.fileExists(atPath: model.path), "uninstall removes the folder, tokenizer.json with it")
    }

    /// Verify, by construction: the files `verifyFiles` checks are the selector's output, and `tokenizer.json` is not in
    /// it — for the pinned sibling list, and for any repo that ships no `tokenizer.json`.
    @Test func tokenizerJSONIsNeverInTheVerifiedSetForThisRepo() {
        let siblings = [".gitattributes", "README.md", "chat_template.json", "config.json", "generation_config.json",
                        "merges.txt", "model.safetensors", "model.safetensors.index.json", "preprocessor_config.json",
                        "tokenizer_config.json", "vocab.json"]
        #expect(Set(ModelFileSelector.filesToDownload(siblings: siblings)) == Set(Self.selected))
        #expect(!ModelFileSelector.filesToDownload(siblings: siblings).contains("tokenizer.json"))
    }
}
