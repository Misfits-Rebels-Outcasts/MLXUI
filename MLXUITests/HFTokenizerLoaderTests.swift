import Foundation
import MLXLMCommon
import Testing

@testable import MLXUI

/// `HFTokenizerLoader`'s fallback for a `tokenizer_class` the pinned swift-transformers doesn't
/// register (journal `2026-496`): Clef Flash declares `Qwen3_5Tokenizer`, and without this the app
/// refuses a checkpoint whose `tokenizer.json` is an ordinary BPE model. The fixture is synthetic —
/// a three-token BPE vocabulary — so the test needs neither the network nor the real model.
struct HFTokenizerLoaderTests {
    private func makeModelFolder(tokenizerClass: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("hf-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let tokenizer = """
        {
          "version": "1.0",
          "truncation": null,
          "padding": null,
          "added_tokens": [],
          "normalizer": null,
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]]
          }
        }
        """
        let config = """
        {"tokenizer_class": "\(tokenizerClass)", "model_max_length": 128}
        """
        try tokenizer.write(to: directory.appendingPathComponent("tokenizer.json"), atomically: true, encoding: .utf8)
        try config.write(to: directory.appendingPathComponent("tokenizer_config.json"), atomically: true, encoding: .utf8)
        return directory
    }

    @Test func anUnregisteredTokenizerClassFallsBackToBPE() async throws {
        let directory = try makeModelFolder(tokenizerClass: "TotallyNewTokenizer")
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        // The vocabulary is a real BPE one, so the fallback is the same tokenizer that a registered
        // class would have built: "ab" merges, "a" and "b" do not.
        #expect(tokenizer.convertTokenToId("a") == 0)
        #expect(tokenizer.convertTokenToId("ab") == 2)
        #expect(tokenizer.encode(text: "ab", addSpecialTokens: false) == [2])
    }

    @Test func aRegisteredClassStillUsesTheStrictPath() async throws {
        let directory = try makeModelFolder(tokenizerClass: "BPETokenizer")
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try await HFTokenizerLoader().load(from: directory)
        #expect(tokenizer.encode(text: "ab", addSpecialTokens: false) == [2])
    }
}
