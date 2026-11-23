import Foundation
import Testing
@testable import MLXUI

/// MC4 review follow-up (issue #8) — the bf16 probe. The shipped 4-bit checkpoint only reaches
/// cosine 0.89-0.94 against the official bf16 reference (a quantization limit, floors ruled on
/// issue #8); this test proves the *forward pass* is right by loading the **official bf16
/// checkpoint** through the same loader and asking for >= 0.999 on the same text cases (the PR
/// author measured >= 0.9997).
///
/// Opt-in and never downloads anything: skipped unless `MLXUI_TEST_QWEN3VL_EMBED_BF16_DIR` (reaching
/// the host as `TEST_RUNNER_MLXUI_TEST_QWEN3VL_EMBED_BF16_DIR`) names a local copy of
/// `Qwen/Qwen3-VL-Embedding-2B` containing `config.json`, `preprocessor_config.json`, the tokenizer
/// files and the `model*.safetensors` weights. Copy it under `/private/tmp` — the sandboxed test
/// host cannot read `~/Library/Application Support`.
enum Qwen3VLEmbedBF16Probe {
    nonisolated static let directory: URL? = {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MLXUI_TEST_QWEN3VL_EMBED_BF16_DIR"]
            ?? env["TEST_RUNNER_MLXUI_TEST_QWEN3VL_EMBED_BF16_DIR"],
              !path.isEmpty else { return nil }
        let url = URL(fileURLWithPath: path)
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
            return nil
        }
        return url
    }()

    nonisolated static var canRun: Bool { directory != nil }
}

struct Qwen3VLEmbedBF16ParityTests {

    /// Each text case against the bf16 reference vector: 2048 dims, norm 1, cosine >= 0.999.
    @Test(.enabled(if: Qwen3VLEmbedBF16Probe.canRun))
    func bf16TextVectorsMatchTheReference() async throws {
        let directory = try #require(Qwen3VLEmbedBF16Probe.directory)
        let golden = try Qwen3VLEmbedRealLoad.golden()
        let cases = golden.cases.filter { $0.kind == "text" }
        #expect(!cases.isEmpty)
        let embedder = try await Qwen3VLEmbedder.load(from: directory)
        for testCase in cases {
            let text = try #require(testCase.text)
            let vector = try await embedder.embedTexts([text])[0]
            let cosine = Qwen3VLEmbedRealLoad.cosine(
                vector, try Qwen3VLEmbedRealLoad.reference(named: testCase.name))
            let norm = Qwen3VLEmbedRealLoad.norm(vector)
            print("QWEN3VLEMBED bf16 text=\(testCase.name) dims=\(vector.count) norm=\(norm) cosine=\(cosine)")
            #expect(vector.count == Qwen3VLEmbeddingRunDimensions.expected)
            #expect(abs(norm - 1) <= 1e-4)
            #expect(cosine >= 0.999, "\(testCase.name): cosine \(cosine) vs the bf16 reference")
        }
    }
}
