import Testing
import Foundation
import MLX
import MLXEmbedders
import MLXLMCommon
@testable import MLXUI

/// CL-4b golden test — `CLMEncoder` against `Fixtures/CLM/{tokens,embeddings}.json`
/// (provenance in `Fixtures/CLM/PROVENANCE.md`). Needs the real encoder weights at
/// `Fixtures/CLM/encoder_weights/` — **not committed to git** (`.gitignore`, unlike the 72 MB
/// heads fixture: this is the ~8 GB Qwen3-8B checkpoint). Re-download from
/// `https://huggingface.co/RealityCat/CLM-v0.1-8B-MLX-8bit/resolve/main/encoder/` (6 small
/// files + the two `model-0000{1,2}-of-00002.safetensors` shards) to run this locally.
@Suite(.serialized)
struct CLMEncoderTests {
    private struct TokensGoldenCase: Decodable {
        var text_length_chars: Int
        var ids: [Int]
        var count: Int
    }

    private struct EmbeddingsGoldenCase: Decodable {
        var norm: Double
        var dim: Int
        var vector: [Double]
    }

    private static let repoRoot: URL = {
        let filePath = #filePath                      // .../MLXUITests/CLMEncoderTests.swift
        return URL(fileURLWithPath: filePath).deletingLastPathComponent().deletingLastPathComponent()
    }()

    private static var texts: [String: String] {
        // `repetitive_2048` was the 0.9941-cosine outlier `embeddingCosineSpreadCheck` found
        // (deleted 2026-09-28, owner ruling) — entirely version skew (Python mlx 0.32.2 vs the
        // mlx-swift 0.31.4-matched 0.31.1), not a Swift bug; folded in here so the committed
        // gate covers it going forward instead of a separate diagnostic fixture.
        let repetitive2048 = String(
            repeating: "This invoice line item was reviewed for the reported discrepancy. ", count: 256
        ).dropLast()
        return [
            "short": "The customer said the shipment arrived three days late.",
            "medium": "Customer: my invoice was charged twice and nobody answers the phone! "
                + "Please refund the second charge and confirm by email.",
            "empty": "",
            "long_over_2048": String(repeating: "This invoice line item was reviewed for accuracy and completeness. ", count: 400),
            "structured": "invoice_id: INV-2044\nsubtotal: 500\ntax: 45\ntotal: 545\n\n"
                + "Does the total match the line items and tax?",
            "repetitive_2048": String(repetitive2048),
        ]
    }

    private func loadGolden<T: Decodable>(_ name: String) throws -> [String: T] {
        let url = Self.repoRoot.appendingPathComponent("Fixtures/CLM/\(name).json")
        return try JSONDecoder().decode([String: T].self, from: Data(contentsOf: url))
    }

    private func encoderDirectory() -> URL {
        Self.repoRoot.appendingPathComponent("Fixtures/CLM/encoder_weights")
    }

    private func requireEncoderWeights() throws {
        guard FileManager.default.fileExists(
            atPath: encoderDirectory().appendingPathComponent("model-00001-of-00002.safetensors").path)
        else {
            throw CLMHeadsError.invalidConfig(
                "Fixtures/CLM/encoder_weights/ not present locally (gitignored, ~8GB) — "
                    + "re-download from the HF repo's encoder/ folder to run this test.")
        }
    }

    private func loadContainer() async throws -> EmbedderModelContainer {
        try requireEncoderWeights()
        return try await EmbedderModelFactory.shared.loadContainer(
            from: encoderDirectory(), using: HFTokenizerLoader())
    }

    @Test func tokenIdsMatchGoldenExactly() async throws {
        let golden: [String: TokensGoldenCase] = try loadGolden("tokens")
        let container = try await loadContainer()
        for (name, text) in Self.texts {
            let expected = try #require(golden[name])
            let (ids, _) = await container.perform { context in
                CLMEncoder.ids(for: text, tokenizer: context.tokenizer)
            }
            #expect(ids == expected.ids, "\(name): token ids differ")
            #expect(ids.count == expected.count, "\(name): token count differs")
        }
    }

    @Test func longTextTruncatesToMaxTokens() async throws {
        let container = try await loadContainer()
        let longText = try #require(Self.texts["long_over_2048"])
        let (ids, truncated) = await container.perform { context in
            CLMEncoder.ids(for: longText, tokenizer: context.tokenizer)
        }
        #expect(ids.count == CLMEncoder.maxTokens)
        #expect(truncated == true)
    }

    @Test func emptyTextEmbedsAsASingleSpace() async throws {
        let container = try await loadContainer()
        let (ids, truncated) = await container.perform { context in
            CLMEncoder.ids(for: "", tokenizer: context.tokenizer)
        }
        #expect(!ids.isEmpty)
        #expect(truncated == false)
    }

    /// `repetitive_2048` gets its own, lower bar (owner ruling 2026-09-29, CL-4b): a
    /// time-boxed investigation (diffing mlx-swift-lm's `MLXEmbedders.Qwen3Model` against
    /// Python mlx_lm's `qwen3.py`/`base.py`, then an A/B against `MLXLLM.Qwen3Model.model`)
    /// found no code difference to align — RoPE, the causal mask, and the SDPA call are all
    /// the same framework primitives on both Swift model classes for the no-cache case, and
    /// the A/B produced the bit-identical 0.9997831312353592 either way. The one found
    /// difference (Python's SwiGLU is `mx.compile`-fused; Swift's is eager `silu(gate) *
    /// up`) is the leading unconfirmed hypothesis for why this specific highly-repetitive,
    /// near-tied-attention sequence amplifies a gap that ordinary text doesn't show — see
    /// the CL-4b journal for the full write-up. Not demoted to non-blocking.
    private static let repetitiveTextCosineFloor = 0.9997
    private static let ordinaryTextCosineFloor = 0.9999

    @Test func embeddingsMatchGoldenCosine() async throws {
        let golden: [String: EmbeddingsGoldenCase] = try loadGolden("embeddings")
        let container = try await loadContainer()
        let names = Array(Self.texts.keys)
        let vectors: [[Float]] = await container.perform { context in
            let ids = names.map { CLMEncoder.ids(for: Self.texts[$0]!, tokenizer: context.tokenizer).ids }
            let embeddings = CLMEncoder.embed(idLists: ids, context: context)
            eval(embeddings)
            return (0 ..< embeddings.dim(0)).map { row in
                embeddings[row ..< row + 1].squeezed(axis: 0).asArray(Float.self)
            }
        }
        for (index, name) in names.enumerated() {
            let expected = try #require(golden[name])
            let vector = vectors[index]
            #expect(vector.count == expected.dim, "\(name): dim mismatch")

            let floor = name == "repetitive_2048" ? Self.repetitiveTextCosineFloor : Self.ordinaryTextCosineFloor
            let cos = cosine(vector.map(Double.init), expected.vector)
            #expect(cos >= floor, "\(name): cosine \(cos) below \(floor)")

            let norm = sqrt(vector.reduce(0.0) { $0 + Double($1) * Double($1) })
            #expect(abs(norm - expected.norm) < 1e-3, "\(name): norm \(norm) vs \(expected.norm)")
        }
    }

    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
        let normA = sqrt(a.reduce(0.0) { $0 + $1 * $1 })
        let normB = sqrt(b.reduce(0.0) { $0 + $1 * $1 })
        return dot / (normA * normB)
    }
}
