import CryptoKit
import Foundation
import MLXLMCommon
import Testing
@testable import MLXUI

// MP7-3b: Klein prompt template + tokenization, against diffusers 0.37.1 (g3b_tokens.json, g3b_tokens_edge.json).
//
// Where the tokenizer comes from. The 11.4 MB tokenizer.json is not committed (2 MB fixture budget). The always-run tests below
// need no tokenizer file: they prove the template string byte for byte against diffusers' own templated strings, and the
// truncate / pad / mask logic with a stand-in encoder that replays the golden token counts. The token-id parity suite is
// gated on a real tokenizer folder, found in this order: env MLXUI_TEST_KLEIN_TOKENIZER_DIR (pass TEST_RUNNER_MLXUI_TEST_KLEIN_TOKENIZER_DIR
// to xcodebuild), the installed FLUX.2 Klein model, then any installed model whose tokenizer.json is byte-identical to the pinned
// Klein one (every Qwen3 checkpoint ships the same file; mlx-community/Qwen3-8B-4bit does). Without one it is skipped with that reason.

private struct GoldenPrompt: Decodable {
    let prompt: String
    let templated: String
    let input_ids: [Int]
    let real_length: Int
    let name: String?
    let full_token_count: Int?
    let cut_token_count: Int?
    let pad_token_id: Int?
    let padding_side: String?
    let truncation_side: String?

    static func load(_ file: String) throws -> [GoldenPrompt] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/Flux2Klein/\(file)")
        return try JSONDecoder().decode([GoldenPrompt].self, from: Data(contentsOf: url))
    }
}

enum KleinTokenizerSource {
    /// sha256 and size of the pinned tokenizer.json (mlx-community/FLUX.2-klein-4B-bf16 @ 2f7dd4ac).
    static let sha256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    static let byteCount = 11_422_654

    nonisolated static let directory: URL? = {
        let fm = FileManager.default
        func usable(_ dir: URL) -> Bool { fm.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path)
            && fm.fileExists(atPath: dir.appendingPathComponent("tokenizer_config.json").path) }
        if let env = ProcessInfo.processInfo.environment["MLXUI_TEST_KLEIN_TOKENIZER_DIR"], usable(URL(fileURLWithPath: env)) {
            return URL(fileURLWithPath: env)
        }
        let klein = ModelStore.shared.directory(forHFModelID: "mlx-community/FLUX.2-klein-4B-bf16").appendingPathComponent("tokenizer")
        if usable(klein) { return klein }
        let models = (try? fm.contentsOfDirectory(at: ModelStore.shared.modelsDirectory, includingPropertiesForKeys: nil)) ?? []
        for dir in models where usable(dir) {
            let file = dir.appendingPathComponent("tokenizer.json")
            guard (try? fm.attributesOfItem(atPath: file.path)[.size] as? Int) == byteCount, let data = try? Data(contentsOf: file) else { continue }
            if SHA256.hash(data: data).map({ String(format: "%02x", $0) }).joined() == sha256 { return dir }
        }
        return nil
    }()
    nonisolated static let enabled = directory != nil
}

@Suite struct Flux2KleinPromptTemplateTests {
    @Test func theTemplateIsDiffusersByteForByte() throws {
        let cases = try GoldenPrompt.load("g3b_tokens.json") + GoldenPrompt.load("g3b_tokens_edge.json")
        #expect(cases.count == 13)
        for c in cases {
            #expect(Flux2KleinPrompt.template(c.prompt) == c.templated, Comment(rawValue: c.name ?? c.prompt))
        }
        #expect(Flux2KleinPrompt.template("") == "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test func aLiteralImageMarkerIsPassedThroughLikeDiffusers() throws {
        let edge = try GoldenPrompt.load("g3b_tokens_edge.json")
        let raw = try #require(edge.first { $0.name == "img_raw" }), stripped = try #require(edge.first { $0.name == "img_stripped" })
        #expect(Flux2KleinPrompt.template(raw.prompt).contains("[IMG]"))
        #expect(Flux2KleinPrompt.template(raw.prompt) == raw.templated, "owner ruling 2026-10-31: no strip")
        #expect(raw.templated != stripped.templated)
    }

    @Test func padCutAndMaskLogicAgainstTheGoldenCounts() throws {
        let edge = try GoldenPrompt.load("g3b_tokens_edge.json")
        for c in edge {
            let full = try #require(c.full_token_count)
            // Stand-in encoder: right number of tokens, ids are their positions + 1000 (never the pad id).
            let e = Flux2KleinPrompt.encode(c.prompt) { _ in Array(1000..<(1000 + full)) }
            #expect(e.ids.count == 512 && e.attentionMask.count == 512)
            #expect(e.realLength == c.real_length && e.cutTokenCount == c.cut_token_count, Comment(rawValue: c.name ?? ""))
            #expect(e.wasTruncated == (c.cut_token_count ?? 0 > 0))
            #expect(e.attentionMask == Array(repeating: 1, count: c.real_length) + Array(repeating: 0, count: 512 - c.real_length))
            #expect(e.ids[..<c.real_length].elementsEqual((0..<c.real_length).map { Int32(1000 + $0) }), "kept tokens are the FIRST 512")
            #expect(e.ids[c.real_length...].allSatisfy { $0 == 151_643 }, "right-padded with <|endoftext|>")
            #expect(c.pad_token_id == 151_643 && c.padding_side == "right" && c.truncation_side == "right")
        }
        #expect(Flux2KleinPrompt.padTokenID == 151_643 && Flux2KleinPrompt.maxLength == 512)
    }

    @Test func exactly512IsNotCutAndOneMoreIsCutAndCounted() {
        func run(_ n: Int) -> Flux2KleinEncodedPrompt { Flux2KleinPrompt.encode("x") { _ in Array(0..<n) } }
        let at = run(512), over = run(513), empty = run(0)
        #expect(at.cutTokenCount == 0 && !at.wasTruncated && at.realLength == 512 && !at.attentionMask.contains(0))
        #expect(over.cutTokenCount == 1 && over.wasTruncated && over.realLength == 512 && over.ids.last == 511)
        #expect(empty.realLength == 0 && empty.attentionMask.allSatisfy { $0 == 0 } && empty.ids.allSatisfy { $0 == 151_643 })
    }
}

@Suite(.enabled(if: KleinTokenizerSource.enabled,
                "needs a Qwen3 tokenizer folder: set TEST_RUNNER_MLXUI_TEST_KLEIN_TOKENIZER_DIR, or install mlx-community/Qwen3-8B-4bit or FLUX.2-klein"))
struct Flux2KleinTokenizerParityTests {
    private func tokenizer() async throws -> any MLXLMCommon.Tokenizer {
        let dir = try #require(KleinTokenizerSource.directory)
        let file = dir.appendingPathComponent("tokenizer.json")
        let sha = SHA256.hash(data: try Data(contentsOf: file)).map { String(format: "%02x", $0) }.joined()
        print("KLEIN TOKENIZER USED: \(file.path) sha256=\(sha) pinned=\(sha == KleinTokenizerSource.sha256)")
        return try await HFTokenizerLoader().load(from: dir)
    }

    @Test func everyGoldenPromptTokenizesToDiffusersIdsAndMask() async throws {
        let tok = try await tokenizer()
        // One `SMOKE|row|check|result|detail` line per prompt group, the format model-smoke.sh counts (checks=N).
        let groups: [(String, [GoldenPrompt])] = [
            ("standard prompts (short, quoted text, empty, accents/CJK/emoji)", try GoldenPrompt.load("g3b_tokens.json")),
        ] + (try {
            let edge = try GoldenPrompt.load("g3b_tokens_edge.json")
            func pick(_ names: [String]) -> [GoldenPrompt] { edge.filter { names.contains($0.name ?? "") } }
            return [("length edges: 511, 512, 513 and 700 tokens (cut at the end)", pick(["at511", "at512", "past513", "past700"])),
                    ("CJK, whitespace, literal special-token text", pick(["cjk", "whitespace", "special_literal"])),
                    ("literal [IMG] passed through, and its stripped twin", pick(["img_raw", "img_stripped"]))]
        }())
        #expect(groups.map(\.1.count).reduce(0, +) == 13)
        for (title, cases) in groups {
            var failures: [String] = []
            for c in cases {
                let e = try Flux2KleinPrompt.encode(c.prompt, tokenizer: tok)
                let name = Comment(rawValue: c.name ?? String(c.prompt.prefix(20)))
                let ok = e.realLength == c.real_length && e.ids[..<e.realLength].map(Int.init) == c.input_ids
                    && e.cutTokenCount == (c.cut_token_count ?? 0) && e.ids.count == 512 && e.attentionMask.reduce(0, +) == Int32(c.real_length)
                #expect(e.realLength == c.real_length, name)
                #expect(e.ids[..<e.realLength].map(Int.init) == c.input_ids, name)
                #expect(e.cutTokenCount == (c.cut_token_count ?? 0), name)
                #expect(e.ids.count == 512 && e.attentionMask.reduce(0, +) == Int32(c.real_length), name)
                if !ok { failures.append(c.name ?? String(c.prompt.prefix(20))) }
            }
            print("SMOKE|3b|tokenizer parity: \(title)|\(failures.isEmpty ? "PASS" : "FAIL")|\(cases.count) prompts equal diffusers ids, length, cut count and mask\(failures.isEmpty ? "" : "; differ: " + failures.joined(separator: ", "))")
        }
    }

    @Test func theNamedEdgeCasesBehaveAsSpecified() async throws {
        let tok = try await tokenizer()
        let edge = try GoldenPrompt.load("g3b_tokens_edge.json")
        func enc(_ n: String) throws -> Flux2KleinEncodedPrompt {
            try Flux2KleinPrompt.encode(try #require(edge.first { $0.name == n }).prompt, tokenizer: tok)
        }
        let empty = try Flux2KleinPrompt.encode("", tokenizer: tok)
        #expect(empty.realLength == 12 && empty.cutTokenCount == 0, "the bare template is 12 tokens")
        let at = try enc("at512"), past = try enc("past513"), long = try enc("past700")
        #expect(at.realLength == 512 && at.cutTokenCount == 0)
        #expect(past.realLength == 512 && past.cutTokenCount == 1)
        #expect(long.realLength == 512 && long.cutTokenCount == 188)
        // The cut is at the end, so a long prompt loses its closing <|im_end|> (151645) and the assistant tail; shorter ones keep it.
        func imEnds(_ e: Flux2KleinEncodedPrompt) -> Int { e.ids.filter { $0 == 151_645 }.count }
        #expect(imEnds(at) == 1 && imEnds(past) == 1 && imEnds(long) == 0)
        let cjk = try enc("cjk")
        #expect(cjk.realLength == 47 && cjk.cutTokenCount == 0)
    }

    @Test func aTokenizerWithAnotherVocabularyIsRefused() throws {
        struct Other: MLXLMCommon.Tokenizer {
            func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
            func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
            func convertTokenToId(_ token: String) -> Int? { token == "<|endoftext|>" ? 50256 : nil }
            func convertIdToToken(_ id: Int) -> String? { nil }
            var bosToken: String? { nil }
            var eosToken: String? { nil }
            var unknownToken: String? { nil }
            func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?, additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
        }
        #expect(throws: Flux2KleinPromptError.padTokenMismatch(found: 50256)) { try Flux2KleinPrompt.encode("a", tokenizer: Other()) }
    }
}
