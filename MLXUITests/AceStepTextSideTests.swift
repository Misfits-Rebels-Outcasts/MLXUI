import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3d: the ACE-Step text side. Formatting and the projector need nothing; the tokenizer golden needs only the real
/// `Qwen3-Embedding-0.6B/tokenizer.json` (run-if-present, light); the encoder's loading path is proved on a tiny synthetic
/// Qwen3; the real hidden-state golden is env-gated (`AceStepTextSideRealTests`).
enum AceStepTextFixtures {
    nonisolated static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/MC5/mc5_tokens.json")

    nonisolated static let defaultModelDirectory = "/Volumes/Ovid/Users/chinpinboo/Documents/PipelineStudio/RSI/notes/modelcandidates/mc5/oracle/Ace-Step1.5"

    /// An installed-layout folder holding the real tokenizer (the oracle copy, or `MLXUI_TEST_ACESTEP_DIR`).
    nonisolated static var modelDirectory: URL? {
        let env = ProcessInfo.processInfo.environment
        let raw = env["MLXUI_TEST_ACESTEP_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_ACESTEP_DIR"] ?? defaultModelDirectory
        let url = URL(fileURLWithPath: raw)
        let tok = url.appendingPathComponent("Qwen3-Embedding-0.6B/tokenizer.json")
        return FileManager.default.fileExists(atPath: tok.path) ? url : nil
    }

    nonisolated static var haveTokenizer: Bool { modelDirectory != nil }

    struct Prompt { let caption: String; let lyrics: String; let instrumental: Bool; let language: String }

    /// The three prompt inputs (from `mc5/scripts/prompts.json`), duration 10 s.
    nonisolated static let prompts: [String: Prompt] = [
        "instrumental": Prompt(caption: "upbeat electronic dance music with heavy bass, punchy drums, festival energy, 128 BPM",
                               lyrics: "[Instrumental]", instrumental: true, language: "unknown"),
        "english_vocal": Prompt(caption: "warm acoustic singer-songwriter ballad, intimate male vocal, fingerpicked guitar, soft strings, 84 BPM",
                                lyrics: "[Verse 1]\nThe city lights are fading slow\nI trace the roads we used to know\nEvery window holds a ghost of you\n\n[Chorus]\nBut I keep walking through the rain\nHolding on to what remains\nCalling out your name into the blue",
                                instrumental: false, language: "en"),
        "chinese_vocal": Prompt(caption: "gentle Mandarin pop ballad, female vocal, piano and strings, emotional, 72 BPM",
                                lyrics: "[Verse 1]\n窗外的雨还在下\n回忆在房间慢慢发芽\n那些没说完的话\n\n[Chorus]\n我还站在原地等你回答\n夜色把街灯都点亮\n你的名字还在心上",
                                instrumental: false, language: "zh"),
    ]

    nonisolated static func golden() throws -> [String: Any] {
        let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL))
        return try #require(obj as? [String: Any])
    }

    nonisolated static func entry(_ id: String) throws -> [String: Any] {
        let prompts = try #require(try golden()["prompts"] as? [String: Any])
        return try #require(prompts[id] as? [String: Any])
    }

    nonisolated static func ints(_ any: Any?) -> [Int] { (any as? [NSNumber])?.map(\.intValue) ?? [] }
}

struct AceStepPromptFormattingTests {

    @Test func textAndLyricStringsAreByteExact() throws {
        for (id, p) in AceStepTextFixtures.prompts {
            let e = try AceStepTextFixtures.entry(id)
            let wantText = try #require(e["text_str"] as? String)
            let wantLyric = try #require(e["lyric_str"] as? String)
            #expect(AceStepPrompt.textString(caption: p.caption, duration: 10) == wantText, "\(id) text")
            #expect(AceStepPrompt.lyricString(lyrics: p.lyrics, language: p.language, instrumental: p.instrumental) == wantLyric, "\(id) lyric")
        }
    }

    @Test func instrumentalUsesTheLyricPathWithUnknown() {
        let s = AceStepPrompt.lyricString(lyrics: "ignored words", language: "en", instrumental: true)
        #expect(s == "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>")
    }

    /// Python `int()` truncates: 59.9 s is "59 seconds", 10.0 is "10 seconds".
    @Test func durationTruncatesLikePythonInt() {
        #expect(AceStepPrompt.textString(caption: "x", duration: 59.9).contains("- duration: 59 seconds\n"))
        #expect(AceStepPrompt.textString(caption: "x", duration: 10).contains("- duration: 10 seconds\n"))
        #expect(AceStepPrompt.textString(caption: "x", duration: 240).hasSuffix("- duration: 240 seconds\n<|endoftext|>\n"))
    }

    @Test func projectorMatchesANumpyComputedValue() {
        // x [1,2,3] @ W.T with W [4,3] = [[1,0,0],[0,2,0],[0,0,3],[.5,.5,.5]]; expected from numpy (x @ W.T).
        let x = MLXArray([Float(1), 2, 3, 0.5, -1, 2], [1, 2, 3])
        let w = MLXArray([Float(1), 0, 0, 0, 2, 0, 0, 0, 3, 0.5, 0.5, 0.5], [4, 3])
        let y = AceStepTextProjector.project(x, weight: w)
        #expect(y.shape == [1, 2, 4])
        let got = y.asArray(Float.self)
        let want: [Float] = [1, 4, 9, 3, 0.5, -2, 6, 0.75]
        #expect(zip(got, want).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(AceStepTextProjector.weightName == "encoder.text_projector.weight")
    }

    @Test func projectorKeepsBF16() {
        let x = MLXArray.ones([1, 2, 1024]).asType(.bfloat16)
        let w = MLXArray.ones([2048, 1024]).asType(.bfloat16)
        let y = AceStepTextProjector.project(x, weight: w)
        #expect(y.shape == [1, 2, 2048] && y.dtype == .bfloat16)
    }
}

@Suite(.enabled(if: AceStepTextFixtures.haveTokenizer, "needs Qwen3-Embedding-0.6B/tokenizer.json (the oracle copy, or MLXUI_TEST_ACESTEP_DIR)"))
struct AceStepTokenizerTests {

    private func tokenizer() async throws -> AceStepTokenizer {
        try await AceStepTokenizer(directory: try #require(AceStepTextFixtures.modelDirectory))
    }

    @Test func idsMatchTheUpstreamGoldenExactly() async throws {
        let tok = try await tokenizer()
        print("ACESTEP-TOKENIZER postProcessorAppendsEndOfText=\(tok.postProcessorAppendsEndOfText)")
        for (id, p) in AceStepTextFixtures.prompts {
            let e = try AceStepTextFixtures.entry(id)
            let text = tok.encodeText(caption: p.caption, duration: 10)
            let lyric = tok.encodeLyrics(lyrics: p.lyrics, language: p.language, instrumental: p.instrumental)
            #expect(text == AceStepTextFixtures.ints(e["text_ids"]), "\(id) text ids")
            #expect(lyric == AceStepTextFixtures.ints(e["lyric_ids"]), "\(id) lyric ids")
            // T25: the template's own literal EOS plus the tokenizer's appended one.
            #expect(Array(text.suffix(3)) == [AceStepPrompt.endOfText, 198, AceStepPrompt.endOfText])
            #expect(Array(lyric.suffix(2)) == [AceStepPrompt.endOfText, AceStepPrompt.endOfText])
        }
    }

    @Test func longPromptsTruncateToTheFirstN() async throws {
        let tok = try await tokenizer()
        let t = try #require(try AceStepTextFixtures.golden()["truncation"] as? [String: Any])
        let textStr = try #require(t["long_text_str"] as? String)
        let lyricStr = try #require(t["long_lyric_str"] as? String)
        let text = tok.encode(textStr, maxLength: AceStepPrompt.maxTextTokens)
        let lyric = tok.encode(lyricStr, maxLength: AceStepPrompt.maxLyricTokens)
        #expect(text.count == 256)
        #expect(Array(text.suffix(3)) == [448, 9450, 151_643])
        #expect(text == AceStepTextFixtures.ints(t["long_text_ids_max256"]))
        #expect(lyric.count == 2048)
        #expect(lyric == AceStepTextFixtures.ints(t["long_lyric_ids_max2048"]))
    }

    @Test func aMissingFolderThrowsAndDoesNotReachForTheNetwork() async throws {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-notok-\(UUID().uuidString)")
        await #expect(throws: AceStepWeightsError.missingFile("Qwen3-Embedding-0.6B/tokenizer.json")) {
            _ = try await AceStepTokenizer(directory: empty)
        }
    }
}

/// The encoder's loading path and shapes on a tiny synthetic Qwen3 (1 layer, hidden 8), so the strict name check, the shared
/// embedding table and the causal, unpooled forward are proved without real weights.
struct AceStepTextEncoderTinyTests {

    static func lcg(_ n: Int, seed: UInt32) -> [Float] {
        var s = seed
        return (0..<n).map { _ in s = s &* 1_664_525 &+ 1_013_904_223; return Float(Int(s >> 9) % 2001 - 1000) / 5000 }
    }

    static func makeFolder(dropping drop: String? = nil) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-tinyqwen-\(UUID().uuidString)")
        let qdir = dir.appendingPathComponent("Qwen3-Embedding-0.6B")
        try FileManager.default.createDirectory(at: qdir, withIntermediateDirectories: true)
        let config = """
        {"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,
         "num_key_value_heads":1,"head_dim":4,"rms_norm_eps":1e-6,"vocab_size":16,"rope_theta":10000,
         "rope_scaling":null,"tie_word_embeddings":true,"max_position_embeddings":64}
        """
        try config.write(to: qdir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        var specs: [AceStepWeightsConversionTests.Spec] = []
        var seed: UInt32 = 1
        func add(_ name: String, _ shape: [Int], ones: Bool = false) {
            seed += 1
            let n = shape.reduce(1, *)
            specs.append(AceStepWeightsConversionTests.f32(ones ? Array(repeating: 1, count: n) : lcg(n, seed: seed), shape, name))
        }
        add("embed_tokens.weight", [16, 8])
        add("layers.0.self_attn.q_proj.weight", [8, 8]); add("layers.0.self_attn.k_proj.weight", [4, 8])
        add("layers.0.self_attn.v_proj.weight", [4, 8]); add("layers.0.self_attn.o_proj.weight", [8, 8])
        add("layers.0.self_attn.q_norm.weight", [4], ones: true); add("layers.0.self_attn.k_norm.weight", [4], ones: true)
        add("layers.0.mlp.gate_proj.weight", [16, 8]); add("layers.0.mlp.up_proj.weight", [16, 8]); add("layers.0.mlp.down_proj.weight", [8, 16])
        add("layers.0.input_layernorm.weight", [8], ones: true); add("layers.0.post_attention_layernorm.weight", [8], ones: true)
        add("norm.weight", [8], ones: true)
        if let drop { specs.removeAll { $0.name == drop } }
        try AceStepWeightsConversionTests.write(specs, to: qdir.appendingPathComponent("model.safetensors"))
        return dir
    }

    @Test func aMissingFolderThrowsAndDoesNotReachForTheNetwork() async throws {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-noqwen-\(UUID().uuidString)")
        await #expect(throws: AceStepWeightsError.missingFile("Qwen3-Embedding-0.6B/config.json")) {
            _ = try await AceStepTextEncoder.load(directory: empty)
        }
        // A folder with a config but no weights throws too (named), never a download.
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-cfgonly-\(UUID().uuidString)")
        let q = dir.appendingPathComponent("Qwen3-Embedding-0.6B")
        try FileManager.default.createDirectory(at: q, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"model_type":"qwen3","hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"rms_norm_eps":1e-6,"vocab_size":16}"#
            .write(to: q.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        await #expect(throws: AceStepWeightsError.missingFile("model.safetensors")) {
            _ = try await AceStepTextEncoder.load(directory: dir)
        }
    }

    @Test func loadsStrictlyAndAMissingTensorFailsTheLoad() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try await AceStepTextEncoder.load(directory: dir)
        #expect(encoder.embedTable.shape == [16, 8])

        let bad = try Self.makeFolder(dropping: "layers.0.mlp.down_proj.weight")
        defer { try? FileManager.default.removeItem(at: bad) }
        await #expect(throws: (any Error).self) { _ = try await AceStepTextEncoder.load(directory: bad) }
    }

    @Test func hiddenStatesAreCausalUnpooledAndLyricRowsAreAnExactGather() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let encoder = try await AceStepTextEncoder.load(directory: dir)
        let a = try encoder.hiddenStates(ids: [3, 5, 7, 9])
        let b = try encoder.hiddenStates(ids: [3, 5, 7, 2])
        #expect(a.shape == [1, 4, 8], "one vector per token, no pooling")
        // Causal: changing the last token leaves the earlier positions unchanged and changes the last.
        let early = abs(a[0..., 0 ..< 3, 0...] - b[0..., 0 ..< 3, 0...]).max().item(Float.self)
        let late = abs(a[0..., 3 ..< 4, 0...] - b[0..., 3 ..< 4, 0...]).max().item(Float.self)
        #expect(early < 1e-5)
        #expect(late > 1e-4)

        let lyric = try encoder.lyricEmbeddings(ids: [4, 4, 11])
        #expect(lyric.shape == [1, 3, 8])
        #expect(throws: AceStepEngineError.emptyTokens("lyric tokens")) { try encoder.lyricEmbeddings(ids: []) }
        let table = encoder.embedTable
        #expect(abs(lyric[0, 0] - table[4]).max().item(Float.self) == 0)
        #expect(abs(lyric[0, 2] - table[11]).max().item(Float.self) == 0)

        let inputs = try encoder.inputs(textIds: [3, 5, 7], lyricIds: [4, 11])
        #expect(inputs.textMask.shape == [1, 3] && inputs.lyricMask.shape == [1, 2])
        #expect(inputs.textMask.asArray(Int32.self) == [1, 1, 1])
        #expect(inputs.textHidden.shape == [1, 3, 8] && inputs.lyricEmbeds.shape == [1, 2, 8])
    }
}
