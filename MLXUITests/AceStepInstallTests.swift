import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// MC5-3b: what the ACE-Step 1.5 install downloads, the "no runner yet" flag, and the bundled `silence_latent`.
@MainActor
struct AceStepInstallTests {

    static let hfID = "ACE-Step/Ace-Step1.5"

    /// The whole pinned sibling list (28 files, path → bytes), `ACE-Step/Ace-Step1.5` @ 19671f4, from
    /// `RSI/notes/modelcandidates/hf/ACE-Step_Ace-Step1.5/tree.json` (directories dropped), including the LM folder.
    static let siblings: [(String, Int)] = [
        (".gitattributes", 1_740),
        ("Qwen3-Embedding-0.6B/added_tokens.json", 707),
        ("Qwen3-Embedding-0.6B/chat_template.jinja", 4_116),
        ("Qwen3-Embedding-0.6B/config.json", 1_359),
        ("Qwen3-Embedding-0.6B/merges.txt", 1_671_853),
        ("Qwen3-Embedding-0.6B/model.safetensors", 1_191_586_416),
        ("Qwen3-Embedding-0.6B/special_tokens_map.json", 613),
        ("Qwen3-Embedding-0.6B/tokenizer.json", 11_423_705),
        ("Qwen3-Embedding-0.6B/tokenizer_config.json", 5_404),
        ("Qwen3-Embedding-0.6B/vocab.json", 2_776_833),
        ("README.md", 5_496),
        ("acestep-5Hz-lm-1.7B/added_tokens.json", 2_217_787),
        ("acestep-5Hz-lm-1.7B/chat_template.jinja", 4_168),
        ("acestep-5Hz-lm-1.7B/config.json", 1_385),
        ("acestep-5Hz-lm-1.7B/merges.txt", 1_671_853),
        ("acestep-5Hz-lm-1.7B/model.safetensors", 3_708_521_528),
        ("acestep-5Hz-lm-1.7B/special_tokens_map.json", 1_824_199),
        ("acestep-5Hz-lm-1.7B/tokenizer.json", 24_321_939),
        ("acestep-5Hz-lm-1.7B/tokenizer_config.json", 14_072_925),
        ("acestep-5Hz-lm-1.7B/vocab.json", 2_776_833),
        ("acestep-v15-turbo/config.json", 1_968),
        ("acestep-v15-turbo/configuration_acestep_v15.py", 13_130),
        ("acestep-v15-turbo/model.safetensors", 4_787_825_604),
        ("acestep-v15-turbo/modeling_acestep_v15_turbo.py", 96_036),
        ("acestep-v15-turbo/silence_latent.pt", 3_841_215),
        ("config.json", 1_968),
        ("vae/config.json", 425),
        ("vae/diffusion_pytorch_model.safetensors", 337_431_388),
    ]

    /// R7, decided per file against what the engine reads. The tokenizer comes from `tokenizer.json` through
    /// `HFTokenizerLoader` (with `tokenizer_config.json`); `vocab.json` / `merges.txt` / `added_tokens.json` /
    /// `special_tokens_map.json` / `chat_template.jinja` are redundant with it (about 4.4 MB) but ride along under the
    /// selector's existing component-folder rule, which is shared by every repo, so no per-file trimming is added.
    static let expected: Set<String> = [
        "config.json",
        "acestep-v15-turbo/config.json",
        "acestep-v15-turbo/model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "Qwen3-Embedding-0.6B/model.safetensors",
        "Qwen3-Embedding-0.6B/config.json",
        "Qwen3-Embedding-0.6B/tokenizer.json",
        "Qwen3-Embedding-0.6B/tokenizer_config.json",
        "Qwen3-Embedding-0.6B/vocab.json",
        "Qwen3-Embedding-0.6B/merges.txt",
        "Qwen3-Embedding-0.6B/added_tokens.json",
        "Qwen3-Embedding-0.6B/special_tokens_map.json",
        "Qwen3-Embedding-0.6B/chat_template.jinja",
    ]

    static func selected() -> [String] {
        ModelFileSelector.filesToDownload(siblings: siblings.map(\.0), forModel: hfID)
    }

    static func selectedBytes() -> Int {
        let size = Dictionary(uniqueKeysWithValues: siblings)
        return selected().reduce(0) { $0 + (size[$1] ?? 0) }
    }

    // MARK: - Download selection

    @Test func selectsExactlyTheR7Set() {
        let got = Self.selected()
        #expect(Set(got) == Self.expected)
        #expect(got.count == Self.expected.count, "no duplicates")
        #expect(!got.contains { $0.hasPrefix("acestep-5Hz-lm-1.7B/") })
        // Global skips stay: remote-code .py, the silence_latent .pt (it ships bundled, R6), docs.
        #expect(!got.contains { $0.hasSuffix(".py") || $0.hasSuffix(".pt") || $0.hasSuffix(".md") })
        #expect(!got.contains(".gitattributes"))
    }

    @Test func selectedBytesMatchTheCatalogFigure() {
        #expect(Self.selectedBytes() == 6_332_732_359)
        // The LM folder is the other 3.76 GB.
        let lm = Self.siblings.filter { $0.0.hasPrefix("acestep-5Hz-lm-1.7B/") }.reduce(0) { $0 + $1.1 }
        #expect(lm == 3_755_412_617)
    }

    /// Without the per-model entry point (and for any other repo id) the LM folder is selected: the exclusion is this
    /// entry's alone and nothing else changes.
    @Test func theExclusionIsThisEntrysAlone() {
        let names = Self.siblings.map(\.0)
        let plain = ModelFileSelector.filesToDownload(siblings: names)
        #expect(plain.contains("acestep-5Hz-lm-1.7B/model.safetensors"))
        #expect(ModelFileSelector.filesToDownload(siblings: names, forModel: "someone/else") == plain)
        #expect(ModelFileSelector.excludedFolders(for: "mlx-community/Qwen3-4B-4bit").isEmpty)
        #expect(ModelFileSelector.excludedFolders(for: "jasonvassallo/mlx-musicgen-small").isEmpty)
        #expect(ModelFileSelector.excludedFolders(for: Self.hfID) == ["acestep-5Hz-lm-1.7B/"])
    }

    // MARK: - ModelSupport flag

    private func entry() throws -> ModelEntry {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    /// MC5-3i removed the "no runner yet" gap: the entry is no longer flagged, and neither is anything else in the catalog.
    @Test func noLongerFlaggedAsHavingNoRunner() throws {
        let e = try entry()
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
        let all = try #require(try? JSONDecoder().decode(BrowserData.self, from: Data(contentsOf:
            URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("MLXUI/Resources/browser.json")))).domains.flatMap { $0.allModels }
        let flagged = all.filter { ModelSupport.unsupportedReason(for: $0) != nil }.map(\.hfModelId)
        #expect(flagged.isEmpty, "nothing is flagged: \(flagged)")
    }

    /// Only the ACE-Step module captures the entry (MC5-3i); MusicGen's claim still needs "musicgen" in the ids.
    @Test func onlyTheAceStepModuleCapturesItAndMusicGenStillOwnsItsOwn() throws {
        let e = try entry()
        #expect(MusicGenSDK().claim(e) == .no)
        #expect(MusicGenUI().claim(e) == .no)
        var claimers: [String] = []
        for module in installedModules {
            let one = ModelRegistry()
            module.register(into: one)
            if one.bestModule(for: e) != nil { claimers.append(module.descriptor.id) }
        }
        #expect(claimers == ["acestep"], "exactly the ACE-Step module claims the entry: \(claimers)")
        // And MusicGen's own entry still resolves to it.
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let mg = try #require(all.first { $0.hfModelId == "jasonvassallo/mlx-musicgen-small" })
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        #expect(full.bestModule(for: mg)?.descriptor.id == "musicgen")
    }

    // MARK: - Bundled silence_latent (R6)

    /// sha256 of `acestep-v15-turbo/silence_latent.pt` (source) and of the bundled raw f32 `[15000, 64]` (output).
    static let sourceSHA256 = "a778e9dd942f5e8b2c09c55370782d318834432b03dabbcdf70e6ed49ad6358b"
    static let resourceSHA256 = "a7ee13d8902f0c02def49249f05a3e5dd99550ae8aed263299be43329b330e23"
    static let frames = 15_000
    static let channels = 64

    private func resourceData() throws -> Data {
        let url = try #require(Bundle.main.url(forResource: "silence_latent_f32", withExtension: "bin"),
                               "silence_latent_f32.bin must be in the app bundle")
        return try Data(contentsOf: url)
    }

    @Test func bundledSilenceLatentHasTheRecordedHashAndSize() throws {
        let data = try resourceData()
        #expect(data.count == Self.frames * Self.channels * 4)
        #expect(data.count == 3_840_000)
        let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #expect(digest == Self.resourceSHA256)
    }

    @Test func silenceLatentIsFiniteAndNotConstant() throws {
        let data = try resourceData()
        let floats: [Float] = data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        #expect(floats.count == Self.frames * Self.channels)
        #expect(floats.allSatisfy { $0.isFinite })
        #expect(Set(floats.prefix(Self.channels)).count > 1, "first frame is not constant")
        #expect(Set(floats.suffix(Self.channels)).count > 1, "last frame is not constant")
        #expect(Set(floats).count > 1000)
    }

    /// The `.pt` holds `[1, 64, 15000]`; the resource is its transpose, `[15000, 64]` row-major, so element
    /// `[frame, channel]` is the source's `[0, channel, frame]`. Expected values were read from the `.pt` with
    /// `torch.load(weights_only=True)` in the pinned oracle venv (`t[0, channel, frame]`, printed as Python floats).
    @Test func resourceIsTheTransposeOfTheSource() throws {
        let data = try resourceData()
        func value(frame: Int, channel: Int) -> Float {
            data.withUnsafeBytes {
                Float(bitPattern: UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: (frame * Self.channels + channel) * 4, as: UInt32.self)))
            }
        }
        let expected: [(Int, Int, Float)] = [
            (0, 0, 0.5706896781921387),
            (0, 63, 0.42015549540519714),
            (1, 0, 0.3040347397327423),
            (7, 5, 0.279321551322937),
            (100, 30, 0.5900188088417053),
            (14_999, 0, 0.39274585247039795),
            (14_999, 63, 1.2298461198806763),
        ]
        for (frame, channel, want) in expected {
            #expect(value(frame: frame, channel: channel) == want, "frame \(frame) channel \(channel)")
        }
    }
}
