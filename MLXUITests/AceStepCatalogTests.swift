import Testing
import Foundation
@testable import MLXUI

/// MC5-3a: the ACE-Step 1.5 catalog entry. Facts are from the pinned HF repo
/// (`ACE-Step/Ace-Step1.5` @ 19671f4, tree and api captured 2026-11-21 under
/// `RSI/notes/modelcandidates/hf/ACE-Step_Ace-Step1.5/`) and the MC5-0b rulings (R3, R7, R8). The engine, the
/// download exclusion of `acestep-5Hz-lm-1.7B/` and the workflow wiring land in MC5-3b … MC5-W; this file pins what
/// the catalog must say.
@MainActor
struct AceStepCatalogTests {

    private static let hfID = "ACE-Step/Ace-Step1.5"

    /// The pinned sibling list (path → bytes), from `tree.json` @ 19671f4.
    private static let siblings: [(String, Int)] = [
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
        ("acestep-v15-turbo/config.json", 1_968),
        ("acestep-v15-turbo/configuration_acestep_v15.py", 13_130),
        ("acestep-v15-turbo/model.safetensors", 4_787_825_604),
        ("acestep-v15-turbo/modeling_acestep_v15_turbo.py", 96_036),
        ("acestep-v15-turbo/silence_latent.pt", 3_841_215),
        ("config.json", 1_968),
        ("vae/config.json", 425),
        ("vae/diffusion_pytorch_model.safetensors", 337_431_388),
    ]

    /// Files under the planner-LM folder, left out of the install (R7). Sizes are from the same tree.
    private static let lmFolderBytes = 3_755_412_617

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func entry() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchTheRepo() throws {
        let e = try entry()
        #expect(e.id == "ACE-Step--Ace-Step1.5")
        #expect(e.displayName == "ACE-Step 1.5")
        #expect(e.family == "ace-step")
        #expect(e.modelType == .music)
        #expect(e.source == .mlx)
        #expect(e.hfRepo == "ACE-Step")
        #expect(e.architecture == "AceStepConditionGenerationModel")
        #expect(e.license == "mit")
        #expect(e.lastUpdated == "2026-02-03")
        #expect(e.taskTags == ["text-to-music", "music-generation"])
        #expect(e.communityDownloads == 93_349)
        #expect(e.communityLikes == 937)
        // R3: bf16, nothing quantized; one variant.
        #expect(e.format == "mlx-bf16")
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "bf16")
        #expect(e.variants?.first?.format == "mlx-bf16")
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.recommended == true)
        #expect(e.variants?.first?.qualityPercent == 100, "published precision, not a benchmark")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
    }

    /// R7: the install is every sibling except `acestep-5Hz-lm-1.7B/`. The list above holds no LM folder files
    /// (those are the excluded 3.76 GB), so its sum is the catalog download figure. The real selector also skips
    /// `.py` / `.md` / `.pt` files, which MC5-3b's selector test pins against this same list.
    @Test func downloadSizeIsTheR7SiblingSum() throws {
        let kept = Self.siblings.reduce(0) { $0 + $1.1 }
        #expect(kept == 6_336_689_976)
        #expect(Self.siblings.allSatisfy { !$0.0.hasPrefix("acestep-5Hz-lm-1.7B/") })
        // Kept files plus the LM folder are the whole repo tree: 10.09 GB (MC5-0 facts).
        #expect(kept + Self.lmFolderBytes == 10_092_102_593)
        let e = try entry()
        #expect(e.downloadSizeGB == 6.34)
        #expect((Double(kept) / 1e9 * 100).rounded() / 100 == e.downloadSizeGB)
    }

    /// AM1: provisional `ramGB` = max(weights x 1.5, download x 1.2). Resident bf16 weights 5.897 GB (T16: the FSQ
    /// tokenizer and detokenizer are not loaded) x 1.5 = 8.846 (rounded up to 0.1 GB), and download x 1.2 = 7.6, so 8.9. AM-T replaces it.
    @Test func ramGBIsTheProvisionalAM1Figure() throws {
        let e = try entry()
        let weights = 5.897
        let provisional = max(weights * 1.5, e.downloadSizeGB * 1.2)
        #expect((provisional * 10).rounded(.up) / 10 == 8.9)
        #expect(e.ramGB == 8.9)
        #expect(e.ramGB >= e.downloadSizeGB, "CatalogGuardTests floor")
        #expect(e.description?.contains("provisional estimate, not a measurement") == true)
        #expect(e.description?.contains("RAM figure is measured") == false)
        #expect(e.description?.contains("measured peak") == true, "says it will be replaced by a measured peak")
        let notes = try catalogNotes()
        #expect(notes.contains("PROVISIONAL"))
        #expect(notes.contains("6,336,689,976"))
    }

    @Test func fieldsThatDoNotApplyAreNil() throws {
        let e = try entry()
        // The LLM weights heuristic and the tokens/s baseline do not describe a music diffusion model.
        #expect(e.speedTokensPerSec == nil)
        #expect(e.speedHardware == nil)
        #expect(e.speedEstimated == nil)
        #expect(e.displaySpeed == "—")
        #expect(e.benchmarks == nil)
        #expect(e.qualityScore == nil)
        #expect(e.paramCountB == nil)
        #expect(e.contextWindow == nil)
        #expect(e.languages == nil)
    }

    /// v1 is a style prompt and optional lyrics to music, with no planner LM. The text must not claim anything that
    /// is not built, and names the bundled Apache-2.0 text encoder (R8). It does not raise the VAE licence question.
    @Test func aboutTextMakesNoClaimBeyondV1() throws {
        let e = try entry()
        let text = ((e.description ?? "") + "\n" + (e.summary ?? "")).lowercased()
        #expect(!text.isEmpty)
        for banned in ["language model", " lm ", "lm-", "planner", "cover", "edit", "repaint", "extract", "lego",
                       "commercial", "50+", "vocal-to", "chain-of-thought", "10 minute", "10-minute"] {
            #expect(!text.contains(banned), "About text must not mention '\(banned)'")
        }
        #expect(text.contains("lyrics"))
        #expect(text.contains("qwen3-embedding-0.6b"))
        #expect(text.contains("apache-2.0"))
        #expect(!text.contains("vae licen"))
        #expect(!text.contains("stable audio"))
        #expect(!text.contains("stability"))
        #expect(text.contains("mit"))
    }

    @Test func liveInTheMusicLeafAppendedLast() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "music" })
        #expect(leaf.allModels.map(\.hfModelId) == ["jasonvassallo/mlx-musicgen-small", Self.hfID], "append, never reorder")
        let section = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(section.domainIds.contains("music"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("music") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["music"] != nil)
    }

    /// The MusicGen entry is untouched by MC5-3a.
    @Test func musicGenEntryIsUnchanged() throws {
        let all = try catalog().domains.flatMap { $0.allModels }
        let m = try #require(all.first { $0.hfModelId == "jasonvassallo/mlx-musicgen-small" })
        #expect(m.id == "jasonvassallo--mlx-musicgen-small")
        #expect(m.displayName == "MusicGen-small")
        #expect(m.family == "MusicGen")
        #expect(m.modelType == .music)
        #expect(m.license == "cc-by-nc-4.0")
        #expect(m.format == "mlx-f32")
        #expect(m.ramGB == 3.7)
        #expect(m.downloadSizeGB == 2.5)
        #expect(m.paramCountB == 0.615)
        #expect(m.variants?.count == 1)
        #expect(m.variants?.first?.qualityPercent == 100)
        #expect(m.summary?.hasPrefix("Meta's smallest MusicGen text-to-music model") == true)
    }

    /// `notes` is not decoded into `ModelEntry`; read it from the raw JSON.
    private func catalogNotes() throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let domains = try #require(root["domains"] as? [[String: Any]])
        let leaf = try #require(domains.first { $0["id"] as? String == "music" })
        let models = try #require(leaf["models"] as? [[String: Any]])
        let m = try #require(models.first { $0["hfModelId"] as? String == Self.hfID })
        return try #require(m["notes"] as? String)
    }
}
