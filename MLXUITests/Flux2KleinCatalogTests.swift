import Testing
import Foundation
@testable import MLXUI

/// MP7-2: the FLUX.2 Klein 4B catalog entry, its install file set, and the guarantees that hold until the Klein module exists
/// (MP7-3h / MP7-4): the FLUX.1 engine must not claim it, Browse flags it "no runner yet", and no workflow resolves it as
/// runnable. Facts are from the pinned HF files in `RSI/notes/modelpack/hf/mlx-community_FLUX.2-klein-4B-bf16/` (repo sha
/// 2f7dd4ac69a9fd9c909f8cc25b072dab2c83781d): `tree.json` for the file list and byte sizes, `api.json` for the licence.
@MainActor
struct Flux2KleinCatalogTests {

    private static let hfID = "mlx-community/FLUX.2-klein-4B-bf16"

    /// Every *file* in the pinned `tree.json` (the five directory entries are not siblings), name and exact byte size, in the tree's order.
    /// Embedded rather than read from `RSI/` (gitignored, absent from CI and the worktree).
    private static let tree: [(name: String, bytes: Int)] = [
        (".gitattributes", 1_723),
        ("README.md", 1_622),
        ("editing.jpg", 2_506_178),
        ("flux-2-klein-4b.safetensors", 7_751_105_712),
        ("model_index.json", 446),
        ("others.jpg", 3_386_946),
        ("realism.jpg", 2_855_711),
        ("scheduler/scheduler_config.json", 486),
        ("text_encoder/config.json", 1_536),
        ("text_encoder/generation_config.json", 214),
        ("text_encoder/model-00001-of-00002.safetensors", 4_967_215_360),
        ("text_encoder/model-00002-of-00002.safetensors", 3_077_766_632),
        ("text_encoder/model.safetensors.index.json", 32_855),
        ("tokenizer/added_tokens.json", 707),
        ("tokenizer/chat_template.jinja", 4_168),
        ("tokenizer/merges.txt", 1_671_853),
        ("tokenizer/special_tokens_map.json", 613),
        ("tokenizer/tokenizer.json", 11_422_654),
        ("tokenizer/tokenizer_config.json", 5_404),
        ("tokenizer/vocab.json", 2_776_833),
        ("transformer/config.json", 541),
        ("transformer/diffusion_pytorch_model.safetensors", 7_751_109_744),
        ("vae/config.json", 821),
        ("vae/diffusion_pytorch_model.safetensors", 168_120_878),
    ]

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func klein() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    // MARK: - Entry

    @Test func entryFactsMatchTheRepo() throws {
        let e = try klein()
        #expect(e.id == "mlx-community--FLUX.2-klein-4B-bf16")
        #expect(e.displayName == "FLUX.2 Klein 4B")
        #expect(e.modelType == .image)
        #expect(e.runnerKind == .image)
        #expect(e.source == .mlx)
        #expect(e.format == "mlx-bf16")
        // SPEC-Q241
        #expect(e.license == "apache-2.0", "the HF card says Apache-2.0 (MP7-0 D); catflow-mlx's manifest disagrees, SPEC-Q241")
        #expect(e.speedTokensPerSec == nil, "meaningless for diffusion")
        #expect(e.downloadSizeGB == 15.98)
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.ramGB == e.ramGB)
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "imagegen" })
        #expect(leaf.allModels.map(\.hfModelId).contains(Self.hfID))
        #expect(leaf.allModels.first?.hfModelId == "mlx-community/Flux-1.lite-8B-MLX-Q4", "append-only: existing entries keep their place")
    }

    /// MP7-4: the RAM figure is the largest measured whole-run footprint peak at 1024 x 1024, quantize 8 (MP7-3h 5.83 GB; the MP7-4 matrix and cold run 5.61 to 6.05 GB),
    /// with its provenance and the 4-bit peak in the description, and no "fits a 16 GB Mac" claim anywhere (owner, 2026-10-31 / 2026-11-01).
    @Test func ramGBIsTheMeasuredFootprintAtQuantize8WithProvenance() throws {
        let e = try klein()
        #expect(e.ramGB == 6.05)
        #expect(e.variants?.first?.ramGB == 6.05)
        #expect(Flux2KleinBooking.bookedBytes == Int64((e.ramGB * 1e9).rounded()), "the lease books exactly the catalog figure")
        let description = try #require(e.description)
        #expect(description.contains("RAM figure is measured"))
        #expect(description.contains("6.05 GB") && description.contains("quantize 8") && description.contains("1024 x 1024"))
        #expect(description.contains("quantize 4") && description.contains("5.4 to 6.1 GB"), "the 4-bit peak is recorded too")
        #expect(!description.contains("not yet measured"))
        for text in [description, e.summary ?? ""] {
            #expect(!text.contains("16 GB Mac"), "no unmeasured fit claim")
        }
        #expect(e.summary?.contains("about 6 GB") == true && e.summary?.contains("8-bit") == true)
    }

    // MARK: - Install file set

    @Test func theEmbeddedTreeIsTheFullPinnedTree() {
        #expect(Self.tree.count == 24)
        #expect(Self.tree.first(where: { $0.name == "flux-2-klein-4b.safetensors" })?.bytes == 7_751_105_712)
    }

    /// Diffusers layout: the root duplicate checkpoint, the three sample JPEGs, the docs and the two index/metadata files nothing
    /// reads are skipped; the 16 selected files total 15,980,098,444 B = 15.98 GB, which is the catalog's `downloadSizeGB`.
    @Test func selectorTakesSixteenFilesTotallingFifteenPointNineEightGB() throws {
        let sizes = Dictionary(uniqueKeysWithValues: Self.tree.map { ($0.name, $0.bytes) })
        let picked = ModelFileSelector.filesToDownload(siblings: Self.tree.map(\.name))
        #expect(picked.count == 16)
        #expect(picked.reduce(0) { $0 + (sizes[$1] ?? 0) } == 15_980_098_444)
        #expect(Double(15_980_098_444) / 1e9 == 15.980098444)
        #expect(try klein().downloadSizeGB == 15.98)

        // skipped: the redundant root checkpoint, the JPEGs, docs, and the two files no loader reads
        for skipped in ["flux-2-klein-4b.safetensors", "editing.jpg", "others.jpg", "realism.jpg", "README.md", ".gitattributes",
                        "model_index.json", "text_encoder/model.safetensors.index.json"] {
            #expect(!picked.contains(skipped), "\(skipped) must not be downloaded")
        }
        // taken: every component's weights and the config/tokenizer files the port reads
        for taken in ["transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors",
                      "text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors",
                      "transformer/config.json", "vae/config.json", "text_encoder/config.json",
                      "scheduler/scheduler_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
                      "tokenizer/chat_template.jinja", "tokenizer/vocab.json", "tokenizer/merges.txt"] {
            #expect(picked.contains(taken), "\(taken) must be downloaded")
        }
        // A loader that lists `*.safetensors` per folder needs no shard index (FluxEngine.loadShardSet does the same).
        #expect(picked.filter { $0.hasPrefix("text_encoder/") && $0.hasSuffix(".safetensors") }.count == 2)
    }

    /// `entry` with its `hfModelId`/`id` replaced (the struct is all `let`s, so round-trip it through JSON).
    private func renamed(_ entry: ModelEntry, hfModelId: String) throws -> ModelEntry {
        var json = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as? [String: Any])
        json["hfModelId"] = hfModelId
        json["id"] = hfModelId.replacingOccurrences(of: "/", with: "--")
        return try JSONDecoder().decode(ModelEntry.self, from: JSONSerialization.data(withJSONObject: json))
    }

    // MARK: - With the Klein module registered (MP7-4)

    /// FluxSDK matches any `.image` + `.mlx` id containing "flux"; Klein must not fall to the FLUX.1 engine (owner, 2026-10-31); the Klein module takes it.
    @Test func theFlux1ModuleDoesNotClaimKleinButStillClaimsFlux1AndKleinGoesToTheKleinModule() throws {
        let k = try klein()
        #expect(FluxSDK().claim(k) == .no)
        let flux1 = try #require(try catalog().domains.flatMap { $0.allModels }.first { $0.hfModelId == "mlx-community/Flux-1.lite-8B-MLX-Q4" })
        #expect(FluxSDK().claim(flux1) == .exact)
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        #expect(registry.bestModule(for: k)?.descriptor.id == "flux2-klein")
        #expect(registry.bestModule(for: flux1)?.descriptor.id == "flux")
        // every other flux-family spelling the guard names
        for hf in ["black-forest-labs/FLUX.2-klein-4B", "someone/flux2-dev-4bit", "x/FLUX-2-anything", "x/klein-9b"] {
            #expect(FluxSDK().claim(try renamed(flux1, hfModelId: hf)) == .no, Comment(rawValue: hf))
        }
    }

    @Test func theMP72GapIsGoneAndFlux1IsUnaffected() throws {
        let k = try klein()
        #expect(ModelSupport.unsupportedReason(for: k) == nil, "the Klein module registered: no more \"no runner yet\"")
        let flux1 = try #require(try catalog().domains.flatMap { $0.allModels }.first { $0.hfModelId == "mlx-community/Flux-1.lite-8B-MLX-Q4" })
        #expect(ModelSupport.unsupportedReason(for: flux1) == nil)
        #expect(!ModelFileSelector.filesToDownload(siblings: Self.tree.map(\.name)).isEmpty)
    }

    /// The pool names "FLUX.2 Klein 4B" (from catflow-mlx). MP7-W added the `CatalogBridge` row (`.requantized` onto the catflow pin `black-forest-labs/FLUX.2-klein-4B`) and the
    /// verbatim manifest, replacing the MP7-2 test `workflowsStillHaveNoBridgeEntryForKlein` ("no CatalogBridge entry until MP7-W"), as that test said it would be.
    @Test func workflowsNowBridgeKlein() throws {
        let k = try klein()
        let entry = try #require(CatalogBridge.entry(for: "FLUX.2 Klein 4B"), "MP7-W added the bridge row")
        #expect(entry.candidates == [Self.hfID] && entry.equivalence == .requantized && entry.manifestFile == "flux2-klein-4b.json")
        #expect(CatalogBridge.entry(forCandidate: Self.hfID)?.display == "FLUX.2 Klein 4B")
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set([k].filter { registry.bestModule(for: $0) != nil }.map(\.id))
        #expect(claimable == [k.id])
    }
}
