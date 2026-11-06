import Foundation
import Testing
@testable import MLXUI

/// MC4-8 — "Qwen3-VL Embedding 2B" is reachable from AI Workflows for `Embed` (text only, decision
/// 7): one `CatalogBridge` row (`.same`, explicit `modelType: .embedding`), one MLXUI-authored
/// curated manifest, and a last-slot append to the `"Embed"` pool so the seed ("BGE-M3") does not
/// move. `Retrieve` and the index format are **unchanged** — decision 8 pins today's behaviour.
/// Written to extend, not edit, `CatFlowCatalogBridgeTests`/`CatFlowIndexStoreTests`.
struct Qwen3VLEmbeddingWorkflowWiringTests {

    private static let display = "Qwen3-VL Embedding 2B"
    private static let hfID = "mlx-community/Qwen3-VL-Embedding-2B-4bit"
    private static let cardID = "mlx-community--Qwen3-VL-Embedding-2B-4bit"

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    @MainActor
    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    // MARK: - the four tables (CLAUDE.md), one pin each

    /// 1. `TaskCatalog`: `Embed` exists, `list(text) -> list(vector)`, a model-class engine task.
    @Test func table1_embedIsInTheTaskCatalogUnchanged() throws {
        let task = try #require(TaskCatalog.get("Embed"))
        #expect(task.refKind == .engine)
        #expect(task.refName == "engines.embed.embed")
    }

    /// 2. `servedRefNamePrefixes`: the executor genuinely serves `engines.embed.*`.
    @Test func table2_theExecutorServesEmbed() {
        #expect(TaskModels.isServedByExecutor("Embed"))
        #expect(TaskModels.runnerKind(forTask: "Embed") == .embedding)
    }

    /// 3. `taskKinds` + the derived pool: Qwen3-VL Embedding 2B is a runnable `.embedding` model per
    /// the live catalog, the registry's claim and `ModelSupport`.
    @MainActor
    @Test func table3_qwen3VLEmbeddingIsInTheDerivedEmbedPool() throws {
        let catalog = try loadCatalog()
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(card.runnerKind == .embedding)
        #expect(ModelSupport.unsupportedReason(for: card) == nil)
        let claimable = claimableIDs(catalog)
        #expect(claimable.contains(Self.cardID))
        let derived = TaskModels.derivedModels(for: "Embed", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.contains { $0.displayName == Self.display })
        // The pool's default is still BGE-M3 (decision 7: appended last, the seed does not move).
        #expect(TaskModels.defaultModel(forTask: "Embed", catalog: catalog, claimableModelIDs: claimable) == "BGE-M3")
    }

    /// 4. `CatalogBridge.entries`: the name the `.cat` writes resolves to the installable card,
    /// `.same`, explicit `.embedding` model type, no substitution note.
    @Test func table4_theBridgeResolvesTheNameTheCatWrites() throws {
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.modelType == .embedding)
        #expect(entry.manifestFile == "qwen3-vl-embedding-2b.json")
        let catalog = try loadCatalog()
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.id == Self.cardID)
            #expect(slot.modelEntry?.modelType == .embedding)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let name, let reason, _):
            Issue.record("\(name) should resolve: \(reason)")
        }
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(TaskModels.displayName(for: card) == Self.display)
    }

    // MARK: - the manifest

    @Test func manifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "qwen3-vl-embedding-2b.json"))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-embed")
        #expect(manifest.tasks == ["Embed"])
        #expect(manifest.settings.isEmpty, "decision 3/7: no instruction, dimension or image-Embed setting")
        #expect(manifest.resources?.diskGB == 1.8)
        #expect(manifest.resources?.ramGB == 2.7)
    }

    @Test func manifestResourcesMatchTheCatalogCard() throws {
        let card = try #require(try loadCatalog().first { $0.id == Self.cardID })
        let manifest = try #require(CuratedManifest.load(manifestFile: "qwen3-vl-embedding-2b.json"))
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    /// `CuratedManifest` does not decode `notes`, `license` or `checksum`, so read the shipped JSON.
    @Test func manifestSaysItIsMLXUIAuthoredAndNamesThePinnedRevision() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/qwen3-vl-embedding-2b.json")
        let json = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        #expect(json["license"] as? String == "Apache-2.0")
        #expect(json["pinned"] as? Bool == true)
        #expect(json["manifest_version"] as? Int == 1)
        #expect(json["source"] as? String == "https://huggingface.co/mlx-community/Qwen3-VL-Embedding-2B-4bit")
        #expect(json["checksum"] == nil, "MLXUI-authored manifests have no catflow checksum to inherit")
        let notes = try #require(json["notes"] as? String)
        #expect(notes.contains("MLXUI-authored"))
        #expect(notes.contains("c713a965783416b9467b419f2d14ca6445847455"))
        #expect(notes.contains("no checksum field"))
    }

    // MARK: - the pool

    /// Appended **last**: the seed ("BGE-M3") is unchanged and stays the default.
    @MainActor
    @Test func appendedLastInTheEmbedPoolWithoutMovingTheSeed() throws {
        #expect(TaskModels.models(forTask: "Embed") == ["BGE-M3", Self.display])
        #expect(TaskModels.seededModels(forTask: "Embed") == ["BGE-M3", Self.display])
        #expect(TaskModels.seededModels(forTask: "Embed").first == "BGE-M3")
        for task in TaskCatalog.allTasks().map(\.name) where task != "Embed" {
            #expect(!TaskModels.models(forTask: task).contains(Self.display),
                    "\(task) must not offer the embedder")
        }
    }

    // MARK: - the row, end to end through a stub stage

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-embed-wiring-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func teardown(_ base: URL) {
        try? FileManager.default.removeItem(at: base)
    }

    /// The `Embed` stage fed by the app's embedder shape produces 2048-d vectors, and `Store Index`
    /// records that embedder and those dims in the manifest.
    @Test func embedRowThroughAStubStageStores2048Dimensions() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)

        let chunk = "A woman playing with her dog on a beach at sunset."
        let vector = [Float](repeating: 0, count: Qwen3VLEmbedder.dimensions)
        let stage = EmbeddingStage(id: "embed.stub", name: "Embedding") { _ in [vector] }
        let output = try await stage.run(.text(chunk), progress: { _ in })
        guard case let .embedding(vectors) = output else {
            Issue.record("expected an embedding, got \(output.kind)")
            return
        }
        #expect(vectors.count == 1)
        #expect(vectors[0].count == 2048)

        let vectorURL = flowDir.appendingPathComponent("v.npy")
        try NpyCodec.save(vectors[0], to: vectorURL)
        let store = StoreIndexTool(
            workspace: ws, flowID: "f", settings: "kb; embedder=\"\(Self.display)\"")
        let indexAsset = try await store.run(inputs: [
            Asset(items: [Item(kind: .text, value: chunk, path: nil, sourceText: nil)]),
            Asset(items: [Item(kind: .vector, value: nil, path: vectorURL, sourceText: nil)]),
        ])
        let indexDir = try #require(indexAsset.items.first?.path)
        let manifest = try IndexFormat.manifest(
            from: Data(contentsOf: indexDir.appendingPathComponent("manifest.json")))
        #expect(manifest.embedder == Self.display)
        #expect(manifest.dims == 2048)
        #expect(manifest.count == 1)
        #expect(manifest.format_version == 1)
    }

    /// Decision 8: `Retrieve` compares dimensions only, and this model's 2048-d vectors cannot be
    /// retrieved from BGE-M3's 1024-d index — today's dims message, nothing else changed.
    @Test func a2048dQueryAgainstA1024dIndexFailsWithTodaysDimsMessage() async throws {
        let (ws, base) = try makeWorkspace()
        defer { teardown(base) }
        let flowDir = ws.directory(for: "f")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)

        // A BGE-M3-shaped index: one 1024-d vector.
        let oldVector = flowDir.appendingPathComponent("v.npy")
        try NpyCodec.save([Float](repeating: 1.0 / 1024.0, count: 1024), to: oldVector)
        let store = StoreIndexTool(workspace: ws, flowID: "f", settings: "kb; embedder=BGE-M3")
        let indexAsset = try await store.run(inputs: [
            Asset(items: [Item(kind: .text, value: "an old chunk", path: nil, sourceText: nil)]),
            Asset(items: [Item(kind: .vector, value: nil, path: oldVector, sourceText: nil)]),
        ])
        let indexDir = try #require(indexAsset.items.first?.path)

        // A 2048-d query from this embedder.
        let queryURL = flowDir.appendingPathComponent("q.npy")
        try NpyCodec.save([Float](repeating: 1.0 / 2048.0, count: 2048), to: queryURL)
        let retrieve = RetrieveTool(workspace: ws, flowID: "f", settings: "top_k=1")
        do {
            _ = try await retrieve.run(inputs: [
                Asset(items: [Item(kind: .index, value: nil, path: indexDir, sourceText: nil)]),
                Asset(items: [Item(kind: .vector, value: nil, path: queryURL, sourceText: nil)]),
            ])
            Issue.record("a 2048-d query must be refused by the dims check")
        } catch let error as FlowError {
            guard case let .stageFailure(row, message) = error else {
                Issue.record("unexpected FlowError: \(error)")
                return
            }
            #expect(row == "Retrieve")
            #expect(message == "query has 2048 dims, index expects 1024")
        }
        // And the index format itself is untouched: the same manifest still decodes.
        let manifest = try IndexFormat.manifest(
            from: Data(contentsOf: indexDir.appendingPathComponent("manifest.json")))
        #expect(manifest.embedder == "BGE-M3")
        #expect(manifest.dims == 1024)
    }
}
