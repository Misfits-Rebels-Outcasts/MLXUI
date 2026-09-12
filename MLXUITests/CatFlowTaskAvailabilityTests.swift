import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-4 — the availability truth. The load-bearing test is the cross-check: enumerate
/// the catalog and assert `TaskAvailability` agrees with `RealExecutor`'s actual dispatch for
/// every instant task (drive the executor, observe whether it throws `unsupportedTask`).
/// Adding a `case` to `RealExecutor` must be the only edit needed to flip a task available.
///
/// Model tasks are `@MainActor` here because the derived pool (CFM-R14-2) reads the registry's
/// claim answer, and the registry is `@MainActor`.
@MainActor
struct CatFlowTaskAvailabilityTests {

    // MARK: - The cross-check: TaskAvailability ⟺ RealExecutor dispatch

    @Test func availabilityMatchesTheExecutorsDispatchForEveryInstantTask() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-avail-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let blob = base.appendingPathComponent("blobs")
        let executor = RealExecutor(workspace: ws, flowID: "probe", blobDirectory: blob,
                                    makeModelStage: { _, _ in
                                        throw StageError.unsupportedModel(id: "probe", kind: .llm)
                                    },
                                    installedModelIDs: [], catalog: [])

        let instant = TaskCatalog.entries.filter { $0.taskClass == .instant }
        var mismatches: [String] = []
        for task in instant {
            let dispatched = await dispatches(task.name, executor: executor)
            // Instant verdicts are catalog-free (CFM-R14-FIX-3) — explicit empties.
            let available = TaskAvailability.isAvailable(task.name, catalog: [], claimableModelIDs: [])
            if dispatched != available {
                mismatches.append("\(task.name): dispatched=\(dispatched) available=\(available)")
            }
        }
        #expect(mismatches.isEmpty, "drift: \(mismatches.joined(separator: "; "))")
        // Every claimed-supported tool is genuinely dispatched (the set isn't lying).
        for name in TaskAvailability.supportedInstantTools {
            #expect(await dispatches(name, executor: executor), "\(name) claimed but not dispatched")
        }
    }

    /// Whether the executor routes `name` to a real tool (does NOT throw unsupportedTask).
    private func dispatches(_ name: String, executor: RealExecutor) async -> Bool {
        let row = Row(id: UUID(), task: name, settings: nil)
        do {
            _ = try await executor.execute(path: "1", row: row, inputs: [],
                                           transcript: nil, context: nil, usedFlowContent: nil)
            return true
        } catch FlowError.unsupportedTask {
            return false
        } catch {
            return true   // a real tool error (missing input/file) — the case exists
        }
    }

    // MARK: - CFM-R14-2 / CFM-R14-FIX-4: the derived model pool's cross-check

    /// The cross-check the tools already have, applied to the model side — driven through the
    /// **executor path**, not the predicate under test (CFM-R14-FIX-4). For every derived
    /// candidate of every model task:
    ///  1. the registry resolves it to a module and that module **builds a stage** through the
    ///     same `bestModule → sdk.makeStage` path `AppFlowExecutorFactory` uses;
    ///  2. `FlowPreflight.run` does **not** bucket a row naming it `blocked` (the pre-Run gate).
    /// Both fail for an unbridged pick until `CatalogBridge.resolve` learns the derived-pool
    /// fallback (CFM-R14-FIX-1) — this is the test that proves FIX-1 is real.
    @Test @MainActor func derivedPoolMembersBuildStagesAndSurvivePreflight() throws {
        let catalog = try bundledCatalog()
        let claimable = claimableIDs(catalog: catalog)
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }

        for task in TaskCatalog.entries where task.taskClass == .model {
            let pool = TaskModels.derivedModels(for: task.name, catalog: catalog,
                                                claimableModelIDs: claimable)
            for slot in pool {
                // This test exercises the real MLX stage-building path, which only a
                // cataloged `ModelEntry` can feed. AFM-FOLLOWUP-2: narrowed from a blanket
                // `guard let model = slot.modelEntry else { continue }` — that silently
                // skipped *any* non-cataloged slot, which was one step too far for what was a
                // legitimate reason (AFM's `.system` is real now; its own dispatch is proven
                // by `CatFlowAppleFoundationTests`, which mocks the executor instead of
                // building a `ModelRegistry` stage). Skip exactly `.system`/`.provider`; a
                // `.cataloged` slot with no `modelEntry` should be impossible, so it still
                // fails loudly if `ModelSlot` ever grows a way to make that true.
                guard let model = slot.modelEntry else {
                    switch slot {
                    case .system, .provider:
                        continue
                    case .cataloged:
                        Issue.record("\(task.name) offers a cataloged slot with no modelEntry — should be impossible")
                        continue
                    }
                }
                let display = slot.displayName
                // (1) the executor's own stage path builds it.
                let resolved = try #require(registry.bestModule(for: model),
                                            "\(task.name) offers \(model.hfModelId) but no module claims it")
                #expect(throws: Never.self) {
                    _ = try resolved.sdk.makeStage(for: model, config: .default)
                }
                // (2) the pre-Run gate does not block a row naming it.
                let doc = FlowDocument(version: "0.8", rows: [
                    Row(id: UUID(), task: task.name, model: display, settings: nil, refs: []),
                ])
                let preflight = FlowPreflight.run(doc, catalog: catalog,
                                                  installedModelIDs: [],
                                                  totalRAMGB: SystemInfo.detect().totalRAMGB)
                #expect(!preflight.blocked.map(\.display).contains(display),
                        "\(task.name) offers \(display) but preflight blocks it — CFM-R14-FIX-1")
            }
        }
    }

    // MARK: - canRun refuses before the install prompt

    @Test func canRunRefusesAFlowWithAnUnportedNetTool() {
        let doc = FlowDocument(version: "0.8", rows: [
            Row(id: UUID(), task: "Web Search", settings: "query=hi"),
        ])
        guard case .notRunnable(let reason) = FlowRunner.canRun(doc) else {
            Issue.record("expected notRunnable")
            return
        }
        #expect(reason.contains("Web Search"))
        #expect(reason.contains("doesn't run yet"))
    }

    @Test func canRunAllowsJoinVideoNowThatItIsPorted() {
        // R13-3: Join Video was the last unported instant tool; a flow using it is runnable.
        let doc = FlowDocument(version: "0.8", rows: [
            Row(id: UUID(), task: "Read Video", settings: "clips/a.mp4"),
            Row(id: UUID(), task: "Join Video", settings: ""),
        ])
        #expect(FlowRunner.canRun(doc) == .runnable)
    }

    @Test func canRunStillAllowsAPortedInstantFlow() {
        let doc = FlowDocument(version: "0.8", rows: [
            Row(id: UUID(), task: "Read Text", settings: "memo.txt"),
            Row(id: UUID(), task: "Save Text", settings: "out.md"),
        ])
        #expect(FlowRunner.canRun(doc) == .runnable)
    }

    // MARK: - The channel states

    @Test func netIsRefusedButStagedNowRuns() {
        // Net/staged verdicts are catalog-free (CFM-R14-FIX-3) — explicit empties.
        let emptyCatalog: [ModelEntry] = []
        let emptyClaim: Set<String> = []
        let web = TaskCatalog.get("Web Search")!
        if case .refusedByChannel = TaskAvailability.state(for: web, catalog: emptyCatalog, claimableModelIDs: emptyClaim) {} else { Issue.record("net not refused") }
        #expect(!TaskAvailability.isAvailable("Web Search", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
        // CFM-R12-8: staged rows now queue a visible outbox entry.
        #expect(TaskAvailability.isAvailable("Stage Send", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
        #expect(TaskAvailability.isAvailable("Stage Post", catalog: emptyCatalog, claimableModelIDs: emptyClaim))
    }

    @Test func agentIsChannelRefusedOnlyUnderAppStore() {
        let improvise = TaskCatalog.get("Improvise")!
        let emptyCatalog: [ModelEntry] = []
        let emptyClaim: Set<String> = []
        #expect(TaskAvailability.state(for: improvise, isAppStore: true, catalog: emptyCatalog, claimableModelIDs: emptyClaim) == .refusedByChannel(reason: "runs only in the direct build"))
        #expect(TaskAvailability.state(for: improvise, isAppStore: false, catalog: emptyCatalog, claimableModelIDs: emptyClaim) == .available)
    }

    // MARK: - DA-2: the named-list verdict table (replaces the tautological marker test)

    /// DA-2 (`RSI/DelegateDeciderBacklog.md`) — replaces `everyUnavailableTaskGetsAPickerMarker`,
    /// which the R-FIX outcome flagged by name. That test compared `marker(for:) != nil`
    /// against `state(for:) != .available`, but `marker` is a plain `switch` over `state`, so
    /// the two agree **by construction** and it could never fail. DA-1's own journal (2026-238)
    /// records it passing *both* with and without the six decider lines in `taskKinds` — the
    /// tautology proving itself.
    ///
    /// This version pins a **named list**: every `.model`-class task in `TaskCatalog`, each with
    /// the verdict it must produce against the bundled catalog + the real registry-claim table,
    /// spelled out as data. Consequences:
    ///  - a new `.model` task added to `TaskCatalog` without a `taskKinds` entry (or without a
    ///    line here) fails the `Set` check, instead of silently shipping a "needs newer support"
    ///    marker nobody vetted;
    ///  - removing any of DA-1's six deciders from `taskKinds` flips its pinned `.available` and
    ///    fails the per-task check.
    /// `unportedInstant == []` is carried over from the retired test — that one is real (every
    /// instant tool has a `RealExecutor.runInstant` case), not tautological.
    @Test @MainActor func everyModelTaskMatchesItsPinnedVerdict() throws {
        let catalog = try bundledCatalog()
        let claimable = claimableIDs(catalog: catalog)

        // name → the verdict `TaskAvailability` must return. `.available` ⟺ the derived pool is
        // non-empty: a claimable, kind-correct catalog model exists AND `RealExecutor` serves
        // the task. `.needsNewerSupport` ⟺ it does not — the inline note says which clause fails.
        let expected: [String: TaskAvailability.State] = [
            // Plain + framed LLM tasks — the shared `.llm` pool.
            "Generate": .available, "Summarize": .available, "Translate": .available,
            "Answer": .available, "Rewrite": .available, "Draft": .available,
            "Ask": .available, "Title": .available, "Critique": .available,
            "Verify": .available, "Revise": .available, "Merge": .available,
            "Text to Table": .available,
            // The six deciders — DA-1. `.llm`; frame-backed except `Decide` (engines.llm.decide).
            "Decide": .available, "Classify": .available, "Gate": .available,
            "Score": .available, "Judge": .available, "Think": .available,
            // Other served engines with a claimable model in the bundled catalog.
            "Transcribe": .available,       // .asr
            "Speak": .available,            // .tts
            "Describe Image": .available,   // .vision
            "OCR": .available,              // .ocr
            "Embed": .available,            // .embedding
            "Generate Image": .available,   // .image — engines.diffusion.generate_image (served exactly)
            "Generate Video": .available,   // .video — engines.diffusion.generate_video
            "Generate Sound": .available,   // .music — engines.diffusion.generate_sound
            "Segment": .available,          // .segmentation — engines.diffusion.segment (CFM-R15-1)
            "Rerank": .available,           // .rerank — MoC-4
            "Extract Structured": .available,   // .llm — engines.llm.extract_structured, ported DA-3a, offered DA-3b

            // `.image` model exists, but `engines.diffusion.edit_image` / `.inpaint` are not on
            // the served allow-list — no stage accepts the tuple these rows hand the executor.
            "Edit Image": .needsNewerSupport, "Instruct Edit": .needsNewerSupport,
            "Inpaint": .needsNewerSupport,
            // The latent family — `.image` kind in `taskKinds`, but not executor-served.
            "Init Latent": .needsNewerSupport, "Encode Latent": .needsNewerSupport,
            "Decode Latent": .needsNewerSupport, "Denoise": .needsNewerSupport,
            // In `taskKinds` (.video) but `engines.diffusion.animate` has no served prefix.
            "Animate": .needsNewerSupport,
            // No `taskKinds` entry and no served prefix.
            "Upscale": .needsNewerSupport, "Estimate Depth": .needsNewerSupport,
            "Load Checkpoint": .needsNewerSupport, "Blend": .needsNewerSupport,
            "Bake LoRA": .needsNewerSupport, "Pin Model": .needsNewerSupport,
        ]

        let modelTasks = Set(TaskCatalog.allTasks()
            .filter { $0.taskClass == .model }.map(\.name))
        let drift = modelTasks.symmetricDifference(Set(expected.keys)).sorted()
        #expect(modelTasks == Set(expected.keys),
                "model-task drift — add the new task to `expected` with its verdict: \(drift)")

        for task in modelTasks.sorted() {
            guard let want = expected[task], let desc = TaskCatalog.get(task) else { continue }
            let got = TaskAvailability.state(for: desc, catalog: catalog,
                                            claimableModelIDs: claimable)
            #expect(got == want, "\(task): pinned \(want) but got \(got)")
        }

        // Carried over from the retired test — real, not tautological: every instant tool in
        // the catalog has a `RealExecutor.runInstant` case (Join Video was the last, R13-3).
        let unportedInstant = TaskCatalog.entries.filter {
            $0.taskClass == .instant && !TaskAvailability.supportedInstantTools.contains($0.name)
        }.map(\.name)
        #expect(unportedInstant == [])
    }

    /// The bundled catalog's flat entries.
    private func bundledCatalog() throws -> [ModelEntry] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(BrowserData.self, from: data)
            .domains.flatMap { $0.allModels }
    }

    /// The registry's own claim answer over the bundled catalog — a real `ModelRegistry`
    /// with every shipped module registered, exactly as `AppState.init` builds it.
    private func claimableIDs(catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    /// CFM-R12-FIX-12 → CFM-R14-2 + FIX-2: a model task's availability agrees with its
    /// **derived pool** — a task is available exactly when the registry-claimable, kind-correct
    /// catalog set is non-empty **and the executor genuinely serves the task**. The old
    /// bridge-as-authority version is retired with the display-name pools; this is the
    /// registry's answer.
    @Test func modelAvailabilityAgreesWithTheDerivedPool() throws {
        let catalog = try bundledCatalog()
        let claimable = claimableIDs(catalog: catalog)
        let modelTasks = TaskCatalog.entries.filter { $0.taskClass == .model }
        var mismatches: [String] = []
        for task in modelTasks {
            let derived = TaskModels.derivedModels(for: task.name, catalog: catalog,
                                                   claimableModelIDs: claimable)
            let available = TaskAvailability.isAvailable(task.name, catalog: catalog,
                                                         claimableModelIDs: claimable)
            if (derived.isEmpty == available) {
                mismatches.append("\(task.name): derived=\(derived.map { $0.displayName }) availability=\(available)")
            }
        }
        #expect(mismatches.isEmpty, "model drift: \(mismatches.joined(separator: "; "))")
        // The specific gaps the review named (CFM-R14-FIX-2 → CFM-R15-1). OCR + Describe
        // Image are bridged; Generate Image derives Flux + SDXL-Turbo (the executor serves
        // generate_image). Segment was pending the owner's CFM-R13-6 ruling and became
        // available 2026-08-27 when the ruling landed option (1) — the executor now serves
        // engines.diffusion.segment and SAM3 derives (CFM-R15-1). The latent family, Edit
        // Image, and Inpaint map to `.image` but no stage accepts what those rows hand the
        // executor, so they stay marked. Upscale and Estimate Depth have no catalog model.
        let available = { (name: String) in
            TaskAvailability.isAvailable(name, catalog: catalog, claimableModelIDs: claimable)
        }
        #expect(available("Segment"))
        #expect(!available("Edit Image"))
        #expect(!available("Inpaint"))
        #expect(!available("Init Latent"))
        #expect(!available("Decode Latent"))
        #expect(available("Generate Image"))
        #expect(!available("Upscale"))
        #expect(!available("Estimate Depth"))
        #expect(available("OCR"))
        #expect(available("Describe Image"))
        #expect(available("Summarize"))
        #expect(available("Transcribe"))
    }

    // MARK: - DA-1: the six deciders join the derived-pool authority

    /// DA-1 (`RSI/DelegateDeciderBacklog.md`, owner ruling 2026-09-09). Until DA-1 the six
    /// decider tasks had no `TaskModels.taskKinds` entry, so `derivedModels` failed on its
    /// **first** guard clause (`guard let kind = taskKinds[task]`), the pool was `[]`, and
    /// every one reported `.needsNewerSupport` — an empty Model menu plus a false yellow
    /// "needs a model" warning on 21 gallery rows that run fine. Giving them `.llm` (the kind
    /// `RealExecutor.runDecider` actually builds — a plain LLM stage for all six) makes the
    /// derived pool the same non-empty pool `Summarize` already has.
    ///
    /// This test genuinely fails with the six lines removed from `taskKinds`: verified locally
    /// by deleting them (all six `available(...)` and all six non-empty-pool expectations went
    /// red), then restoring them — see journal `2026-238`.
    @Test func theSixDecidersBecomeAvailableWithDA1() throws {
        let catalog = try bundledCatalog()
        let claimable = claimableIDs(catalog: catalog)
        let available = { (name: String) in
            TaskAvailability.isAvailable(name, catalog: catalog, claimableModelIDs: claimable)
        }
        // Control: a frame-backed `.llm` model task that was already available before DA-1.
        #expect(available("Summarize"))
        // The six deciders — Classify/Gate/Score/Judge/Think are `.frame`, Decide is `.engine`
        // (`engines.llm.decide`). All six resolve to a plain LLM stage in `runDecider`.
        for name in ["Decide", "Classify", "Gate", "Score", "Judge", "Think"] {
            #expect(available(name), "\(name) must be available after DA-1")
            #expect(!TaskModels.derivedModels(for: name, catalog: catalog,
                                              claimableModelIDs: claimable).isEmpty,
                    "\(name) must have a non-empty derived pool after DA-1")
        }
        // DA-3b flipped this: `Extract Structured` now has a real `RealExecutor` path
        // (`ExtractStructuredStage`, DA-3a) *and* its `taskKinds` entry, so it is offered.
        // Was `#expect(!available(...))` through DA-3a — the assertion the port kept red on
        // purpose so the offer stayed a separate, revertable commit.
        #expect(available("Extract Structured"),
                "Extract Structured must be available after DA-3b")
        #expect(!TaskModels.derivedModels(for: "Extract Structured", catalog: catalog,
                                          claimableModelIDs: claimable).isEmpty)
    }

    /// CFM-R14-FIX-2 — the executor-served allow-list is the authority: a model task is
    /// offerable only when `RealExecutor` genuinely has a path for it, not merely because a
    /// catalog model exists for its kind. This pins the FIX-2 decision and why, so a future
    /// "just add the kind" can't silently re-offer a task nobody can run.
    @Test func offerableModelTasksAreExecutorServed() {
        // Served: an executor path exists for each (LLM frames, ASR, TTS, VLM, embed, the
        // diffusion generators, and — since the owner ruled CFM-R13-6 option (1) 2026-08-27 —
        // Segment). Frame-backed LLM tasks are covered by `refKind == .frame`.
        let served = ["Generate", "Summarize", "Translate", "Answer", "Rewrite", "Draft",
                      "Ask", "Title", "Critique", "Verify", "Revise", "Merge",
                      "Extract Structured", "Text to Table",
                      "Transcribe", "Speak", "Describe Image", "OCR", "Embed",
                      "Generate Image", "Generate Video", "Generate Sound",
                      "Segment", "Rerank"]
        for name in served {
            #expect(TaskModels.isServedByExecutor(name), "\(name) should be executor-served")
        }
        // NOT served: a catalog model may exist for the kind, but the executor has no path.
        // The latent family and the edit/inpaint rows have no stage accepting what they hand
        // the executor; Upscale/Estimate Depth have no served prefix at all. `Rerank` moved to
        // the served list in MoC-3-1 (RSI/DelegateMoCBacklog.md) — `engines.rerank.` is now
        // allow-listed, even though no `.rerank` catalog entry exists yet (MoC-4). Served and
        // "has a runnable candidate" are different questions: `derivedModels(for: "Rerank", …)`
        // stays `[]` until a model lands, exactly as it did before this entry.
        let unserved = ["Edit Image", "Instruct Edit", "Inpaint",
                        "Init Latent", "Encode Latent", "Decode Latent", "Denoise",
                        "Upscale", "Estimate Depth", "Animate"]
        for name in unserved {
            #expect(!TaskModels.isServedByExecutor(name), "\(name) should NOT be executor-served")
        }
    }

    // MARK: - MoC-4: Rerank stopped being honestly empty once MoC-4-1/4-3 landed

    /// Updated from MoC-3-1's "stays empty until MoC-4" version now that MoC-4-1 (catalog
    /// entry), MoC-4-3 (`RerankSDK`, claims it), and MoC-4-4 (the `CatalogBridge` entry
    /// connecting the catalog's raw `displayName` to the pool's spaced display name) have
    /// all landed — the CFM-R14-FIX-2 discipline this guards is now "the derived pool
    /// matches what the runtime can actually do," not "empty." `defaultModel` now returns
    /// `"Qwen3 Reranker 0.6B"` — the bridge display name, which is also `taskModels["Rerank"]`'s
    /// pool string — for the first time since `Rerank` existed as a task.
    @Test func rerankDerivedPoolNowContainsTheLandedModel() throws {
        let catalog = try bundledCatalog()
        let claimable = claimableIDs(catalog: catalog)
        let derived = TaskModels.derivedModels(for: "Rerank", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.count == 1)
        #expect(derived.first?.modelEntry?.hfModelId == "mlx-community/Qwen3-Reranker-0.6B-4bit")
        #expect(TaskModels.defaultModel(forTask: "Rerank", catalog: catalog, claimableModelIDs: claimable) == "Qwen3 Reranker 0.6B")
    }

    /// The bundled catalog decodes with the `rerank` domain leaf now carrying the
    /// Qwen3-Reranker entry MoC-4-1 added (the leaf itself was created empty in MoC-3-1).
    @Test func rerankDomainDecodesTheLandedModel() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("MLXUI/Resources/browser.json")
        let data = try Data(contentsOf: url)
        let browserData = try JSONDecoder().decode(BrowserData.self, from: data)
        let rerank = try #require(browserData.domains.first { $0.id == "rerank" })
        #expect(rerank.allModels.count == 1)
        #expect(rerank.allModels.first?.hfModelId == "mlx-community/Qwen3-Reranker-0.6B-4bit")
    }
}
