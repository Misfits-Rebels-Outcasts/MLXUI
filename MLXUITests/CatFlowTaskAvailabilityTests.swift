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
            for model in pool {
                let display = TaskModels.displayName(for: model)
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

    @Test @MainActor func everyUnavailableTaskGetsAPickerMarker() throws {
        // The marker set and the availability state agree by construction. Model tasks need
        // the real catalog + claim table (CFM-R14-FIX-3); instant/net verdicts ignore them.
        let catalog = try bundledCatalog()
        let claimable = claimableIDs(catalog: catalog)
        let expectedUnavailable = TaskCatalog.allTasks().filter {
            TaskAvailability.marker(for: $0, catalog: catalog, claimableModelIDs: claimable) != nil
        }
        let computedUnavailable = TaskCatalog.allTasks().filter { task in
            switch TaskAvailability.state(for: task, catalog: catalog, claimableModelIDs: claimable) {
            case .available: return false
            case .needsNewerSupport, .refusedByChannel: return true
            }
        }
        #expect(expectedUnavailable.map(\.name).sorted() == computedUnavailable.map(\.name).sorted())
        // Every one is labeled, and every available one is not.
        for task in TaskCatalog.allTasks() {
            #expect((TaskAvailability.marker(for: task, catalog: catalog, claimableModelIDs: claimable) != nil)
                    != TaskAvailability.isAvailable(task.name, catalog: catalog, claimableModelIDs: claimable))
        }
        // Every instant tool is ported since R13-3 (Join Video was the last).
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
                mismatches.append("\(task.name): derived=\(derived.map { $0.hfModelId }) availability=\(available)")
            }
        }
        #expect(mismatches.isEmpty, "model drift: \(mismatches.joined(separator: "; "))")
        // The specific gaps the review named (CFM-R14-FIX-2). OCR + Describe Image are bridged;
        // Generate Image derives Flux + SDXL-Turbo (the executor serves generate_image).
        // Segment has a catalog model (sam3) but **no sanctioned executor path** — hazard H2 /
        // CFM-R13-6 is the owner's to answer, so it stays marked. The latent family, Edit
        // Image, and Inpaint map to `.image` but no stage accepts what those rows hand the
        // executor, so they stay marked. Upscale and Estimate Depth have no catalog model.
        let available = { (name: String) in
            TaskAvailability.isAvailable(name, catalog: catalog, claimableModelIDs: claimable)
        }
        #expect(!available("Segment"))
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

    /// CFM-R14-FIX-2 — the executor-served allow-list is the authority: a model task is
    /// offerable only when `RealExecutor` genuinely has a path for it, not merely because a
    /// catalog model exists for its kind. This pins the FIX-2 decision and why, so a future
    /// "just add the kind" can't silently re-offer a task nobody can run.
    @Test func offerableModelTasksAreExecutorServed() {
        // Served: an executor path exists for each (LLM frames, ASR, TTS, VLM, embed, the
        // diffusion generators). Frame-backed LLM tasks are covered by `refKind == .frame`.
        let served = ["Generate", "Summarize", "Translate", "Answer", "Rewrite", "Draft",
                      "Ask", "Title", "Critique", "Verify", "Revise", "Merge",
                      "Extract Structured", "Text to Table",
                      "Transcribe", "Speak", "Describe Image", "OCR", "Embed",
                      "Generate Image", "Generate Video", "Generate Sound"]
        for name in served {
            #expect(TaskModels.isServedByExecutor(name), "\(name) should be executor-served")
        }
        // NOT served: a catalog model may exist for the kind, but the executor has no path.
        // Segment is pending the owner's CFM-R13-6 ruling; the latent family and the
        // edit/inpaint rows have no stage accepting what they hand the executor.
        let unserved = ["Segment", "Edit Image", "Instruct Edit", "Inpaint",
                        "Init Latent", "Encode Latent", "Decode Latent", "Denoise",
                        "Upscale", "Estimate Depth", "Animate", "Rerank"]
        for name in unserved {
            #expect(!TaskModels.isServedByExecutor(name), "\(name) should NOT be executor-served")
        }
    }
}
