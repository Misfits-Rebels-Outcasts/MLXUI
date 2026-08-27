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
            let available = TaskAvailability.isAvailable(task.name)
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
        let web = TaskCatalog.get("Web Search")!
        if case .refusedByChannel = TaskAvailability.state(for: web) {} else { Issue.record("net not refused") }
        #expect(!TaskAvailability.isAvailable("Web Search"))
        // CFM-R12-8: staged rows now queue a visible outbox entry.
        #expect(TaskAvailability.isAvailable("Stage Send"))
        #expect(TaskAvailability.isAvailable("Stage Post"))
    }

    @Test func agentIsChannelRefusedOnlyUnderAppStore() {
        let improvise = TaskCatalog.get("Improvise")!
        #expect(TaskAvailability.state(for: improvise, isAppStore: true) == .refusedByChannel(reason: "runs only in the direct build"))
        #expect(TaskAvailability.state(for: improvise, isAppStore: false) == .available)
    }

    @Test func everyUnavailableTaskGetsAPickerMarker() {
        // The marker set and the availability state agree by construction.
        let expectedUnavailable = TaskCatalog.allTasks().filter {
            TaskAvailability.marker(for: $0) != nil
        }
        let computedUnavailable = TaskCatalog.allTasks().filter { task in
            switch TaskAvailability.state(for: task) {
            case .available: return false
            case .needsNewerSupport, .refusedByChannel: return true
            }
        }
        #expect(expectedUnavailable.map(\.name).sorted() == computedUnavailable.map(\.name).sorted())
        // Every one is labeled, and every available one is not.
        for task in TaskCatalog.allTasks() {
            #expect((TaskAvailability.marker(for: task) != nil) != TaskAvailability.isAvailable(task.name))
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

    /// CFM-R12-FIX-12 → CFM-R14-2: a model task's availability agrees with its **derived
    /// pool** — a task is available exactly when the registry-claimable, kind-correct catalog
    /// set is non-empty. The old bridge-as-authority version is retired with the display-name
    /// pools; this is the registry's answer.
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
        // The specific gaps the review named. CFM-R13-9/12 bridged OCR + Describe Image;
        // R14-2 derives Generate Image (Flux + SDXL-Turbo) and Segment (sam3) into
        // availability. Upscale and Estimate Depth have **no catalog model at all** — those
        // stay marked. Every assertion passes the catalog + claim table: the derived pool is
        // the availability authority since R14-2.
        let available = { (name: String) in
            TaskAvailability.isAvailable(name, catalog: catalog, claimableModelIDs: claimable)
        }
        #expect(available("Segment"))
        #expect(available("Generate Image"))
        #expect(available("Edit Image"))
        #expect(!available("Upscale"))
        #expect(!available("Estimate Depth"))
        #expect(available("OCR"))
        #expect(available("Describe Image"))
        #expect(available("Summarize"))
        #expect(available("Transcribe"))
    }
}
