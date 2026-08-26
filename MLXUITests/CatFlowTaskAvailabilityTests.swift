import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-4 — the availability truth. The load-bearing test is the cross-check: enumerate
/// the catalog and assert `TaskAvailability` agrees with `RealExecutor`'s actual dispatch for
/// every instant task (drive the executor, observe whether it throws `unsupportedTask`).
/// Adding a `case` to `RealExecutor` must be the only edit needed to flip a task available.
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

    /// CFM-R12-FIX-12: a model task's availability agrees with `CatalogBridge` — a task is
    /// available exactly when one of its pool's models resolves.
    @Test func modelAvailabilityAgreesWithTheBridge() {
        let modelTasks = TaskCatalog.entries.filter { $0.taskClass == .model }
        var mismatches: [String] = []
        for task in modelTasks {
            let hasModel = TaskModels.defaultModel(forTask: task.name) != nil
            if hasModel != TaskAvailability.isAvailable(task.name) {
                mismatches.append("\(task.name): bridge=\(hasModel) availability=\(TaskAvailability.isAvailable(task.name))")
            }
        }
        #expect(mismatches.isEmpty, "model drift: \(mismatches.joined(separator: "; "))")
        // The specific gaps the review named. CFM-R13-9/12 bridged OCR and Describe Image;
        // Segment, Upscale, Estimate Depth and Generate Image still have no catalog model.
        #expect(!TaskAvailability.isAvailable("Segment"))
        #expect(!TaskAvailability.isAvailable("Upscale"))
        #expect(!TaskAvailability.isAvailable("Estimate Depth"))
        #expect(!TaskAvailability.isAvailable("Generate Image"))
        #expect(TaskAvailability.isAvailable("OCR"))
        #expect(TaskAvailability.isAvailable("Describe Image"))
        #expect(TaskAvailability.isAvailable("Summarize"))
    }
}
