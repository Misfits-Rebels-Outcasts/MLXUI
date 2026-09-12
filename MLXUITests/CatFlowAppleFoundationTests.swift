import Testing
import Foundation
@testable import MLXUI

/// Phase AFM — Apple Foundation Models as a selectable model, entirely mock-driven (the real
/// device path is excluded from CI, same convention as MLX's model-test marker).
/// `AppleFoundationAvailability.checkerOverride`/`executorOverride`/`simulateOSUnavailable`
/// are the seams; every test resets them, since they're process-global.
struct CatFlowAppleFoundationTests {

    /// A fixture `Readiness`, injected via `checkerOverride` so a test can drive the pool
    /// through every state without a real device.
    private struct FixedChecker: AppleFoundationAvailabilityChecking {
        let readiness: Readiness
    }

    /// A fixture executor, injected via `executorOverride`. Records the last call for
    /// assertions that care what was actually asked.
    private final class MockExecutor: AppleFoundationExecuting, @unchecked Sendable {
        var textToReturn = "a short summary"
        var tagToReturn = "ship"
        private(set) var lastPrompt: String?
        private(set) var lastTags: [String]?

        func generate(instructions: String, prompt: String) async throws -> String {
            lastPrompt = prompt
            return textToReturn
        }

        func generateTag(instructions: String, prompt: String, tags: [String]) async throws -> String {
            lastPrompt = prompt
            lastTags = tags
            return tagToReturn
        }
    }

    /// Resets every override — call at the start of a test that touches them, and rely on
    /// `defer` to reset again on exit, so a failure mid-test can't leak state into the next.
    private func resetOverrides() {
        AppleFoundationAvailability.simulateOSUnavailable = false
        AppleFoundationAvailability.checkerOverride = nil
        AppleFoundationAvailability.executorOverride = nil
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    private func issues(for text: String) throws -> [FlowIssue] {
        let flow = try CatParser.parseForValidation(text)
        return FlowValidator.checkFlow(flow)
    }

    private func realExecutor(catalog: [ModelEntry] = []) -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "test",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [],
            catalog: catalog)
    }

    // MARK: - AFM-1: the manifest, decoded

    @Test func manifestDecodesKindEngineTasksAndCredentials() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "apple-foundation.json"))
        #expect(manifest.id == "apple/foundation-models")
        #expect(manifest.display == "apple-foundation @ system")
        #expect(manifest.kind == "system")
        #expect(manifest.engine == "apple-foundation-models")
        #expect(manifest.credentials == nil)
        #expect(manifest.capabilities?.rejectedSettings == nil)
        let tasks = try #require(manifest.tasks)
        for task in ["Summarize", "Rewrite", "Translate", "Answer", "Draft", "Title",
                     "Critique", "Verify", "Merge", "Revise",
                     "Classify", "Gate", "Score", "Judge", "Decide"] {
            #expect(tasks.contains(task), "manifest should list \(task)")
        }
        // AFM-2: not `Generate` — the manifest's curated list is narrower than the
        // reference's `@frames-text` group, which also includes it.
        #expect(!tasks.contains("Generate"))
    }

    // MARK: - AFM-1: absent below macOS 26, present when ready — MS-2's third section

    @Test func appleFoundationIsAbsentFromThePoolWhenTheOSIsSimulatedUnavailable() throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.simulateOSUnavailable = true
        let catalog = try loadCatalog()
        let pool = TaskModels.derivedModels(for: "Summarize", catalog: catalog, claimableModelIDs: [])
        #expect(!pool.contains { if case .system = $0 { return true }; return false })
    }

    @Test func appleFoundationAppearsInEveryManifestTaskWhenReady() throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(readiness: .ready)
        let catalog = try loadCatalog()
        let manifestTasks = try #require(CuratedManifest.load(manifestFile: "apple-foundation.json")?.tasks)
        for task in manifestTasks {
            let pool = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: [])
            #expect(pool.contains { $0.displayName == "apple-foundation @ system" },
                    "\(task) should offer the system slot once the manifest lists it")
        }
    }

    /// AFM-1's exit condition, made concrete: the build machine this suite actually runs on
    /// is macOS 26+ (the SDK is installed — `RSI/journal/2026-258` verified this against the
    /// real `FoundationModels` framework), so **the default, no-override path reflects this
    /// machine's real Apple Intelligence state, never `nil`.** Only `simulateOSUnavailable`
    /// lets a test see the pre-AFM "absent from the pool" shape on a real macOS 26 machine —
    /// which is exactly what every pre-AFM test file (MS's `CatFlowModelSlotTests` included)
    /// runs under, since none of them touch this override.
    @Test func everyExistingPoolTestRunsUnderTheSimulatedAbsentShape() throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.simulateOSUnavailable = true
        let catalog = try loadCatalog()
        let pool = TaskModels.derivedModels(for: "Summarize", catalog: catalog, claimableModelIDs: [])
        #expect(!pool.contains { $0.displayName == "apple-foundation @ system" })
    }

    // MARK: - AFM-1: readiness states reach the picker via ModelSlot (fixture-driven)

    @Test func needsSetupReadinessKeepsTheSlotSelectable() {
        resetOverrides()
        defer { resetOverrides() }
        let ref = SystemModelRef(id: "apple/foundation-models", displayName: "apple-foundation @ system",
                                 readiness: .needsSetup(reason: "Turn on Apple Intelligence in System Settings",
                                                        action: .enableAppleIntelligence),
                                 resourceNote: "Built into macOS")
        let slot = ModelSlot.system(ref)
        #expect(FlowRowInspectorView.isModelButtonEnabled(slot, installedModelIDs: [], totalRAMGB: 8))
    }

    @Test func deviceNotEligibleReadinessDisablesTheSlot() {
        let ref = SystemModelRef(id: "apple/foundation-models", displayName: "apple-foundation @ system",
                                 readiness: .unavailable(reason: "This Mac can't run Apple Intelligence"),
                                 resourceNote: "Built into macOS")
        let slot = ModelSlot.system(ref)
        #expect(!FlowRowInspectorView.isModelButtonEnabled(slot, installedModelIDs: [], totalRAMGB: 8))
    }

    // MARK: - AppleFoundationAvailability's own plumbing

    @Test func currentReadinessHonorsTheSimulatedUnavailableOverride() {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.simulateOSUnavailable = true
        #expect(AppleFoundationAvailability.currentReadiness() == nil)
    }

    @Test func currentReadinessHonorsTheCheckerOverride() {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(readiness: .needsDownload(gb: 0))
        #expect(AppleFoundationAvailability.currentReadiness() == .needsDownload(gb: 0))
    }

    @Test func makeExecutorReturnsTheOverrideWhenSet() async throws {
        resetOverrides()
        defer { resetOverrides() }
        let mock = MockExecutor()
        AppleFoundationAvailability.executorOverride = mock
        let executor = try #require(AppleFoundationAvailability.makeExecutor())
        _ = try await executor.generate(instructions: "", prompt: "hi")
        #expect(mock.lastPrompt == "hi")
    }

    // MARK: - AFM-1: `isRemoteRow` — the trap the plan named

    @Test func appleFoundationRowIsNotRemote() throws {
        let doc = try CatParser.parseForValidation("""
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Summarize    (1)  apple-foundation @ system
        3. Save Text    out.md

        models:
          apple-foundation @ system = apple/foundation-models
        """)
        guard case .some(let row) = doc.rows.first(where: { $0.task == "Summarize" }) else {
            Issue.record("expected a Summarize row")
            return
        }
        #expect(!FlowValidator.isRemoteRow(row))
    }

    @Test func aFlowNamingAppleFoundationValidatesWithNoE120() throws {
        // E120 has no raise site in this tree yet (verified: `grep -rn "\"E120\"" MLXUI/`
        // matches only the ErrorCatalog template) — this pins intent, not a live regression,
        // and will start meaning something the moment a direct offdevice check is added.
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Summarize    (1)  apple-foundation @ system
        3. Save Text    out.md

        models:
          apple-foundation @ system = apple/foundation-models
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E120" })
    }

    /// The trap's real, *live* consequence on this tree: `isRemoteRow` feeds E605's
    /// "costly row" classification. Before the fix, a trigger-headed flow naming AFM with no
    /// rate budget would wrongly demand one, as if AFM made a network call.
    @Test func aTriggerHeadedFlowUsingAppleFoundationDoesNotWronglyDemandARateBudget() throws {
        let text = """
        mlxflow 0.8; events
        1. On Schedule
        2. Read Text    (1)  memo.txt
        3. Summarize    (2)  apple-foundation @ system
        4. Save Text    out.md

        models:
          apple-foundation @ system = apple/foundation-models
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E605" },
                "an on-device model must never count as a costly row for the rate-budget check")
    }

    // MARK: - AFM-3: FlowPreflight plans an empty download set when ready

    @Test func preflightPlansAnEmptyDownloadSetWhenAppleFoundationIsReady() throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(readiness: .ready)
        let doc = FlowDocument(version: "0.8", rows: [
            Row(task: "Summarize", model: "apple-foundation @ system", settings: "\"x\""),
        ])
        let result = FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 16)
        #expect(result.toDownload.isEmpty)
        #expect(result.installed.count == 1)
        #expect(!result.isBlocked)
    }

    @Test func preflightSurfacesNeedsSetupSeparatelyFromBlocked() throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(
            readiness: .needsSetup(reason: "Turn on Apple Intelligence in System Settings",
                                   action: .enableAppleIntelligence))
        let doc = FlowDocument(version: "0.8", rows: [
            Row(task: "Summarize", model: "apple-foundation @ system", settings: "\"x\""),
        ])
        let result = FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 16)
        #expect(result.needsSetup.count == 1)
        #expect(result.blocked.isEmpty)
        #expect(!result.isBlocked)   // needsSetup alone doesn't hard-block
        #expect(FlowPreflight.blockedReason(result, totalRAMGB: 16) == "Turn on Apple Intelligence in System Settings")
        #expect(FlowPreflight.blockedAction(result) == .enableAppleIntelligence)
    }

    // MARK: - AFM-2: RealExecutor dispatches a frame-backed row to Apple Foundation Models

    @Test func summarizeRowDispatchesToTheMockExecutorWhenReady() async throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(readiness: .ready)
        let mock = MockExecutor()
        mock.textToReturn = "a short summary"
        AppleFoundationAvailability.executorOverride = mock

        let executor = realExecutor()
        let row = Row(task: "Summarize", model: "apple-foundation @ system", settings: nil)
        let input = Asset(items: [Item(kind: .text, value: "a long document about many things",
                                       path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input])
        #expect(output.items.first?.value == "a short summary")
        #expect(mock.lastPrompt?.contains("a long document about many things") == true)
    }

    @Test func summarizeRowRefusesPlainlyWhenNotReady() async throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(
            readiness: .unavailable(reason: "This Mac can't run Apple Intelligence"))
        let executor = realExecutor()
        let row = Row(task: "Summarize", model: "apple-foundation @ system", settings: nil)
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        await #expect(throws: (any Error).self) {
            _ = try await executor.execute(path: "1", row: row, inputs: [input])
        }
    }

    // MARK: - AFM-2: deciders — guided generation, never F004's parse-and-retry

    @Test func gateRowReturnsATagFromTheDeclaredSetViaGuidedGeneration() async throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(readiness: .ready)
        let mock = MockExecutor()
        mock.tagToReturn = "ship"
        AppleFoundationAvailability.executorOverride = mock

        let executor = realExecutor()
        let row = Row(task: "Gate", model: "apple-foundation @ system",
                     settings: "\"Ready to ship?\"", tags: ["ship", "hold"])
        let input = Asset(items: [Item(kind: .text, value: "looks solid", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input])
        #expect(executor.lastTag == "ship")
        #expect(mock.lastTags == ["ship", "hold"])
        #expect(output.items.first?.value == "looks solid")   // passthrough, per Spec §7.4 R1
    }

    @Test func aGuidedTagOutsideTheDeclaredSetRefusesRatherThanGuess() async throws {
        resetOverrides()
        defer { resetOverrides() }
        AppleFoundationAvailability.checkerOverride = FixedChecker(readiness: .ready)
        let mock = MockExecutor()
        mock.tagToReturn = "not-a-real-tag"
        AppleFoundationAvailability.executorOverride = mock

        let executor = realExecutor()
        let row = Row(task: "Gate", model: "apple-foundation @ system",
                     settings: "\"Ready to ship?\"", tags: ["ship", "hold"])
        let input = Asset(items: [Item(kind: .text, value: "looks solid", path: nil, sourceText: nil)])
        await #expect(throws: (any Error).self) {
            _ = try await executor.execute(path: "1", row: row, inputs: [input])
        }
    }
}
