import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R4-3: `65-VoiceoverBed` — `Generate Sound` (MusicGen) → `Save Audio`. The
/// `.substitute` bridge path (MusicGen → `jasonvassallo/mlx-musicgen-small`) is exercised
/// through preflight, and the flow runs under the mock executor. See
/// `RSI/DelegateMergeBacklog.md` CFM-R4-3.
struct CatFlowVoiceoverBedTests {

    private func decode(_ flowID: String) throws -> FlowDocument {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).cat")
        return try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    // MARK: - Runnable + bridge

    @Test func voiceoverBedIsRunnable() throws {
        let doc = try decode("65-VoiceoverBed")
        #expect(FlowRunner.canRun(doc) == .runnable)
    }

    @Test func voiceoverBedResolvesMusicGenThroughSubstitute() throws {
        let doc = try decode("65-VoiceoverBed")
        let catalog = try loadCatalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16)

        // Row 1 is a model row (Generate Sound / MusicGen), not blocked.
        #expect(result.isBlocked == false)
        let need = try #require(result.needs.first)
        #expect(need.display == "MusicGen")
        #expect(need.model?.modelType == .music)
        // The bridge equivalence is .substitute → the inspector shows a note.
        #expect(need.equivalence == .substitute)

        let session = FlowRunSession()
        session.prepareInstall(result, doc: doc)
        let note = session.substitutionNotes[doc.rows[0].id]
        #expect(note?.contains("MusicGen") == true)
    }

    // MARK: - First-row prompt fallback (smoke-34: "needs an input, but got 0")

    @Test func generateSoundUsesSettingsPromptWhenNoInput() async throws {
        // 65-VoiceoverBed row 1 is the flow's first row — it has no upstream input. The
        // Python `_prompt` falls back to `Settings.first_bare()`, so the Swift must feed
        // the quoted settings prompt to the stage rather than refuse with "got 0".
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let catalogURL = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: catalogURL))
            .domains.flatMap { $0.allModels }
        let music = try #require(catalog.first { $0.hfModelId == "jasonvassallo/mlx-musicgen-small" })

        let capture = CapturingStage()
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "65-VoiceoverBed",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in capture },
            installedModelIDs: [music.id],
            catalog: catalog)

        let row = Row(task: "Generate Sound", model: "MusicGen",
                      settings: "seed=7\n\"calm ambient background music, soft piano\"")
        let asset = try await executor.execute(path: "1", row: row, inputs: [])
        // The stub echoes its text prompt; it must be the quoted settings prompt, not a
        // "needs an input" error.
        #expect(asset.items.first?.value?.contains("calm ambient background music") == true)
    }

    // MARK: - Runs under mock

    @Test func voiceoverBedRunsEndToEndUnderMock() async throws {
        let doc = try decode("65-VoiceoverBed")
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-voiceover-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let blob = base.appendingPathComponent("blobs")
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let context = FlowRunner.RunContext(flowID: "65-VoiceoverBed", workspace: workspace,
                                            blobDirectory: blob,
                                            executor: MockExecutor(blobDirectory: blob))

        let runner = FlowRunner()
        var started: [UUID] = []
        var finished: [UUID] = []
        for await event in runner.run(doc, context: context) {
            if case .started(let id) = event { started.append(id) }
            if case .finished(let id, _) = event { finished.append(id) }
        }
        #expect(started.map(\.uuidString) == doc.rows.map(\.id.uuidString))
        #expect(finished.map(\.uuidString) == doc.rows.map(\.id.uuidString))
    }
}

/// A stub `text → text` stage that echoes its input — the smoke-34 harness: it proves the
/// settings prompt reached the stage instead of a "needs an input" refusal.
private struct CapturingStage: PipelineStage {
    let id = "stub.capture"
    let name = "Capture"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        progress(1.0)
        return input
    }
}
