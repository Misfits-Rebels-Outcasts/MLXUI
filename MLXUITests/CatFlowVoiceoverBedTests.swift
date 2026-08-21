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
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).parse.json")
        return try JSONDecoder().decode(FlowDocument.self, from: Data(contentsOf: url))
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
