import Testing
import Foundation
@testable import MLXUI

/// ER-10 (SPEC-Q238 / SPEC-Q239, owner rulings R1 and R11): `Identify Speakers` and `Clean Audio`. The four tables, the no-settings
/// refusals (E723 / E724), the table shaping, and execution through stub stages (no weights).
struct CatFlowAudioRowsTests {

    private static let sortformer = "Sortformer 4spk v2.1"
    private static let deepFilter = "DeepFilterNet 3"

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    @MainActor
    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("er10-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func wav(in dir: URL, rate: Int = 16000) throws -> Asset {
        let url = dir.appendingPathComponent("in.wav")
        try AudioWriter.writeWAV(AudioBuffer(samples: (0..<rate / 2).map { sin(Float($0) * 0.05) * 0.3 }, sampleRate: rate), to: url)
        return Asset(items: [Item(kind: .audio, value: nil, path: url, sourceText: nil)])
    }

    // MARK: - table 1 and 4: catalog and bridge

    @Test func theTasksAreModelRowsWithTheRightShapes() throws {
        let speakers = try #require(TaskCatalog.get("Identify Speakers"))
        #expect(speakers.taskClass == .model && speakers.refKind == .engine)
        #expect(speakers.refName == "engines.diarize.identify_speakers")
        #expect(speakers.accepts == .single(.audio) && speakers.gives == .single(.table))
        let clean = try #require(TaskCatalog.get("Clean Audio"))
        #expect(clean.taskClass == .model && clean.refKind == .engine)
        #expect(clean.refName == "engines.audio_clean.clean_audio")
        #expect(clean.accepts == .single(.audio) && clean.gives == .single(.audio))
    }

    @Test func theBridgeRowsResolveToTheCatalogCards() throws {
        let catalog = try loadCatalog()
        for (display, id, kind, file) in [
            (Self.sortformer, "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16", RunnerKind.diarization, "sortformer-4spk-v2.1.json"),
            (Self.deepFilter, "mlx-community--DeepFilterNet-mlx", RunnerKind.enhance, "deepfilternet-3.json"),
        ] {
            let entry = try #require(CatalogBridge.entry(for: display))
            #expect(entry.equivalence == .same)
            #expect(entry.manifestFile == file)
            switch CatalogBridge.resolve(display, catalog: catalog) {
            case .runnable(let slot, let equivalence, let note):
                #expect(slot.modelEntry?.id == id)
                #expect(slot.modelEntry?.runnerKind == kind)
                #expect(equivalence == .same)
                #expect(note == nil)
            case .notRunnable(let name, let reason, _):
                Issue.record("\(name) should resolve: \(reason)")
            }
        }
    }

    @Test func theManifestsDeclareTheTaskAndNoSettings() throws {
        let sortformer = try #require(CuratedManifest.load(manifestFile: "sortformer-4spk-v2.1.json"))
        #expect(sortformer.display == Self.sortformer && sortformer.tasks == ["Identify Speakers"] && sortformer.settings.isEmpty)
        let deep = try #require(CuratedManifest.load(manifestFile: "deepfilternet-3.json"))
        #expect(deep.display == Self.deepFilter && deep.tasks == ["Clean Audio"] && deep.settings.isEmpty)
        let catalog = try loadCatalog()
        for (manifest, id) in [(sortformer, "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16"), (deep, "mlx-community--DeepFilterNet-mlx")] {
            let card = try #require(catalog.first { $0.id == id })
            #expect(manifest.resources?.ramGB == card.ramGB)
            #expect(manifest.resources?.diskGB == card.downloadSizeGB)
        }
    }

    // MARK: - tables 2 and 3: served, and the derived pool

    @MainActor
    @Test func theRowsAreServedAndOfferedOnlyWhenAModelIsClaimable() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        for (task, display, kind) in [("Identify Speakers", Self.sortformer, RunnerKind.diarization), ("Clean Audio", Self.deepFilter, RunnerKind.enhance)] {
            #expect(TaskModels.isServedByExecutor(task))
            #expect(TaskModels.runnerKind(forTask: task) == kind)
            #expect(TaskModels.seededModels(forTask: task) == [display])
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == display)
            #expect(TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable).map(\.displayName) == [display])
            // With nothing claimable the pool is empty: the row is not offered.
            #expect(TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: []).isEmpty)
        }
    }

    // MARK: - check 25 (E723 / E724)

    private func issues(_ text: String) throws -> [FlowIssue] {
        FlowValidator.checkFlow(try CatParser.parseForValidation(text))
    }

    @Test func settingsAreRefusedByNameStatically() throws {
        let pins = "\n\nmodels:\n  \(Self.sortformer) = nvidia/diar_streaming_sortformer_4spk-v2.1\n  \(Self.deepFilter) = mlx-community/DeepFilterNet-mlx\n"
        let clean = try issues("mlxflow 0.8\n1. Identify Speakers   \(Self.sortformer)\(pins)")
        #expect(!clean.contains { $0.code == "E723" }, "\(clean)")
        let a = try issues("mlxflow 0.8\n1. Identify Speakers   \(Self.sortformer); max_speakers=3\(pins)")
        #expect(a.first { $0.code == "E723" }?.message
                == "Row 1's `Identify Speakers` doesn't take settings -- the model is named on the row and v1 has nothing else to set; remove `max_speakers=`.",
                "\(a)")
        let b = try issues("mlxflow 0.8\n1. Clean Audio   \(Self.deepFilter); strength=high\(pins)")
        #expect(b.first { $0.code == "E724" }?.message
                == "Row 1's `Clean Audio` doesn't take settings -- the model is named on the row and v1 has nothing else to set; remove `strength=`.",
                "\(b)")
    }

    @Test func theNoSettingsParseNamesTheFirstItemGiven() {
        #expect(AudioRowSettings.firstGiven(nil) == nil)
        #expect(AudioRowSettings.firstGiven("") == nil)
        #expect(AudioRowSettings.firstGiven("loud") == "`loud`")
        #expect(AudioRowSettings.firstGiven("threshold=0.4") == "`threshold=`")
        #expect(throws: AudioRowSettingsError.self) { try AudioRowSettings.parseNone("x=1") }
        #expect((try? AudioRowSettings.parseNone(nil)) != nil)
    }

    // MARK: - table shaping

    @Test func theTableIsSortedByStartWithTheStagesLabels() throws {
        let csv = "speaker,start,end\n2,5.000,7.250\n1,0.500,4.000\n1,8.000,9.000\n"
        let rows = try IdentifySpeakersTable.rows(fromCSV: csv)
        #expect(rows.count == 3)
        #expect(rows.map { $0[0] as? String } == ["Speaker 1", "Speaker 2", "Speaker 1"])
        #expect(rows.map { $0[1] as? Double } == [0.5, 5.0, 8.0])
        #expect(rows.map { $0[2] as? Double } == [4.0, 7.25, 9.0])
        #expect(IdentifySpeakersTable.columns == ["speaker", "start", "end"])
        #expect(throws: (any Error).self) { _ = try IdentifySpeakersTable.rows(fromCSV: "nonsense") }
        #expect(try IdentifySpeakersTable.rows(fromCSV: "speaker,start,end\n").isEmpty)
    }

    // MARK: - execution

    private func executor(_ dir: URL, stage: @escaping @Sendable () -> any PipelineStage) throws -> RealExecutor {
        let catalog = try loadCatalog()
        let ids = Set(catalog.filter { [Self.sortformer, Self.deepFilter].contains($0.displayName) }.map(\.id))
        return RealExecutor(workspace: FlowWorkspace(root: dir.appendingPathComponent("flows")), flowID: "er10",
                            blobDirectory: dir.appendingPathComponent("blobs"),
                            makeModelStage: { _, _ in stage() }, installedModelIDs: ids, catalog: catalog)
    }

    @Test func identifySpeakersWritesATableFromTheStagesCSV() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let csv = "speaker,start,end\n1,0.500,4.000\n2,4.000,9.500\n"
        let ex = try executor(dir) { CSVStub(csv: csv) }
        let out = try await ex.execute(path: "2", row: Row(task: "Identify Speakers", model: Self.sortformer, settings: nil),
                                       inputs: [try wav(in: dir)])
        let item = try #require(out.items.first)
        #expect(item.kind == .table)
        let table = try TableTool.readTable(from: try #require(item.path))
        #expect(table.columns == ["speaker", "start", "end"])
        #expect(table.rows.count == 2)
        #expect(table.rows[1][0] as? String == "Speaker 2")
    }

    @Test func identifySpeakersRefusesASettingAndANonAudioInput() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ex = try executor(dir) { CSVStub(csv: "speaker,start,end\n") }
        do {
            _ = try await ex.execute(path: "2", row: Row(task: "Identify Speakers", model: Self.sortformer, settings: "threshold=0.4"),
                                     inputs: [try wav(in: dir)])
            Issue.record("a setting should be refused")
        } catch {
            #expect("\(error)".contains("`threshold=`"), "\(error)")
        }
        await #expect(throws: (any Error).self) {
            _ = try await ex.execute(path: "2", row: Row(task: "Identify Speakers", model: Self.sortformer, settings: nil),
                                     inputs: [Asset(items: [Item(kind: .text, value: "hi", path: nil, sourceText: nil)])])
        }
    }

    @Test func cleanAudioReturnsTheStagesAudioAtItsNativeRate() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ex = try executor(dir) { CleanStub() }
        let out = try await ex.execute(path: "2", row: Row(task: "Clean Audio", model: Self.deepFilter, settings: nil),
                                       inputs: [try wav(in: dir)])
        let item = try #require(out.items.first)
        #expect(item.kind == .audio)
        let data = try Data(contentsOf: try #require(item.path))
        // A RIFF/WAVE file whose sample-rate field (bytes 24...27, little-endian) is 48 000.
        #expect(data.count > 44 && String(decoding: data.prefix(4), as: UTF8.self) == "RIFF")
        let rate = data[24..<28].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
        #expect(rate == 48000)
        await #expect(throws: (any Error).self) {
            _ = try await ex.execute(path: "2", row: Row(task: "Clean Audio", model: Self.deepFilter, settings: "strength=2"), inputs: [try wav(in: dir)])
        }
    }

    @Test func theMockMatchesPython() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let mock = MockExecutor(blobDirectory: dir)
        let table = try await mock.execute(path: "1", row: Row(task: "Identify Speakers", model: Self.sortformer, settings: nil),
                                           inputs: [try wav(in: dir)], transcript: nil, context: nil, usedFlowContent: nil)
        let read = try TableTool.readTable(from: try #require(table.items.first?.path))
        #expect(read.columns == ["speaker", "start", "end"] && read.rows.count == 4)
        let input = try wav(in: dir)
        let cleaned = try await mock.execute(path: "2", row: Row(task: "Clean Audio", model: Self.deepFilter, settings: nil),
                                             inputs: [input], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(cleaned.items.first?.path == input.items.first?.path)
        await #expect(throws: (any Error).self) {
            _ = try await mock.execute(path: "1", row: Row(task: "Clean Audio", model: Self.deepFilter, settings: "x=1"),
                                       inputs: [input], transcript: nil, context: nil, usedFlowContent: nil)
        }
    }
}

private nonisolated struct CSVStub: PipelineStage {
    let csv: String
    let id = "stub.csv", name = "Stub"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .audio)  // as the real `SortformerStage` does
        return .text(csv)
    }
}

private nonisolated struct CleanStub: PipelineStage {
    let id = "stub.clean", name = "Stub"
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .audio }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        .audio(AudioBuffer(samples: [Float](repeating: 0.1, count: 480), sampleRate: 48000))
    }
}
