import Foundation
import Testing
@testable import MLXUI

/// FA-6 for issue #13 (decision 8, SPEC-Q261): `Align` in AI Workflows — the four tables (catalog entry, kinds + served
/// prefix, the derived pool, the bridge), the MLXUI-authored curated manifest, the row grammar `(1,2)`, the `lang=`
/// setting (refused by name when wrong), the two-input contract, and the `text,start,end` table the executor writes.
/// No weights: `makeModelStage` is injected (the `TranslateGemmaWorkflowWiringTests` shape).
private nonisolated final class AlignWiringSeenBuild: @unchecked Sendable {
    private let lock = NSLock()
    private var config: StageConfig?
    private var calls = 0
    private var text: String?
    func noteRun(_ transcript: String?) { lock.lock(); text = transcript; lock.unlock() }
    /// `AlignTranscript.current` as the stage saw it when it ran.
    var transcriptAtRun: String? { lock.lock(); defer { lock.unlock() }; return text }
    func note(_ config: StageConfig) { lock.lock(); self.config = config; calls += 1; lock.unlock() }
    var observed: StageConfig? { lock.lock(); defer { lock.unlock() }; return config }
    var callCount: Int { lock.lock(); defer { lock.unlock() }; return calls }
}

nonisolated struct ForcedAlignerWorkflowWiringTests {

    private static let display = "Qwen3 Aligner 0.6B"
    private static let hfID = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    private static let cardID = "mlx-community--Qwen3-ForcedAligner-0.6B-8bit"
    private static let manifestFile = "qwen3-forcedaligner-0.6b-8bit.json"
    private static let csv = "text,start,end\nHello,0.100,0.500\nworld,0.500,1.000\n"

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

    // MARK: - The four tables

    @Test func table1theRowIsACatalogModelTaskServedByTheASRPrefix() throws {
        let desc = try #require(TaskCatalog.get("Align"))
        #expect(desc.taskClass == .model)
        #expect(desc.refKind == .engine)
        #expect(desc.refName == "engines.asr.align")
        #expect(TaskModels.runnerKind(forTask: "Align") == .align)
        #expect(TaskModels.isServedByExecutor("Align"), "engines.asr. is already on the served allow-list")
        #expect(TaskFamilies.family(of: "Align") == "Audio, image, video")
    }

    @MainActor
    @Test func table2theAlignerIsTheAlignDefaultAndIsInNoOtherPool() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        let card = try #require(catalog.first { $0.id == Self.cardID })
        #expect(card.runnerKind == .align)
        #expect(claimable.contains(Self.cardID), "the model is claimable, so it derives")
        let alignPool = TaskModels.derivedModels(for: "Align", catalog: catalog, claimableModelIDs: claimable)
        #expect(alignPool.contains { $0.modelEntry?.id == Self.cardID })
        #expect(TaskModels.defaultModel(forTask: "Align", catalog: catalog, claimableModelIDs: claimable) == Self.display)
        for task in TaskCatalog.allTasks().map(\.name) where task != "Align" {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(!derived.contains { $0.modelEntry?.id == Self.cardID }, "\(task) must not offer the aligner")
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) != Self.display,
                    "\(task)'s seed must not move")
        }
    }

    @Test func table3thePoolHasOnlyTheAlignerAndChangesNoOtherPool() throws {
        #expect(TaskModels.models(forTask: "Align") == [Self.display])
        #expect(!TaskModels.models(forTask: "Transcribe").contains(Self.display),
                "the aligner is in no Transcribe pool")
        #expect(TaskModels.runnerKind(forTask: "Transcribe") == .asr)
    }

    @Test func table4theBridgeResolvesTheNameTheCatWrites() throws {
        let catalog = try loadCatalog()
        let entry = try #require(CatalogBridge.entry(for: Self.display))
        #expect(entry.pinnedID == Self.hfID)
        #expect(entry.candidates == [Self.hfID])
        #expect(entry.equivalence == .same)
        #expect(entry.manifestFile == Self.manifestFile)
        #expect(entry.modelType == nil)
        switch CatalogBridge.resolve(Self.display, catalog: catalog) {
        case .runnable(let slot, let equivalence, let note):
            #expect(slot.modelEntry?.hfModelId == Self.hfID)
            #expect(slot.modelEntry?.runnerKind == .align)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
        #expect(CatalogBridge.entry(forCandidate: Self.hfID, modelType: .align)?.display == Self.display)
        let card = try #require(catalog.first { $0.hfModelId == Self.hfID })
        #expect(TaskModels.displayName(for: card) == Self.display)
    }

    // MARK: - The manifest

    @Test func theManifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-align")
        #expect(manifest.tasks == ["Align"])
        #expect(manifest.settings.keys.sorted() == ["lang"])
        #expect(manifest.settings["lang"]?.type == "enum")
        #expect(manifest.settings["lang"]?.values == ForcedAlignerEngine.languageCodes)
        #expect(manifest.settings["lang"]?.defaultValue == .string("en"))
        let raw = try String(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/\(Self.manifestFile)"),
                             encoding: .utf8)
        #expect(raw.contains("\"license\": \"Apache-2.0\""))
        #expect(raw.contains("0e1a68e91d815300c7c9754b2a7639378b23db15"), "the manifest names the pinned revision")
    }

    @Test func manifestResourcesMatchTheCatalogEntry() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        let card = try #require(try loadCatalog().first { $0.hfModelId == Self.hfID })
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    // MARK: - The row grammar and its settings

    @Test func theRowParsesWithTwoInputs() throws {
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Read Audio   memo.m4a
        2. Read Text    memo.txt
        3. Align (1,2)  Qwen3 Aligner 0.6B; lang=en
        """)
        let row = try #require(doc.rows.first { $0.task == "Align" })
        #expect(row.refs.count == 2)
        #expect(row.model == "Qwen3 Aligner 0.6B")
        #expect(row.settings?.contains("lang=en") == true)
    }

    @Test func theLanguageDefaultsToEnglishAndTakesTheElevenCodes() throws {
        #expect(try AlignRowRequest.languageCode(settings: nil) == "en")
        #expect(try AlignRowRequest.languageCode(settings: "") == "en")
        for code in ForcedAlignerEngine.languageCodes {
            #expect(try AlignRowRequest.languageCode(settings: "lang=\(code)") == code)
        }
        #expect(try AlignRowRequest.languageCode(settings: "lang=ZH") == "zh")
    }

    @Test func anUnsupportedLanguageIsRefusedByName() {
        do {
            _ = try AlignRowRequest.languageCode(settings: "lang=xx", rowLabel: "Row 3")
            Issue.record("lang=xx must be refused")
        } catch {
            let sentence = String(describing: error)
            #expect(sentence.contains("Row 3"))
            #expect(sentence.contains("lang"))
            #expect(sentence.contains("xx"))
            #expect(sentence.contains("zh, en, yue, fr, de, it, ja, ko, pt, ru, es"))
        }
    }

    @Test func anyOtherSettingIsRefusedByName() {
        for settings in ["foo=1", "lang=en; foo=1", "speed=2"] {
            do {
                _ = try AlignRowRequest.languageCode(settings: settings, rowLabel: "Row 3")
                Issue.record("\(settings) must be refused")
            } catch {
                let sentence = String(describing: error)
                #expect(sentence.contains("Row 3"))
                #expect(sentence.contains("foo") || sentence.contains("speed"), "\(sentence)")
                #expect(sentence.contains("lang="), "\(sentence)")
            }
        }
    }

    // MARK: - The executor path

    private func executor(flowID: String = "f",
                          seen: AlignWiringSeenBuild = AlignWiringSeenBuild(),
                          stub: @escaping @Sendable () -> String = { ForcedAlignerWorkflowWiringTests.csv })
        throws -> RealExecutor {
        let catalog = try loadCatalog()
        let ws = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("forcedaligner-wiring-\(UUID().uuidString)"))
        return RealExecutor(workspace: ws, flowID: flowID,
                            blobDirectory: ws.root.appendingPathComponent("blobs"),
                            makeModelStage: { _, config in
                                seen.note(config)
                                return AlignWiringStubStage(csv: stub(), onRun: { seen.noteRun($0) })
                            },
                            installedModelIDs: [Self.cardID], catalog: catalog)
    }

    private func audioAsset() throws -> Asset {
        // A real WAV the executor can decode; the stub ignores its content.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fa-wiring-\(UUID().uuidString).wav")
        var buffer = [Float]()
        for i in 0..<16_000 { buffer.append(Float(sin(Double(i) / 16_000.0 * 2 * .pi * 440) * 0.2)) }
        try AudioWriter.writeWAV(AudioBuffer(samples: buffer, sampleRate: 16_000), to: url)
        return Asset(items: [Item(kind: .audio, value: nil, path: url, sourceText: nil)])
    }

    private func textAsset(_ text: String) -> Asset {
        Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }

    @Test func theRowRunsThroughTheStageAndWritesTheTable() async throws {
        let seen = AlignWiringSeenBuild()
        let exec = try executor(seen: seen)
        let out = try await exec.execute(path: "3",
                                        row: Row(task: "Align", model: Self.display, settings: "lang=zh"),
                                        inputs: [try audioAsset(), textAsset("你好世界")],
                                        transcript: nil, context: nil, usedFlowContent: nil)
        let config = try #require(seen.observed)
        #expect(seen.transcriptAtRun == "你好世界", "the row's second input is the run's transcript")
        #expect(config.language == "zh", "the row's lang= reaches the stage")
        let item = try #require(out.items.first)
        #expect(item.kind == .table, "Align produces a table asset")
        #expect(item.value == Self.csv, "the table also carries its CSV for Save Text")
        let tablePath = try #require(item.path)
        let table = try TableTool.readTable(from: tablePath)
        // Owner ruling 4 (Boo, 2026-11-27): the table has "columns exactly `text`, `start`, `end`".
        #expect(AlignmentExport.columns == ["text", "start", "end"])
        #expect(table.columns == AlignmentExport.columns)
        #expect(table.rows.count == 2)
        #expect(table.rows[0][0] as? String == "Hello")
    }

    @Test func oneInputIsRefusedNamingTheRow() async throws {
        let exec = try executor()
        do {
            _ = try await exec.execute(path: "3", row: Row(task: "Align", model: Self.display, settings: "lang=en"),
                                       inputs: [try audioAsset()], transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("Align with one input must fail")
        } catch {
            let sentence = String(describing: error)
            #expect(sentence.contains("3"))
            #expect(sentence.contains("two inputs"))
        }
    }

    @Test func theInputsMustBeAudioFirstAndTextSecond() async throws {
        let exec = try executor()
        do {
            _ = try await exec.execute(path: "3", row: Row(task: "Align", model: Self.display, settings: "lang=en"),
                                       inputs: [textAsset("hello"), try audioAsset()],
                                       transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("text first must fail")
        } catch {
            #expect(String(describing: error).contains("an audio input first"))
        }
        do {
            _ = try await exec.execute(path: "3", row: Row(task: "Align", model: Self.display, settings: "lang=en"),
                                       inputs: [try audioAsset(), try audioAsset()],
                                       transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("two audio inputs must fail")
        } catch {
            #expect(String(describing: error).contains("a text input second"))
        }
    }

    @Test func aBadSettingFailsBeforeAnyStageIsBuilt() async throws {
        let seen = AlignWiringSeenBuild()
        let exec = try executor(seen: seen)
        for settings in ["lang=xx", "foo=1"] {
            do {
                _ = try await exec.execute(path: "3", row: Row(task: "Align", model: Self.display, settings: settings),
                                           inputs: [try audioAsset(), textAsset("hello")],
                                           transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("\(settings) must fail the row")
            } catch {
                #expect(String(describing: error).contains("3"))
            }
        }
        #expect(seen.callCount == 0, "a bad setting is refused before the model stage")
    }

    /// The issue's chain: `Read Audio` → `Read Text` → `Align (1,2)` → `Save Text` writes the expected CSV.
    @Test func theFlowWritesTheExpectedCSV() async throws {
        let flowID = "align-flow"
        let exec = try executor(flowID: flowID)
        let workspace = exec.workspace
        let dir = workspace.directory(for: flowID)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workspace.root) }
        try FileManager.default.copyItem(at: repoRoot.appendingPathComponent("MLXUI/Resources/Samples/sample-audio.m4a"),
                                         to: dir.appendingPathComponent("memo.m4a"))
        try "The quick brown fox.".write(to: dir.appendingPathComponent("memo.txt"), atomically: true, encoding: .utf8)

        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Read Audio   memo.m4a
        2. Read Text    memo.txt
        3. Align (1,2)  Qwen3 Aligner 0.6B; lang=en
        4. Save Text (3)  out.csv
        """)
        let context = FlowRunner.RunContext(flowID: flowID, workspace: workspace,
                                           blobDirectory: exec.blobDirectory, executor: exec)
        var failure: String?
        for await event in FlowRunner().run(doc, context: context) {
            if case .failed(_, let error) = event { failure = "\(error)" }
        }
        if let failure { print("FAFLOW failure: \(failure)") }
        #expect(failure == nil, "\(failure ?? "")")
        let saved = try String(contentsOf: dir.appendingPathComponent("out.csv"), encoding: .utf8)
        #expect(saved == Self.csv)
    }
}

/// The stub stage `Align` rows are handed in these tests: audio in, the fixed CSV out (the executor turns it into the
/// `text,start,end` table). The real stage is `ForcedAlignerStage`, tested in `ForcedAlignerModuleTests`.
nonisolated struct AlignWiringStubStage: PipelineStage, @unchecked Sendable {
    let csv: String
    var onRun: (@Sendable (String?) -> Void)?
    var id: String { "align-stub" }
    var name: String { "Qwen3 Aligner 0.6B" }
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        guard case .audio = input else { throw StageError.kindMismatch(expected: .audio, got: input.kind) }
        onRun?(AlignTranscript.current)
        progress(1)
        return .text(csv)
    }
}
