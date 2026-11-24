import Foundation
import Testing
@testable import MLXUI

/// PF-9 (SPEC-Q251, owner ruling 2026-11-18) — the `Redact` row in AI Workflows: the four tables, the curated manifest,
/// and `RealExecutor.runModel`'s `.redact` branch with its `only=` subset and its refusals. No weights are loaded; the
/// stage is a stub, and every text here is synthetic (decision 10).
@MainActor
struct PrivacyFilterWorkflowWiringTests {

    private static let display = "Privacy Filter"
    private static let hfID = "openai/privacy-filter"
    private static let cardID = "openai--privacy-filter"
    private static let manifestFile = "openai-privacy-filter.json"
    private static let synthetic = "Alex Sample wrote to alex.sample@example.com and left a phone number."

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    private func registry() -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return registry
    }

    private func claimableIDs(_ catalog: [ModelEntry]) -> Set<String> {
        let registry = registry()
        return Set(catalog.filter { registry.bestModule(for: $0) != nil }.map(\.id))
    }

    /// What the stub stage builder saw.
    private final class Seen: @unchecked Sendable {
        private let lock = NSLock()
        private var configs: [StageConfig] = []
        func note(_ config: StageConfig) { lock.lock(); configs.append(config); lock.unlock() }
        var observed: [StageConfig] { lock.lock(); defer { lock.unlock() }; return configs }
        var last: StageConfig? { observed.last }
    }

    private nonisolated struct StubRedactStage: PipelineStage {
        let id = "privacy-filter.stub"
        let name = "Privacy Filter"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        let seen: Seen
        let text: String
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            guard case .text = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
            progress(1)
            return .text(text)
        }
    }

    private func input(_ text: String = PrivacyFilterWorkflowWiringTests.synthetic) -> Asset {
        Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }

    private func executor(catalog: [ModelEntry], installed: Set<String>,
                          blob: URL? = nil, seen: Seen) throws -> RealExecutor {
        let directory = blob ?? FileManager.default.temporaryDirectory
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return RealExecutor(
            workspace: FlowWorkspace(root: directory),
            flowID: "pf-redact",
            blobDirectory: directory.appendingPathComponent("blobs"),
            makeModelStage: { _, config in
                seen.note(config)
                return StubRedactStage(seen: seen, text: "<PRIVATE_PERSON> wrote to <PRIVATE_EMAIL> and left a phone number.")
            },
            installedModelIDs: installed,
            catalog: catalog)
    }

    // MARK: - Table 1: the task

    @Test func table1_theRowIsAModelTaskInTheLanguageFamily() throws {
        let desc = try #require(TaskCatalog.get("Redact"))
        #expect(desc.accepts == .single(.text))
        #expect(desc.gives == .single(.text))
        #expect(desc.taskClass == .model)
        #expect(desc.refKind == .engine)
        #expect(desc.refName == "engines.redact.redact")
        #expect(TaskFamilies.family(of: "Redact") == "Language models")
        #expect(CatParser.taskNames.contains("Redact"), "the parser's single-word list must know it")
        #expect(TaskCatalog.allTasks().count == 121)   // main added 120 before Redact (ER-7..ER-11); +1
    }

    @Test func table1_theRedactTaskIsAvailableAndDefaultsToThisModel() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let desc = try #require(TaskCatalog.get("Redact"))
        #expect(TaskAvailability.state(for: desc, catalog: catalog, claimableModelIDs: claimable) == .available)
        #expect(TaskModels.isServedByExecutor("Redact"))
        #expect(TaskModels.runnerKind(forTask: "Redact") == .redact)
        #expect(TaskModels.seededModels(forTask: "Redact") == [Self.display])
        #expect(TaskModels.defaultModel(forTask: "Redact", catalog: catalog, claimableModelIDs: claimable)
                == Self.display)
        let derived = TaskModels.derivedModels(for: "Redact", catalog: catalog, claimableModelIDs: claimable)
        #expect(derived.map(\.id) == [entry.id])
        #expect(TaskModels.displayName(for: entry) == Self.display)
    }

    // MARK: - Table 2/3: the pool

    @Test func table2_theModelIsOfferedForRedactOnly() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        #expect(entry.runnerKind == .redact)
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
        for task in TaskCatalog.allTasks().map(\.name) where task != "Redact" {
            let derived = TaskModels.derivedModels(for: task, catalog: catalog, claimableModelIDs: claimable)
            #expect(!derived.contains { $0.modelEntry?.id == entry.id },
                    "\(task) must not offer Privacy Filter")
        }
    }

    @Test func table3_thePoolAppendMovesNoExistingSeed() throws {
        let catalog = try loadCatalog()
        let claimable = claimableIDs(catalog)
        #expect(TaskModels.models(forTask: "Redact") == [Self.display])
        // The LLM-keyed rows still seed Ministral 3B; the decision tasks still seed Laya; the translator is unchanged.
        for task in ["Generate", "Summarize", "Answer", "Rewrite", "Draft", "Ask", "Title", "Critique",
                     "Verify", "Revise", "Merge", "Text to Table"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B",
                    "\(task)'s seed moved")
            #expect(!TaskModels.models(forTask: task).contains(Self.display), "\(task)'s pool names Privacy Filter")
        }
        for task in ["Classify", "Gate", "Score"] {
            #expect(TaskModels.defaultModel(forTask: task, catalog: catalog, claimableModelIDs: claimable) == "Laya 0.4B")
        }
        #expect(TaskModels.defaultModel(forTask: "Translate", catalog: catalog, claimableModelIDs: claimable) == "Ministral 3B")
    }

    // MARK: - Table 4: the bridge

    @Test func table4_theBridgeResolvesTheNameTheCatWrites() throws {
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
            #expect(slot.modelEntry?.runnerKind == .redact)
            #expect(equivalence == .same)
            #expect(note == nil)
        case .notRunnable(let display, let reason, _):
            Issue.record("\(display) should resolve: \(reason)")
        }
        #expect(CatalogBridge.entry(forCandidate: Self.hfID, modelType: .redact)?.display == Self.display)
        #expect(CatalogBridge.entries.filter { $0.candidates.contains(Self.hfID) }.count == 1)
    }

    // MARK: - The curated manifest

    @Test func theManifestLoadsFromTheAppBundle() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        #expect(manifest.id == Self.hfID)
        #expect(manifest.display == Self.display)
        #expect(manifest.kind == "local")
        #expect(manifest.engine == "mlx-redact")
        #expect(manifest.tasks == ["Redact"])
        #expect(manifest.settings.keys.sorted() == ["only"])
        #expect(manifest.settings["only"]?.defaultValue?.settingString
                == "person,email,phone,url,address,date,account,secret")
        #expect(manifest.settings["only"]?.mapsTo == "only")
        // Licence, pinned revision and the "MLXUI-authored" note live in the file; read it as the bundle shipped it.
        let raw = try String(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models/\(Self.manifestFile)"),
                             encoding: .utf8)
        #expect(raw.contains("\"license\": \"Apache-2.0\""))
        #expect(raw.contains("7ffa9a043d54d1be65afb281eddf0ffbe629385b"), "the manifest names the pinned revision")
        #expect(raw.contains("SPEC-Q251"))
    }

    @Test func manifestResourcesMatchTheCatalogEntry() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: Self.manifestFile))
        let card = try #require(try loadCatalog().first { $0.hfModelId == Self.hfID })
        #expect(manifest.resources?.ramGB == card.ramGB)
        #expect(manifest.resources?.diskGB == card.downloadSizeGB)
    }

    // MARK: - The executor branch

    @Test func theRowRunsTheRegistryStageAndPersistsTheRedactedText() async throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let seen = Seen()
        let exec = try executor(catalog: catalog, installed: [entry.id], seen: seen)
        let row = Row(task: "Redact", model: Self.display, settings: nil)
        let output = try await exec.execute(path: "2", row: row, inputs: [input()],
                                            transcript: nil, context: nil, usedFlowContent: nil)
        let text = try #require(output.items.first?.value)
        #expect(text == "<PRIVATE_PERSON> wrote to <PRIVATE_EMAIL> and left a phone number.")
        // No settings: the manifest's default is all eight, carried on the stage config.
        #expect(seen.last?.redactOnly == Set(PrivacyFilterCategory.allCases))
    }

    @Test func onlyAcceptsACommaSeparatedSubset() async throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        for (settings, expected) in [
            ("only=\"email,phone\"", Set([PrivacyFilterCategory.email, .phone])),
            ("only=person", Set([PrivacyFilterCategory.person])),
            ("only=\"email, phone , secret\"", Set([PrivacyFilterCategory.email, .phone, .secret])),
        ] {
            let seen = Seen()
            let exec = try executor(catalog: catalog, installed: [entry.id], seen: seen)
            _ = try await exec.execute(path: "3", row: Row(task: "Redact", model: Self.display, settings: settings),
                                       inputs: [input()], transcript: nil, context: nil, usedFlowContent: nil)
            #expect(seen.last?.redactOnly == expected, "\(settings)")
        }
    }

    @Test func aBadSettingFailsTheRowByItsOwnName() async throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        for (settings, offending) in [("foo=1", "foo"), ("only=\"ssn\"", "ssn")] {
            let seen = Seen()
            let exec = try executor(catalog: catalog, installed: [entry.id], seen: seen)
            do {
                _ = try await exec.execute(path: "4", row: Row(task: "Redact", model: Self.display, settings: settings),
                                           inputs: [input()], transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("\(settings) must fail the row")
            } catch {
                let sentence = String(describing: error)
                #expect(sentence.contains(offending), "the sentence names the offending name: \(sentence)")
                #expect(sentence.contains("Redact"), "the sentence names the row: \(sentence)")
                #expect(sentence.contains(Self.display) || sentence.contains("category"),
                        "the sentence says what the row does take: \(sentence)")
            }
            #expect(seen.observed.isEmpty, "a bad setting must fail before any model stage is built (\(settings))")
        }
    }

    @Test func aRedactRowNeedsExactlyOneTextInput() async throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let seen = Seen()
        let exec = try executor(catalog: catalog, installed: [entry.id], seen: seen)
        let row = Row(task: "Redact", model: Self.display, settings: nil)
        for inputs in [[], [input(), input()]] as [[Asset]] {
            do {
                _ = try await exec.execute(path: "5", row: row, inputs: inputs,
                                           transcript: nil, context: nil, usedFlowContent: nil)
                Issue.record("\(inputs.count) inputs must be refused")
            } catch let error as FlowError {
                guard case .badInputCardinality = error else {
                    Issue.record("expected badInputCardinality, got \(error)")
                    continue
                }
            }
        }
        #expect(seen.observed.isEmpty, "a cardinality failure must not build a model stage")
    }

    /// `Read Text` → `Redact Privacy Filter` → `Save Text`, run row by row through the real executor with a stub stage:
    /// the file the flow writes is the stage's redacted text.
    @Test func theFlowWritesTheRedactedText() async throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("pf-flow-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = FlowWorkspace(root: root)
        let inputURL = try workspace.resolve("input.txt", flowID: "pf-redact")
        try FileManager.default.createDirectory(at: inputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.synthetic.write(to: inputURL, atomically: true, encoding: .utf8)

        let seen = Seen()
        let exec = RealExecutor(
            workspace: workspace, flowID: "pf-redact",
            blobDirectory: root.appendingPathComponent("blobs"),
            makeModelStage: { _, config in
                seen.note(config)
                return StubRedactStage(seen: seen, text: "<PRIVATE_PERSON> wrote to <PRIVATE_EMAIL> and left a phone number.")
            },
            installedModelIDs: [entry.id], catalog: catalog)

        let read = try await exec.execute(path: "1", row: Row(task: "Read Text", model: nil, settings: "input.txt"),
                                          inputs: [], transcript: nil, context: nil, usedFlowContent: nil)
        let redacted = try await exec.execute(path: "2", row: Row(task: "Redact", model: Self.display, settings: nil),
                                              inputs: [read], transcript: nil, context: nil, usedFlowContent: nil)
        _ = try await exec.execute(path: "3", row: Row(task: "Save Text", model: nil, settings: "out.txt"),
                                   inputs: [redacted], transcript: nil, context: nil, usedFlowContent: nil)
        let outURL = try workspace.resolve("out.txt", flowID: "pf-redact")
        let written = try String(contentsOf: outURL, encoding: .utf8)
        #expect(written == "<PRIVATE_PERSON> wrote to <PRIVATE_EMAIL> and left a phone number.")
        #expect(seen.last?.redactOnly == Set(PrivacyFilterCategory.allCases))
    }

    // MARK: - The run screen's stage claims the model (Trap 10)

    @Test func theSDKClaimsByFamilyAndKindOnly() throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        #expect(PrivacyFilterSDK().claim(entry) == .exact)
        #expect(registry().bestModule(for: entry)?.descriptor.id == "privacy-filter")
        // Trap 10: the claim is the family AND the `.redact` kind AND `source == .mlx` — never the architecture string.
        // An `.llm` card of the same source is left to chat.
        let llm = try #require(catalog.first { $0.modelType == .llm && $0.source == .mlx })
        #expect(PrivacyFilterSDK().claim(llm) == .no)
        // A `.redact`-kind card of another family is not claimed either: build one from the entry by changing the
        // family through the same memberwise initialiser.
        let lookalike = ModelEntry(
            id: "other--redactor", family: "redact-lookalike", displayName: "Redact Lookalike",
            paramSize: entry.paramSize, paramCountB: entry.paramCountB, modelType: .redact,
            source: .mlx, format: entry.format, platforms: entry.platforms,
            minMacOSVersion: entry.minMacOSVersion, hfRepo: "other", hfModelId: "other/redactor",
            ramGB: entry.ramGB, downloadSizeGB: entry.downloadSizeGB, contextWindow: entry.contextWindow,
            license: entry.license, licenseUrl: entry.licenseUrl, description: nil, summary: nil,
            descriptionSource: nil, architecture: entry.architecture, languages: nil, lastUpdated: nil,
            taskTags: nil, benchmarks: nil, speedTokensPerSec: nil, speedHardware: nil,
            speedEstimated: nil, communityDownloads: nil, communityLikes: nil, variants: nil)
        #expect(PrivacyFilterSDK().claim(lookalike) == .no, "family, not architecture, is the claim")
    }
}
