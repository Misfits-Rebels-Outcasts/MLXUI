import Testing
import Foundation
@testable import MLXUI

/// ER-7 — `Save File` (SPEC-Q253) and `Replace` (SPEC-Q254): the Swift ports of catflow-mlx's
/// ER-1 rows. Refusal wording is pinned by the synced `er1_refusals.json` golden; the rest is
/// behaviour — byte-exact save, the shared confinement, shared-span with `Extract`, literal
/// `with=`, German and French text.
struct CatFlowEdgeRowsSaveReplaceTests {

    // MARK: - helpers

    private struct RefusalCase: Decodable {
        struct Issue: Decodable { var row: String; var code: String; var message: String }
        var name: String
        var flow: String
        var issues: [Issue]
    }

    private var goldenURL: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/goldens/check/er1_refusals.json")
    }

    private func makeWorkspace() throws -> (FlowWorkspace, URL) {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("catflow-er7-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return (FlowWorkspace(root: base.appendingPathComponent("flows")), base)
    }

    private func fileAsset(_ dir: URL, name: String, bytes: Data) throws -> Asset {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try bytes.write(to: url)
        return Asset(items: [Item(kind: .file, value: nil, path: url, sourceText: nil)])
    }

    private func text(_ s: String) -> Asset { Asset(items: [Item(kind: .text, value: s, path: nil, sourceText: nil)]) }

    private func replace(_ settings: String, _ input: String) async throws -> String {
        let out = try await ReplaceTool(settings: settings).run(text(input)) { _ in }
        return try #require(out.items.first?.value)
    }

    private func extract(_ settings: String, _ input: String) async throws -> [String] {
        let out = try await ExtractTool(settings: settings).run(text(input)) { _ in }
        return out.items.compactMap(\.value)
    }

    private func issues(_ cat: String) throws -> [FlowIssue] {
        FlowValidator.checkFlow(try CatParser.parseForValidation(cat))
    }

    // MARK: - catalog tables

    @Test func bothRowsAreInTheFourTables() throws {
        let save = try #require(TaskCatalog.catalog["Save File"])
        #expect(save.accepts == Shape.single(Kind.file) && save.gives == Shape.single(Kind.status))
        let replace = try #require(TaskCatalog.catalog["Replace"])
        #expect(replace.accepts == Shape.single(Kind.text) && replace.gives == Shape.single(Kind.text))
        #expect(CacheKey.neverCache.contains("Save File"))
        #expect(!CacheKey.neverCache.contains("Replace"))
        #expect(TaskAvailability.supportedInstantTools.contains("Save File"))
        #expect(TaskAvailability.supportedInstantTools.contains("Replace"))
        #expect(CatParser.taskNames.contains("Save File") && CatParser.taskNames.contains("Replace"))
        #expect(FlowInterpreter.namePlaceholderTasks.contains("Save File"))
    }

    // MARK: - refusals: the synced golden, case by case

    @Test func everyEr1RefusalMatchesThePythonGolden() throws {
        let cases = try JSONDecoder().decode([String: [RefusalCase]].self, from: Data(contentsOf: goldenURL))["cases"] ?? []
        #expect(cases.count == 14)
        for c in cases {
            let actual = try issues(c.flow).map { RefusalCase.Issue(row: $0.row, code: $0.code, message: $0.message) }
            #expect(actual.map(\.code) == c.issues.map(\.code), "\(c.name)")
            #expect(actual.map(\.message) == c.issues.map(\.message), "\(c.name)")
            #expect(actual.map(\.row) == c.issues.map(\.row), "\(c.name)")
        }
    }

    @Test func goodRowsPassCheck25() throws {
        #expect(try issues("mlxflow 0.8\n1. Read Text  n.txt\n2. Replace  find=\"a\"; with=\"b\"\n").isEmpty)
        #expect(try issues("mlxflow 0.8\n1. Read Text  n.txt\n2. Replace  pattern=emails\n").isEmpty)
        #expect(try issues("mlxflow 0.8\n1. Read Text  n.txt\n2. Replace  pattern=\"\\d+\"; with=\"\"\n").isEmpty)
    }

    // MARK: - Replace behaviour

    @Test func namedEmailsDefaultToTheTag() async throws {
        let out = try await replace("pattern=emails", "write to a@b.co or c@d.org")
        #expect(out == "write to <EMAIL> or <EMAIL>")
        #expect(try await replace("pattern=dates", "on 2026-11-22 and 3/4/2026") == "on <DATE> and <DATE>")
        #expect(try await replace("pattern=urls", "see https://example.com/x now") == "see <URL> now")
    }

    @Test func replaceAndExtractMarkIdenticalSpans() async throws {
        let sample = "write to a@b.co or c@d.org, or e.f@g-h.example.com today"
        let spans = try await extract("emails", sample)
        #expect(spans.count == 3)
        // Replacing each span Extract returned, by hand, equals Replace's own output.
        var expected = sample
        for span in spans { expected = expected.replacingOccurrences(of: span, with: "<EMAIL>") }
        #expect(try await replace("pattern=emails", sample) == expected)
        // And the table is the very one Extract holds.
        #expect(ReplaceSettings.namedDefaults.keys.sorted() == ExtractTool.presets.keys.sorted())
    }

    @Test func namedPatternIsCaseInsensitiveByName() async throws {
        #expect(try await replace("pattern=EMAILS", "a@b.co") == "<EMAIL>")
    }

    @Test func namedPatternTakesAnExplicitWithAndEmptyDeletes() async throws {
        #expect(try await replace("pattern=emails; with=\"[x]\"", "a@b.co") == "[x]")
        #expect(try await replace("pattern=emails; with=\"\"", "hi a@b.co!") == "hi !")
    }

    @Test func findIsLiteralCaseSensitiveAndReplacesEveryOccurrence() async throws {
        #expect(try await replace("find=\"a.b\"; with=\"X\"", "a.b axb a.b") == "X axb X")
        #expect(try await replace("find=\"Cat\"; with=\"Dog\"", "Cat cat Cat") == "Dog cat Dog")
    }

    @Test func germanAndFrenchTextReplaces() async throws {
        #expect(try await replace("find=\"Straße\"; with=\"Str.\"", "Hauptstraße, Straße 5, Strasse") == "Hauptstraße, Str. 5, Strasse")
        #expect(try await replace("find=\"Straße\"; with=\"Str.\"", "Die Straße der Müller") == "Die Str. der Müller")
        #expect(try await replace("find=\"rue\"; with=\"avenue\"", "la rue de l'été, ça va") == "la avenue de l'été, ça va")
        #expect(try await replace("find=\"é\"; with=\"e\"", "été à Zürich, déjà") == "ete à Zürich, dejà")
    }

    @Test func withIsLiteralNoGroupReferences() async throws {
        // A regex with a group, and `$1` / `\1` / a backslash in `with=` stay literal.
        #expect(try await replace("pattern=\"(a)b\"; with=\"$1\"", "ab") == "$1")
        #expect(try await replace("pattern=\"(a)b\"; with=\"\\\\1\"", "ab") == "\\1")
        #expect(try await replace("find=\"x\"; with=\"a\\\\b$0\"", "x") == "a\\b$0")
    }

    @Test func zeroMatchesReturnsTheInputUnchanged() async throws {
        #expect(try await replace("find=\"zzz\"; with=\"y\"", "nothing here") == "nothing here")
        #expect(try await replace("pattern=emails", "no address") == "no address")
    }

    @Test func applyingANamedPatternTwiceIsANoOp() async throws {
        let once = try await replace("pattern=emails", "a@b.co")
        #expect(try await replace("pattern=emails", once) == once)
    }

    @Test func replaceRefusesBadSettingsAtRunTimeNamingTheRow() async throws {
        for bad in ["", "find=\"x\"", "find=\"x\"; pattern=\"y\"; with=\"z\"", "pattern=\"(\"; with=\"z\"", "case=no; pattern=emails"] {
            do {
                _ = try await ReplaceTool(settings: bad).run(text("x")) { _ in }
                Issue.record("expected a refusal for `\(bad)`")
            } catch let e as FlowError {
                #expect(String(describing: e).contains("Replace"))
            }
        }
    }

    // MARK: - Save File behaviour

    @Test func saveFileWritesIdenticalBytesAndNamesThePath() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let bytes = Data((0..<4096).map { UInt8(truncatingIfNeeded: $0 &* 31) } + [0, 255, 10, 13])
        let input = try fileAsset(ws.directory(for: "f"), name: "blob.bin", bytes: bytes)
        let out = try await SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"out/report.pdf\"")
            .run(input) { _ in }
        #expect(out.items.first?.kind == .status)
        #expect(out.items.first?.value == "saved to out/report.pdf")
        #expect(try Data(contentsOf: ws.directory(for: "f").appendingPathComponent("out/report.pdf")) == bytes)
    }

    @Test func saveFileStatusMatchesSaveImagesWording() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let input = try fileAsset(ws.directory(for: "f"), name: "b.bin", bytes: Data([1, 2, 3]))
        let file = try await SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"a.bin\"").run(input) { _ in }
        let image = try await SaveImageTool(workspace: ws, flowID: "f", settings: "b.bin").run(input) { _ in }
        #expect(file.items.first?.value == "saved to a.bin")
        #expect(image.items.first?.value == "saved to b.bin")
    }

    @Test func saveFileOverwriteGoesThroughTheTrashLikeSaveImage() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let dir = ws.directory(for: "f")
        let first = try fileAsset(dir, name: "one.bin", bytes: Data([1]))
        let second = try fileAsset(dir, name: "two.bin", bytes: Data([2]))
        let tool = SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"out.bin\"")
        _ = try await tool.run(first) { _ in }
        _ = try await tool.run(second) { _ in }
        #expect(try Data(contentsOf: dir.appendingPathComponent("out.bin")) == Data([2]))
        let trash = (try? FileManager.default.contentsOfDirectory(atPath: dir.appendingPathComponent(".trash").path)) ?? []
        #expect(trash.contains { $0.hasPrefix("out.bin.") })
    }

    @Test func saveFileRefusesAnEscapeByRowNameAndWritesNothing() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let input = try fileAsset(ws.directory(for: "f"), name: "b.bin", bytes: Data([1]))
        for to in ["../../etc/x", "/tmp/er7-absolute-escape.bin", "a/../../b.bin"] {
            do {
                _ = try await SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"\(to)\"").run(input) { _ in }
                Issue.record("expected a confinement refusal for \(to)")
            } catch let e as FlowError {
                let s = String(describing: e)
                #expect(s.contains("Save File") && s.contains(to), "\(s)")
            }
        }
        #expect(!FileManager.default.fileExists(atPath: "/tmp/er7-absolute-escape.bin"))
    }

    @Test func saveFileRefusesBadSettingsAndAnInputWithNoFile() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let input = try fileAsset(ws.directory(for: "f"), name: "b.bin", bytes: Data([1]))
        for bad in ["", "out.pdf", "path=\"out.pdf\"", "to=\"\""] {
            await #expect(throws: FlowError.self) {
                _ = try await SaveFileTool(workspace: ws, flowID: "f", settings: bad).run(input) { _ in }
            }
        }
        await #expect(throws: FlowError.self) {
            _ = try await SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"o.bin\"").run(Asset(items: [])) { _ in }
        }
    }

    // MARK: - executors (real dispatch and mock)

    private func realExecutor(_ ws: FlowWorkspace, _ base: URL) -> RealExecutor {
        RealExecutor(workspace: ws, flowID: "f", blobDirectory: base,
                     makeModelStage: { _, _ in ER7Stub() }, installedModelIDs: [], catalog: [])
    }

    @Test func realExecutorRunsBothRows() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let ex = realExecutor(ws, base)
        let rep = try await ex.execute(path: "2", row: Row(task: "Replace", settings: "find=\"a\"; with=\"b\""),
                                       inputs: [text("banana")], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(rep.items.first?.value == "bbnbnb")
        let input = try fileAsset(ws.directory(for: "f"), name: "x.bin", bytes: Data([9, 8, 7]))
        let sv = try await ex.execute(path: "2", row: Row(task: "Save File", settings: "to=\"y.bin\""),
                                      inputs: [input], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(sv.items.first?.value == "saved to y.bin")
        #expect(try Data(contentsOf: ws.directory(for: "f").appendingPathComponent("y.bin")) == Data([9, 8, 7]))
    }

    @Test func mockExecutorFabricatesAndWritesNothing() async throws {
        let (ws, base) = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let input = try fileAsset(ws.directory(for: "f"), name: "x.bin", bytes: Data([1]))
        let sv = try await mock.execute(path: "2", row: Row(task: "Save File", settings: "to=\"never.bin\""),
                                        inputs: [input], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(sv.items.first?.kind == .status)
        #expect(!FileManager.default.fileExists(atPath: ws.directory(for: "f").appendingPathComponent("never.bin").path))
        let rep = try await mock.execute(path: "2", row: Row(task: "Replace", settings: "find=\"a\"; with=\"b\""),
                                         inputs: [text("a")], transcript: nil, context: nil, usedFlowContent: nil)
        #expect(rep.items.first?.kind == .text)
    }
}

private nonisolated struct ER7Stub: PipelineStage {
    let id = "stub.er7"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}
