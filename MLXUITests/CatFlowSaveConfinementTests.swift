import Testing
import Foundation
@testable import MLXUI

/// ER-FIX-5 (SPEC-Q253; owner ruling F2, Boo, 2026-11-22): a literal `Save File` `to=` that cannot stay inside the flow's folder is
/// refused at check time, once, in the Python's words (synced golden `erfix5_refusals.json`); a tool refusal reads "Row N failed:" once.
struct CatFlowSaveConfinementTests {
    private struct RefusalCase: Decodable {
        struct Issue: Decodable { var row: String; var code: String; var message: String }
        var name: String
        var flow: String
        var issues: [Issue]
    }

    private var fixtures: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Fixtures/CatFlow")
    }

    private func issues(_ cat: String) throws -> [FlowIssue] {
        FlowValidator.checkFlow(try CatParser.parseForValidation(cat))
    }

    @Test func everyCheckTimeCaseMatchesThePythonGolden() throws {
        let data = try Data(contentsOf: fixtures.appendingPathComponent("goldens/check/erfix5_refusals.json"))
        let cases = try JSONDecoder().decode([String: [RefusalCase]].self, from: data)["cases"] ?? []
        #expect(cases.count == 5)
        for c in cases {
            let actual = try issues(c.flow).map { RefusalCase.Issue(row: $0.row, code: $0.code, message: $0.message) }
            #expect(actual.map(\.code) == c.issues.map(\.code), "\(c.name)")
            #expect(actual.map(\.message) == c.issues.map(\.message), "\(c.name)")
            #expect(actual.map(\.row) == c.issues.map(\.row), "\(c.name)")
            #expect(actual.allSatisfy { $0.message.hasPrefix("Row 3's `Save File` ") && !$0.message.contains("failed") }, "\(c.name)")
        }
    }

    @Test func theConformanceCaseRefusesTheSameWay() throws {
        let cat = try String(contentsOf: fixtures.appendingPathComponent("conformance/erfix5_save_file_escape_refused.cat"), encoding: .utf8)
        let found = try issues(cat)
        #expect(found.map(\.code) == ["E717"])
        #expect(found.first?.message.contains("climbs out with `..`") == true)
    }

    @Test func aPlaceholderAndAnInsidePathPassCheck() throws {
        #expect(try issues("mlxflow 0.8\n1. Read Text  n.txt\n2. Make PDF\n3. Save File  to=\"out/{name}.pdf\"\n").isEmpty)
        #expect(try issues("mlxflow 0.8\n1. Read Text  n.txt\n2. Make PDF\n3. Save File  to=\"out/report.pdf\"\n").isEmpty)
    }

    @Test func aToolRefusalReadsRowNFailedOnlyOnce() {
        let inner = FlowError.stageFailure(row: "Save File", message: "Save File can't write to `x` -- outside").description
        #expect(inner.hasPrefix("Row Save File failed: "))
        #expect(FlowRunner.unprefixed(inner) == "Save File can't write to `x` -- outside")
        let shown = FlowError.stageFailure(row: "3", message: FlowRunner.unprefixed(inner) ?? "").description
        #expect(shown == "Row 3 failed: Save File can't write to `x` -- outside")
        // a message that is not a task-name prefix is left alone
        #expect(FlowRunner.unprefixed("Row 2 failed: boom") == "Row 2 failed: boom")
        #expect(FlowRunner.unprefixed("plain") == "plain")
        #expect(FlowRunner.unprefixed(nil) == nil)
    }

    @Test func aSymlinkOutOfTheFlowFolderIsRefusedByRowNameAtRunTime() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("catflow-erfix5-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "f")
        let outside = base.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("link"), withDestinationURL: outside)
        let src = dir.appendingPathComponent("b.bin")
        try Data([1]).write(to: src)
        let input = Asset(items: [Item(kind: .file, value: nil, path: src, sourceText: nil)])
        do {
            _ = try await SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"link/out.bin\"").run(input) { _ in }
            Issue.record("expected a confinement refusal")
        } catch let e as FlowError {
            guard case .stageFailure(_, let message) = e else { Issue.record("wrong case: \(e)"); return }
            #expect(message.hasPrefix("Save File can't write to `link/out.bin`") && !message.contains("Row "), "\(message)")
        }
        #expect(!FileManager.default.fileExists(atPath: outside.appendingPathComponent("out.bin").path))
    }

    /// ER-CLOSE-2: Save Image and Save File refuse an escaping destination through the ONE `SaveCopy` check: the same error type,
    /// each message naming its own row first, nothing written.
    @Test func saveImageAndSaveFileShareOneConfinementRefusal() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("catflow-erclose2-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "f")
        let outside = base.appendingPathComponent("outside")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(at: dir.appendingPathComponent("link"), withDestinationURL: outside)
        let src = dir.appendingPathComponent("b.bin")
        try Data([1]).write(to: src)
        func refusal(_ stage: any AssetStage, kind: Kind) async -> (String, String)? {
            let input = Asset(items: [Item(kind: kind, value: nil, path: src, sourceText: nil)])
            do { _ = try await stage.run(input) { _ in } } catch let e as FlowError {
                if case .stageFailure(let row, let message) = e { return (row, message) }
            } catch {}
            return nil
        }
        let file = await refusal(SaveFileTool(workspace: ws, flowID: "f", settings: "to=\"link/f.bin\""), kind: .file)
        let image = await refusal(SaveImageTool(workspace: ws, flowID: "f", settings: "path=\"link/i.bin\""), kind: .image)
        #expect(file?.0 == "Save File" && image?.0 == "Save Image")
        #expect(file?.1.hasPrefix("Save File can't write to `link/f.bin`") == true, "\(String(describing: file))")
        #expect(image?.1.hasPrefix("Save Image can't write to `link/i.bin`") == true, "\(String(describing: image))")
        #expect((try FileManager.default.contentsOfDirectory(atPath: outside.path)).isEmpty)
    }
}
