import Testing
import Foundation
@testable import MLXUI

/// FILE-1 (`RSI/DelegateDeciderBacklog.md`) — the row Properties file chooser. The chooser
/// copies a pick into the flow folder (R11-0b, so a zipped flow still runs). Once the panel
/// can open *at* the flow folder, a pick can land on a folder already inside it — and the
/// per-item `removeItem` + `copyItem` loop would delete each file and then copy the file it
/// just deleted. The guard: a pick already inside the flow folder is recorded, never copied.
struct CatFlowFileChooserTests {

    private func tempFlowDir() throws -> (root: URL, flowDir: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("file1-\(UUID().uuidString)")
        let flowDir = root.appendingPathComponent("test-flow", isDirectory: true)
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        return (root, flowDir)
    }

    private func seed(_ dir: URL, files: [String: String]) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (name, body) in files {
            try body.write(to: dir.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
    }

    // MARK: - the guard: a pick already inside the flow folder

    @Test func pickingAFolderAlreadyInsideTheFlowFolderCopiesNothing() throws {
        let (_, flowDir) = try tempFlowDir()
        let receipts = flowDir.appendingPathComponent("receipts", isDirectory: true)
        let contents = ["jan.png": "PNG-JAN", "feb.png": "PNG-FEB", "mar.png": "PNG-MAR"]
        try seed(receipts, files: contents)

        let stored = try FlowRowInspectorView.copyIn(receipts, toFlowDir: flowDir)

        #expect(stored == "receipts/", "a directory keeps its trailing slash")
        // Every file still present and byte-identical — nothing was deleted then re-copied.
        for (name, body) in contents {
            let data = try String(contentsOf: receipts.appendingPathComponent(name), encoding: .utf8)
            #expect(data == body, "\(name) must be untouched")
        }
        // And no `receipts/receipts` duplication.
        #expect(!FileManager.default.fileExists(atPath: receipts.appendingPathComponent("receipts").path))
    }

    @Test func pickingAFileAlreadyInsideStoresItsRelativePath() throws {
        let (_, flowDir) = try tempFlowDir()
        let nested = flowDir.appendingPathComponent("receipts", isDirectory: true)
        try seed(nested, files: ["scan1.png": "ONE"])
        let file = nested.appendingPathComponent("scan1.png")

        let stored = try FlowRowInspectorView.copyIn(file, toFlowDir: flowDir)

        #expect(stored == "receipts/scan1.png")   // no trailing slash for a file
        #expect(try String(contentsOf: file, encoding: .utf8) == "ONE")
    }

    // MARK: - picking from outside still copies in, as before

    @Test func pickingAFolderFromOutsideStillCopiesIn() throws {
        let (root, flowDir) = try tempFlowDir()
        let outside = root.appendingPathComponent("Desktop-photos", isDirectory: true)
        try seed(outside, files: ["a.png": "A", "b.png": "B"])

        let stored = try FlowRowInspectorView.copyIn(outside, toFlowDir: flowDir)

        #expect(stored == "Desktop-photos")
        let copied = flowDir.appendingPathComponent("Desktop-photos")
        #expect(try String(contentsOf: copied.appendingPathComponent("a.png"), encoding: .utf8) == "A")
        #expect(try String(contentsOf: copied.appendingPathComponent("b.png"), encoding: .utf8) == "B")
    }

    // MARK: - a nested relative path round-trips through the security boundary

    @Test func aNestedSubfolderRoundTripsThroughFlowWorkspaceResolve() throws {
        let (root, flowDir) = try tempFlowDir()
        let jan = flowDir.appendingPathComponent("receipts/jan", isDirectory: true)
        try seed(jan, files: ["r1.png": "R1"])

        let stored = try FlowRowInspectorView.copyIn(jan, toFlowDir: flowDir)
        #expect(stored == "receipts/jan/")

        let ws = FlowWorkspace(root: root)
        let resolved = try ws.resolve(stored, flowID: "test-flow")
        #expect(resolved.resolvingSymlinksInPath().path == jan.resolvingSymlinksInPath().path)
    }

    // MARK: - the stored token re-parses as a path, not a model

    @Test func theStoredNestedPathReParsesAsAPathNotAModel() throws {
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Read Images   receipts/jan/
        """)
        #expect(doc.rows[0].task == "Read Images")
        #expect(doc.rows[0].settings == "receipts/jan/")
        #expect(doc.rows[0].model == nil, "the trailing slash keeps it off the HF-repo-id branch")

        // Survives a serialize → re-parse.
        let text = CatSerializer.serialize(doc)
        let again = try CatParser.parse(text)
        #expect(again.rows[0].settings == "receipts/jan/")
        #expect(again.rows[0].model == nil)
    }

    // MARK: - relativePath unit

    @Test func relativePathIsNilForSomethingOutside() throws {
        let (root, flowDir) = try tempFlowDir()
        let outside = root.appendingPathComponent("elsewhere", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        #expect(FlowRowInspectorView.relativePath(of: outside, under: flowDir) == nil)
        #expect(FlowRowInspectorView.relativePath(of: flowDir, under: flowDir) == nil, "the folder itself is not 'inside'")
    }
}
