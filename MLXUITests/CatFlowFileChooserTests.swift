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

/// FILE-2 (`RSI/DelegateDeciderBacklog.md`, Addendum 9) — the in-flow list, "Create a folder
/// here", and the pieces that keep a stored path honestly represented even when it isn't one
/// of the flat picks.
struct CatFlowFileListTests {

    private func tempFlowDir() throws -> (root: URL, flowDir: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("file2-\(UUID().uuidString)")
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

    // MARK: - the list excludes what it should, for both task shapes

    @Test func folderTaskListsOnlyRealSubdirectoriesNotTheCatOrDotfilesOrFiles() throws {
        let (_, flowDir) = try tempFlowDir()
        try seed(flowDir, files: ["ReceiptsToExpense.cat": "mlxflow 0.8", "notes.txt": "hi"])
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent(".blobs"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent("used"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent("receipts"), withIntermediateDirectories: true)

        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Images", flowDir: flowDir, wantsFolder: true)

        #expect(entries.map(\.token) == ["receipts/"], "the .cat, .blobs/, used/ and the plain file are all excluded")
    }

    @Test func fileTaskListsOnlyMatchingFilesNotFoldersOrTheCat() throws {
        let (_, flowDir) = try tempFlowDir()
        try seed(flowDir, files: ["DraftMemo.cat": "mlxflow 0.8", "draft.txt": "hi", "photo.png": "not text"])
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent("receipts"), withIntermediateDirectories: true)

        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Text", flowDir: flowDir, wantsFolder: false)

        #expect(entries.map(\.token) == ["draft.txt"], "the .cat, the folder, and the non-text file are all excluded")
    }

    @Test func aTaskWithNoUTTypeFilterListsEveryRootFile() throws {
        let (_, flowDir) = try tempFlowDir()
        try seed(flowDir, files: ["a.bin": "x", "b.dat": "y"])

        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Index", flowDir: flowDir, wantsFolder: false)

        #expect(Set(entries.map(\.token)) == ["a.bin", "b.dat"])
    }

    // MARK: - picking from the list writes the identical token FILE-1's guard writes

    @Test func theListedTokenMatchesRelativePathForTheSameFile() throws {
        let (_, flowDir) = try tempFlowDir()
        try seed(flowDir, files: ["draft.txt": "hi"])
        let fileURL = flowDir.appendingPathComponent("draft.txt")

        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Text", flowDir: flowDir, wantsFolder: false)
        let listedToken = try #require(entries.first?.token)

        #expect(listedToken == FlowRowInspectorView.relativePath(of: fileURL, under: flowDir))
    }

    @Test func theListedFolderTokenMatchesRelativePathAndCopiesNothing() throws {
        let (_, flowDir) = try tempFlowDir()
        let receipts = flowDir.appendingPathComponent("receipts", isDirectory: true)
        try seed(receipts, files: ["a.png": "A"])

        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Images", flowDir: flowDir, wantsFolder: true)
        let listedToken = try #require(entries.first?.token)
        #expect(listedToken == FlowRowInspectorView.relativePath(of: receipts, under: flowDir))

        // Picking it goes through the same `copyIn`/relativePath seam the panel path uses —
        // same-path, so it copies nothing (FILE-1's guard, unaffected).
        let stored = try FlowRowInspectorView.copyIn(receipts, toFlowDir: flowDir)
        #expect(stored == listedToken)
        #expect(try String(contentsOf: receipts.appendingPathComponent("a.png"), encoding: .utf8) == "A")
    }

    // MARK: - a stored path not in the flat list is flagged, never dropped (owner ruling, 2026-09-11)

    @Test func currentPickStatusIsInListWhenPresent() throws {
        let (_, flowDir) = try tempFlowDir()
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent("receipts"), withIntermediateDirectories: true)
        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Images", flowDir: flowDir, wantsFolder: true)

        #expect(FlowRowInspectorView.currentPickStatus(token: "receipts/", entries: entries, flowDir: flowDir) == .inList)
    }

    @Test func currentPickStatusIsNotInListForANestedPathThatStillExists() throws {
        let (_, flowDir) = try tempFlowDir()
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent("receipts/jan"), withIntermediateDirectories: true)
        // FLAT: `inFlowEntries` never sees `receipts/jan/` — only its parent `receipts/`.
        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Images", flowDir: flowDir, wantsFolder: true)

        #expect(FlowRowInspectorView.currentPickStatus(token: "receipts/jan/", entries: entries, flowDir: flowDir) == .notInList)
    }

    @Test func currentPickStatusIsMissingForADeletedPath() throws {
        let (_, flowDir) = try tempFlowDir()
        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Images", flowDir: flowDir, wantsFolder: true)

        #expect(FlowRowInspectorView.currentPickStatus(token: "receipts/", entries: entries, flowDir: flowDir) == .missing)
    }

    // MARK: - tokensMatch (owner-reported, 2026-09-11): a bare seeded folder name must not
    // read as "not in this list" against the same folder's slash-terminated listed token

    @Test func tokensMatchIgnoresATrailingSlashEitherSide() {
        #expect(FlowRowInspectorView.tokensMatch("sample-images", "sample-images/"))
        #expect(FlowRowInspectorView.tokensMatch("sample-images/", "sample-images"))
        #expect(FlowRowInspectorView.tokensMatch("sample-images/", "sample-images/"))
        #expect(FlowRowInspectorView.tokensMatch("draft.txt", "draft.txt"))
        #expect(!FlowRowInspectorView.tokensMatch("receipts", "receipts/jan"))
    }

    @Test func aBareSeededFolderTokenIsRecognizedAsInList() throws {
        // Reproduces the exact report: a freshly-added `Read Images` row seeds the bare
        // "sample-images" (no trailing slash, predating FILE-1's convention) while the
        // in-flow list — built from the real directory — names it "sample-images/". Before
        // `tokensMatch`, these compared unequal and the current pick was wrongly flagged
        // "not in this list" alongside the real, unmarked listed entry.
        let (_, flowDir) = try tempFlowDir()
        try FileManager.default.createDirectory(at: flowDir.appendingPathComponent("sample-images"),
                                                 withIntermediateDirectories: true)
        let entries = FlowRowInspectorView.inFlowEntries(task: "Read Images", flowDir: flowDir, wantsFolder: true)

        #expect(entries.map(\.token) == ["sample-images/"])
        #expect(FlowRowInspectorView.currentPickStatus(token: "sample-images", entries: entries, flowDir: flowDir) == .inList)
    }

    // MARK: - canReveal: every real folder row gets "Reveal in Finder" (owner, 2026-09-11)

    @Test func canRevealIsTrueForAnyRealFolderRow() {
        let listed = FlowRowInspectorView.InFlowEntry(token: "receipts/", isDirectory: true, count: 3)
        let flaggedButPresent = FlowRowInspectorView.InFlowEntry(token: "receipts/jan/", isDirectory: true,
                                                                  count: nil, note: "not in this list")
        #expect(FlowRowInspectorView.canReveal(listed))
        #expect(FlowRowInspectorView.canReveal(flaggedButPresent))
    }

    @Test func canRevealIsFalseForAFileOrAMissingFolder() {
        let file = FlowRowInspectorView.InFlowEntry(token: "draft.txt", isDirectory: false, count: nil)
        let missing = FlowRowInspectorView.InFlowEntry(token: "receipts/", isDirectory: true,
                                                        count: nil, note: "missing")
        #expect(!FlowRowInspectorView.canReveal(file))
        #expect(!FlowRowInspectorView.canReveal(missing))
    }

    // MARK: - "Create a folder here"

    @Test func createFolderMakesAnEmptyDirectoryThatResolvesThroughFlowWorkspace() throws {
        let (root, flowDir) = try tempFlowDir()

        let token = try FlowRowInspectorView.createFolder(named: "receipts", in: flowDir)
        #expect(token == "receipts/")

        let ws = FlowWorkspace(root: root)
        let resolved = try ws.resolve(token, flowID: "test-flow")
        var isDir: ObjCBool = false
        #expect(FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDir))
        #expect(isDir.boolValue)
        let contents = try FileManager.default.contentsOfDirectory(atPath: resolved.path)
        #expect(contents.isEmpty)
    }

    @Test func createFolderRefusesACollidingName() throws {
        let (_, flowDir) = try tempFlowDir()
        _ = try FlowRowInspectorView.createFolder(named: "receipts", in: flowDir)

        do {
            _ = try FlowRowInspectorView.createFolder(named: "receipts", in: flowDir)
            Issue.record("expected throw")
        } catch let error as FlowFolderCreateError {
            #expect(error == .alreadyExists("receipts"))
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    @Test func createFolderRefusesAnEmptyOrPathLikeName() throws {
        let (_, flowDir) = try tempFlowDir()
        for badName in ["  ", "a/b"] {
            do {
                _ = try FlowRowInspectorView.createFolder(named: badName, in: flowDir)
                Issue.record("expected throw for '\(badName)'")
            } catch let error as FlowFolderCreateError {
                #expect(error == .invalidName)
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
    }
}
