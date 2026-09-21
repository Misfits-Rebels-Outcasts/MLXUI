import Testing
import Foundation
@testable import MLXUI

/// OV-1: `FlowSavedFile.presentation(url:task:)` classifies a saved file by its own extension
/// first, the row's task name only as a fallback — the fix underneath "open the saved file in
/// something that can read it" (`RSI/DelegateOutputViewerBacklog.md` §0.1). `Save Text` writes
/// whatever extension the `.cat` names it, so `kind(forTask:)` alone can't tell a finished
/// HTML page from a Markdown note; this can.
struct CatFlowSavedFilePresentationTests {

    private func tempFile(named name: String) throws -> (url: URL, cleanup: () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-presentation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent(name)
        try Data().write(to: url)
        return (url, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - Every extension §0.1 counted, plus the spec's extra coverage set

    @Test(arguments: [
        ("run-today.md", "Save Text", SavedFilePresentation.text),
        ("cull-list.txt", "Save Text", SavedFilePresentation.text),
        ("policy-changes.diff", "Save Text", SavedFilePresentation.text),
        ("ai-news.html", "Save Text", SavedFilePresentation.web),
        ("table.otsl", "Save Text", SavedFilePresentation.text),
        ("photo-2x.png", "Save Image", SavedFilePresentation.image),
        ("memo-tldr.wav", "Save Audio", SavedFilePresentation.audio),
        ("report.pdf", "Save Text", SavedFilePresentation.pdf),
        ("data.csv", "Save Text", SavedFilePresentation.table),
        ("clip.mp4", "Save Video", SavedFilePresentation.video),
        ("photo.jpeg", "Save Image", SavedFilePresentation.image),
    ])
    func classifiesByExtension(name: String, task: String, expected: SavedFilePresentation) throws {
        let (url, cleanup) = try tempFile(named: name)
        defer { cleanup() }
        #expect(FlowSavedFile.presentation(url: url, task: task) == expected)
    }

    /// The extension alone decides it — no task needed. Proves step 2 (UTType conformance)
    /// really does run before, and independently of, the task-name fallback.
    @Test func extensionAloneClassifiesWithNoTaskAtAll() throws {
        let (url, cleanup) = try tempFile(named: "diagram.png")
        defer { cleanup() }
        #expect(FlowSavedFile.presentation(url: url, task: nil) == .image)
    }

    /// No extension, and no task the classifier recognizes → `.other`, not a crash or a guess.
    @Test func noExtensionAndNoTaskFallsToOther() throws {
        let (url, cleanup) = try tempFile(named: "README")
        defer { cleanup() }
        #expect(FlowSavedFile.presentation(url: url, task: nil) == .other)
    }

    /// No extension, but the task still classifies it — the fallback alone is enough.
    @Test func noExtensionFallsBackToTheTask() throws {
        let (url, cleanup) = try tempFile(named: "README")
        defer { cleanup() }
        #expect(FlowSavedFile.presentation(url: url, task: "Save Text") == .text)
    }

    /// `Save Context` has no `Kind` and therefore no fallback — an unrecognized extension under
    /// it lands on `.other`, same as any other row `kind(forTask:)` doesn't classify.
    @Test func saveContextWithAnUnknownExtensionFallsToOther() throws {
        let (url, cleanup) = try tempFile(named: "session.ctx")
        defer { cleanup() }
        #expect(FlowSavedFile.presentation(url: url, task: "Save Context") == .other)
    }

    // MARK: - A real directory, regardless of task

    @Test func aRealDirectoryClassifiesAsFolderRegardlessOfTask() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-presentation-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(FlowSavedFile.presentation(url: dir, task: "Save Images") == .folder)
        #expect(FlowSavedFile.presentation(url: dir, task: nil) == .folder)
    }

    // MARK: - Corpus: every shipped Save row classifies to something a viewer exists for

    /// Rows the Output tab can present as a saved file at all (same scope as BF-2's corpus
    /// test, `CatFlowSavedFileTests.saveRows` — `FlowSavedFile.kind` classifies them; `Save
    /// Context` has no `Kind` and is out of scope here for the identical reason it is there).
    private static func saveRows(in rows: [Row]) -> [Row] {
        rows.flatMap { row -> [Row] in
            var found: [Row] = []
            if FlowSavedFile.kind(forTask: row.task) != nil {
                found.append(row)
            }
            found.append(contentsOf: saveRows(in: row.children))
            return found
        }
    }

    @Test func everyShippedSaveRowClassifiesToSomethingOtherThanOther() throws {
        let resourceDirs = ["Gallery", "BasicGallery", "Workspaces"]
        let resourcesRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources")

        var checkedRows = 0
        for dirName in resourceDirs {
            let dir = resourcesRoot.appendingPathComponent(dirName)
            let catFiles = try FileManager.default.contentsOfDirectory(atPath: dir.path)
                .filter { $0.hasSuffix(".cat") }
                .sorted()
            for catFile in catFiles {
                let text = try String(contentsOf: dir.appendingPathComponent(catFile), encoding: .utf8)
                let doc = try CatParser.parse(text)
                for row in Self.saveRows(in: doc.rows) {
                    checkedRows += 1
                    guard let rawPath = FlowSettings(row.settings).pathValue() else {
                        Issue.record("\(catFile): \(row.task ?? "?") row has no path")
                        continue
                    }

                    // `.folder` is only ever detected by asking the filesystem (step 1 of
                    // `presentation`), never by extension or task-name fallback — so the row's
                    // path has to exist for real, exactly like BF-2's corpus test.
                    let base = FileManager.default.temporaryDirectory
                        .appendingPathComponent("catflow-presentation-corpus-\(UUID().uuidString)")
                    defer { try? FileManager.default.removeItem(at: base) }
                    let url = base.appendingPathComponent(rawPath)
                    if row.task == "Save Images" || rawPath.hasSuffix("/") {
                        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
                    } else {
                        try FileManager.default.createDirectory(
                            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                        try Data().write(to: url)
                    }

                    let presentation = FlowSavedFile.presentation(url: url, task: row.task)
                    #expect(presentation != .other, """
                        \(catFile) \(row.task ?? "?") '\(rawPath)' classified .other — a new \
                        extension shipped with no viewer path. Add it to OV-1's UTType table \
                        or its kind(forTask:) fallback.
                        """)
                }
            }
        }
        // Pins the same count BF-2's corpus test pins (85) — see that test's failure message
        // for what to do when this fires.
        #expect(checkedRows == 85, """
            Corpus row count changed from 85 — if a new Save * row shipped, confirm it \
            classifies correctly above, then bump this count; if a row disappeared, \
            confirm that was intentional.
            """)
    }
}
