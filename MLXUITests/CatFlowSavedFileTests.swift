import Testing
import Foundation
@testable import MLXUI

/// The Save-row preview resolution (CFM-R11-2 UX): the inspector can present a `Save *` row's
/// written file, resolved against the flow's working folder.
struct CatFlowSavedFileTests {

    @Test func resolvesAnExistingSavedAudioFile() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("memo-tldr.wav")
        try AudioWriter.writeWAV(AudioBuffer(samples: [0.1, 0.2], sampleRate: 24_000), to: file)
        defer { try? FileManager.default.removeItem(at: base) }

        let row = Row(id: UUID(), task: "Save Audio", settings: "memo-tldr.wav")
        let url = FlowSavedFile.resolved(row: row, flowID: "flow-1", workspace: ws)
        #expect(url?.lastPathComponent == "memo-tldr.wav")
    }

    @Test func resolvesPathPairValue() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let file = dir.appendingPathComponent("out.txt")
        try Data("hello".utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: base) }

        let row = Row(id: UUID(), task: "Save Text", settings: "path=out.txt")
        #expect(FlowSavedFile.resolved(row: row, flowID: "flow-1", workspace: ws) != nil)
    }

    @Test func nilWhenNotASaveRowOrFileMissing() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        defer { try? FileManager.default.removeItem(at: base) }

        let notSave = Row(id: UUID(), task: "Read Audio", settings: "memo.m4a")
        #expect(FlowSavedFile.resolved(row: notSave, flowID: "flow-1", workspace: ws) == nil)

        let missing = Row(id: UUID(), task: "Save Audio", settings: "nope.wav")
        #expect(FlowSavedFile.resolved(row: missing, flowID: "flow-1", workspace: ws) == nil)
    }

    @Test func kindFollowsTheTask() {
        #expect(FlowSavedFile.kind(forTask: "Save Audio") == .audio)
        #expect(FlowSavedFile.kind(forTask: "Save Text") == .text)
        #expect(FlowSavedFile.kind(forTask: "Save Image") == .image)
        #expect(FlowSavedFile.kind(forTask: "Save Images") == .folder)
        #expect(FlowSavedFile.kind(forTask: "Save Video") == .video)
        #expect(FlowSavedFile.kind(forTask: "Read Audio") == nil)
        #expect(FlowSavedFile.kind(forTask: nil) == nil)
    }

    /// BF-1: `Save Images` writes a folder (`vacation-web/`, Gallery flow 21 row 3), which
    /// `.image` cannot render — `Data(contentsOf:)` on a directory fails and the Output tab
    /// showed "Couldn't load the image." Pinned separately from `Save Image` so the two can
    /// never be collapsed back into one case (§7 trap 1 — no new `Kind` case; `.folder` is
    /// already one of the fourteen).
    @Test func saveImagesIsNeverConfusedWithSaveImage() {
        #expect(FlowSavedFile.kind(forTask: "Save Images") != FlowSavedFile.kind(forTask: "Save Image"))
        #expect(FlowSavedFile.kind(forTask: "Save Images") == .folder)
        #expect(FlowSavedFile.kind(forTask: "Save Image") == .image)
    }

    /// BF-1, against a real temp flow directory: Gallery flow 21's row 3
    /// (`3. Save Images   vacation-web/; naming="{name}-web"`) resolves to the folder it
    /// wrote, typed `.folder`.
    @Test func resolvesFlow21Row3AsAFolder() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-21")
        let folder = dir.appendingPathComponent("vacation-web")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }

        let row = Row(id: UUID(), task: "Save Images", settings: "vacation-web/; naming=\"{name}-web\"")
        let url = FlowSavedFile.resolved(row: row, flowID: "flow-21", workspace: ws)
        #expect(url?.lastPathComponent == "vacation-web")
        #expect(FlowSavedFile.kind(forTask: row.task) == .folder)
    }

    // MARK: - BF-2: the reader goes through the same boundary the writer does

    @Test func resolvedRefusesADotDotEscape() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        // A file that really exists just outside the flow directory — the escape would
        // otherwise resolve and `fileExists` would say yes.
        try Data("leaked".utf8).write(to: base.appendingPathComponent("escape.txt"))
        defer { try? FileManager.default.removeItem(at: base) }

        let row = Row(id: UUID(), task: "Save Text", settings: "path=../escape.txt")
        #expect(FlowSavedFile.resolved(row: row, flowID: "flow-1", workspace: ws) == nil)
    }

    @Test func resolvedRefusesAnAbsolutePath() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let outside = base.appendingPathComponent("x.txt")
        try Data("leaked".utf8).write(to: outside)
        defer { try? FileManager.default.removeItem(at: base) }

        let row = Row(id: UUID(), task: "Save Text", settings: "path=\(outside.path)")
        #expect(FlowSavedFile.resolved(row: row, flowID: "flow-1", workspace: ws) == nil)
    }

    @Test func resolvedRefusesASymlinkAtTheSavedPath() throws {
        let base = FileManager.default.temporaryDirectory
            .resolvingSymlinksInPath()
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let outsideTarget = base.appendingPathComponent("outside.txt")
        try Data("leaked".utf8).write(to: outsideTarget)
        let link = dir.appendingPathComponent("out.txt")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: outsideTarget)
        defer { try? FileManager.default.removeItem(at: base) }

        let row = Row(id: UUID(), task: "Save Text", settings: "out.txt")
        #expect(FlowSavedFile.resolved(row: row, flowID: "flow-1", workspace: ws) == nil)
    }

    /// `./summary.md` and `summary.md` name the same file to the writer
    /// (`normalizedSavePath`) and must name the same file to the reader.
    @Test func resolvedNormalizesADotSlashPrefixLikeTheWriterDoes() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("catflow-saved-\(UUID().uuidString)")
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let dir = ws.directory(for: "flow-1")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("hello".utf8).write(to: dir.appendingPathComponent("summary.md"))
        defer { try? FileManager.default.removeItem(at: base) }

        let dotSlash = Row(id: UUID(), task: "Save Text", settings: "./summary.md")
        let bare = Row(id: UUID(), task: "Save Text", settings: "summary.md")
        let urlA = FlowSavedFile.resolved(row: dotSlash, flowID: "flow-1", workspace: ws)
        let urlB = FlowSavedFile.resolved(row: bare, flowID: "flow-1", workspace: ws)
        #expect(urlA != nil)
        #expect(urlA?.path == urlB?.path)
    }

    /// Every `Save *` row in the three shipped resource trees resolves to the same URL BF-2
    /// resolves it to as it did on `8052b16` (the raw, un-boundary-checked
    /// `appendingPathComponent`) — for a **legitimate** path, the two formulas must never
    /// disagree. Nothing else in the suite exercises this function against the shipped
    /// galleries, so a silent change here would leave the Output tab empty across all three
    /// shelves without a single other test noticing (§7 trap 11).
    @Test func everyShippedSaveRowResolvesToTheSameURLAsBefore() throws {
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

                    // .resolvingSymlinksInPath() here is load-bearing, not decoration: resolve()
                    // resolves the flow directory internally, and /tmp → /private/tmp on macOS,
                    // so the full-path comparison below would fail on that prefix alone without
                    // matching it up front on both sides.
                    let base = FileManager.default.temporaryDirectory
                        .resolvingSymlinksInPath()
                        .appendingPathComponent("catflow-corpus-\(UUID().uuidString)")
                    let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
                    let flowDir = ws.directory(for: "flow-1")
                    try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
                    defer { try? FileManager.default.removeItem(at: base) }

                    // Reproduce 8052b16's formula: a raw appendingPathComponent, no
                    // normalization, no containment check.
                    let oldURL = flowDir.appendingPathComponent(rawPath)
                    if row.task == "Save Images" || rawPath.hasSuffix("/") {
                        try FileManager.default.createDirectory(at: oldURL, withIntermediateDirectories: true)
                    } else {
                        try FileManager.default.createDirectory(
                            at: oldURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                        try Data().write(to: oldURL)
                    }

                    let newURL = FlowSavedFile.resolved(row: row, flowID: "flow-1", workspace: ws)
                    #expect(newURL?.path == oldURL.path,
                            "\(catFile) \(row.task ?? "?") '\(rawPath)': \(String(describing: newURL?.path)) != \(oldURL.path)")
                }
            }
        }
        // Pins §0.1's count (71 + 11 + 2 + 1 + 0 = 85) so a new gallery flow's Save rows are
        // never silently skipped by this corpus walk.
        #expect(checkedRows == 85, """
            Corpus row count changed from 85 — if a new Save * row shipped, confirm it \
            resolves correctly above, then bump this count; if a row disappeared, confirm \
            that was intentional.
            """)
    }

    /// Rows the Output tab can present as a saved file — i.e. those `FlowSavedFile.kind`
    /// classifies (`Save Audio/Text/Image/Images/Video`; `Save Context` has no viewer and no
    /// entry in §0.1's count, so it's out of scope here as it is there).
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
}
