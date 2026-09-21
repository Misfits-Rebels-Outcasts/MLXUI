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
}
