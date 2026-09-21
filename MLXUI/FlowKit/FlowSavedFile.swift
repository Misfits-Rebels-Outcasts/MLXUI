import Foundation

/// CFM-R11-2 UX (Save-row preview): resolves the file a `Save *` row wrote, so the inspector
/// can play/view the saved result instead of only its status sentence. Pure enough to test.
nonisolated enum FlowSavedFile {
    /// The file a `Save *` row wrote, resolved against the flow's working folder — nil when
    /// the row isn't a Save row, has no path, the path was refused (absolute, `..`, or a
    /// symlink escaping the flow directory), or the file hasn't been written (yet).
    ///
    /// BF-2: routed through `workspace.resolve`, the same boundary every Save *tool* writes
    /// through (`SaveImageTools.swift`) — this used to build the URL with a raw
    /// `appendingPathComponent`, so a path the writer refused could still resolve to a URL
    /// here and `fileExists` could say yes for it, showing the Output tab a file the row
    /// never produced. `normalizedSavePath` matches the writer's own `./out.png` → `out.png`
    /// normalization so both sides agree byte for byte.
    static func resolved(row: Row, flowID: String, workspace: FlowWorkspace) -> URL? {
        guard let task = row.task, task.hasPrefix("Save"),
              let rawPath = FlowSettings(row.settings).pathValue(),
              let url = try? workspace.resolve(normalizedSavePath(rawPath), flowID: flowID)
        else { return nil }
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// The saved file's kind, driving the inspector's presentation (audio/image/text) —
    /// nil for Save rows with no viewer (falls back to a "Show in Finder" row).
    static func kind(forTask task: String?) -> Kind? {
        switch task {
        case "Save Audio": return .audio
        case "Save Text": return .text
        case "Save Image": return .image
        case "Save Images": return .folder
        case "Save Video": return .video
        default: return nil
        }
    }
}
