import Foundation

/// CFM-R11-2 UX (Save-row preview): resolves the file a `Save *` row wrote, so the inspector
/// can play/view the saved result instead of only its status sentence. Pure enough to test.
nonisolated enum FlowSavedFile {
    /// The file a `Save *` row wrote, resolved against the flow's working folder — nil when
    /// the row isn't a Save row, has no path, or the file hasn't been written (yet).
    static func resolved(row: Row, flowID: String, workspace: FlowWorkspace) -> URL? {
        guard let task = row.task, task.hasPrefix("Save"),
              let path = FlowSettings(row.settings).pathValue() else { return nil }
        let url = workspace.directory(for: flowID).appendingPathComponent(path)
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
