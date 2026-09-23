import Foundation
import UniformTypeIdentifiers

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

    /// The saved file's `Kind` (audio/text/image/folder/video), keyed off the row's task name
    /// alone — nil for a Save row with no classification (`Save Context`). Used directly by
    /// pre-OV-1 call sites, and as `presentation`'s fallback when the file's own extension is
    /// unrecognized (OV-1 — `table.otsl` has no system type and relies on exactly this).
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

    /// How the Output tab should present a saved file (OV-1) — the file's own extension first,
    /// the row's task name only as a fallback. `Save Text` writes whatever extension the
    /// `.cat` names (`.md`, `.txt`, `.diff`, `.html`, `.otsl`, …), so `kind(forTask:)` alone
    /// can't tell a finished HTML page from a Markdown note; this can.
    static func presentation(url: URL, task: String?) -> SavedFilePresentation {
        let isDirectory = (try? url.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true
        if isDirectory { return .folder }

        if let type = UTType(filenameExtension: url.pathExtension) {
            if type.conforms(to: .html) { return .web }
            if type.conforms(to: .pdf) { return .pdf }
            if type.conforms(to: .image) { return .image }
            // `.audio` is checked before `.movie`/`.audiovisualContent`: in Apple's hierarchy
            // `public.audio` itself conforms to `public.audiovisual-content` (the umbrella
            // both audio and video sit under), so checking the broad type first would classify
            // every `.wav` as `.video`. `.audio` first, `.movie`/`.audiovisualContent` catches
            // everything else av-shaped that isn't audio.
            if type.conforms(to: .audio) { return .audio }
            if type.conforms(to: .movie) || type.conforms(to: .audiovisualContent) { return .video }
            if type.conforms(to: .commaSeparatedText) || type.conforms(to: .tabSeparatedText) { return .table }
            if type.conforms(to: .text) { return .text }
        }

        // The extension is unrecognized by the system (`.otsl`, and `.diff` on some installs)
        // — fall back to the task-name classifier so `table.otsl` keeps rendering as text
        // exactly as it does today.
        switch kind(forTask: task) {
        case .text: return .text
        case .image: return .image
        case .audio: return .audio
        case .video: return .video
        default: return .other
        }
    }
}
