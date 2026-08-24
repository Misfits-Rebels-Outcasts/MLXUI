import Foundation

/// CFM-R12-5 — the `Save Image` / `Save Images` / `Save Video` tools (the Python
/// `files.py::_save_any` / `save_images`), the port that unblocks every image flow: the
/// gallery can generate an image but could never write it to disk.
///
/// Path resolution and status-asset shape follow the established `SaveTextTool`/
/// `SaveAudioTool` (`SpokenSummaryTools.swift`): every path resolves through
/// `FlowWorkspace.resolve` (the CFM-R1-4 boundary) and the status sentence matches the
/// Python byte for byte (`saved to {dest}` / `saved {n} image(s) to {folder}`). Writes are
/// copy-to-temp-then-replace, so a failed copy never leaves a half-written file.

/// `Save Image` (any → status): write the input's file-backed image to the resolved path.
nonisolated struct SaveImageTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .anyKind }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let src = input.items.first?.path else {
            throw FlowError.missingInlineValue(row: "Save Image", kind: .image)
        }
        guard let rawPath = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Save Image", kind: .file)
        }
        let dest = try workspace.resolve(rawPath, flowID: flowID)
        let fm = FileManager.default
        guard fm.fileExists(atPath: src.path) else {
            throw FlowError.fileReadFailed(row: "Save Image", path: src.lastPathComponent)
        }
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = dest.deletingLastPathComponent()
            .appendingPathComponent(".\(dest.lastPathComponent).tmp\(UUID().uuidString)")
        try fm.copyItem(at: src, to: tmp)
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: tmp, to: dest)
        progress(1.0)
        return Asset(items: [Item(kind: .status, value: "saved to \(rawPath)",
                                  path: nil, sourceText: nil)])
    }
}

/// `Save Video` (any → status): the same copy for a file-backed video item.
nonisolated struct SaveVideoTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .anyKind }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let src = input.items.first?.path else {
            throw FlowError.missingInlineValue(row: "Save Video", kind: .video)
        }
        guard let path = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Save Video", kind: .file)
        }
        let dest = try workspace.resolve(path, flowID: flowID)
        let fm = FileManager.default
        guard fm.fileExists(atPath: src.path) else {
            throw FlowError.fileReadFailed(row: "Save Video", path: src.lastPathComponent)
        }
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        let tmp = dest.deletingLastPathComponent()
            .appendingPathComponent(".\(dest.lastPathComponent).tmp\(UUID().uuidString)")
        try fm.copyItem(at: src, to: tmp)
        if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
        try fm.moveItem(at: tmp, to: dest)
        progress(1.0)
        return Asset(items: [Item(kind: .status, value: "saved to \(path)",
                                  path: nil, sourceText: nil)])
    }
}

/// `Save Images` (list of image → status): write every item into `folder=` (or the first
/// bare token, else `.`) under `naming=` (default `item_{n}{ext}`; `{n}` = 1-based index,
/// `{ext}` = the source extension, `{name}` = the source stem).
nonisolated struct SaveImagesTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .listOf(.image) }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let s = FlowSettings(settings)
        let folderRaw = s.value(for: "folder") ?? s.firstBare() ?? "."
        let naming = s.value(for: "naming") ?? "item_{n}{ext}"
        let folder = try workspace.resolve(folderRaw, flowID: flowID)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        var saved = 0
        for (i, item) in input.items.enumerated() {
            guard let src = item.path else { continue }
            guard FileManager.default.fileExists(atPath: src.path) else { continue }
            let name = naming
                .replacingOccurrences(of: "{n}", with: String(i + 1))
                .replacingOccurrences(of: "{ext}", with: src.pathExtension.isEmpty ? "" : ".\(src.pathExtension)")
                .replacingOccurrences(of: "{name}", with: src.deletingPathExtension().lastPathComponent)
            let dest = folder.appendingPathComponent(name)
            let fm = FileManager.default
            let tmp = folder.appendingPathComponent(".\(name).tmp\(UUID().uuidString)")
            try fm.copyItem(at: src, to: tmp)
            if fm.fileExists(atPath: dest.path) { try fm.removeItem(at: dest) }
            try fm.moveItem(at: tmp, to: dest)
            saved += 1
        }
        progress(1.0)
        let folderName = folder.lastPathComponent.isEmpty ? "." : folder.lastPathComponent
        return Asset(items: [Item(kind: .status, value: "saved \(saved) image(s) to \(folderName)",
                                  path: nil, sourceText: nil)])
    }
}
