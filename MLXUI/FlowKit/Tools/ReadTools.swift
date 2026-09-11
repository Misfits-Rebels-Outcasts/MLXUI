import Foundation
import PDFKit

/// CFM-R11-0c — the four newly-ported `Read *` tools (`Read Image`, `Read Images`,
/// `Read Files`, `Read PDF`), plus CFM-R11-0b's **one list**.
///
/// The one-list rule (R11-0c): seeding and tool availability read the same `SampleSeed.samples`
/// table, so a task can never be seedable without being runnable — the false affordance
/// R11-0b exists to remove. The picker marks any `Read *` task missing from the list with a
/// "needs newer support" marker instead of appearing identical to the runnable ones.
nonisolated enum SampleSeed {
    /// The single source of truth: a `Read *` task → the sample its row gets (the settings
    /// value) and the `(source, destination)` pairs `FlowWorkspace.prepare` copies in.
    /// **Runnable ⟺ present in this table.**
    static let samples: [String: (value: String, assets: [(source: String, destination: String)])] = [
        "Read Audio": ("sample-audio.m4a", [("sample-audio.m4a", "sample-audio.m4a")]),
        "Read Text": ("sample-text.txt", [("sample-text.txt", "sample-text.txt")]),
        "Read Image": ("sample-image.png", [("sample-image.png", "sample-image.png")]),
        "Read PDF": ("sample-pdf.pdf", [("sample-pdf.pdf", "sample-pdf.pdf")]),
        "Read CSV": ("sample-data.csv", [("sample-data.csv", "sample-data.csv")]),
        "Read JSON": ("sample-data.json", [("sample-data.json", "sample-data.json")]),
        "Read Images": ("sample-images", [
            ("sample-img-01.png", "sample-images/sample-img-01.png"),
            ("sample-img-02.png", "sample-images/sample-img-02.png"),
            ("sample-img-03.png", "sample-images/sample-img-03.png"),
        ]),
        "Read Files": ("sample-files; pattern=*", [
            ("sample-note-01.txt", "sample-files/sample-note-01.txt"),
            ("sample-note-02.txt", "sample-files/sample-note-02.txt"),
            ("sample-note-03.txt", "sample-files/sample-note-03.txt"),
        ]),
    ]

    /// The settings value a seeded row stores (a bare path: `sample-audio.m4a`, `sample-images`),
    /// or nil when the task is unseeded.
    static func seedValue(for task: String) -> String? { samples[task]?.value }

    /// The copy pairs to bring a task's sample into a flow folder, or nil when unseeded.
    static func seedAssets(for task: String) -> [(source: String, destination: String)]? {
        samples[task]?.assets
    }

    /// Whether a `Read *` task has a runnable tool in this build — the one list, nothing else.
    static func isRunnable(_ task: String) -> Bool { samples[task] != nil }

    /// Every `Read *` task in the catalog, for the picker's marker (runnable or not).
    static let readTaskNames = ["Read Audio", "Read Text", "Read Image", "Read Images",
                                "Read Files", "Read PDF", "Read Video", "Read Index",
                                "Read Context", "Read CSV", "Read JSON"]
}

/// Shared path resolution for the `Read *` tools — mirrors `tools/files.py::_resolve_path`:
/// an upstream item of the matching kind wins, else the settings' `path=`/first bare token,
/// resolved through the flow's working directory (the CFM-R1-4 security boundary).
private enum ReadPath {
    static func resolve(workspace: FlowWorkspace, flowID: String, settings: String,
                        inputs: [Asset], kind: Kind, row: String) throws -> URL {
        if let item = inputs.first?.items.first, item.kind == kind, let path = item.path {
            return path
        }
        guard let raw = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: row, kind: kind)
        }
        return try workspace.resolve(raw, flowID: flowID)
    }
}

/// `Read Image` (file → image): resolve the path and return a file-backed `.image` item —
/// the same contract as `tools/files.py::read_image` (no decode; downstream stages load it).
nonisolated struct ReadImageTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let url = try ReadPath.resolve(workspace: workspace, flowID: flowID, settings: settings,
                                       inputs: [input], kind: .file, row: "Read Image")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FlowError.fileReadFailed(row: "Read Image", path: url.path)
        }
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
    }
}

/// A tiny glob matcher for the `pattern=` setting (`*.m4a`, `*`, `?`) — the subset of
/// `Path.glob` the gallery's patterns actually use. Case-sensitive, anchored, no `**`.
nonisolated enum GlobMatch {
    static func matches(_ name: String, glob: String) -> Bool {
        var regex = "^"
        for scalar in glob.unicodeScalars {
            switch scalar {
            case "*": regex += ".*"
            case "?": regex += "."
            case ".", "(", ")", "[", "]", "{", "}", "+", "^", "$", "\\", "|":
                regex += "\\\(Character(scalar))"
            default:
                regex.append(Character(scalar))
            }
        }
        regex += "$"
        guard let re = try? NSRegularExpression(pattern: regex) else { return false }
        let ns = NSRange(name.startIndex..<name.endIndex, in: name)
        return re.firstMatch(in: name, range: ns) != nil
    }
}

/// `Read Images` (folder → list of image): enumerate the folder's image files (`pattern=`
/// glob, else the fixed image-extension set — `tools/files.py::read_images`). Sorted by name.
nonisolated struct ReadImagesTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.folder) }
    var produces: Shape { .listOf(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let folder = try ReadPath.resolve(workspace: workspace, flowID: flowID, settings: settings,
                                          inputs: [input], kind: .folder, row: "Read Images")
        let fm = FileManager.default
        let isDir = (try? folder.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
        // QR12R2-3 note: skip dot-directories (e.g. the Save-* `.trash/`) so a re-run's
        // folder glob doesn't pick up trashed files.
        guard isDir, let contents = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil,
                                                                options: [.skipsHiddenFiles]) else {
            throw FlowError.fileReadFailed(row: "Read Images", path: folder.path)
        }
        let pattern = FlowSettings(settings).value(for: "pattern")
        let imageExts = ["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff"]
        let files = contents
            .filter { $0.hasDirectoryPath == false }
            .filter { url in
                let name = url.lastPathComponent
                if let pattern { return GlobMatch.matches(name, glob: pattern) }
                return imageExts.contains(url.pathExtension.lowercased())
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        // FILE-2 / SPEC-Q219: a folder that resolved fine but yielded nothing must say so — an
        // enclosing `<each>` over an empty list runs zero times and the flow completes green
        // having done nothing. Deliberate divergence from `tools/files.py::read_images`, which
        // returns an empty list here.
        guard !files.isEmpty else {
            throw FlowError.emptyFolder(row: "Read Images", path: folder.path)
        }
        progress(1.0)
        return Asset(items: files.map { Item(kind: .image, value: nil, path: $0, sourceText: nil) })
    }
}

/// `Read Files` (folder → list of file): enumerate the folder's files (`pattern=` glob or
/// `*`, optional `recursive=true`) — `tools/files.py::read_files`. Sorted by name.
nonisolated struct ReadFilesTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.folder) }
    var produces: Shape { .listOf(.file) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let folder = try ReadPath.resolve(workspace: workspace, flowID: flowID, settings: settings,
                                          inputs: [input], kind: .folder, row: "Read Files")
        let fm = FileManager.default
        let isDir = (try? folder.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
        guard isDir else {
            throw FlowError.fileReadFailed(row: "Read Files", path: folder.path)
        }
        let s = FlowSettings(settings)
        let pattern = s.value(for: "pattern") ?? s.firstBare() ?? "*"
        let recursive = (s.value(for: "recursive", default: "false") ?? "false")
            .lowercased() == "true"
        var files: [URL] = []
        if recursive {
            // R13-8: same `.skipsHiddenFiles` treatment as the flat branch — the Save-*
            // tools' `.trash/` lives in the folder they overwrite, and a re-run's glob must
            // not pick up the trashed copies.
            if let enumerator = fm.enumerator(at: folder, includingPropertiesForKeys: nil,
                                              options: [.skipsHiddenFiles]),
               let contents = enumerator.allObjects as? [URL] {
                files = contents.filter { !$0.hasDirectoryPath }
            }
        } else if let contents = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil,
                                                             options: [.skipsHiddenFiles]) {
            files = contents.filter { !$0.hasDirectoryPath }
        }
        let matched = files.filter { GlobMatch.matches($0.lastPathComponent, glob: pattern) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        // FILE-2 / SPEC-Q219: a folder that resolved fine but yielded nothing must say so — an
        // enclosing `<each>` over an empty list runs zero times and the flow completes green
        // having done nothing. Deliberate divergence from `tools/files.py::read_files`, which
        // returns an empty list here.
        guard !matched.isEmpty else {
            throw FlowError.emptyFolder(row: "Read Files", path: folder.path)
        }
        progress(1.0)
        return Asset(items: matched.map { Item(kind: .file, value: nil, path: $0, sourceText: nil) })
    }
}

/// `Read PDF` (file → text): extract each page's selectable text with PDFKit (the
/// system-provided `pdftotext` analogue — `tools/files.py::read_pdf`). A scanned PDF yields
/// empty text, honestly.
nonisolated struct ReadPDFTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let url = try ReadPath.resolve(workspace: workspace, flowID: flowID, settings: settings,
                                       inputs: [input], kind: .file, row: "Read PDF")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FlowError.fileReadFailed(row: "Read PDF", path: url.path)
        }
        guard let document = PDFDocument(url: url) else {
            throw FlowError.fileReadFailed(row: "Read PDF", path: url.path)
        }
        progress(0.4)
        var pages: [String] = []
        for index in 0..<document.pageCount {
            if let page = document.page(at: index), let text = page.string {
                pages.append(text)
            }
        }
        progress(1.0)
        return Asset(items: [Item(kind: .text, value: pages.joined(separator: "\n\n"),
                                  path: nil, sourceText: nil)])
    }
}
