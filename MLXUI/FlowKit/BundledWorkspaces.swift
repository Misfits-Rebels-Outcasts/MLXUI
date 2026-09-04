import Foundation

/// CFM-R17-5 — the workspaces that ship in the app bundle. A bundled workspace is the
/// multi-flow analogue of a gallery flow: its `.cat` files ship flat in `Contents/Resources/`
/// (the synchronized root group flattens `MLXUI/Resources/Workspaces/**`, so a colliding
/// filename ships under an `<id>--` prefix, exactly as `GalleryLoader.bundledAssets` does)
/// and are copied into `workspaces/<id>/` — side by side — on first use.
///
/// `uses_example` is `catflow-mlx/library/uses_example/` ported verbatim (source commit
/// `5e0f622`, 2026-08-07): `AskYourDocs.cat` calls `RagQuery.cat` through
/// `uses: RagQuery = ./RagQuery.cat`. It is the first thing to actually exercise the app's
/// `uses:` expansion, sibling resolution, and the App Store E117 channel guarantee end to end.
///
/// `ask_your_docs` (CFM-R17-6) is `16-IngestFolder` + `18-DocChat` sharing one `library.index`:
/// `Ingest.cat` builds the index from the workspace's own `docs/` PDFs, `DocChat.cat` reads and
/// chats against it. It ships **no prebuilt index** — Build creates it (CFM-R17-4's card).
nonisolated enum BundledWorkspaces {

    /// One file inside a bundled workspace: the flat bundle resource name (no extension), the
    /// extension, and the path it must have inside the workspace (may name a subdirectory).
    struct BundledFile: Sendable {
        let resource: String
        let ext: String
        let destination: String
    }

    struct Meta: Sendable, Identifiable {
        /// The workspace id — `workspaces/<id>/`.
        let id: String
        let title: String
        let description: String
        /// The flow file a caller opens/runs first.
        let entryFlow: String
        let files: [BundledFile]
    }

    static let all: [Meta] = [
        Meta(id: "uses_example",
             title: "Ask Your Docs — uses: demo",
             description: "A worked `uses:` pair: AskYourDocs.cat calls RagQuery.cat as a composite instead of pasting its retrieval rows in.",
             entryFlow: "AskYourDocs.cat",
             files: [
                BundledFile(resource: "uses_example--AskYourDocs", ext: "cat", destination: "AskYourDocs.cat"),
                BundledFile(resource: "uses_example--RagQuery", ext: "cat", destination: "RagQuery.cat"),
             ]),
        Meta(id: "ask_your_docs",
             title: "Ask Your Docs",
             description: "Ingest.cat builds library.index from the docs/ folder; DocChat.cat reads it and answers questions, looping so you can keep asking.",
             entryFlow: "DocChat.cat",
             files: [
                BundledFile(resource: "ask_your_docs--Ingest", ext: "cat", destination: "Ingest.cat"),
                BundledFile(resource: "ask_your_docs--DocChat", ext: "cat", destination: "DocChat.cat"),
                // The corpus — no prebuilt library.index. `doc-{a,b,c}.pdf` are the same flat
                // bundle resources `16-IngestFolder` ships (globally unique, read-only copy).
                BundledFile(resource: "doc-a", ext: "pdf", destination: "docs/doc-a.pdf"),
                BundledFile(resource: "doc-b", ext: "pdf", destination: "docs/doc-b.pdf"),
                BundledFile(resource: "doc-c", ext: "pdf", destination: "docs/doc-c.pdf"),
             ]),
    ]

    /// The ids of every bundled workspace — `WorkspaceStore.scan`'s `bundledWorkspaceIDs`
    /// excludes these from a user's "my workspaces" list (their materialized copy is scratch,
    /// like a gallery flow's working directory).
    static var ids: Set<String> { Set(all.map(\.id)) }

    static func meta(id: String) -> Meta? { all.first { $0.id == id } }

    /// Copy `meta`'s flow files into `workspaces/<id>/` if they aren't already there.
    /// **Idempotent** — a file the user has since edited is left alone (only missing files
    /// are written), matching `FlowWorkspace.prepare`.
    static func prepare(_ meta: Meta, workspace: FlowWorkspace, bundle: Bundle = .main) throws {
        let fm = FileManager.default
        let dir = workspace.directory(for: meta.id)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        for file in meta.files {
            let dest = dir.appendingPathComponent(file.destination)
            if fm.fileExists(atPath: dest.path) { continue }
            guard let src = bundle.url(forResource: file.resource, withExtension: file.ext) else {
                throw BundledWorkspaceError.missingResource("\(file.resource).\(file.ext)")
            }
            try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: src, to: dest)
        }
    }
}

nonisolated enum BundledWorkspaceError: Error, CustomStringConvertible, Equatable {
    case missingResource(String)

    var description: String {
        switch self {
        case .missingResource(let name):
            return "A bundled workspace file is missing from the app: \(name)."
        }
    }
}
