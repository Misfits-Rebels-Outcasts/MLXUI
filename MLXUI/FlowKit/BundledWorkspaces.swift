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
/// It also ships the two assets its rows read — a sample `question.txt` and the prebuilt
/// `kb.index/` (`RagQuery.cat` row 2) — so opening it and pressing Run works once BGE-M3 and
/// Qwen3 8B are installed, exactly as `gallery/17-AskYourDocs.cat` (the same flow with the
/// retrieval rows written out) does. The Python reference leaves both files for the caller to
/// supply (`tests/test_uses_example.py::_install`); the app has no such step, so a shelf user
/// would otherwise fail on row 1 (CFM-R17-FIX-6). `kb.index/` is the same corpus gallery 17
/// ships (API-key docs) — hence the sample question.
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
                // The two assets the rows read. `question.txt` — `AskYourDocs.cat` row 1;
                // `kb.index/` — `RagQuery.cat` row 2, the same prebuilt corpus `16/17/18`
                // ship (globally unique flat resources, read-only copy).
                BundledFile(resource: "uses_example--question", ext: "txt", destination: "question.txt"),
                BundledFile(resource: "library-chunks", ext: "jsonl", destination: "kb.index/chunks.jsonl"),
                BundledFile(resource: "library-manifest", ext: "json", destination: "kb.index/manifest.json"),
                BundledFile(resource: "library-vectors", ext: "bin", destination: "kb.index/vectors.bin"),
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

    /// The ids of every bundled workspace. Used to tell a bundled workspace's Remove/Restore
    /// (a tombstone — CFM-R17-FIX-1) from a user workspace's delete, and to keep a stale id
    /// out of the tombstone set. A bundled workspace **does** show in "My Workflows" — its
    /// materialised copy is a real, editable workspace, not scratch — so this is *not* a
    /// scan-exclusion list (contrast `GalleryLoader` + `UserFlowStore`).
    static var ids: Set<String> { Set(all.map(\.id)) }

    static func meta(id: String) -> Meta? { all.first { $0.id == id } }

    /// True when `workspaceID` names a workspace that ships in the app bundle. A bundled
    /// workspace's Remove is a **tombstone** (CFM-R17-FIX-1), not an outright delete — the
    /// user can bring it back — so the callers word its dialog and wire its restore path
    /// differently from a user workspace.
    static func isBundled(_ workspaceID: String) -> Bool { ids.contains(workspaceID) }

    /// Materialise every bundled workspace whose id is **not** in `removed` into
    /// `workspaces/<id>/`. Idempotent (`prepare` only writes missing files); a tombstoned id
    /// is left absent so a removed bundled workspace does not resurrect on the next reload
    /// (CFM-R17-FIX-1).
    static func prepareAll(into workspace: FlowWorkspace, removed: Set<String> = [],
                           bundle: Bundle = .main) {
        for meta in all where !removed.contains(meta.id) {
            try? prepare(meta, workspace: workspace, bundle: bundle)
        }
    }

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

/// CFM-R17-FIX-1 — the ids of bundled workspaces the user has removed, persisted in
/// `UserDefaults`. `AppState.reloadWorkspaces()` passes this set to `prepareAll` so a
/// removed bundled workspace stays removed across launches; "Restore bundled workspaces"
/// clears it. Stale ids (a workspace that no longer ships) are dropped on read and write.
nonisolated enum BundledWorkspaceTombstones {
    static let defaultsKey = "removedBundledWorkspaceIDs"

    static func load(from defaults: UserDefaults = .standard) -> Set<String> {
        let stored = (defaults.array(forKey: defaultsKey) as? [String]) ?? []
        return Set(stored).intersection(BundledWorkspaces.ids)
    }

    static func save(_ ids: Set<String>, to defaults: UserDefaults = .standard) {
        let live = ids.intersection(BundledWorkspaces.ids)   // never persist a stale id
        if live.isEmpty {
            defaults.removeObject(forKey: defaultsKey)
        } else {
            defaults.set(live.sorted(), forKey: defaultsKey)
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
