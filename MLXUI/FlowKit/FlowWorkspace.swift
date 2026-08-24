import Foundation
import AppKit

/// The working directory for a flow, and the **only** way `.cat` paths resolve.
///
/// Every flow gets a directory at `ModelStore.shared.baseDirectory/flows/<flow-id>/`; every
/// unqualified path in a `.cat` resolves against that directory via `resolve(_:flowID:)`.
/// This is a deliberate divergence from the Python runtime (which resolves relative to the
/// current working directory) — the sandbox requires it, and `SPEC_QUESTIONS.md` records it.
///
/// **Security boundary.** `resolve` is not a convenience: it rejects any path that escapes
/// the flow directory after resolution (`..`, absolute paths, symlinks pointing outside).
/// Bundled gallery inputs are **copied** into the working directory on first open
/// (`prepare(flowID:)`), never read in place — the bundle is read-only and a flow may want
/// to overwrite its own inputs.
///
/// Injectable `root` mirrors `ModelStore(baseDirectory:)` so tests never touch the real
/// Application Support directory (`RSI/policies.md` puts user data off-limits).
nonisolated struct FlowWorkspace: Sendable {
    /// `…/flows/` — the parent of every flow directory. Inject a temp dir in tests.
    let root: URL

    init(root: URL) {
        self.root = root
    }

    /// The app-wide instance, rooted at `ModelStore.shared.flowsDirectory`.
    static let shared = FlowWorkspace(root: ModelStore.shared.flowsDirectory)

    /// A flow's own directory: `root/<flow-id>/`.
    func directory(for flowID: String) -> URL {
        root.appendingPathComponent(flowID, isDirectory: true)
    }

    // MARK: - First-open asset copy

    /// Create the flow directory if absent and copy any bundled input assets in on first
    /// use. **Idempotent**: a second call leaves files the user has since changed alone —
    /// only missing files are copied. Never reads assets in place from the bundle.
    ///
    /// `bundledAssets` is a list of `(sourceName, relativeDestination)` pairs. The Xcode
    /// synchronized root group flattens `MLXUI/Resources/Gallery/**` into the bundle's
    /// `Contents/Resources/` root, so the source name is the flat filename and the
    /// destination is the path the flow expects (e.g. `vacation/photo-01.png` for
    /// `21-PhotoWebPrep`, whose `.cat` reads `vacation/`).
    func prepare(flowID: String, sourceDir: URL?, bundledAssets: [(source: String, destination: String)]) throws {
        let fm = FileManager.default
        let dir = directory(for: flowID)
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)

        guard let sourceDir else { return }
        for asset in bundledAssets {
            let src = sourceDir.appendingPathComponent(asset.source)
            let dest = dir.appendingPathComponent(asset.destination)
            if fm.fileExists(atPath: dest.path) { continue }        // copy only what's missing
            guard fm.fileExists(atPath: src.path) else { continue }  // asset not shipped
            try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.copyItem(at: src, to: dest)
        }
    }

    // MARK: - Path resolution (the security boundary)

    /// Resolve a `.cat`-relative path against the flow directory. Throws `FlowError` on:
    /// absolute paths, `..` traversal, or symlinks whose target escapes the flow directory.
    /// This is the **single** function every path in a `.cat` goes through (CFM-R2-4).
    ///
    /// `URL.resolvingSymlinksInPath()` silently leaves an existing symlink unresolved when
    /// the *final* component doesn't exist yet (a `.cat` path often names a file the flow
    /// will create), so this walks the components and resolves each **existing** one —
    /// a symlink is followed and checked for containment even when the path beyond it
    /// doesn't exist. That is the escape case that matters.
    func resolve(_ relativePath: String, flowID: String) throws -> URL {
        let fm = FileManager.default
        let flowDir = directory(for: flowID)
        let resolvedFlowDir = flowDir.resolvingSymlinksInPath()

        let trimmed = relativePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw FlowWorkspaceError.emptyPath
        }
        // Absolute paths are never allowed — a `.cat` must name something inside its flow.
        guard !(trimmed as NSString).isAbsolutePath else {
            throw FlowWorkspaceError.escapesFlow(relativePath)
        }
        // Reject any `..` component outright (don't clamp).
        let components = trimmed.split(separator: "/").map(String.init)
        guard !components.contains("..") else {
            throw FlowWorkspaceError.escapesFlow(relativePath)
        }

        func contained(_ url: URL) -> Bool {
            let path = url.path
            return path == resolvedFlowDir.path || path.hasPrefix(resolvedFlowDir.path + "/")
        }

        var current = flowDir
        for component in components {
            let next = current.appendingPathComponent(component)
            // A symlink — even a dangling one whose target doesn't exist yet — is refused
            // outright (H6). `fileExists` traverses links, so it reports `false` for a
            // dangling link and the write would later escape the workspace through it; the
            // `.isSymbolicLinkKey` probe reads the link itself and catches both cases.
            if isSymbolicLink(next) {
                throw FlowWorkspaceError.escapesFlow(relativePath)
            }
            // Only resolve components that exist; a symlink that points outside is caught
            // even when the target file beyond it doesn't.
            if fm.fileExists(atPath: next.path) || fm.fileExists(atPath: next.path + "/") {
                let resolved = next.resolvingSymlinksInPath()
                guard contained(resolved) else {
                    throw FlowWorkspaceError.escapesFlow(relativePath)
                }
                current = resolved
            } else {
                current = next
            }
        }
        return current
    }

    /// Whether `url` is a symbolic link, reading the link itself (not its target), so a
    /// dangling link is still detected.
    private func isSymbolicLink(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
    }

    // MARK: - Finder

    /// Reveal the flow's working directory in Finder (`NSWorkspace`).
    func revealInFinder(flowID: String) {
        let dir = directory(for: flowID)
        let fm = FileManager.default
        if !fm.fileExists(atPath: dir.path) {
            try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        NSWorkspace.shared.activateFileViewerSelecting([dir])
    }
}

/// Path-resolution failures. Error voice: one plain sentence implying the fix.
nonisolated enum FlowWorkspaceError: Error, CustomStringConvertible, Equatable {
    case emptyPath
    case escapesFlow(String)

    var description: String {
        switch self {
        case .emptyPath:
            return "A flow path was empty — every row needs a file name to work with."
        case .escapesFlow(let path):
            return "The path '\(path)' escapes the flow's folder — paths must stay inside the flow's working directory."
        }
    }
}
