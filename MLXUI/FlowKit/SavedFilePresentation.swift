import Foundation

/// How the Output tab presents a saved file — a **presentation** fact about the actual file on
/// disk (its real extension, or its task-name fallback), not a `Kind`. `Kind` is ported
/// verbatim from `catflow-mlx/core/kinds.py` and is the type system `check` runs on; it is
/// never extended for display concerns (OV-1, §7 trap 1 in
/// `RSI/DelegateOutputViewerBacklog.md`). `SavedFilePresentation` sits beside it instead.
///
/// `Save Text` writes whatever extension the `.cat` names — 71 shipped rows cover five real
/// file types (`.md`, `.txt`, `.diff`, `.html`, `.otsl`), all classified `Kind.text` alike.
/// `SavedFilePresentation` (`FlowSavedFile.presentation(url:task:)`) is what tells them apart.
nonisolated enum SavedFilePresentation: String, CaseIterable, Sendable {
    case text, web, image, audio, video, pdf, table, folder, other

    /// Whether the Output tab offers a Quick Look button for this presentation (OV-2) — every
    /// case except `.folder`: Quick Look previews a file, and a folder already has its own
    /// "Show in Finder" affordance for browsing what's inside it.
    var allowsQuickLook: Bool { self != .folder }
}
