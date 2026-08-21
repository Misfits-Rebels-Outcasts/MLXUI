import Foundation

/// The fourteen asset kinds — ported verbatim from `catflow-mlx/src/catflow/core/kinds.py`
/// (`Kind`, 14 cases). These are the type system `catflow check` runs on; FlowKit's
/// `Asset`/`Shape` are built from them. `Media`/`MediaKind` (Core/Media.swift) stay
/// untouched; `Asset` sits beside them and the bridge between the two worlds lands in
/// CFM-R2-3.
///
/// Notes carried over from the Python docstrings:
/// - `context` (the journal) and `occurrence` (a trigger's fired event) were added at v0.7.
/// - `model` is the one categorical exception — pipeline-only **by doctrine** (it is not
///   *what* a row transforms but *how*), so a `.cat` forbids it (E713).
/// - `latent` is an ordinary kind (pipeline-only by accident of which rows exist), checked
///   by latent *space*, not by the producing model.
nonisolated enum Kind: String, Sendable, CaseIterable {
    case text, audio, image, video, file, folder, index, vector, table, status,
         context, occurrence, model, latent
}
