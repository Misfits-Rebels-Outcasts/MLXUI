import Foundation
import Observation
import CoreGraphics

/// A decoded thumbnail crossing from a background task to the main actor.
nonisolated struct RailThumbnail: @unchecked Sendable {
    let image: CGImage
}

/// KR-3 — the kind rail's per-run preview cache, held by `FlowRunSession`. Previews are built
/// lazily, one row at a time, off the main actor, from the output the session **already holds**
/// (`FlowRunSession.outputs`); the store never runs, loads or downloads anything. It is cleared
/// when a new run starts and per row when that row finishes again, so a preview can't outlive the
/// result it was made from.
@Observable
final class RailPreviewStore {
    private(set) var previews: [UUID: RailPreview] = [:]
    private(set) var thumbnails: [UUID: CGImage] = [:]
    /// Bumped by `clear()`; part of the views' task key so they re-request after a new run.
    private(set) var generation = 0
    /// Bumped per row by `invalidate(_:)`.
    private(set) var versions: [UUID: Int] = [:]
    /// How many previews have been built — the "no preview work" tests read this.
    @ObservationIgnored private(set) var buildCount = 0
    @ObservationIgnored private var attempted: Set<UUID> = []

    /// Forget everything — a new run is starting.
    func clear() {
        previews = [:]
        thumbnails = [:]
        attempted = []
        versions = [:]
        generation += 1
    }

    /// Forget one row — it has a fresh result (or was edited).
    func invalidate(_ rowID: UUID) {
        previews[rowID] = nil
        thumbnails[rowID] = nil
        attempted.remove(rowID)
        versions[rowID, default: 0] += 1
    }

    /// Build `rowID`'s preview from `asset`, once. A no-op when the setting is off (`enabled`
    /// false: nothing is decoded), when there is no result, or when it was already built.
    func request(rowID: UUID, asset: Asset?, enabled: Bool,
                 probe: RailPreviewProbe = .live) async {
        guard enabled, let asset, !attempted.contains(rowID) else { return }
        attempted.insert(rowID)
        buildCount += 1
        let generation = generation, version = versions[rowID, default: 0]
        let built = await Task.detached(priority: .utility) { () -> (RailPreview?, RailThumbnail?) in
            let preview = RailPreviews.preview(for: asset, probe: probe)
            var thumbnail: RailThumbnail?
            if case .image(let url)? = preview,
               let full = ImageLoader.decodedCGImage(fromSecurityScoped: url) {
                // 16-pt chip: a 32 × 32 px budget covers @2x.
                thumbnail = RailThumbnail(image: ImageLoader.downscaled(full, maxPixels: 32 * 32))
            }
            return (preview, thumbnail)
        }.value
        // A new run or a newer result for this row landed while we worked: drop ours.
        guard generation == self.generation, version == versions[rowID, default: 0] else { return }
        previews[rowID] = built.0
        thumbnails[rowID] = built.1?.image
    }
}

/// What a row's preview task re-runs on: a new run, a fresh result for this row, or the result
/// arriving at all.
nonisolated struct RailPreviewKey: Hashable {
    let rowID: UUID
    let generation: Int
    let version: Int
    let hasResult: Bool
}
