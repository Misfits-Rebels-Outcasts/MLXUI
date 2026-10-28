import Foundation

/// The `PipelineStage` the module template requires. RIFE is a **video → video** transform and `Media` has no video kind, so this
/// stage cannot do the real work: the Run UI (RIFE-4) drives `RIFEVideoPipeline` directly with a file URL and its own options (2×/4×,
/// scale). It exists so `ModelSDK.makeStage` has something honest to return, and it refuses a direct run with a plain sentence rather
/// than pretend an `image → image` transform. Nothing in AI Workflows names the kind (RIFE-W skipped, owner ruling 2026-10-28); WanVideo's
/// `text → image` first-frame stage is the nearest precedent. Implementer's call, pending owner confirmation.
nonisolated struct RIFEStage: PipelineStage {
    let id: String
    let name: String
    /// The installed folder (`config.json` + `model.safetensors`) the video pipeline loads from.
    let modelDirectory: URL
    var accepts: MediaKind { .image }
    var produces: MediaKind { .image }

    init(modelID: String) {
        self.id = "rife.\(modelID)"
        self.name = "RIFE 4.25"
        self.modelDirectory = RIFEEngine.installedModelDirectory(id: modelID)
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        throw RIFEVideoError.needsWholeVideo
    }
}
