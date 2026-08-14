import Foundation

/// `ModelSDK` for SAM3 image segmentation (SA-AM4).
///
/// Claims `.segmentation` runner-kind entries whose id contains `"sam"`.  This is
/// intentionally narrow so that any future non-SAM segmentation model (e.g. a
/// SegFormer port) can register its own SDK without a priority conflict.
nonisolated struct SegmentAnythingSDK: ModelSDK {
    let id = "segment-anything"

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .segmentation else { return .no }
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        guard haystack.contains("sam") else { return .no }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        SegmentAnythingStage(modelID: model.id)
    }
}
