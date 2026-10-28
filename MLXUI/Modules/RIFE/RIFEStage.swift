import Foundation

/// `text → text` stage backed by `RIFEVideoPipeline`: the input is the **path of a video file**, the output is the **path of the new
/// MP4**. `Media` has no video kind, so a file path is how a video crosses the stage boundary (WanVideo's `text → image` first-frame
/// stage is the nearest precedent). Making it a real stage, and not a stub, is what lets the Run screen book memory through
/// `LeasingStage` like every other model. Options (2× or 4×, scale) ride on the stage because `StageConfig` has no field for
/// them and nothing in AI Workflows names this kind (RIFE-W skipped, owner ruling 2026-10-28). Implementer's call, pending owner
/// confirmation.
nonisolated struct RIFEStage: PipelineStage {
    let id: String
    let name: String
    /// The installed folder (`config.json` + `model.safetensors`).
    let modelDirectory: URL
    let options: RIFEVideoOptions
    /// Where the interpolated MP4 is written (created by the run; an existing file is replaced).
    let outputURL: URL
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    init(modelDirectory: URL, options: RIFEVideoOptions = RIFEVideoOptions(), outputURL: URL = RIFEStage.temporaryOutputURL()) {
        self.id = "rife.\(modelDirectory.lastPathComponent)"
        self.name = "RIFE 4.25"
        self.modelDirectory = modelDirectory
        self.options = options
        self.outputURL = outputURL
    }

    init(modelID: String, options: RIFEVideoOptions = RIFEVideoOptions(), outputURL: URL = RIFEStage.temporaryOutputURL()) {
        self.init(modelDirectory: RIFEEngine.installedModelDirectory(id: modelID), options: options, outputURL: outputURL)
    }

    static func temporaryOutputURL() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("rife-\(UUID().uuidString).mp4")
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case let .text(path) = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
        progress(0.01)
        try await RIFEVideoPipeline.run(input: URL(fileURLWithPath: path), output: outputURL, modelDirectory: modelDirectory,
                                        options: options, progress: progress)
        return .text(outputURL.path)
    }
}
