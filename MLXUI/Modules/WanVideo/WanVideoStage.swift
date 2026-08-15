import Foundation
import CoreGraphics

/// `text → image` (first frame) stage for WAN 2.1 T2V. Full video export is AM5.
nonisolated struct WanVideoStage: PipelineStage {
    let id:   String
    let name: String
    var accepts:  MediaKind { .text }
    var produces: MediaKind { .image }

    private let generate: @Sendable (String, @Sendable (Double) -> Void) async throws -> CGImage

    init(
        id:   String,
        name: String,
        generate: @escaping @Sendable (String, @Sendable (Double) -> Void) async throws -> CGImage
    ) {
        self.id       = id
        self.name     = name
        self.generate = generate
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case let .text(prompt) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        progress(0.05)
        let image = try await generate(prompt, progress)
        progress(1.0)
        return .image(ImageMedia(cgImage: image))
    }
}

extension WanVideoStage {
    init(modelID: String) {
        self.init(id: "wanvideo.\(modelID)", name: "WAN 2.1 (\(modelID))") { prompt, progress in
            try await WanVideoEngine.generate(prompt: prompt, modelID: modelID, progress: progress)
        }
    }
}
