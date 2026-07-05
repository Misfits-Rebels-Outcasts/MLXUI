import Foundation

/// A sink tool node (`text → text`): writes the upstream text to `url` as UTF-8, then
/// passes it through unchanged so the value stays inspectable downstream. The audio
/// counterpart (`SaveWAV`) lands with Tools_Audio in M5.
nonisolated struct SaveTextStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    private let url: URL

    init(url: URL, id: String = "tool.save_text", name: String = "Save Text") {
        self.url = url
        self.id = id
        self.name = name
    }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        guard case let .text(text) = input else {
            throw StageError.kindMismatch(expected: .text, got: input.kind)
        }
        do {
            try text.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw StageError.engineFailure(stage: name, underlying: error)
        }
        progress(1.0)
        return input
    }
}
