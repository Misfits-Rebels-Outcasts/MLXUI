import Testing
import CoreGraphics
@testable import MLXUI

/// Covers the `Core` media contracts (M1): `Media.kind` routing, `MediaKind`
/// completeness, `AudioBuffer` round-trip, and the `PipelineStage.require` guard.
/// These are the foundation every stage is validated against (RSI/evals/eval-plan.md G2).
struct CoreMediaTests {

    // MARK: Media.kind

    @Test func mediaKindMatchesPayloadForEveryCase() throws {
        #expect(Media.text("hi").kind == .text)
        #expect(Media.audio(AudioBuffer(samples: [0], sampleRate: 16_000)).kind == .audio)
        #expect(Media.embedding([[0.0]]).kind == .embedding)

        let cg = try makeImage()
        #expect(Media.image(ImageMedia(cgImage: cg)).kind == .image)
    }

    // MARK: MediaKind

    @Test func mediaKindHasAllFourCasesWithStableRawValues() {
        #expect(MediaKind.allCases.count == 4)
        #expect(Set(MediaKind.allCases.map(\.rawValue)) == ["text", "audio", "image", "embedding"])
    }

    // MARK: AudioBuffer round-trip

    @Test func audioBufferRoundTripsThroughMedia() throws {
        let buffer = AudioBuffer(samples: [-1.0, 0.0, 0.25, 1.0], sampleRate: 24_000)
        let media = Media.audio(buffer)
        guard case let .audio(unwrapped) = media else {
            Issue.record("Media.audio did not unwrap to an AudioBuffer")
            return
        }
        #expect(unwrapped == buffer)
        #expect(unwrapped.samples == [-1.0, 0.0, 0.25, 1.0])
        #expect(unwrapped.sampleRate == 24_000)
    }

    // MARK: PipelineStage.require

    @Test func requirePassesWhenKindMatches() throws {
        let stage = PassthroughStage()
        try stage.require(.text("ok"), .text)   // must not throw
    }

    @Test func requireThrowsKindMismatchOnWrongKind() {
        let stage = PassthroughStage()
        #expect(throws: StageError.self) {
            try stage.require(.audio(AudioBuffer(samples: [], sampleRate: 16_000)), .text)
        }
    }

    // MARK: helpers

    /// A 1×1 image so `ImageMedia` can be exercised without bundled assets.
    /// Uses `#require` rather than force-unwrap to honor the project style rule.
    private func makeImage() throws -> CGImage {
        let space = CGColorSpaceCreateDeviceGray()
        let ctx = try #require(CGContext(data: nil, width: 1, height: 1, bitsPerComponent: 8,
                                         bytesPerRow: 1, space: space,
                                         bitmapInfo: CGImageAlphaInfo.none.rawValue))
        return try #require(ctx.makeImage())
    }
}

/// Minimal stage used only to reach the `require` extension and `StageError` in tests.
/// `nonisolated` because `PipelineStage: Sendable` and this test module defaults to
/// main-actor isolation.
private nonisolated struct PassthroughStage: PipelineStage {
    let id = "passthrough"
    let name = "Passthrough"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }

    func run(
        _ input: Media,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> Media {
        try require(input, .text)
        progress(1.0)
        return input
    }
}
