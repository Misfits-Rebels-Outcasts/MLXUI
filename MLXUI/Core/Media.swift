import CoreGraphics

// The Core media contracts are concurrency-neutral value types that flow between
// pipeline stages running off the main actor. The app target defaults to main-actor
// isolation, so each type opts out explicitly with `nonisolated`.

/// A typed payload passed between pipeline stages.
///
/// The pipeline is assembled at runtime (the user chains models), so connections
/// are validated by `kind` rather than by compile-time generics — see
/// `Design/pipeline-stage-sketch.md`. This is the shared `Core` contract every
/// stage (LLM, ASR, TTS, tools) consumes and produces.
nonisolated enum Media: Sendable {
    case text(String)
    case audio(AudioBuffer)
    case image(ImageMedia)
    case embedding([[Float]])      // batch of vectors

    /// The "shape" of this value — used to validate stage connections without
    /// unwrapping the payload.
    var kind: MediaKind {
        switch self {
        case .text:      .text
        case .audio:     .audio
        case .image:     .image
        case .embedding: .embedding
        }
    }
}

/// The shape of a `Media` value, independent of its payload. Used to check that
/// one stage produces what the next one accepts.
nonisolated enum MediaKind: String, Sendable, CaseIterable {
    case text, audio, image, embedding
}

/// Mono PCM in [-1, 1]. ASR consumes 16 kHz; Kokoro TTS produces 24 kHz — the
/// sample rate travels with the samples so downstream code never has to guess.
nonisolated struct AudioBuffer: Sendable, Equatable {
    var samples: [Float]
    var sampleRate: Int

    init(samples: [Float], sampleRate: Int) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

/// `CGImage` is thread-safe in practice but not marked `Sendable`, so we vouch
/// for it explicitly here rather than leaking `@unchecked` across the codebase.
nonisolated struct ImageMedia: @unchecked Sendable {
    let cgImage: CGImage

    init(cgImage: CGImage) {
        self.cgImage = cgImage
    }
}
