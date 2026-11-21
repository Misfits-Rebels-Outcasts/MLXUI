import Foundation

/// What a flow row / Run window can set for an ACE-Step run. Only duration and seed: steps, size and quantize are refused by name in `AceStepSDK`.
nonisolated struct AceStepRunSettings: Sendable, Hashable {
    /// SPEC-Q (placeholder, MC5-W / S2 owns the real answer): `StageConfig` has no duration field, so a flow row gets this default (seconds).
    static let flowDefaultDuration = 30.0
    var duration: Double = AceStepRunSettings.flowDefaultDuration
    var seed: UInt64?
}

/// `text -> audio` stage for ACE-Step 1.5 turbo (text2music, instrumental unless a lyrics-taking caller uses the engine directly). The generator is
/// injectable so the stage's contract is testable without a model. `produces` is the repo's mono `AudioBuffer` (48 kHz mixdown of the stereo result);
/// the Run UI (MC5-3j) calls the engine for the stereo WAV.
nonisolated struct AceStepStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .audio }
    let settings: AceStepRunSettings

    private let generate: @Sendable (AceStepRequest, @Sendable @escaping (AceStepPhase) -> Void) async throws -> AceStepAudio

    init(id: String, name: String, settings: AceStepRunSettings = AceStepRunSettings(),
         generate: @escaping @Sendable (AceStepRequest, @Sendable @escaping (AceStepPhase) -> Void) async throws -> AceStepAudio) {
        self.id = id
        self.name = name
        self.settings = settings
        self.generate = generate
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case let .text(prompt) = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
        let seed = settings.seed ?? UInt64.random(in: 0 ..< (1 << 32))
        let request = AceStepRequest(caption: prompt, duration: settings.duration, seed: seed)
        progress(0)
        let audio = try await generate(request) { progress($0.fraction) }
        progress(1)
        return .audio(audio.monoMixdown)
    }
}

extension AceStepStage {
    /// Stage for an installed model id (the A2 pattern: only files `InstallManager` already downloaded; nothing is fetched).
    init(modelID: String, settings: AceStepRunSettings = AceStepRunSettings()) {
        self.init(id: "acestep.\(modelID)", name: "ACE-Step 1.5 (\(modelID))", settings: settings) { request, onPhase in
            let directory = ModelStore.shared.directory(forModelID: modelID)
            guard FileManager.default.fileExists(atPath: directory.path) else { throw StageError.modelNotInstalled(id: modelID) }
            var engine = AceStepEngine(directory: directory)
            engine.observer = { event in if case .phase(let p) = event { onPhase(p) } }
            return try await engine.generate(request).audio
        }
    }
}
