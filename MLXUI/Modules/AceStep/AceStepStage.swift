import Foundation

/// What a flow row / Run window can set for an ACE-Step run. Only duration and seed: steps, size and quantize are refused by name in `AceStepSDK`.
nonisolated struct AceStepRunSettings: Sendable, Hashable {
    /// SPEC-Q (placeholder, MC5-W / S2 owns the real answer): `StageConfig` has no duration field, so a flow row gets this default (seconds).
    static let flowDefaultDuration = 30.0
    var duration: Double = AceStepRunSettings.flowDefaultDuration
    var seed: UInt64?
    /// SPEC-Q251 (MC5-W): the flow row's `steps=`, 1...20 per the MLXUI-authored manifest; nil = the turbo default 8.
    var steps: Int?
    static let stepsRange = 1 ... 20
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
        let base = AceStepRequest(caption: prompt, duration: settings.duration, seed: seed)
        var request = base
        if let steps = settings.steps { request.steps = steps }
        progress(0)
        let audio = try await generate(request) { progress($0.fraction) }
        progress(1)
        return .audio(audio.monoMixdown)
    }
}

extension AceStepStage {
    /// Stage for an installed model id (the A2 pattern: only files `InstallManager` already downloaded; nothing is fetched).
    init(modelID: String, settings: AceStepRunSettings = AceStepRunSettings(), directory: URL? = nil) {
        self.init(id: "acestep.\(modelID)", name: "ACE-Step 1.5 (\(modelID))", settings: settings, generate: AceStepEngine.generator(modelID: modelID, directory: directory))
    }
}

extension AceStepEngine {
    /// The one place a run's engine is built from an installed model folder (the Run screen and the flow stage both use it). `directory` nil = the app's store
    /// (`ModelStore.shared`); the AM-T smoke suite passes the checkpoint folder it was given, so it runs this same code on those weights.
    static func generator(modelID: String, directory: URL? = nil)
        -> @Sendable (AceStepRequest, @Sendable @escaping (AceStepPhase) -> Void) async throws -> AceStepAudio {
        { request, onPhase in
            let folder = directory ?? ModelStore.shared.directory(forModelID: modelID)
            guard FileManager.default.fileExists(atPath: folder.path) else { throw StageError.modelNotInstalled(id: modelID) }
            var engine = AceStepEngine(directory: folder)
            engine.observer = { event in if case .phase(let p) = event { onPhase(p) } }
            return try await engine.generate(request).audio
        }
    }
}
