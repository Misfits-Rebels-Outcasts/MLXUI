import CoreGraphics
import Foundation

// FLUX.2 Klein 4B text -> image stage (MP7-4). Wraps `Flux2KleinEngine` (MP7-3h): the per-run settings (size, steps, seed, quantize) ride on the stage because
// `StageConfig` has no `quantize` field and the task says not to add one without need (RIFE's options ride on its stage for the same reason). `seed` / `width` /
// `height` / `steps` also arrive through `StageConfig` when a flow row names them (`Flux2KleinSDK.makeStage`).
//
// The Local Server never serves this model and nothing here registers it there.

/// Where a run is, for the Run UI's progress line. Derived from the engine's lifecycle events, never from timing.
nonisolated enum Flux2KleinPhase: Equatable, Sendable {
    case loadingTextEncoder
    case encoding
    case loadingImageModel
    case denoising(step: Int, of: Int)
    case loadingDecoder
    case decoding

    /// The phase a lifecycle event starts, if it starts one. `steps` is the request's step count.
    static func phase(after event: Flux2KleinLifecycleEvent, steps: Int) -> Flux2KleinPhase? {
        switch event {
        case let .cacheLimitApplied(component, _):         // the first event of every stage, before its component is read from disk
            switch component {
            case .textEncoder: return .loadingTextEncoder
            case .transformer: return .loadingImageModel
            case .vae: return .loadingDecoder
            }
        case let .loaded(component):
            switch component {
            case .textEncoder: return .encoding
            case .transformer: return .denoising(step: 0, of: steps)
            case .vae: return .decoding
            }
        case let .denoiseStep(index): return .denoising(step: index + 1, of: steps)
        case .embeddingsEvaluated, .latentsEvaluated, .released, .cacheCleared: return nil
        }
    }

    /// 0...1 for the progress bar: the encoder is about 6% of a 1024 x 1024 run, the DiT (load plus 4 steps) about 86%, the decoder about 5% (MP7-3h, 36.9 s warm).
    var fraction: Double {
        switch self {
        case .loadingTextEncoder: return 0.01
        case .encoding: return 0.04
        case .loadingImageModel: return 0.07
        case let .denoising(step, steps): return 0.12 + 0.80 * Double(min(max(step, 0), max(steps, 1))) / Double(max(steps, 1))
        case .loadingDecoder: return 0.94
        case .decoding: return 0.96
        }
    }

    var label: String {
        switch self {
        case .loadingTextEncoder: return "Loading the text encoder…"
        case .encoding: return "Reading the prompt…"
        case .loadingImageModel: return "Loading the image model…"
        case let .denoising(step, steps): return step == 0 ? "Starting to denoise (\(steps) steps)…" : "Denoising: step \(step) of \(steps)"
        case .loadingDecoder: return "Loading the decoder…"
        case .decoding: return "Decoding the picture…"
        }
    }
}

/// What a run needs beyond the prompt. Defaults are the curated manifest's (1024 x 1024, 4 steps, quantize 8); `seed == nil` means "pick one at run time".
nonisolated struct Flux2KleinRunSettings: Sendable, Equatable {
    static let defaultSize = 1024
    static let defaultSteps = 4
    static let defaultQuantize = 8

    var width = Flux2KleinRunSettings.defaultSize
    var height = Flux2KleinRunSettings.defaultSize
    var steps = Flux2KleinRunSettings.defaultSteps
    var seed: UInt64?
    var quantize = Flux2KleinRunSettings.defaultQuantize

    func request(prompt: String, seed concreteSeed: UInt64, row: String) -> Flux2KleinRequest {
        Flux2KleinRequest(prompt: prompt, width: width, height: height, steps: steps, seed: concreteSeed, quantize: quantize, row: row)
    }

    /// A random seed that fits a 32-bit key (what the Run UI shows and the user can type back).
    static func randomSeed() -> UInt64 { UInt64.random(in: 0 ..< (1 << 32)) }
}

nonisolated struct Flux2KleinStage: PipelineStage {
    /// How a row is named in refusals ("<row>: FLUX.2 Klein 4B runs at quantize 4 or 8 ..."). The engine's default label is the model's own name, which doubles it in a
    /// sentence ("FLUX.2 Klein 4B: FLUX.2 Klein 4B runs ..."), so a stage always carries a real label: the flow row's ("Row 3") where a row builds it, "This image" in the Run UI.
    static let runWindowLabel = "This image"

    let id: String
    let name: String
    let settings: Flux2KleinRunSettings
    let rowLabel: String
    var accepts: MediaKind { .text }
    var produces: MediaKind { .image }

    /// Builds the engine for a run; the observer is how phases reach `onPhase`. Production: the installed folder. Tests: the tiny stage source.
    typealias EngineFactory = @Sendable (_ observer: @escaping @Sendable (Flux2KleinLifecycleEvent) -> Void) throws -> Flux2KleinEngine

    private let makeEngine: EngineFactory
    /// Called, in order, as the run moves through its stages (the Run UI's "Denoising: step 2 of 4"). Not called from the main actor.
    private let onPhase: (@Sendable (Flux2KleinPhase) -> Void)?

    init(id: String, name: String, settings: Flux2KleinRunSettings = Flux2KleinRunSettings(), rowLabel: String = Flux2KleinStage.runWindowLabel,
         onPhase: (@Sendable (Flux2KleinPhase) -> Void)? = nil, engine: @escaping EngineFactory) {
        self.id = id
        self.name = name
        self.settings = settings
        self.rowLabel = rowLabel
        self.onPhase = onPhase
        self.makeEngine = engine
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        guard case let .text(prompt) = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
        let seed = settings.seed ?? Flux2KleinRunSettings.randomSeed()
        let request = settings.request(prompt: prompt, seed: seed, row: rowLabel)
        let steps = request.steps
        let onPhase = self.onPhase
        @Sendable func report(_ phase: Flux2KleinPhase) {
            progress(phase.fraction)
            onPhase?(phase)
        }
        // Everything the engine would refuse, refused before the first byte of weights is read (and before a seed is spent).
        try Flux2KleinEngine.validate(request, budgetBytes: Flux2KleinMemoryPlan.defaultBudgetBytes)
        try Task.checkCancellation()
        let engine = try makeEngine { event in
            if let phase = Flux2KleinPhase.phase(after: event, steps: steps) { report(phase) }
        }
        let result = try await engine.generate(request)
        progress(1.0)
        return .image(ImageMedia(cgImage: result.image))
    }
}

extension Flux2KleinStage {
    /// Stage for a catalog model id, backed by the installed model folder (the A2 pattern: nothing is downloaded).
    init(modelID: String, settings: Flux2KleinRunSettings = Flux2KleinRunSettings(), rowLabel: String = Flux2KleinStage.runWindowLabel,
         onPhase: (@Sendable (Flux2KleinPhase) -> Void)? = nil) {
        self.init(id: "flux2klein.\(modelID)", name: "FLUX.2 Klein 4B (\(modelID))", settings: settings, rowLabel: rowLabel, onPhase: onPhase) { observer in
            let directory = ModelStore.shared.directory(forModelID: modelID)
            guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("transformer").path) else {
                throw StageError.modelNotInstalled(id: modelID)
            }
            return Flux2KleinEngine(stages: Flux2KleinInstalledStages(modelDirectory: directory), observer: observer)
        }
    }
}
