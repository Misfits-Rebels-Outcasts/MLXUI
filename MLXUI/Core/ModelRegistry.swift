import SwiftUI

/// How strongly a `ModelSDK` claims a given `ModelEntry`. The registry resolves a model
/// to the module whose SDK claims it most strongly. `Comparable` compares `rawValue`, so
/// `.exact > .specific > .generic > .no`.
/// See `Design/aisdk-aiui-architecture.md`.
enum ClaimScore: Int, Comparable, Sendable {
    case no, generic, specific, exact

    static func < (lhs: ClaimScore, rhs: ClaimScore) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// The runnable half of a model module: decides whether it handles a model and, if so,
/// turns it into a `PipelineStage`. `nonisolated` + `Sendable` so a resolved stage can be
/// built off the main actor and captured in a run `Task`.
///
/// Simplified vs. the doc: no `files:` param — Whisper needs no installed files, and that
/// type doesn't exist. `config: StageConfig` already exists (`Core/StageFactory.swift`).
protocol ModelSDK: Sendable {
    /// Stable module id, e.g. `"whisper"`.
    var id: String { get }
    /// How strongly this SDK claims `model` (`.no` means "not mine").
    func claim(_ model: ModelEntry) -> ClaimScore
    /// Build a runnable stage for `model`. Throws `StageError` when it can't run.
    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage
}

/// The UI half of a model module: produces the SwiftUI run surface for a resolved model +
/// stage. `makeRunView` is `@MainActor` (it builds views); `claim` mirrors the SDK so the
/// registry could resolve UI independently if ever needed.
protocol ModelUI {
    func claim(_ model: ModelEntry) -> ClaimScore
    @MainActor func makeRunView(for model: ModelEntry, stage: any PipelineStage) -> AnyView
}

/// Static metadata describing a module — surfaced in about/diagnostics and used to record
/// the backing package and its license obligations.
struct ModelModuleDescriptor: Sendable {
    let id: String
    let displayName: String
    let modalities: [String]
    let modelTypes: [ModelType]
    let backingPackage: String
    let packageLicense: String
    let maintainers: [String]
    let notes: String
}

/// A module bundles an SDK + UI + descriptor and registers itself into the registry.
protocol ModelModule {
    static var descriptor: ModelModuleDescriptor { get }
    @MainActor static func register(into registry: ModelRegistry)
}

/// Registry of model modules. Stores `(sdk, ui, descriptor)` in registration order and
/// resolves a `ModelEntry` to the best-claiming module. `@MainActor` so it can hold a
/// non-`Sendable` `any ModelUI` without raising concurrency diagnostics.
@MainActor
final class ModelRegistry {
    /// A module resolved for a specific model: its SDK, UI, and descriptor.
    struct Resolved {
        let sdk: any ModelSDK
        let ui: any ModelUI
        let descriptor: ModelModuleDescriptor
    }

    private var modules: [Resolved] = []

    func add(sdk: any ModelSDK, ui: any ModelUI, descriptor: ModelModuleDescriptor) {
        modules.append(Resolved(sdk: sdk, ui: ui, descriptor: descriptor))
    }

    /// The module whose SDK claims `model` most strongly, or `nil` if none claim it.
    /// Replaces the running best only on a **strictly greater** score, so a tie keeps the
    /// earliest-registered module.
    func bestModule(for model: ModelEntry) -> Resolved? {
        var best: Resolved?
        var bestScore: ClaimScore = .no
        for module in modules {
            let score = module.sdk.claim(model)
            if score == .no { continue }
            if best == nil || score > bestScore {
                best = module
                bestScore = score
            }
        }
        return best
    }
}
