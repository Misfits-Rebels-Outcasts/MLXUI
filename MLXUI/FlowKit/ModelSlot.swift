import Foundation

/// MS-1 — a model MLXUI can run, whether or not it is a download. A `browser.json` row
/// (`.cataloged`) is everything the app has run until now; `.system` (Phase AFM, e.g. Apple
/// Foundation Models) and `.provider` (Phase RM, e.g. Claude/GPT/DeepSeek) are both
/// non-downloadable and answer the same three questions instead of forcing a synthetic
/// `ModelEntry` into `browser.json`'s world (backlog §1 — the trap: `ramGB: 0` + a fake id
/// compiles fine and is wrong in a way that looks right: the install planner would plan a
/// download for something with no files, the RAM gate would wave through a model whose real
/// constraint is "is Apple Intelligence on?", and Browse would show a card you can't browse).
///
/// Deliberately **no `ramGB` property** here. RAM is `.cataloged`'s business alone
/// (`modelEntry?.ramGB`, compared against *this Mac's* total at the point of use) — hoisting
/// it back to the top, so every case could answer the same question uniformly, is precisely
/// how this refactor fails: it would compile for `.system`/`.provider` and be meaningless.
nonisolated enum ModelSlot: Sendable, Equatable, Identifiable {
    case cataloged(ModelEntry)
    case system(SystemModelRef)        // Phase AFM
    case provider(ProviderModelRef)    // Phase RM

    var id: String {
        switch self {
        case .cataloged(let entry): return entry.id
        case .system(let ref): return ref.id
        case .provider(let ref): return ref.id
        }
    }

    /// The `.cat`-visible name — `TaskModels.displayName` (the bridge display when bridged,
    /// else the catalog's own name) for `.cataloged`, so this is byte-identical to what the
    /// picker wrote before MS.
    var displayName: String {
        switch self {
        case .cataloged(let entry): return TaskModels.displayName(for: entry)
        case .system(let ref): return ref.displayName
        case .provider(let ref): return ref.displayName
        }
    }

    /// What a Model-menu row prints on its trailing edge. `.cataloged`'s is its RAM
    /// footprint — unchanged from the pre-MS picker, which showed `ramGB` here, never
    /// download size (the total download size is the *section header's* business, summed
    /// separately). `.system`/`.provider` supply their own note once they exist.
    var resourceNote: String? {
        switch self {
        case .cataloged(let entry): return String(format: "%.1f GB", entry.ramGB)
        case .system(let ref): return ref.resourceNote
        case .provider(let ref): return ref.resourceNote
        }
    }

    /// The underlying catalog entry, when this slot is a download — the escape hatch for
    /// code that genuinely needs `ramGB`/`hfModelId`/install tracking (the picker's RAM-fit
    /// dimming, `RealExecutor`'s MLX load path). `nil` for `.system`/`.provider`: exposing the
    /// raw entry when one exists is not the same trap as hoisting `ramGB` onto every case.
    var modelEntry: ModelEntry? {
        if case .cataloged(let entry) = self { return entry }
        return nil
    }

    /// Whether this slot can run right now. `.cataloged`'s answer needs `installedModelIDs`
    /// (install tracking lives in `AppState`, never on `ModelEntry` itself, so it can't be a
    /// bare stored property); `.system`/`.provider` carry their own precomputed answer —
    /// Apple Intelligence's availability, a Keychain check — which is Phase AFM/RM's job to
    /// compute, not this one's.
    func readiness(installedModelIDs: Set<String> = []) -> Readiness {
        switch self {
        case .cataloged(let entry):
            return installedModelIDs.contains(entry.id) ? .ready : .needsDownload(gb: entry.downloadSizeGB)
        case .system(let ref): return ref.readiness
        case .provider(let ref): return ref.readiness
        }
    }
}

/// MS-1 — the answer to "can this slot run right now, and if not, what does it need?" — the
/// picker's enabled/disabled state, the run sheet's buckets, and the refusal banner all read
/// this. Deliberately excludes anything RAM-shaped: whether a *cataloged* model fits *this
/// Mac* is a separate, per-render comparison (`ModelSlot.modelEntry?.ramGB` vs. the detected
/// total), never baked into a static readiness value here — a model that doesn't fit is still
/// `.ready`/`.needsDownload` in principle; the picker just disables that particular row.
nonisolated enum Readiness: Sendable, Equatable {
    case ready
    case needsDownload(gb: Double)
    case needsSetup(reason: String, action: SetupAction?)
    case unavailable(reason: String)
}

/// MS-1 — the fix-it action a `.needsSetup` readiness or a `.notRunnable` resolution can
/// attach, so the banner naming the problem can also offer the one button that solves it (the
/// owner's fifth ask, MS-4). Nothing produces a non-nil one until Phase AFM
/// (`.enableAppleIntelligence`) or Phase KEY/RM/WS (`.openSettings`) — MS only builds the
/// plumbing and tests it with a fixture, so those phases are one line each.
nonisolated enum SetupAction: Sendable, Equatable {
    case openSettings(SettingsPane)
    case enableAppleIntelligence
    case installModel(ModelEntry)
}

/// MS-1 — which section of Settings a `.openSettings` action should land on.
/// `Views/Settings/SettingsView.swift` has no pane/tab navigation yet (one scrolling `Form`
/// with two sections); wiring an action to an actual scroll/selection is Phase KEY's job (the
/// "Model & Search Providers" section KEY-2 adds). This enum exists now so `SetupAction`
/// compiles and is reviewable; it names no pane no phase has built yet.
nonisolated enum SettingsPane: Sendable, Equatable {
    /// Where a system/provider model's setup would be offered (Phase AFM/RM).
    case models
    /// KEY-2's "Model & Search Providers" section — provider API keys.
    case providers
}

/// Phase AFM's system-model reference (e.g. `apple-foundation @ system`) — a `ModelSlot
/// .system` payload. Shape-only until AFM-1 gives it Apple Foundation Models' actual
/// `SystemLanguageModel.default.availability` mapping; MS only needs the type to exist so
/// `ModelSlot` is a complete, compiling type today, with nothing constructing one yet (empty
/// registry — see `TaskModels`'s `systemModels`).
nonisolated struct SystemModelRef: Sendable, Equatable {
    let id: String
    let displayName: String
    let readiness: Readiness
    let resourceNote: String?
}

/// Phase RM's provider-model reference (e.g. `claude-sonnet @ anthropic`) — a `ModelSlot
/// .provider` payload. Same shape-only status as `SystemModelRef`; RM-1/RM-2 give it real
/// manifests and a Keychain-backed readiness.
nonisolated struct ProviderModelRef: Sendable, Equatable {
    let id: String
    let displayName: String
    let readiness: Readiness
    let resourceNote: String?
}
