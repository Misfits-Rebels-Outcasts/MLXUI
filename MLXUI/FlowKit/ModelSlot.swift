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
    ///
    /// MS-FOLLOWUP-1: **required, no default.** A `= []` default silently means "nothing is
    /// installed," so an omitted argument would report every cataloged model as
    /// `.needsDownload` — the same mistake CFM-R14-FIX-3 already fixed in
    /// `TaskAvailability.swift`: "the derived pool is the availability authority, and a
    /// silently-empty catalog would report every model task as unavailable — a wrong answer
    /// that looks correct." A caller with genuinely no install state passes an explicit `[]`,
    /// the same convention `FlowRunner` already follows for `TaskAvailability`.
    func readiness(installedModelIDs: Set<String>) -> Readiness {
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
/// SET-1 (`RSI/DelegateSettingsBacklog.md`) turned Settings into a real `Settings` scene with a
/// `TabView`; `CaseIterable` + `title` + `systemImage` + the `String` raw value are that
/// phase's G2 seam — the raw value is the persisted tab key `SettingsRootView`'s
/// `@AppStorage("settingsPane")` reads/writes (SET-2), so its cases must stay stable strings
/// once shipped. `.privacy` has no tab yet — it arrives in SET-5.
nonisolated enum SettingsPane: String, CaseIterable, Sendable, Equatable {
    /// Where a system/provider model's setup would be offered (Phase AFM/RM), and — per SET-1
    /// §3 — where model files come from and where they live on this Mac.
    case models
    /// The provider API key rows (Phase KEY/RM/WS).
    case providers
    /// SET-2 — what a model may do on this Mac (per-tool on/off, approval, limits). Renamed
    /// from the codebase's "Agent Tools" to "Tools" per OG-5: "Agent" is our word, not the
    /// user's. Grouping and per-tool descriptions are SET-3's job.
    case agentTools
    /// SET-5 — what leaves this Mac, rendered from `RM-6-privacy-disclosure.md` §1. No tab
    /// yet; the case exists now so `SettingsPane.allCases`' order is settled ahead of it.
    case privacy

    /// The tab title, and the pane's own heading inside the window.
    var title: String {
        switch self {
        case .models: return "Models"
        case .providers: return "Providers"
        case .agentTools: return "Tools"
        case .privacy: return "Privacy"
        }
    }

    /// The tab's SF Symbol.
    var systemImage: String {
        switch self {
        case .models: return "shippingbox"
        case .providers: return "key"
        case .agentTools: return "wrench.and.screwdriver"
        case .privacy: return "hand.raised"
        }
    }
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
/// .provider` payload. RM-2: carries the full `CuratedManifest` rather than a handful of
/// flattened fields — dispatch (`ProviderExecutor`) needs `engine`/`baseURL`/`credentials`
/// and settings resolution (`CuratedManifest.resolveEngineSettings`) needs the manifest's
/// own `settings` dict, so re-deriving a second, parallel shape here would just be the
/// same trap `ModelSlot`'s own header warns against in miniature: a stripped-down copy
/// that "happens to work" until dispatch needs a field nobody carried forward.
nonisolated struct ProviderModelRef: Sendable, Equatable {
    let manifest: CuratedManifest
    let readiness: Readiness

    var id: String { manifest.id }
    var displayName: String { manifest.display }
    /// RM-2, verbatim from the backlog: said on the Model menu row, not only in a run-time
    /// refusal — this is a *Human In Control* product, so the picker is the right place to
    /// disclose that picking this row means the row's data leaves the machine.
    var resourceNote: String? { "Runs on \(providerName) — leaves this Mac" }
    /// The ` @ provider` suffix of the display name (`"anthropic"`, `"openai"`, or the
    /// literal `"http://mac-studio.local:8080"` for the LAN case — its own display already
    /// names the endpoint, so this reads correctly without inventing a separate label).
    var providerName: String {
        guard let range = manifest.display.range(of: " @ ") else { return manifest.display }
        return String(manifest.display[range.upperBound...])
    }
}

/// KEY-3 — the Keychain-backed half of `ProviderModelRef.readiness`, built now so RM-2
/// wires it in as one line (`readiness: ProviderCredential.readiness(providerName:)`),
/// the same "build the plumbing, later phase plugs it in" pattern as `SetupAction`
/// itself (MS-4) and `SystemModelRef.readiness` (AFM-1).
///
/// The `reason:` string here is picker/banner prose — parallel to
/// `SystemLanguageModelAvailabilityChecker.readiness`'s "Turn on Apple Intelligence in
/// System Settings" — not `ErrorCatalog`'s `R910`. `R910` is a distinct, row-numbered
/// message a *remote executor* raises mid-run when a row it is about to dispatch turns
/// out to need a key that isn't set; that raise site is RM's job once a remote executor
/// exists, exactly as `R904` sits in the catalog unraised today, waiting for the same
/// executor. This function only answers "can the row start at all" — the
/// `.needsSetup` readiness it returns is what already keeps `FlowPreflight` from
/// starting a run that names an unconfigured provider, via the same mechanism Phase
/// AFM's `.needsSetup` readiness uses today.
nonisolated enum ProviderCredential {
    static func readiness(providerName: String) -> Readiness {
        let account = KeychainHelper.providerAccount(providerName)
        guard KeychainHelper.get(account: account) != nil else {
            return .needsSetup(reason: "Add your \(providerName) key in Settings",
                                action: .openSettings(.providers))
        }
        return .ready
    }

    /// RM-2's actual one line: `TaskModels.providerModels` calls this per manifest.
    /// `manifest.credentials == nil` is the credential-less LAN case (`egress: "lan"`, e.g.
    /// the now-removed `macstudio-qwen3-32b.json`) — always `.ready`, never asks for a key
    /// that manifest never declared.
    static func readiness(for manifest: CuratedManifest) -> Readiness {
        guard let name = manifest.credentials else { return .ready }
        return readiness(providerName: name)
    }
}
