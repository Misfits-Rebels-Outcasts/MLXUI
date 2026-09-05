import Foundation

/// How closely a `browser.json` candidate matches the catflow manifest's pinned model.
/// `.same`/`.requantized` run silently; `.sameFamily`/`.substitute` surface a note on the
/// row (R3's inspector). A display name with **no** candidate is not runnable — never guess
/// a substitution that isn't in the table. See `RSI/DelegateMergeBacklog.md` CFM-R2-2.
nonisolated enum Equivalence: String, Sendable, Equatable, CaseIterable {
    case same, requantized, sameFamily, substitute

    /// A one-line note for `.sameFamily`/`.substitute`; `nil` for the silent cases.
    /// Surfaces *what actually loads* so a substitution is never hidden.
    func note(display: String, substitutedID: String) -> String? {
        switch self {
        case .same, .requantized:
            return nil
        case .sameFamily, .substitute:
            return "running \(display) as \(substitutedID)"
        }
    }
}

/// One bridge row: the `.cat`-visible display name → a ranked list of installable
/// `browser.json` `hfModelId`s, with the catflow manifest's pinned id (the settings
/// authority) and the equivalence class. Explicit and reviewable — never derived.
nonisolated struct BridgeEntry: Sendable, Equatable {
    let display: String        // the name a .cat writes, e.g. "Whisper Large v3"
    let pinnedID: String       // catflow's manifest id — the settings authority
    let candidates: [String]   // browser.json hfModelIds, best first
    let equivalence: Equivalence
    /// The curated-manifest filename (under `Resources/CatFlow/models/`) whose settings
    /// (enums, ranges, defaults) govern this display name.
    let manifestFile: String

    init(display: String, pinnedID: String, candidates: [String],
         equivalence: Equivalence, manifestFile: String) {
        self.display = display
        self.pinnedID = pinnedID
        self.candidates = candidates
        self.equivalence = equivalence
        self.manifestFile = manifestFile
    }
}

/// The outcome of resolving a display name against the catalog.
nonisolated enum CatalogBridgeResolution: Sendable, Equatable {
    /// Runnable: the `ModelEntry` to install/run, plus the equivalence and any note.
    case runnable(ModelEntry, equivalence: Equivalence, note: String?)
    /// Not runnable: a plain sentence naming the model and why (never a silent `nil`).
    case notRunnable(display: String, reason: String)
}

/// Maps a `.cat` display name ("Whisper Large v3") to an installable `browser.json`
/// `ModelEntry`. Exact-id matching is **never** used — the two catalogs don't intersect by
/// HF id (7 of 32; Whisper Large v3 and Kokoro 82M both miss). Settings still come from
/// the curated manifest (`Resources/CatFlow/models/`); only the *weights* are substituted.
nonisolated enum CatalogBridge {

    /// The seventeen display names the bridge runs. **`SAM Base`** joined 2026-08-27 (CFM-R15-1):
    /// hazard H2 / `CFM-R13-6` was **ruled option (1)** — a headless default — and the bridge
    /// maps the reference's own `SAM Base` id onto the one installable segmentation entry,
    /// `sam3-4bit`, as a `.substitute` so the substitution is shown on the row, never hidden.
    /// CFM-R13-9/12: OCR (`olmOCR-2 7B`, `dots.ocr`) and `Describe Image` (`LFM2-VL 1.6B`,
    /// `Gemma 3 4B`) joined 2026-08-26 — all four models were already in `browser.json`, so
    /// this is a table change only, no catalog intake. `olmOCR-2 7B` resolves to the
    /// `-mlx` build of the same 1025 checkpoint (a different repo → `.sameFamily`, so the
    /// substitution is surfaced, never hidden); `LFM2-VL 1.6B` is pinned 8-bit in the
    /// manifest but ships 4-bit in the catalog (`.requantized`, silent).
    /// CFM-R14-1: the three ASR gaps (`Whisper Tiny`, `Whisper Small`, `Voxtral Mini 4B
    /// Realtime`) joined 2026-08-26, so every `browser.json` ASR entry is reachable from
    /// the `Transcribe` row's Model menu.
    static let entries: [BridgeEntry] = [
        BridgeEntry(
            display: "Whisper Large v3",
            pinnedID: "mlx-community/whisper-large-v3-mlx",
            candidates: ["mlx-community/whisper-large-v3-asr-fp16"],
            equivalence: .requantized,
            manifestFile: "whisper-large-v3.json"),
        // CFM-R14-1 — the ASR gaps. whisper-tiny/whisper-small have no catflow manifest for
        // Whisper Small (author one below); Whisper Tiny pins the catflow `whisper-tiny`
        // manifest id but loads the `-asr-fp16` catalog build (same checkpoint, packaged for
        // the app — `.requantized`, silent). Voxtral resolves exactly.
        BridgeEntry(
            display: "Whisper Tiny",
            pinnedID: "mlx-community/whisper-tiny",
            candidates: ["mlx-community/whisper-tiny-asr-fp16"],
            equivalence: .requantized,
            manifestFile: "whisper-tiny.json"),
        BridgeEntry(
            display: "Whisper Small",
            pinnedID: "mlx-community/whisper-small-asr-fp16",
            candidates: ["mlx-community/whisper-small-asr-fp16"],
            equivalence: .same,
            manifestFile: "whisper-small.json"),
        BridgeEntry(
            display: "Voxtral Mini 4B Realtime",
            pinnedID: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            candidates: ["mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"],
            equivalence: .same,
            manifestFile: "voxtral-mini-4b-realtime-2602-4bit.json"),
        BridgeEntry(
            display: "Qwen3 8B",
            pinnedID: "mlx-community/Qwen3-8B-4bit",
            candidates: ["mlx-community/Qwen3-8B-4bit"],
            equivalence: .same,
            manifestFile: "qwen3-8b-4bit.json"),
        BridgeEntry(
            display: "Kokoro 82M",
            pinnedID: "mlx-community/Kokoro-82M-4bit",
            candidates: ["mlx-community/Kokoro-82M-bf16"],
            equivalence: .requantized,
            manifestFile: "kokoro-82m-4bit.json"),
        BridgeEntry(
            display: "Ministral 3B",
            pinnedID: "mlx-community/Ministral-3-3B-Instruct-2512-4bit",
            candidates: ["mlx-community/Ministral-3-3B-Instruct-2512-4bit"],
            equivalence: .same,
            manifestFile: "ministral-3b-4bit.json"),
        BridgeEntry(
            display: "BGE-M3",
            pinnedID: "mlx-community/bge-m3-mlx-4bit",
            candidates: ["mlx-community/bge-m3-mlx-fp16"],
            equivalence: .requantized,
            manifestFile: "bge-m3.json"),
        BridgeEntry(
            display: "MusicGen",
            pinnedID: "meta/musicgen-small",
            candidates: ["jasonvassallo/mlx-musicgen-small"],
            equivalence: .substitute,
            manifestFile: "musicgen-small.json"),
        BridgeEntry(
            display: "Llama 3.1 8B",
            pinnedID: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            candidates: ["mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"],
            equivalence: .same,
            manifestFile: "llama-3.1-8b-4bit.json"),
        // CFM-R13-9 — OCR.
        BridgeEntry(
            display: "olmOCR-2 7B",
            pinnedID: "mlx-community/olmOCR-2-7B-1025-4bit",
            candidates: ["mlx-community/olmOCR-2-7B-1025-mlx-4bit"],
            equivalence: .sameFamily,
            manifestFile: "olmocr-2-7b-1025-4bit.json"),
        BridgeEntry(
            display: "dots.ocr",
            pinnedID: "mlx-community/dots.ocr-4bit",
            candidates: ["mlx-community/dots.ocr-4bit"],
            equivalence: .same,
            manifestFile: "dots-ocr-4bit.json"),
        // MoC-1-4 (RSI/DelegateMoCBacklog.md) — GLM-OCR. No reference manifest exists in
        // catflow-mlx, so `glm-ocr-4bit.json` is MLXUI-authored (see its own "notes"). Not
        // added to `taskModels["OCR"]` — Decision C is unruled, so it stays picker-reachable
        // but never the default seed (CFM-R14-2: pools order, never filter).
        BridgeEntry(
            display: "GLM-OCR",
            pinnedID: "mlx-community/GLM-OCR-4bit",
            candidates: ["mlx-community/GLM-OCR-4bit"],
            equivalence: .same,
            manifestFile: "glm-ocr-4bit.json"),
        // CFM-R13-12 — Describe Image.
        BridgeEntry(
            display: "LFM2-VL 1.6B",
            pinnedID: "mlx-community/LFM2-VL-1.6B-8bit",
            candidates: ["mlx-community/LFM2-VL-1.6B-4bit"],
            equivalence: .requantized,
            manifestFile: "lfm2-vl-1.6b-8bit.json"),
        BridgeEntry(
            display: "Gemma 3 4B",
            pinnedID: "mlx-community/gemma-3-4b-it-4bit",
            candidates: ["mlx-community/gemma-3-4b-it-4bit"],
            equivalence: .same,
            manifestFile: "gemma-3-4b-it-4bit.json"),
        // SV-AM-W — SeedVR2 3B super-resolution (ported in SV-AM4/AM5, journal/2026-162).
        // Only the int8 build ships in browser.json today; the fp16 variant
        // (`mlx-community/SeedVR2-3B-mlx`) has no catalog entry, so it is **not** listed —
        // the golden `everyCandidateExistsInBrowserCatalog` exists exactly to catch a
        // candidate that names no catalog model (CFM-R15 drive-by fix).
        BridgeEntry(
            display: "SeedVR2 3B",
            pinnedID: "mlx-community/SeedVR2-3B-mlx-int8",
            candidates: ["mlx-community/SeedVR2-3B-mlx-int8"],
            equivalence: .same,
            manifestFile: "seedvr2-3b.json"),
        // CFM-R15-1 — Segment. `63-CutOutSubject` names `SAM Base` (the reference manifest's
        // own id `mlx/sam-base`); the only installable segmentation entry is SAM3. `.substitute`
        // renders "running SAM Base as mlx-community/sam3-4bit" on the row, so the different
        // generation is *shown* — the MusicGen precedent (`meta/musicgen-small` → MusicGen).
        BridgeEntry(
            display: "SAM Base",
            pinnedID: "mlx/sam-base",
            candidates: ["mlx-community/sam3-4bit"],
            equivalence: .substitute,
            manifestFile: "sam-base.json"),
    ]

    static func entry(for display: String) -> BridgeEntry? {
        entries.first { $0.display == display }
    }

    /// The bridge entry that lists `candidateID` as one of its candidates — the reverse
    /// lookup the derived model pool uses to name a `.cat` display for a catalog model
    /// (CFM-R14-2). `nil` when the model is unbridged (then its `hfModelId` is the name).
    static func entry(forCandidate candidateID: String) -> BridgeEntry? {
        entries.first { $0.candidates.contains(candidateID) }
    }

    /// Resolve a display name to an installable `ModelEntry`. `catalog` is the flat list
    /// of `browser.json` entries. Picks the first candidate present in the catalog
    /// (best-first order is the table's).
    ///
    /// **CFM-R14-FIX-1 — the derived-pool fallback.** R14-2's derived pool offers catalog
    /// models with no bridge row; their display name is either the `hfModelId` itself or (after
    /// FIX-7) the catalog `displayName`. A bridge miss is therefore **not** "can't run" anymore —
    /// fall back to matching the display against the catalog by id or display name, and resolve
    /// `.same` with no curated manifest (a manifest-less model is non-stochastic: `withSeed`
    /// already tolerates the missing file). Without this, 20 of the 34 catalog entries would be
    /// selectable and pinned yet refused at Run.
    static func resolve(_ display: String, catalog: [ModelEntry]) -> CatalogBridgeResolution {
        if let entry = entry(for: display) {
            for candidate in entry.candidates {
                if let model = catalog.first(where: { $0.hfModelId == candidate }) {
                    let note = entry.equivalence.note(display: display, substitutedID: candidate)
                    return .runnable(model, equivalence: entry.equivalence, note: note)
                }
            }
            return .notRunnable(display: display,
                                reason: "\(display) isn't installed in the model catalog — add it to the catalog before this flow can run.")
        }
        // No bridge row: an R14-2/7 derived pick. Match the catalog directly.
        if let model = catalog.first(where: { $0.hfModelId == display || $0.displayName == display }) {
            return .runnable(model, equivalence: .same, note: nil)
        }
        return .notRunnable(display: display,
                            reason: "\(display) isn't in the runnable-model table — this flow needs a model MLXUI can't run yet.")
    }
}

// MARK: - Curated manifest (the settings authority)

/// A value that can be a string, number, or boolean — manifests' setting defaults are
/// mixed-type (`"default": "af_heart"`, `"default": 1.0`, `"default": true`).
nonisolated enum FlexValue: Codable, Sendable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let n = try? c.decode(Double.self) { self = .number(n); return }
        throw DecodingError.dataCorruptedError(in: c, debugDescription: "setting default must be a string, number, or bool")
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let s): try c.encode(s)
        case .number(let n): try c.encode(n)
        case .bool(let b):   try c.encode(b)
        }
    }
}

/// One curated manifest's setting spec — the enums/ranges/defaults the Python's
/// `resolve_settings` validates against. Ported from `models/curated/*.json` (`settings`).
nonisolated struct SettingSpec: Codable, Sendable, Equatable {
    var type: String
    var values: [String]?
    var min: Double?
    var max: Double?
    var defaultValue: FlexValue?
    var mapsTo: String?
    var positionalOK: Bool?
    var autoOK: Bool?

    enum CodingKeys: String, CodingKey {
        case type, values, min, max, mapsTo
        case defaultValue = "default"
        case positionalOK = "positional_ok"
        case autoOK = "auto_ok"
    }
}

/// A manifest resource footprint (`resources`).
nonisolated struct ManifestResources: Codable, Sendable, Equatable {
    var diskGB: Double?
    var ramGB: Double?
    var bits: Int?

    enum CodingKeys: String, CodingKey {
        case diskGB = "disk_gb"
        case ramGB = "ram_gb"
        case bits
    }
}

/// A curated manifest — the settings authority for a bridge display name. Only the fields
/// FlowKit consumes are decoded; the rest of the file is ignored.
nonisolated struct CuratedManifest: Codable, Sendable, Equatable {
    var id: String
    var display: String
    var settings: [String: SettingSpec]
    var resources: ManifestResources?

    enum CodingKeys: String, CodingKey {
        case id, display, settings, resources
    }

    init(id: String, display: String, settings: [String: SettingSpec], resources: ManifestResources?) {
        self.id = id
        self.display = display
        self.settings = settings
        self.resources = resources
    }

    static func load(bundle: Bundle = .main, manifestFile: String) -> CuratedManifest? {
        // The `Resources/CatFlow/models/` group is a synchronized folder, and Xcode's
        // resource copy flattens it to the top of `Contents/Resources/` — every curated
        // manifest ships there, not under a `CatFlow/models` subdirectory. Confirmed against
        // the built product for every existing manifest, not just this one.
        guard let url = bundle.url(forResource: (manifestFile as NSString).deletingPathExtension,
                                   withExtension: "json"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(CuratedManifest.self, from: data)
    }
}
