import Foundation

/// Flags catalog models whose **architecture has no MLX runner yet**, so the UI can warn
/// before a Run that would only produce an `unsupportedModelType(...)` error. These are
/// tracked for iterative porting in `RSI/backlog.md` (§ "Model support gaps"). The backing
/// packages' type registries (`VLMTypeRegistry`, `EmbedderTypeRegistry`) enumerate what's
/// runnable; the entries below are the browser.json models whose `model_type` isn't in them.
///
/// Keyed on `hfModelId`/`id` substrings (precise to the catalog entries) so we don't need the
/// model installed to flag it. Update when an architecture is ported (remove its entry).
nonisolated enum ModelSupport {
    private struct Gap {
        let idFragment: String   // matched case-insensitively against id / hfModelId
        let reason: String
    }

    private static let gaps: [Gap] = [
        // All catalog OCR / embedder architectures now have MLX runners:
        // PaddleOCR-VL → vendored PaddleOCRVL pipeline (SUP-1, journal/2026-42).
        // dots.ocr / dots.mocr → custom `dots_ocr` model registered into MLXVLM (SUP-3, journal/2026-44/45).
        // DeepSeek-OCR-2 → custom `deepseekocr_2` model registered into MLXVLM (SUP-2, journal/2026-47…51).
        // ModernBERT (nomicai-modernbert-embed) → standalone runner (SUP-4, journal/2026-41).
        // SAM3 (sam3_video / Sam3VideoModel) → SegmentAnythingEngine ported in SA-AM4.
        // WAN 2.1 → WanVideoEngine ported in WAN-AM4 (journal/2026-93).
        // SeedVR2 3B → SeedVR2Engine ported in SV-AM4 (journal/2026-162).
        // Laya 0.4B (decision encoder) → LayaEngine ported in LY-4 (journal/2026-334).
        // CLM 8B (decision encoder) → CLMEngine ported in CL-4b, Run UI in CL-5 (journal/2026-341).
        // Clef Flash (decision) → ClefEngine ported in CF-5b (journal/2026-498); CF-2's gap is gone.
        // Sortformer 4spk v2.1 (diarization) → SortformerEngine in MP4-3a; the MP4-1 "no runner yet" gap is gone.
        // DeepFilterNet 3 (enhance) → DeepFilterNetEngine in MP5-3a; the MP5-1 "no runner yet" gap is gone.
        // LFM2.5-Audio (voice) → LFMAudioEngine in MP6-3a; the MP6-1 "no runner yet" gap is gone.
        // RIFE 4.25 (interpolate) → RIFEEngine + RIFERunView in RIFE-3/4; the RIFE-1 "no runner yet" gap is gone.
        // FLUX.2 Klein 4B (image) → Flux2KleinModule in MP7-4 (engine MP7-3a…3h); the MP7-2 "no runner yet" gap is gone.
    ]

    /// WW2-3-7 (owner ruling 2026-10-23; the WW6 "show it, don't fail it" rule): the OS a model's runner needs, shown
    /// **dimmed** — Browse card, detail, Run UI and the flow Model menu — instead of offered and then refused.
    ///
    /// Only **Kokoro**: on macOS < 15 its audio can't be read safely (`MLXAudioThread` needs `TaskExecutor`, macOS 15+; the
    /// array mlx-audio-swift's `interpolate1d` returns can't be evaluated off its thread under mlx-swift 0.32, journal
    /// 2026-557/558). The other TTS families don't use the pattern and stay available. The read guard
    /// (`MLXErrorGuard.read`) stays, so a run that is attempted anyway fails with an error, never a crash.
    /// **Remove this gate when upstream fixes `interpolate1d`** (links: `RSI/notes/ww2/upstream-issues.md`).
    ///
    /// `osVersion` is injectable so tests can stand in for macOS 14.
    static func osRequirement(forID id: String,
                              osVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion) -> String? {
        guard id.lowercased().contains("kokoro"), osVersion.majorVersion < 15 else { return nil }
        return "needs macOS 15"
    }

    static func osRequirement(for model: ModelEntry,
                              osVersion: OperatingSystemVersion = ProcessInfo.processInfo.operatingSystemVersion) -> String? {
        osRequirement(forID: model.id + " " + model.hfModelId, osVersion: osVersion)
    }

    /// "needs macOS 15" → "Needs macOS 15", for a label or sentence start.
    static func osRequirementTitle(_ needs: String) -> String {
        needs.prefix(1).uppercased() + String(needs.dropFirst())
    }

    /// A human-readable reason this model can't run yet, or `nil` if it should run.
    ///
    /// `ignoringGaps` (MP7-3a) lists `idFragment`s to treat as already removed, so a test can show what a model does once
    /// its gap is gone (the Klein module registering in MP7-3h). Empty by default; production never passes it.
    static func unsupportedReason(for model: ModelEntry, ignoringGaps: [String] = []) -> String? {
        let haystack = (model.id + " " + model.hfModelId).lowercased()
        return gaps.first { haystack.contains($0.idFragment) && !ignoringGaps.contains($0.idFragment) }?.reason
    }
}
