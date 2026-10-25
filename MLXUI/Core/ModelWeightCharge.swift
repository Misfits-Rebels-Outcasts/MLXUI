import Foundation

/// R63 — **the one function that says how many bytes a loaded model's weights are booked at** in the shared
/// `MemoryBudget` ledger.
///
/// The ledger used to book the catalog's `ramGB` (the 1.5× RAM heuristic: Qwen3.5-9B 8.96 GiB) for a model whose
/// safetensors are 5.54 GiB on disk and whose resident footprint after a generation measures 4.9–5.3 GiB. The
/// 3.4 GiB of phantom weight meant a ~2 GB prompt-cache snapshot "didn't fit" above ~60k tokens (journal `2026-452`).
///
/// The charge is the **safetensors bytes on disk × `loadFactor`**:
///
/// * Measured (journal `2026-453`, fresh process per model, `phys_footprint` after load and one short generation, over
///   the on-disk safetensors): Qwen3.5-9B 4-bit 0.91×, gemma-2-9b 4-bit 1.09×. `1.10` covers the larger of the two with a
///   little over; it is **not** a KV allowance (a request's KV is reserved separately — S3-8B's working set).
/// * **No safetensors on disk** (an unreadable or non-standard directory) → the caller's catalog figure, today's
///   conservative behaviour; and with neither → the whole directory's size.
///
/// `ServeMemoryAdmission` reads the same `safetensorsBytes` (S3-5a), so admission and the ledger share one basis; admission
/// uses the raw bytes (its `need` already carries `overheadFactor`), the ledger the loaded charge.
nonisolated enum ModelWeightCharge {
    /// What loading adds over the files on disk (measured, ≥ 1.0 — never charge less than the files).
    static let loadFactor = 1.10

    /// The sum of the `*.safetensors` files in `directory` (not tokenizer/config files). `0` when there are none.
    static func safetensorsBytes(in directory: URL) -> Int64 {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])) ?? []
        return files.filter { $0.pathExtension == "safetensors" }
            .reduce(Int64(0)) { $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    }

    /// The ledger charge for the model in `directory`. `catalogFallback` is the catalog's `ramGB` in bytes, used only
    /// when there are no safetensors to measure.
    static func bytes(directory: URL, catalogFallback: Int64?) -> Int64 {
        let onDisk = safetensorsBytes(in: directory)
        if onDisk > 0 { return charge(onDiskBytes: onDisk) }
        return catalogFallback ?? Int64(ModelStore.directorySize(at: directory))
    }

    static func charge(onDiskBytes: Int64) -> Int64 {
        Int64((Double(onDiskBytes) * loadFactor).rounded())
    }
}


/// SC-2c — what every **non-LLM** load books, in one small table (journal `2026-474-cc-sc2c-real-sizes`).
///
/// `ModelWeightCharge` above is the LLM rule (safetensors × 1.10). The other engines load more than their files: a
/// vision tower, activations, image and audio buffers, and MLX's buffer cache keeps the freed ones. SC-2b's live
/// check booked them at the catalog `ramGB` (dots.ocr 1.3 GiB against ≈ 5.7 GB real) and an OCR → VLM pair reached
/// 11 GB on an 8 GiB cap. Each factor below is `phys_footprint` added by the engine's real path (two runs, peak under
/// a 25 ms sampler) over the model's safetensors bytes, measured by `RealSizesMeasureTests` and rounded **up** to the
/// next 0.05:
///
/// | kind | model measured | files | peak added | × files | table |
/// |---|---|---|---|---|---|
/// | OCR, own engine | dots.ocr-4bit | 3.28 GiB | 5.64 | 1.72 | 1.75 |
/// | ASR | Voxtral-Mini-4B | 2.92 | 5.94 | 2.03 | 2.05 |
/// | decision | Laya | 0.78 | 0.93 | 1.18 | 1.20 |
/// | VLM (the pool) | gemma-3-4b-it-4bit | 3.17 | 6.35 | 2.00 | weights 1.10 + a 2 GiB working set per call |
///
/// A kind with **no** measurement (TTS, diffusion, music, segmentation, embedding, rerank, upscale, video) books
/// `max(files × 1.10, catalog ramGB)`: never below the files, never below what it booked before. Each measured row is
/// **one model**; the kind's other models inherit it (a Whisper checkpoint will be overbooked, not under). Implementer's
/// calls, pending owner confirmation: rounding up, and the 2 GiB VLM working set (measured with a ≈ 1 MP image).
nonisolated enum ModelLoadProfile {
    struct Entry: Equatable {
        /// Booked bytes per safetensors byte (what the engine holds once loaded and running, MLX's cache included).
        let factor: Double
        /// Extra bytes a single call needs on top of the resident weights (VLM image + vision tokens), reserved for
        /// the call only; 0 where `factor` already covers a run.
        let workingSetBytes: Int64
        /// Measured on a real model (`false` = the conservative default).
        let measured: Bool
    }

    static let gib = Int64(1_073_741_824)

    static func entry(for kind: RunnerKind) -> Entry {
        switch kind {
        case .llm:      return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 0, measured: true)
        case .ocr:      return Entry(factor: 1.75, workingSetBytes: 0, measured: true)
        case .asr:      return Entry(factor: 2.05, workingSetBytes: 0, measured: true)
        case .decision: return Entry(factor: 1.20, workingSetBytes: 0, measured: true)
        // MP4-3b: Sortformer, measured 2026-10-25 with MLX's own peak counter (`SortformerLongClipMeasurement.measureEnginePath`),
        // at the longest clip each path takes: offline 60 s = 1.000 GiB (1,073,802,220 B), streaming 3,583 s = 0.77 GiB (0.89 GiB
        // at 1,896 s). Weights are 236,108,132 B, so 1,073,802,220 / 236,108,132 = 4.55, rounded up to 4.6. This is MLX's peak,
        // not `phys_footprint` like the rows above (it leaves out the process baseline); measuring `phys_footprint` is a follow-up.
        // The engine caps the one-pass path at 60 s so the figure holds for any length up to its 3,583 s limit (the longest
        // streaming clip measured; owner ruling 2026-10-25). The load-profile factor itself is the implementer's call.
        case .diarization: return Entry(factor: 4.6, workingSetBytes: 0, measured: true)
        case .vision:   return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 2 * gib, measured: true)
        default:        return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 0, measured: false)
        }
    }

    /// Every `*.safetensors` under `directory`, subdirectories included (Laya and CLM keep their encoder and heads in
    /// subfolders; `ModelWeightCharge.safetensorsBytes` reads the top level only, which is right for an LLM folder).
    static func safetensorsBytes(underneath directory: URL) -> Int64 {
        guard let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var total: Int64 = 0
        for case let url as URL in walker where url.pathExtension == "safetensors" {
            total += Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    /// What a `kind` engine loaded from `directory` books **while it is resident and running**. With no safetensors on
    /// disk: the catalog figure (`catalogFallback`), else the directory's size.
    static func charge(kind: RunnerKind, directory: URL, catalogFallback: Int64?) -> Int64 {
        let onDisk = safetensorsBytes(underneath: directory)
        guard onDisk > 0 else { return catalogFallback ?? Int64(ModelStore.directorySize(at: directory)) }
        let entry = entry(for: kind)
        let booked = Int64((Double(onDisk) * entry.factor).rounded())
        return entry.measured ? booked : max(booked, catalogFallback ?? 0)
    }

    /// The per-call extra for `kind` (0 for most).
    static func workingSetBytes(for kind: RunnerKind) -> Int64 { entry(for: kind).workingSetBytes }
}
