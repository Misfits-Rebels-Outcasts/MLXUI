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
        // MP4-3d: Sortformer, `phys_footprint` added by the Run UI's real path (decode, resample, engine, CSV; two runs, 20 ms
        // sampler, `SortformerLongClipMeasurement.measureFootprint`, 2026-10-25) at the longest one-pass clip: 60 s offline went
        // from 115 MiB (decoded) to a 1,673 MiB peak, 1,558 MiB added = 1.521 GiB over the 0.220 GiB of weights (236,108,132 B)
        // = 6.92, rounded up to 6.95. Streaming clips add less (1,104 MiB at 3,583 s). MLX's own peak
        // counter reads 1.000 GiB for the same run and leaves out Metal and driver memory; it is not the booking figure.
        case .diarization: return Entry(factor: 6.95, workingSetBytes: 0, measured: true)
        // MP5-3d: DeepFilterNet 3, `phys_footprint` peak of the whole Run UI path (decode, resample, 8 s / 1 s windows, model, WAV bytes)
        // on the longest clip it accepts, 60 minutes: 2.349 GB (lead's run, 2026-10-25, journal 2026-598); 0.62 GB at 1 minute, 0.86 GB at
        // 10. The growth is host arrays, not weights (the 25.6 MB installed files, v1 to v3), so a factor over the files would be
        // meaningless (about 92x) and would under-book if the install ever shrank to v3 alone. It is booked as a per-call working
        // set instead (the VLM row's form): files x 1.10 + 2.35 GB. Over-books by the files' own size, never under.
        // Implementer's call, pending owner confirmation (the precedent .diarization row uses a factor; its weights are 236 MB).
        // Provisional with ramGB 2.35 (owner, 2026-10-25): the final figure is the in-app Activity Monitor peak on the 60 min clip.
        case .enhance:  return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 2_350_000_000, measured: true)
        case .vision:   return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 2 * gib, measured: true)
        // RIFE-5c: RIFE 4.25, whole-process `phys_footprint` peak of the largest run the Run screen offers, Case E (4K 4x, Scale 0.5, the
        // 1 GiB MLX cache cap on): 6061 MiB = 6.36 GB, 2026-10-29 (test host; `RIFEMemoryMeasurementTests`). The growth is frame buffers and
        // flow maps at 4K, not the 22.7 MB of weights, so it is a per-call working set (the .enhance row's form), booked as 6.4 GB
        // (owner, 2026-10-29: "ramGB 6.4 decimal GB: Confirmed"). 1080p peaks near 3 GB, so smaller clips are over-booked, never under.
        case .interpolate: return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 6_400_000_000, measured: true)
        // MC1-12 (R2): SAM-Audio Base, `phys_footprint` peak of the whole Run screen path (WAV decode, 30 s windows, model, two
        // tracks) on the longest recording it accepts, 30 minutes: 13,528 MiB = 14.19 GB, 2026-11-03 (AM-T `SAMAudioSmokeTests`,
        // MC1-11; 12.63 GB at 60 s, 11.52 GB at 10 s). The growth is window activations and host arrays, not the weights alone, so
        // it is a per-call working set (the .enhance row's form): files x 1.10 + 7.70 GB. The files counted are the SAM folder
        // INCLUDING t5/ (`safetensorsBytes(underneath:)`): 5,923,698,622 B x 1.10 = 6,516,068,484 B; with the working set the
        // booking is 14,216,068,484 B, just over the measured 14.19 GB (the working set is 14.2 GB less the files' charge, 7.6839 GB,
        // rounded up to the next 0.05). NOT 14.2 GB on top of the weights, which would book about 20.7 GB.
        // On a 19 GB Mac the shared budget is about 11.4 GiB, so this model always runs alone: the budget evicts everything else,
        // then proceeds over budget. That is intended. Implementer's call, pending owner confirmation.
        case .separate: return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 7_700_000_000, measured: true)
        // MC2-11 (R2): TranslateGemma 4B, whole-process `phys_footprint` peak of the Run screen path at the longest input the engine
        // accepts, 2,048 rendered tokens: 4.08 GB (AM-T `TranslateGemmaSmokeTests`, 2026-11-07, 20 ms sampler, test host baseline
        // included; one resident copy, MLX active 2,082 MiB). The contributor measured 2.80 / 3.95 / 4.18 GB on an M3 Pro. It is not
        // the `.llm` row (files x 1.10 alone, ~2.4 GB booked): KV cache and activations at 2K tokens add ~1.4 GB above the weights,
        // so it is a per-call working set (the .enhance row's form): files x 1.10 + 2.0 GB = 2.22 GB x 1.10 + 2.0 = ~4.4 GB, at or
        // over the measured peak. Shorter inputs are over-booked, never under. Implementer's call, pending owner confirmation.
        case .translate: return Entry(factor: ModelWeightCharge.loadFactor, workingSetBytes: 2_000_000_000, measured: true)
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
