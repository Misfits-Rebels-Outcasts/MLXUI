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
