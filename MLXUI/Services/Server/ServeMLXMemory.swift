import Foundation
import MLX

/// S3-5c — give memory back after a large request (journal `2026-442`).
///
/// MLX keeps freed buffers in a pool (`Memory.cacheMemory`) up to `Memory.cacheLimit`, which **defaults to the
/// memory limit (18.4 GB on this 18 GiB Mac)**: INV-1's "13 GB and swapping" was mostly this pool. API verified
/// in `mlx-swift` 0.31.4, `Source/MLX/Memory.swift`: `Memory.cacheLimit` (get/set, bytes; "If using more than the
/// given limit, free memory will be reclaimed from the cache on the next allocation"), `Memory.cacheMemory`,
/// `Memory.activeMemory`, `Memory.peakMemory`, `Memory.snapshot()`, and `Memory.clearCache()` ("Cause all cached
/// buffers to be deallocated").
///
/// **The choice** (measured; the numbers are in the journal): a *scoped cache limit* around each server
/// generation, not "clear the cache afterwards". Clearing afterwards returned the idle footprint but left the
/// request's own peak (13.8 GB, swapping 6.5 GB) untouched; the limit lowers both (peak 11.1 GB, idle 8.0 GB
/// vs 13.7 GB, 20k-token request: 79 s vs 365 s in the same session) and a short chat's tokens/s did not move
/// (29.6 → 29.7). The limit is applied only while a server generation runs and the previous value is put back
/// afterwards, so flows and the in-app chat (same process, same global) keep whatever they set — `EngineCache`
/// sets the limit to the budget — and only a generation that overlaps a server request sees the lower one.
nonisolated final class ServeMLXMemory: @unchecked Sendable {

    /// The pool a server generation may keep: **1 GiB** (R52). S3-5c shipped 256 MiB (lowest peak), but its
    /// long-decode data showed the *second* back-to-back reply slowest with 256 MiB in all three pairs, and
    /// agents send back-to-back requests; 1 GiB was as fast as the default on a single decode and still sits
    /// ~1.5 GB under the default's peak. A decode step needs tens of MB of temporaries and a 512-token prefill
    /// chunk a few hundred.
    static let generationCacheLimit = 1_024 * 1_048_576

    static let shared = ServeMLXMemory(read: { Memory.cacheLimit }, write: { Memory.cacheLimit = $0 })

    private let read: @Sendable () -> Int
    private let write: @Sendable (Int) -> Void
    private let lock = NSLock()
    private var depth = 0
    private var saved = 0

    /// - Parameters: the cache limit's getter and setter (MLX's, or fakes in tests).
    init(read: @escaping @Sendable () -> Int, write: @escaping @Sendable (Int) -> Void) {
        self.read = read
        self.write = write
    }

    /// SC-2a-b: whether a server generation's scoped cap is in force right now (read-only; changes nothing).
    var isCapActive: Bool {
        lock.lock(); defer { lock.unlock() }
        return depth > 0
    }

    /// The limit during a generation: never *raises* a lower limit someone already set.
    static func effectiveLimit(current: Int, cap: Int = generationCacheLimit) -> Int { min(current, cap) }

    /// Entering a generation. Nested or overlapping generations share one saved value.
    func begin(cap: Int = generationCacheLimit) {
        lock.lock(); defer { lock.unlock() }
        if depth == 0 {
            saved = read()
            write(Self.effectiveLimit(current: saved, cap: cap))
        }
        depth += 1
    }

    /// Leaving it: the last one out puts the previous limit back.
    func end() {
        lock.lock(); defer { lock.unlock() }
        guard depth > 0 else { return }
        depth -= 1
        if depth == 0 { write(saved) }
    }
}
