import Foundation
import MLX

/// PF-7 — the Privacy Filter engine: one text in, the redacted text plus its detected spans out.
///
/// The heavy lifting is the decode side's ``PrivacyFilterInference`` (tokenize → model log-probabilities → constrained
/// Viterbi labels → spans → redacted text). This type owns only the paths around it:
///
///  * **Input limits before any model work** (decision 5): an empty/whitespace-only text is refused here; the
///    128,000-token limit is refused by ``PrivacyFilterInference`` after tokenization and **before** the forward pass,
///    with one plain sentence carrying only the token count.
///  * **The model is leased, not loaded per run.** ``PrivacyFilterEngineCache`` keeps one loaded inference per installed
///    folder on the shared `MemoryBudget`, so a Run window and a flow row share one ~3 GB copy and the load is visible
///    to chat/flow/server accounting (PF-7's "take a lease" rule). A lease that cannot fit waits and then throws
///    `MemoryBudget.WaitError`'s sentence — never an OOM.
///  * **Stop cancels between windows**: the `isCancelled` callback is consulted before the model call and between
///    windows (Trap 3, the long-input path); a cancelled run is reported as ``Outcome/cancelled``, never as a failure.
///  * **Errors are one plain sentence that never quotes the input** (decision 10). Metadata — token and span counts —
///    is the only thing an error may carry.
nonisolated enum PrivacyFilterEngine {

    /// What one run produced. A Stop is `cancelled` with no redacted text, never a partial result.
    struct Outcome: Sendable {
        let redactedText: String
        /// Every detected span, **all categories** — the Run screen's checkboxes only change which of these are
        /// replaced in `redactedText`; they never re-run the model (decision 6).
        let spans: [PrivacyFilterSpan]
        let tokenCount: Int
        let cancelled: Bool

        static let cancelled = Outcome(redactedText: "", spans: [], tokenCount: 0, cancelled: true)
    }

    /// Where `InstallManager` places the model's files.
    nonisolated static func modelDirectory(for modelID: String) -> URL {
        ModelStore.shared.directory(forModelID: modelID)
    }

    /// Everything that can be refused before the model is even leased.
    static func validate(_ text: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw PrivacyFilterInferenceError.emptyText
        }
    }

    /// Redact one text. `categories == nil` replaces all eight categories; a subset only changes which detected spans
    /// are replaced (the spans themselves are always the model's full output).
    static func redact(
        _ text: String,
        categories: Set<PrivacyFilterCategory>? = nil,
        modelDirectory: URL,
        footprintBytes: Int64? = nil,
        holder: MemoryBudget.HolderKind = .runWindow,
        isCancelled: @Sendable () -> Bool = { Task.isCancelled }
    ) async throws -> Outcome {
        try validate(text)
        guard FileManager.default.fileExists(atPath: modelDirectory.path) else {
            throw StageError.modelNotInstalled(id: modelDirectory.lastPathComponent)
        }
        do {
            let lease = try await PrivacyFilterEngineCache.shared.lease(
                for: modelDirectory, holder: holder, catalogFallback: footprintBytes)
            defer { lease.release() }
            let box = lease.value
            let redaction = try MLXErrorGuard.guardedRun {
                try box.inference.redact(text, categories: categories, progress: nil, isCancelled: isCancelled)
            }
            return Outcome(redactedText: redaction.redactedText, spans: redaction.spans,
                           tokenCount: redaction.tokenCount, cancelled: false)
        } catch is CancellationError {
            return .cancelled
        } catch let error as MemoryBudget.WaitError {
            throw error          // SC-2a-b: the sentence names the model and who holds it
        } catch let error as StageError {
            throw error
        } catch let error as PrivacyFilterInferenceError {
            throw error
        } catch {
            // A cancelled run is never reported as an engine failure (MP5-3g's rule).
            if Task.isCancelled || isCancelled() { return .cancelled }
            throw StageError.engineFailure(stage: "PII redaction", underlying: error)
        }
    }
}

/// One loaded ``PrivacyFilterInference`` boxed for the memory ledger.
///
/// ``PrivacyFilterInference`` is `nonisolated` but not declared `Sendable` (its `PrivacyFilterModel` holds MLX arrays);
/// MLXUI's precedent for an engine a shared cache loads and tasks use is an explicit `@unchecked Sendable` box (Laya,
/// CLM, Clef). Arrays cross to Swift only through `MLXErrorGuard` on the thread that evaluates them (Trap 7), and every
/// entry point into the inference here is synchronous, so there is no `await` between building an array and reading it.
nonisolated final class PrivacyFilterInferenceBox: @unchecked Sendable {
    let inference: PrivacyFilterInference
    init(inference: PrivacyFilterInference) { self.inference = inference }
}

/// PF-7 — one loaded Privacy Filter per installed model directory, **on the shared `MemoryBudget`**.
///
/// The same pattern as `LayaEngineCache`/`CLMEngineCache`: real bytes (the safetensors under the directory × the
/// `.redact` row's measured factor, `ModelLoadProfile`), a lease per use, held entries skipped by eviction. The Run
/// window and a flow row both lease from here, so one install is resident at most once.
actor PrivacyFilterEngineCache {
    static let shared = PrivacyFilterEngineCache()

    private let pool: ResidentPool<PrivacyFilterInferenceBox>

    init(budget: MemoryBudget = .shared) {
        pool = ResidentPool(budget: budget)
    }

    /// What loading `directory` books: the `.redact` row's weights charge (PF-8's measured factor over the
    /// `*.safetensors` files under the folder; the catalog `ramGB` is the fallback when nothing is measurable) plus the
    /// row's per-run working set. That combination is the figure PF-8 books so it is never under the highest measured
    /// peak — the ledger, not just the pre-run "make room?" question, must hold it (`EngineCache.bytesNeeded` is where
    /// the other stages add their working set; this cache is this kind's ledger).
    nonisolated static func charge(directory: URL, catalogFallback: Int64? = nil) -> Int64 {
        ModelLoadProfile.charge(kind: .redact, directory: directory, catalogFallback: catalogFallback)
            + ModelLoadProfile.workingSetBytes(for: .redact)
    }

    /// The inference, **leased**: in use until `release()`. Eviction skips it; a load that can only fit by evicting it
    /// waits (`maxWait`) and then throws `MemoryBudget.WaitError` instead of going over budget.
    func lease(for directory: URL, holder: MemoryBudget.HolderKind? = nil,
               maxWait: Duration = MemoryBudget.defaultMaxWait,
               catalogFallback: Int64? = nil) async throws -> ResidentPool<PrivacyFilterInferenceBox>.Lease {
        let bytes = Self.charge(directory: directory, catalogFallback: catalogFallback)
        return try await pool.lease(for: directory.path, bytes: bytes, maxWait: maxWait, holder: holder) {
            PrivacyFilterInferenceBox(inference: try await PrivacyFilterInference(modelDirectory: directory))
        }
    }

    /// Unleased (tests, callers that only need the load to have happened).
    func inference(for directory: URL) async throws -> PrivacyFilterInference {
        let bytes = Self.charge(directory: directory)
        return try await pool.value(for: directory.path, bytes: bytes) {
            PrivacyFilterInferenceBox(inference: try await PrivacyFilterInference(modelDirectory: directory))
        }.inference
    }

    func evict(directory: URL) async { await pool.evict(key: directory.path) }
    func clear() async { await pool.clear() }
}

/// PF-7 — the catalog-model SDK for OpenAI Privacy Filter.
///
/// **Trap 10: claim by family, never by architecture.** The claim is `.redact` + `source == .mlx` +
/// `family == "privacy-filter"` — the catalog's `OpenAIPrivacyFilterForTokenClassification` string is not a claim
/// key, and no other SDK claims the `.redact` kind.
///
/// `StageBooksOwnWeights`: the engine leases its inference from ``PrivacyFilterEngineCache`` on the shared budget, so
/// the `EngineCache` must not book those 3 GB a second time (the VLM/Laya arrangement).
nonisolated struct PrivacyFilterSDK: ModelSDK, StageBooksOwnWeights {
    let id = "privacy-filter"

    /// The model takes no free-text prompt: the only setting is the `only=` category subset, carried on the stage.
    var promptSupport: PromptSupport { .none }

    func claim(_ model: ModelEntry) -> ClaimScore {
        guard model.runnerKind == .redact,
              model.source == .mlx,
              model.family.lowercased() == "privacy-filter" else {
            return .no
        }
        return .exact
    }

    func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage {
        let footprint = Int64(model.ramGB * 1_073_741_824)
        return PrivacyFilterStage(modelID: model.id, displayName: model.displayName,
                                  only: config.redactOnly, footprintBytes: footprint)
    }
}
