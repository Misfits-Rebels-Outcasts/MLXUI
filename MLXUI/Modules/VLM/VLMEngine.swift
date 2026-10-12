import Foundation
import MLX
import MLXVLM
import MLXLMCommon
import CoreImage

/// MLX-native vision-language generation via mlx-swift-lm's `VLMModelFactory`. Mirrors
/// `LLMEngine` (same `loadContainer` + `MLXLMCommon.generate` path) but feeds an image
/// alongside the prompt. Loads the **installed** mlx-community VLM safetensors (the files
/// `InstallManager` downloads) — **no Apple Vision, no temp file**: the `CGImage` goes in
/// directly as `UserInput.Image.ciImage`. Isolates the `MLXVLM` import to one file.
enum VLMEngine {
    /// Cap on input image pixels. VLM vision attention is O(patches²) and these repos ship
    /// `max_pixels: null`, so a 4K image (8 MP) OOMs Metal (~34 GB alloc). ~1 MP keeps the
    /// vision-token count safe while remaining legible for VQA/OCR (journal/2026-39).
    nonisolated static let maxInputPixels = 1_048_576

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }

    /// The installed-model directory keyed by HF repo id rather than catalog card id (MoC-6,
    /// `RSI/DelegateMoCBacklog.md`). Needed when a catalog card's own id diverges from the repo
    /// its files are actually stored under — today, only the `Qwen3.5-9B Vision` card, which
    /// shares its download with the separate `Qwen3.5-9B` (`llm`) card (`MoC-5`).
    nonisolated static func installedModelDirectory(hfModelID: String) -> URL {
        ModelStore.shared.directory(forHFModelID: hfModelID)
    }

    /// Load the VLM at `modelDir` and answer `prompt` about `image`. Loads per call
    /// (matches `LLMEngine`); instance caching is a later optimization.
    nonisolated static func generate(
        prompt: String,
        image: CGImage,
        modelDir: URL,
        maxTokens: Int,
        pool: ModelContainerPool = .vlm,
        holder: MemoryBudget.HolderKind? = .flow
    ) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        do {
            // SC-2b: the weights are booked in the shared budget at their real size (`ModelWeightCharge`, via the
            // VLM `ModelContainerPool` below) and leased for the call, instead of loading outside the ledger. The
            // container then stays resident (evictable, LRU) rather than being reloaded on every call.
            let lease = try await pool.lease(directory: modelDir, holder: holder)
            defer { lease.release() }
            let container = lease.container
            // SC-2c: and the call's own working set (image + vision tokens — ≈ 2 GiB for a 1 MP image, measured),
            // reserved for the call like a server request's KV: other idle models are evicted for it, a model in use is
            // waited for, and a call that still can't fit is refused instead of running over budget.
            let working = try await reserveCallWorkingSet(pool: pool, modelDir: modelDir)
            defer { pool.budget.release(working) }
            // Cap resolution before inference so vision attention can't OOM (see maxInputPixels).
            let bounded = ImageLoader.downscaled(image, maxPixels: maxInputPixels)
            return try await container.perform { context in
                // The image must ride on a chat *message* (not the flat `images:` list): the
                // per-model message generator reads `message.images` to inject the vision
                // placeholders the chat template needs. Passing `UserInput(prompt:images:)`
                // yields zero placeholders → an empty-array reshape crash in the vision merge.
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [
                        .user(prompt, images: [.ciImage(CIImage(cgImage: bounded))])
                    ]))
                // OCP-FIX-1-2 (`RSI/DelegateOCRPromptBacklog.md` §2, journal `2026-230`):
                // consume **raw token ids**, not `.chunk` text. The decoded-text path routes
                // every chunk through mlx-swift-lm's `ToolCallProcessor`, which drops the text
                // before a `<` whenever the `<…` tail partially matches `<tool_call>` — so
                // GLM-OCR's `</td><td>` arrives as `</td<td>`. OCR/VQA never emit tool calls,
                // so we skip that scanner entirely and detokenize the full id stream once
                // (also strictly more correct than streaming reconstruction).
                let stream = try MLXLMCommon.generateTokens(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxTokens),
                    context: context)
                var tokenIds: [Int] = []
                for await generation in stream {
                    if case .token(let id) = generation { tokenIds.append(id) }
                }
                return context.tokenizer.decode(tokenIds: tokenIds)
            }
        } catch let error as StageError {
            throw error
        } catch let error as MemoryBudget.WaitError {
            throw error          // the sentence that names the model and who holds it (SC-2a-b)
        } catch {
            throw StageError.engineFailure(stage: "VLM", underlying: error)
        }
    }

    /// SC-2c: reserve one call's working set (`ModelLoadProfile` `.vision`) beside the model's resident weights. Other
    /// idle models are evicted for it; a model in use is waited for (`MemoryBudget.defaultMaxWait`) and then a
    /// `WaitError` naming it is thrown; a call that cannot fit even alone throws `insufficientRAM`. The caller releases
    /// the reservation when the call ends.
    nonisolated static func reserveCallWorkingSet(pool: ModelContainerPool, modelDir: URL,
                                                  maxWait: Duration = MemoryBudget.defaultMaxWait) async throws -> MemoryBudget.Reservation {
        let weights = ModelWeightCharge.bytes(directory: modelDir, catalogFallback: nil)
        let need = weights + ModelLoadProfile.workingSetBytes(for: .vision)
        switch await pool.reserveWorkingSetReporting(totalNeedBytes: need, directory: modelDir, maxWait: maxWait) {
        case .granted(let reservation): return reservation
        case .busy(let blockers): throw MemoryBudget.WaitError(label: modelDir.lastPathComponent, blockers: blockers)
        case .tooLarge:
            throw StageError.insufficientRAM(requiredGB: Double(need) / 1_073_741_824,
                                             availableGB: Double(pool.budget.capacityBytes) / 1_073_741_824)
        }
    }

    /// The pool's loader for a VLM directory: the config normalisation, then `VLMModelFactory`.
    nonisolated static func loadContainer(directory: URL, architecture: String?) async throws -> ModelContainer {
        // Some checkpoints (e.g. olmOCR-2, fine-tuned from Qwen2.5-VL) ship a config.json
        // missing the special-token ids the stock Qwen2.5-VL decoder requires. Fill them
        // in before the factory decodes it, else load fails with `keyNotFound`.
        VLMConfigNormalizer.normalizeConfigIfNeeded(at: directory)
        return try await VLMModelFactory.shared.loadContainer(from: directory, using: HFTokenizerLoader())
    }
}

extension ModelContainerPool {
    /// SC-2b: the VLM weights' pool — the same shared `MemoryBudget` as the LLM pool and `EngineCache`, its own
    /// loader. Chosen over `EngineCache` because the thing to book and lease is a loaded `ModelContainer` (the stage
    /// is a closure that owns nothing), which is exactly what `ModelContainerPool` already does for LLM weights.
    static let vlm = ModelContainerPool(budget: .shared, loader: VLMEngine.loadContainer)
}

/// The VLM and OCR SDKs build a `VLMStage`, whose weights `VLMEngine` books in the pool above; the engine cache
/// must book those stages at 0 (see `StageBooksOwnWeights`).
extension VLMSDK: StageBooksOwnWeights {}
extension OCRSDK: StageBooksOwnWeights {}
