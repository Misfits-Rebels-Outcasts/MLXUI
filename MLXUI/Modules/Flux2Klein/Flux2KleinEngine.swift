import CoreGraphics
import Foundation
import MLX

// FLUX.2 Klein staged engine (MP7-3h): prompt text in, a picture out, with the three big components resident one at a time.
//
//   tokenize -> TEXT ENCODER loaded (streamed, >= 8 bit) -> embeddings evaluated -> encoder released -> cache cleared
//            -> DiT loaded (streamed, quantized) -> denoise (4 steps) -> latents evaluated -> DiT released -> cache cleared
//            -> VAE loaded -> decode -> VAE released -> cache cleared -> CGImage
//
// Never two of {encoder, DiT, VAE} at once (owner done-when a); the stage order is observable through `Flux2KleinLifecycleEvent`, and the components come from a
// `Flux2KleinStageSource` so a test runs the whole thing with tiny configs and no real weights.
//
// Memory.cacheLimit is process-global (the Local Server sets its own around a generation). A run saves the value it found, caps it for every stage and puts
// it back on EVERY exit: done, thrown, cancelled (`Flux2KleinCacheLimitGuard`). Sizes are checked against a cost model built from measured peaks
// (`Flux2KleinMemoryPlan`) before anything loads.
//
// This file registers nothing (no module, no Stage, no Run UI, nothing in the Local Server): MP7-4 owns those. The Local Server never serves this model.
// Implementer's call, pending owner confirmation, wherever a comment below says so.

nonisolated enum Flux2KleinComponent: String, Sendable, CaseIterable {
    case textEncoder, transformer, vae
}

/// What the engine reports, in order, as it works. A test reads the sequence; production passes no observer.
nonisolated enum Flux2KleinLifecycleEvent: Equatable, Sendable {
    /// `Memory.cacheLimit` was set for this stage (the value written, never above what the run found).
    case cacheLimitApplied(Flux2KleinComponent, bytes: Int)
    case loaded(Flux2KleinComponent)
    case embeddingsEvaluated
    case denoiseStep(index: Int)
    case latentsEvaluated
    case released(Flux2KleinComponent)
    case cacheCleared
}

// MARK: - Request, result, errors

nonisolated struct Flux2KleinRequest: Sendable, Equatable {
    var prompt: String
    var width: Int
    var height: Int
    var steps: Int = 4
    var seed: UInt64 = 0
    /// The row's `quantize`: the DiT's bit width (3, 4, 5, 6 or 8; Browse default 8). The text encoder is built at `max(quantize, 8)`.
    var quantize: Int = 8
    /// The row, for sentences ("FLUX.2 Klein 4B can't make ...").
    var row: String = Flux2KleinEngine.modelName
    /// Evaluate after every transformer block (done-when d). ON by default (implementer's call, pending owner confirmation): measured at 1024 x 1024, int8, MLX peak 5.67 GB -> 5.22 GB
    /// (footprint 6.00 -> 5.44 GB) at the same speed within noise (7.31 s vs 7.47 s); lower at every size measured (MP7-3h-report.md). Same arithmetic: the tiny golden shows max-abs 0.
    var evaluatesEachBlock: Bool = true
}

/// Test-facing injection (never set in production): a fixed noise tensor (patchified NCHW float32 `[1, 128, H/16, W/16]`, what diffusers' `prepare_latents` draws before packing)
/// and/or fixed text embeddings `[1, rows, joint_attention_dim]` (skips the encoder stage entirely).
nonisolated struct Flux2KleinInjection: @unchecked Sendable {
    var noise: MLXArray?
    var embeddings: MLXArray?
    /// Called with the step index and the (evaluated) packed latents `[1, tokens, channels]` after each denoise step.
    var stepProbe: (@Sendable (Int, MLXArray) -> Void)?
    init(noise: MLXArray? = nil, embeddings: MLXArray? = nil) { self.noise = noise; self.embeddings = embeddings }
}

nonisolated struct Flux2KleinProgress: Sendable, Equatable {
    var component: Flux2KleinComponent
    /// 1-based denoise step; 0 for the other stages' start.
    var step: Int
    var steps: Int
}

nonisolated struct Flux2KleinStageReport: Sendable, Equatable {
    var component: Flux2KleinComponent
    /// Wall time of the whole stage (load + work + release), seconds.
    var seconds: Double
    /// `Memory.peakMemory` since the stage began (MLX allocations only, not the process footprint), bytes.
    var mlxPeakBytes: Int
}

nonisolated struct Flux2KleinResult: @unchecked Sendable {
    let image: CGImage
    /// The decoder's output `[1, 3, H, W]`, float32, roughly [-1, 1], evaluated.
    let pixels: MLXArray
    let width: Int
    let height: Int
    let seed: UInt64
    let stages: [Flux2KleinStageReport]
}

nonisolated enum Flux2KleinEngineError: Error, Equatable, LocalizedError {
    case invalidSteps(Int)
    case sizeExceedsMemory(row: String, width: Int, height: Int, quantize: Int, neededGB: Double, availableGB: Double, largestSide: Int)
    case imageConversionFailed
    case injectedShapeMismatch(String)
    case stageComponentUnavailable

    var errorDescription: String? {
        switch self {
        case let .invalidSteps(steps):
            return "FLUX.2 Klein needs at least 1 step; the row asks for \(steps)."
        case let .sizeExceedsMemory(row, width, height, quantize, needed, available, largest):
            let head = "\(row) can't make a \(width) x \(height) image at quantize \(quantize) on this Mac: it needs about \(Self.gb(needed)) GB at its busiest step and this Mac can give it \(Self.gb(available)) GB."
            if largest >= Flux2KleinLatents.sizeMultiple {
                return head + " The largest it can make here is \(largest) x \(largest) (or any other shape with no more pixels)."
            }
            return head + " It can't make an image of any size here at this quantize."
        case .imageConversionFailed:
            return "FLUX.2 Klein could not turn the decoded picture into an image."
        case let .injectedShapeMismatch(what):
            return "FLUX.2 Klein injected input: \(what)."
        case .stageComponentUnavailable:
            return "FLUX.2 Klein lost a model component mid-run; this is a bug, please report it."
        }
    }

    private static func gb(_ v: Double) -> String { String(format: "%.1f", v) }
}

// MARK: - Memory plan (done-when c)

/// What each stage costs, from numbers MEASURED on this Mac (decimal GB; reports MP7-3c, 3f, 3g, 3h) and linear in pixels where the stage is. The stages are never co-resident, so
/// a size fits when the BUSIEST stage fits (not the sum). Compared against `MemoryBudget.shared.capacityBytes` (`max(4 GB, RAM x 0.6)`: 11.6 GB on a 19.3 GB Mac).
///
/// Implementer's call, pending owner confirmation: the budget is the whole capacity, not capacity minus what other holders reserve right now (MP7-4 leases through
/// `LeasingStage` / `MemoryBudget.reserve`, which is where "what is free now" is decided); the figures are MLX peaks (`Memory.peakMemory`), not process footprint.
nonisolated enum Flux2KleinMemoryPlan {
    /// Text encoder at its 8-bit floor: MLX peak while streaming the load plus one forward (MP7-3c). Independent of the image size.
    static let encoderPeakGB = 3.83
    /// VAE decoder at bf16 compute with per-layer evaluation (MP7-3g MLX peaks: 512 -> 1.66 GB, 1024 -> 5.74 GB, 1280 -> 8.92 GB): the 0.17 GB of weights plus about 0.08 GB of fixed
    /// working memory, then a per-output-pixel part. An upper envelope of the three points.
    static let vaeWeightsGB = 0.25
    static let vaeGBPerPixel = 5.4e-6
    /// DiT weights resident after the streamed quantize: int8 4.128 GB, int4 2.196 GB, bf16 7.751 GB (MP7-3f); linear in the bit width between them.
    static let ditWeightsGBAtInt8 = 4.128
    static let ditWeightsGBPerBit = (4.128 - 2.196) / 4
    /// Extra MLX peak while a block is being quantized during the streamed load: measured peaks int8 4.50, int4 2.56 GB (bf16 7.75) over their weights of 4.128 / 2.196 GB, i.e. 0.37;
    /// rounded UP to 0.38 so the plan is never under a measured peak.
    static let ditLoadTransientGB = 0.38
    /// DiT working memory in one forward on top of its weights, linear in output pixels, measured at 512 / 768 / 1024 / 1280 / 2048 square (MP7-3h-report.md; MLX peak minus weights, cache
    /// capped at 256 MB): with per-block evaluation 0.38 / 0.69 / 1.09 / 1.59 / 3.46 GB, without 1.42 / 1.46 / 1.54 / 1.97 (2048: 3.97) GB. The constants are upper envelopes of those
    /// points (every measured value is at or under its line), so a size is never admitted on an optimistic fit.
    static let ditActivationBaseGB = 0.20
    static let ditActivationGBPerPixel = 9.0e-7
    /// The same without per-block evaluation (the request's `evaluatesEachBlock` off).
    static let ditActivationBaseNoEvalGB = 1.45
    static let ditActivationGBPerPixelNoEval = 6.5e-7

    static func ditWeightsGB(quantize: Int) -> Double {
        ditWeightsGBAtInt8 + ditWeightsGBPerBit * Double(quantize - 8)
    }

    struct Peaks: Equatable {
        var encoder: Double
        var transformer: Double
        var vae: Double
        var busiest: Double { max(encoder, transformer, vae) }
    }

    static func peaks(width: Int, height: Int, quantize: Int, evaluatesEachBlock: Bool = true) -> Peaks {
        let pixels = Double(width) * Double(height)
        let weights = ditWeightsGB(quantize: quantize)
        let forward = evaluatesEachBlock ? ditActivationBaseGB + ditActivationGBPerPixel * pixels
                                         : ditActivationBaseNoEvalGB + ditActivationGBPerPixelNoEval * pixels
        return Peaks(encoder: encoderPeakGB,
                     transformer: weights + max(ditLoadTransientGB, forward),
                     vae: vaeWeightsGB + vaeGBPerPixel * pixels)
    }

    /// The largest SQUARE side (a multiple of 16) whose busiest stage fits in `budgetBytes`; 0 when not even the smallest does.
    static func largestSquareSide(quantize: Int, budgetBytes: Int64, evaluatesEachBlock: Bool = true) -> Int {
        let multiple = Flux2KleinLatents.sizeMultiple
        var lo = 0, hi = 1024                       // sides in units of `multiple`
        func fits(_ units: Int) -> Bool {
            let side = units * multiple
            return peaks(width: side, height: side, quantize: quantize, evaluatesEachBlock: evaluatesEachBlock).busiest * 1e9 <= Double(budgetBytes)
        }
        guard fits(1) else { return 0 }
        lo = 1
        while lo < hi {
            let mid = (lo + hi + 1) / 2
            if fits(mid) { lo = mid } else { hi = mid - 1 }
        }
        return lo * multiple
    }

    /// Throws `Flux2KleinEngineError.sizeExceedsMemory` naming the row, the size asked and the largest that fits. Assumes the size is already a valid multiple of 16.
    static func check(width: Int, height: Int, quantize: Int, row: String, budgetBytes: Int64, evaluatesEachBlock: Bool = true) throws {
        let busiest = peaks(width: width, height: height, quantize: quantize, evaluatesEachBlock: evaluatesEachBlock).busiest
        guard busiest * 1e9 > Double(budgetBytes) else { return }
        throw Flux2KleinEngineError.sizeExceedsMemory(row: row, width: width, height: height, quantize: quantize, neededGB: busiest,
                                                      availableGB: Double(budgetBytes) / 1e9,
                                                      largestSide: largestSquareSide(quantize: quantize, budgetBytes: budgetBytes, evaluatesEachBlock: evaluatesEachBlock))
    }
}

// MARK: - Process-global cache limit (done-when b)

/// Per-stage MLX buffer-cache caps, bytes. Every stage gets one (done-when b: "cap it in every stage, not only the VAE": the 3g runs reached 13.6 GB with the default cache
/// while 5.7 GB was live). The values are the implementer's call, pending owner confirmation, set from the MP7-3h measurements.
nonisolated struct Flux2KleinCacheLimits: Sendable, Equatable {
    var textEncoder: Int
    var transformer: Int
    var vae: Int

    static let standard = Flux2KleinCacheLimits(textEncoder: 256 * 1_048_576, transformer: 256 * 1_048_576, vae: 256 * 1_048_576)

    func limit(for component: Flux2KleinComponent) -> Int {
        switch component {
        case .textEncoder: return textEncoder
        case .transformer: return transformer
        case .vae: return vae
        }
    }
}

/// One run's hold on the global `Memory.cacheLimit`. Reads the value it finds at creation, writes `min(found, cap)` per stage (never RAISES a lower limit someone
/// set), and `restore()` puts the found value back. Idempotent. Same rule as `ServeMLXMemory.end` (MP6-3h ruling): if someone else changed the limit while the run held it
/// (the Local Server entering a generation, `EngineCache.init`), their value is left alone rather than overwritten.
nonisolated final class Flux2KleinCacheLimitGuard: @unchecked Sendable {
    private let read: @Sendable () -> Int
    private let write: @Sendable (Int) -> Void
    private let lock = NSLock()
    private let found: Int
    private var lastApplied: Int?
    private var restored = false

    init(read: @escaping @Sendable () -> Int = { Memory.cacheLimit }, write: @escaping @Sendable (Int) -> Void = { Memory.cacheLimit = $0 }) {
        self.read = read
        self.write = write
        self.found = read()
    }

    /// Sets the limit for a stage and returns what was written.
    @discardableResult
    func apply(cap: Int) -> Int {
        lock.lock(); defer { lock.unlock() }
        let target = min(found, cap)
        write(target)
        lastApplied = target
        return target
    }

    func restore() {
        lock.lock(); defer { lock.unlock() }
        guard !restored else { return }
        restored = true
        guard let lastApplied, read() == lastApplied else { return }
        write(found)
    }
}

// MARK: - Stage source (the lifecycle seam)

/// Where the three components and the tokenizer come from. Production: the installed model folder. Tests: tiny random or golden-weight modules.
nonisolated protocol Flux2KleinStageSource: Sendable {
    /// The dtype the denoise loop runs in: bfloat16 in production, float32 for the tiny golden runs.
    var computeDType: DType { get }
    /// The packed latent's channel count (128 = 32 VAE channels x 2 x 2); only the tiny test models differ.
    var packedChannels: Int { get }
    func encodePrompt(_ prompt: String) async throws -> Flux2KleinEncodedPrompt
    func makeTextEncoder(quantize: Int) throws -> Flux2KleinTextEncoder
    func makeTransformer(quantize: Int) throws -> Flux2KleinTransformer
    func makeVAE() throws -> Flux2KleinVAE
}

nonisolated extension Flux2KleinStageSource {
    var packedChannels: Int { Flux2KleinLatents.patchedChannels }
}

/// The installed `mlx-community/FLUX.2-klein-4B-bf16` folder (the A2 pattern: only files `InstallManager` already downloaded; nothing is fetched).
nonisolated struct Flux2KleinInstalledStages: Flux2KleinStageSource {
    let modelDirectory: URL
    private let vaeDType: DType?

    /// Production: the VAE decodes in the checkpoint's bf16 (owner, 2026-11-01: "Keep bf16").
    init(modelDirectory: URL) { self.modelDirectory = modelDirectory; self.vaeDType = nil }

    /// TEST-ONLY: an fp32 VAE decode (twice the peak memory). Not reachable from a row or the Run UI.
    init(modelDirectory: URL, vaeComputeDTypeForTesting: DType?) { self.modelDirectory = modelDirectory; self.vaeDType = vaeComputeDTypeForTesting }

    var computeDType: DType { .bfloat16 }

    func encodePrompt(_ prompt: String) async throws -> Flux2KleinEncodedPrompt {
        let tokenizer = try await HFTokenizerLoader().load(from: modelDirectory.appendingPathComponent("tokenizer"))
        return try Flux2KleinPrompt.encode(prompt, tokenizer: tokenizer)
    }

    func makeTextEncoder(quantize: Int) throws -> Flux2KleinTextEncoder {
        try Flux2KleinTextEncoder.load(textEncoderDirectory: modelDirectory.appendingPathComponent("text_encoder"), bits: quantize)
    }

    func makeTransformer(quantize: Int) throws -> Flux2KleinTransformer {
        try Flux2KleinTransformer.load(transformerDirectory: modelDirectory.appendingPathComponent("transformer"), bits: quantize)
    }

    func makeVAE() throws -> Flux2KleinVAE {
        try Flux2KleinVAE.load(vaeDirectory: modelDirectory.appendingPathComponent("vae"), computeDType: vaeDType)
    }
}

// MARK: - The engine

nonisolated struct Flux2KleinEngine: Sendable {
    static let modelName = "FLUX.2 Klein 4B"

    let stages: any Flux2KleinStageSource
    let cacheLimits: Flux2KleinCacheLimits
    let budgetBytes: @Sendable () -> Int64
    let readCacheLimit: @Sendable () -> Int
    let writeCacheLimit: @Sendable (Int) -> Void
    let observer: (@Sendable (Flux2KleinLifecycleEvent) -> Void)?

    init(stages: any Flux2KleinStageSource,
         cacheLimits: Flux2KleinCacheLimits = .standard,
         budgetBytes: @escaping @Sendable () -> Int64 = { MemoryBudget.shared.capacityBytes },
         readCacheLimit: @escaping @Sendable () -> Int = { Memory.cacheLimit },
         writeCacheLimit: @escaping @Sendable (Int) -> Void = { Memory.cacheLimit = $0 },
         observer: (@Sendable (Flux2KleinLifecycleEvent) -> Void)? = nil) {
        self.stages = stages
        self.cacheLimits = cacheLimits
        self.budgetBytes = budgetBytes
        self.readCacheLimit = readCacheLimit
        self.writeCacheLimit = writeCacheLimit
        self.observer = observer
    }

    // MARK: Validation (makeStage calls the same)

    /// `quantize` for the WHOLE row, in a sentence about the image model; then the size (multiple of 16, SPEC-Q242), then the memory cap. Everything the engine would refuse, refused
    /// before anything loads. The text encoder's own check (its 8-bit floor) can never fire from here because the encoder is only ever asked for `max(quantize, 8)`.
    static func validate(_ request: Flux2KleinRequest, budgetBytes: Int64) throws {
        try validate(quantize: request.quantize)
        guard request.steps >= 1 else { throw Flux2KleinEngineError.invalidSteps(request.steps) }
        try Flux2KleinLatents.validateSize(width: request.width, height: request.height)   // SPEC-Q242
        try Flux2KleinMemoryPlan.check(width: request.width, height: request.height, quantize: request.quantize, row: request.row, budgetBytes: budgetBytes,
                                          evaluatesEachBlock: request.evaluatesEachBlock)
    }

    /// 3, 4, 5, 6 or 8 (MLX affine widths with group 64; 7 is not one). The sentence is the image model's (`Flux2KleinTransformerError.unsupportedBits`).
    static func validate(quantize: Int) throws {
        guard Flux2KleinQuantization.ditSupportedBits.contains(quantize) else { throw Flux2KleinTransformerError.unsupportedBits(quantize) }
    }

    // MARK: Generate

    /// Runs the staged pipeline. `@concurrent`: a nonisolated async function runs on the caller's executor, and the forward passes would beachball the main actor.
    /// Callers on the main actor wrap it in `MLXErrorGuard.guardedRun` (KICKOFF 17c); readbacks inside use `MLXErrorGuard`.
    @concurrent func generate(_ request: Flux2KleinRequest, injection: Flux2KleinInjection? = nil,
                              progress: @escaping @Sendable (Flux2KleinProgress) -> Void = { _ in }) async throws -> Flux2KleinResult {
        try Self.validate(request, budgetBytes: budgetBytes())
        let (gridWidth, gridHeight) = try Flux2KleinLatents.validateSize(width: request.width, height: request.height)
        let imageSeqLen = gridWidth * gridHeight
        let dtype = stages.computeDType
        if let noise = injection?.noise, noise.shape != [1, stages.packedChannels, gridHeight, gridWidth] {
            throw Flux2KleinEngineError.injectedShapeMismatch("noise must be [1, \(stages.packedChannels), \(gridHeight), \(gridWidth)], got \(noise.shape)")
        }

        let cache = Flux2KleinCacheLimitGuard(read: readCacheLimit, write: writeCacheLimit)
        defer { cache.restore() }        // done, thrown or cancelled: the process-global limit goes back
        var reports: [Flux2KleinStageReport] = []
        try Task.checkCancellation()

        // 1. Text encoder -> embeddings (evaluated) -> encoder released -> cache cleared.
        let embeddings: MLXArray
        if let injected = injection?.embeddings {
            embeddings = injected.asType(dtype)
        } else {
            let encoded = try await stages.encodePrompt(request.prompt)
            try Task.checkCancellation()
            embeddings = try await resident(.textEncoder, cache: cache, reports: &reports,
                                            load: { try stages.makeTextEncoder(quantize: request.quantize) }) { encoder in
                let out = try encoder.embeddings(for: encoded)        // evaluated
                emit(.embeddingsEvaluated)
                return out.asType(dtype)
            }
        }
        try MLXErrorGuard.evaluate(embeddings)
        try Task.checkCancellation()

        // 2. DiT (streamed) -> denoise -> latents evaluated -> DiT released -> cache cleared.
        let latents: MLXArray = try await resident(.transformer, cache: cache, reports: &reports,
                                                   load: { try stages.makeTransformer(quantize: request.quantize) }) { dit in
            let noise = try injection?.noise ?? Flux2KleinLatents.makeNoise(width: request.width, height: request.height, seed: request.seed,
                                                                      channels: stages.packedChannels)
            var x = Flux2KleinLatents.packToSequence(noise.asType(dtype))
            try MLXErrorGuard.evaluate(x)
            let imgIDs = Flux2KleinLatents.imageIDs(gridHeight: gridHeight, gridWidth: gridWidth)
            let txtIDs = Flux2KleinLatents.textIDs(count: embeddings.dim(1))
            let sigmas = Flux2KleinSampler.sigmas(steps: request.steps, imageSeqLen: imageSeqLen)
            for i in 0 ..< request.steps {
                try Task.checkCancellation()
                let sigma = MLXArray([Flux2KleinSampler.transformerSigma(sigmas[i], dtype: dtype)])
                let prediction = try dit(hidden: x, encoder: embeddings, sigma: sigma, imgIDs: imgIDs, txtIDs: txtIDs, evaluatesEachBlock: request.evaluatesEachBlock)
                x = Flux2KleinSampler.stepUpcasting(sample: x, velocity: prediction, sigma: sigmas[i], sigmaNext: sigmas[i + 1])
                try MLXErrorGuard.evaluate(x)
                injection?.stepProbe?(i, x)
                emit(.denoiseStep(index: i))
                progress(Flux2KleinProgress(component: .transformer, step: i + 1, steps: request.steps))
            }
            let unpacked = Flux2KleinLatents.unpackFromSequence(x, gridHeight: gridHeight, gridWidth: gridWidth)
            try MLXErrorGuard.evaluate(unpacked)
            emit(.latentsEvaluated)
            return unpacked
        }
        try Task.checkCancellation()

        // 3. VAE -> pixels -> VAE released -> cache cleared.
        let (pixels, image): (MLXArray, CGImage) = try await resident(.vae, cache: cache, reports: &reports, load: { try stages.makeVAE() }) { vae in
            let pixels = try vae.decode(packedLatents: latents)                 // [1, 3, H, W] float32, evaluated
            return (pixels, try Self.cgImage(from: pixels))
        }
        return Flux2KleinResult(image: image, pixels: pixels, width: request.width, height: request.height, seed: request.seed, stages: reports)
    }

    // MARK: Stage plumbing

    private func emit(_ event: Flux2KleinLifecycleEvent) { observer?(event) }

    /// Holds exactly one component for the duration of `run`: caps the cache, loads, runs, then drops the component, empties MLX's buffer cache and says so,
    /// on every exit (the thrown and cancelled paths too). `run` must hand back evaluated arrays (the component's weights are gone when it returns).
    private func resident<R: AnyObject, T>(_ component: Flux2KleinComponent, cache: Flux2KleinCacheLimitGuard, reports: inout [Flux2KleinStageReport],
                                           load: () throws -> R, run: (R) async throws -> T) async throws -> T {
        let applied = cache.apply(cap: cacheLimits.limit(for: component))
        emit(.cacheLimitApplied(component, bytes: applied))
        Memory.peakMemory = 0
        let started = Date()
        let slot = Flux2KleinSlot<R>()
        func release() {
            slot.value = nil
            emit(.released(component))
            Memory.clearCache()
            emit(.cacheCleared)
        }
        do { slot.value = try load() } catch { Memory.clearCache(); throw error }
        emit(.loaded(component))
        let result: T
        do { result = try await run(try slot.require()) } catch { release(); throw error }
        let peak = Memory.peakMemory
        release()
        reports.append(Flux2KleinStageReport(component: component, seconds: Date().timeIntervalSince(started), mlxPeakBytes: peak))
        return result
    }

    // MARK: Pixels

    /// `[1, 3, H, W]` in roughly [-1, 1] -> 8-bit RGB: `x / 2 + 1 / 2`, clamp, x 255, round (diffusers' `VaeImageProcessor.postprocess` + `numpy_to_pil`).
    private static func cgImage(from nchw: MLXArray) throws -> CGImage {
        let height = nchw.dim(2), width = nchw.dim(3)
        let scaled = clip(nchw.transposed(0, 2, 3, 1).asType(.float32) / 2 + 0.5, min: 0, max: 1) * 255
        let bytes = try MLXErrorGuard.read(round(scaled).asType(.uint8), as: UInt8.self)
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for p in 0 ..< (width * height) {
            rgba[p * 4] = bytes[p * 3]
            rgba[p * 4 + 1] = bytes[p * 3 + 1]
            rgba[p * 4 + 2] = bytes[p * 3 + 2]
        }
        guard let ctx = CGContext(data: &rgba, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
              let image = ctx.makeImage() else { throw Flux2KleinEngineError.imageConversionFailed }
        return image
    }
}

/// A reference holder the engine can empty on demand, so a component is really released (not merely out of scope: a debug build keeps locals alive to the end of the function).
nonisolated final class Flux2KleinSlot<R: AnyObject>: @unchecked Sendable {
    var value: R?
    func require() throws -> R {
        guard let value else { throw Flux2KleinEngineError.stageComponentUnavailable }
        return value
    }
}
