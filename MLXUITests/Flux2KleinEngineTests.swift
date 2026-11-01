import CoreGraphics
import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXUI

// MP7-3h: the staged Klein engine, on tiny modules and tiny goldens only (no real weights here; see Flux2KleinEngineRealTests for those).
//  * stage order and "never two of encoder / DiT / VAE resident" (a stage-source seam with weak references),
//  * Memory.cacheLimit saved, capped in every stage, restored on done / thrown / cancelled,
//  * quantize validated for the whole row, sizes refused above a cost-model cap, the sigmas bit-exact against diffusers,
//  * the whole tiny run against the diffusers glue golden (g3g_glue.safetensors, never edited).

// MARK: - Sampler exactness (done-when e)

struct Klein3hSchedulerCase: Decodable {
    let image_seq_len: Int
    let steps: Int
    let mu: Double
    let input_sigmas: [Double]
    let sigmas: [Double]
    let timesteps: [Double]
}

@Suite struct Flux2KleinSamplerExactnessTests {
    static func cases() throws -> [Klein3hSchedulerCase] {
        let url = Klein3d.fixtures.appendingPathComponent("g3f_scalars.json")
        return try JSONDecoder().decode([Klein3hSchedulerCase].self, from: Data(contentsOf: url))
    }

    /// The sigmas the engine passes are the ones diffusers 0.37.1's scheduler stores: float32, bit for bit (not "within 2e-7").
    @Test func sigmasAreBitIdenticalToDiffusersInEveryGoldenCase() throws {
        let cases = try Self.cases()
        #expect(cases.count == 6)
        for c in cases {
            let mine = Flux2KleinSampler.sigmas(steps: c.steps, imageSeqLen: c.image_seq_len)
            let want = c.sigmas.map { Float($0) }
            #expect(mine == want, "(\(c.image_seq_len), \(c.steps)): \(mine) vs \(want)")
            // the input to the shift is np.linspace(1, 1/N, N) in float64 -> float32; the same values must be reproducible
            #expect(c.input_sigmas.count == c.steps)
            // and the scheduler timesteps are float32 sigma x 1000
            let t = Flux2KleinSampler.timesteps(steps: c.steps, imageSeqLen: c.image_seq_len)
            #expect(t == c.timesteps.map { Float($0) }, "timesteps (\(c.image_seq_len), \(c.steps))")
            #expect(Flux2KleinSampler.empiricalMu(imageSeqLen: c.image_seq_len, steps: c.steps) == c.mu)
        }
    }

    @Test func theSafetensorsSchedulerGoldenIsBitIdenticalToo() throws {
        let g = try loadArrays(url: Klein3d.fixtures.appendingPathComponent("g3f_scheduler.safetensors"))
        for (seq, steps) in [(256, 4), (1024, 4), (4096, 4), (4096, 8), (4096, 1), (16384, 4)] {
            let want = try Klein3gFix.floats(try #require(g["s\(seq)_n\(steps).sigmas"]))
            #expect(Flux2KleinSampler.sigmas(steps: steps, imageSeqLen: seq) == want, "(\(seq), \(steps))")
        }
    }

    /// The tiny 2-step glue run's schedule (2 x 2 grid), the one the end-to-end golden below walks.
    @Test func theGlueRunsSigmasAreBitIdentical() throws {
        let glue = try loadArrays(url: Klein3d.fixtures.appendingPathComponent("g3g_glue.safetensors"))
        #expect(Flux2KleinSampler.sigmas(steps: 2, imageSeqLen: 4) == (try Klein3gFix.floats(try #require(glue["out.sigmas"]))))
    }

    /// What the transformer is fed in a bf16 pipeline: `bf16(bf16(bf16(sigma x 1000) / 1000) x 1000)` (diffusers 0.37.1 pipeline + `Flux2Transformer2DModel.forward`), worked by hand
    /// for the real 1024 x 1024, 4-step schedule: 1000 -> 1000, 967.38 -> 968, 908.14 -> 908, 767.20 -> 768.
    @Test func theTransformerSigmaFollowsDiffusersBF16RoundingChain() {
        let sigmas = Flux2KleinSampler.sigmas(steps: 4, imageSeqLen: 4096)
        let want: [Float] = [1.0, 0.968, 0.908, 0.768]
        for (s, w) in zip(sigmas, want) {
            #expect(abs(Flux2KleinSampler.transformerSigma(s, dtype: .bfloat16) - w) < 1e-6, "sigma \(s)")
        }
        // float32 pipeline: (sigma x 1000) / 1000, no rounding to a coarser type
        for s in sigmas.dropLast() { #expect(Flux2KleinSampler.transformerSigma(s, dtype: .float32) == (s * 1000) / 1000) }
        #expect(Flux2KleinSampler.transformerSigma(0, dtype: .bfloat16) == 0)
    }

    @Test func softwareBF16RoundingMatchesMLXsCast() throws {
        let values: [Float] = [0, 1, 967.384, 0.96875, 0.968, 908.14, 767.2, 1e-3, 0.7041117, 123456.7, -3.14159, 65504.5, 2.5e-7]
        let viaMLX = try MLXErrorGuard.read(MLXArray(values).asType(.bfloat16).asType(.float32), as: Float.self)
        for (v, m) in zip(values, viaMLX) { #expect(Flux2KleinSampler.roundToBF16(v) == m, "\(v)") }
    }

    @Test func theUpcastingStepEqualsThePlainStepInFloat32AndKeepsTheDtypeInBF16() throws {
        MLXRandom.seed(3)
        let x = MLXRandom.normal([1, 6, 8]), v = MLXRandom.normal([1, 6, 8])
        let plain = Flux2KleinSampler.step(sample: x, velocity: v, sigma: 0.9, sigmaNext: 0.7)
        let up = Flux2KleinSampler.stepUpcasting(sample: x, velocity: v, sigma: 0.9, sigmaNext: 0.7)
        #expect(up.dtype == .float32)
        #expect(try Klein3fTiny.maxAbs(plain, up) == 0)
        let b = Flux2KleinSampler.stepUpcasting(sample: x.asType(.bfloat16), velocity: v.asType(.bfloat16), sigma: 0.9, sigmaNext: 0.7)
        #expect(b.dtype == .bfloat16)
    }
}

// MARK: - Tiny stage source with a lifecycle log

/// One shared log for a tiny run: lifecycle events, and what was resident / what the cache limit was whenever a component was made.
final class Klein3hLog: @unchecked Sendable {
    private let lock = NSLock()
    private var _events: [Flux2KleinLifecycleEvent] = []
    private var _residentAtMake: [(Flux2KleinComponent, Int)] = []
    private var _limitAtMake: [(Flux2KleinComponent, Int)] = []
    private var _mainThreadAtMake: [Bool] = []
    private var _aliveAtRelease: [(Flux2KleinComponent, Bool)] = []
    var events: [Flux2KleinLifecycleEvent] { lock.lock(); defer { lock.unlock() }; return _events }
    var residentAtMake: [(Flux2KleinComponent, Int)] { lock.lock(); defer { lock.unlock() }; return _residentAtMake }
    var limitAtMake: [(Flux2KleinComponent, Int)] { lock.lock(); defer { lock.unlock() }; return _limitAtMake }
    var mainThreadAtMake: [Bool] { lock.lock(); defer { lock.unlock() }; return _mainThreadAtMake }
    var aliveAtRelease: [(Flux2KleinComponent, Bool)] { lock.lock(); defer { lock.unlock() }; return _aliveAtRelease }
    func record(_ e: Flux2KleinLifecycleEvent) { lock.lock(); _events.append(e); lock.unlock() }
    func made(_ c: Flux2KleinComponent, alive: Int, limit: Int) {
        lock.lock(); _residentAtMake.append((c, alive)); _limitAtMake.append((c, limit)); _mainThreadAtMake.append(Thread.isMainThread); lock.unlock()
    }
    func releasedCheck(_ c: Flux2KleinComponent, alive: Bool) { lock.lock(); _aliveAtRelease.append((c, alive)); lock.unlock() }
}

/// A fake `Memory.cacheLimit`.
final class Klein3hFakeLimit: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Int
    init(_ v: Int) { _value = v }
    var value: Int { get { lock.lock(); defer { lock.unlock() }; return _value } set { lock.lock(); _value = newValue; lock.unlock() } }
}

/// Tiny components (g3a transformer, g3c encoder, g3d VAE weights, all fp32), with weak references so the test can ask whether a component is still alive.
final class Klein3hTinyStages: Flux2KleinStageSource, @unchecked Sendable {
    let log: Klein3hLog
    let limit: Klein3hFakeLimit?
    var failVAE = false
    var onMakeVAE: (@Sendable () -> Void)?
    private weak var encoder: Flux2KleinTextEncoder?
    private weak var dit: Flux2KleinTransformer?
    private weak var vae: Flux2KleinVAE?

    init(log: Klein3hLog = Klein3hLog(), limit: Klein3hFakeLimit? = nil) { self.log = log; self.limit = limit }

    var computeDType: DType { .float32 }
    var packedChannels: Int { 16 }

    static let transformerShape = Flux2KleinTransformerShape(embedder: Klein3d.shape, heads: 4, headDim: 16, mlpRatio: 3.0, doubleBlocks: 1, singleBlocks: 2)

    private var aliveCount: Int { (encoder == nil ? 0 : 1) + (dit == nil ? 0 : 1) + (vae == nil ? 0 : 1) }

    /// Token ids the tiny vocabulary (128) can hold, shaped like the real thing: 512 ids, the first 8 real.
    func encodePrompt(_ prompt: String) async throws -> Flux2KleinEncodedPrompt {
        let ids = (0 ..< 512).map { Int32($0 < 8 ? ($0 * 7 + 3) % 120 : 100) }
        return Flux2KleinEncodedPrompt(ids: ids, attentionMask: (0 ..< 512).map { $0 < 8 ? 1 : 0 }, realLength: 8, cutTokenCount: 0)
    }

    func makeTextEncoder(quantize: Int) throws -> Flux2KleinTextEncoder {
        log.made(.textEncoder, alive: aliveCount, limit: limit?.value ?? -1)
        let e = try Flux2KleinTextEncoderTests.encoder()
        encoder = e
        return e
    }

    func makeTransformer(quantize: Int) throws -> Flux2KleinTransformer {
        log.made(.transformer, alive: aliveCount, limit: limit?.value ?? -1)
        var w: [String: MLXArray] = [:]
        for (k, v) in Klein3d.g3a where k.hasPrefix("w.") { w[String(k.dropFirst(2))] = v }
        let m = try Flux2KleinTransformer.load(shape: Self.transformerShape, shards: [{ w }], bits: nil)
        dit = m
        return m
    }

    func makeVAE() throws -> Flux2KleinVAE {
        log.made(.vae, alive: aliveCount, limit: limit?.value ?? -1)
        onMakeVAE?()
        if failVAE { throw Klein3hError.injected }
        let g = try Klein3gFix.tiny()
        let v = try Flux2KleinVAE.load(shape: g.shape, tensors: g.weights, computeDType: nil)
        vae = v
        return v
    }

    /// Called by the observer on `.released(c)`: is that component's object really gone?
    func checkReleased(_ c: Flux2KleinComponent) {
        switch c {
        case .textEncoder: log.releasedCheck(c, alive: encoder != nil)
        case .transformer: log.releasedCheck(c, alive: dit != nil)
        case .vae: log.releasedCheck(c, alive: vae != nil)
        }
    }
}

enum Klein3hError: Error { case injected }

private func tinyEngine(_ stages: Klein3hTinyStages, limit: Klein3hFakeLimit? = nil, budget: Int64 = 40_000_000_000) -> Flux2KleinEngine {
    Flux2KleinEngine(stages: stages, budgetBytes: { budget },
                     readCacheLimit: { limit?.value ?? Memory.cacheLimit }, writeCacheLimit: { v in if let limit { limit.value = v } else { Memory.cacheLimit = v } },
                     observer: { e in stages.log.record(e); if case let .released(c) = e { stages.checkReleased(c) } })
}

/// A 32 x 32 request (a 2 x 2 packed grid: the tiny VAE's 8x upsample of a 4 x 4 latent).
private let tinyRequest = Flux2KleinRequest(prompt: "a tiny test prompt", width: 32, height: 32, steps: 2, seed: 7, quantize: 8)

// MARK: - Stage order, residency, cache limit

@Suite(.serialized) struct Flux2KleinEngineLifecycleTests {
    @Test func stagesRunInTheOwnersOrder() async throws {
        let stages = Klein3hTinyStages()
        let result = try await tinyEngine(stages).generate(tinyRequest)
        let e = stages.log.events
        let want = ["limit:textEncoder", "loaded:textEncoder", "embeddings", "released:textEncoder", "cleared",
                    "limit:transformer", "loaded:transformer", "step:0", "step:1", "latents", "released:transformer", "cleared",
                    "limit:vae", "loaded:vae", "released:vae", "cleared"]
        #expect(e.map(Self.name) == want, "events: \(e)")
        #expect(result.width == 32 && result.height == 32 && result.image.width == 32 && result.image.height == 32)
        #expect(result.pixels.shape == [1, 3, 32, 32])
        #expect(result.stages.map(\.component) == [.textEncoder, .transformer, .vae])
    }

    private static func name(_ e: Flux2KleinLifecycleEvent) -> String {
        switch e {
        case let .cacheLimitApplied(c, _): return "limit:\(c.rawValue)"
        case let .loaded(c): return "loaded:\(c.rawValue)"
        case .embeddingsEvaluated: return "embeddings"
        case let .denoiseStep(i): return "step:\(i)"
        case .latentsEvaluated: return "latents"
        case let .released(c): return "released:\(c.rawValue)"
        case .cacheCleared: return "cleared"
        }
    }

    /// Done-when (a): a component is built only after the previous one is really gone (weak references), and nothing is alive at any `.released`.
    @Test func noTwoOfEncoderDiTAndVAEAreEverResidentTogether() async throws {
        let stages = Klein3hTinyStages()
        _ = try await tinyEngine(stages).generate(tinyRequest)
        let made = stages.log.residentAtMake
        #expect(made.map(\.0) == [.textEncoder, .transformer, .vae])
        for (component, alive) in made { #expect(alive == 0, "\(component) was built while \(alive) other component(s) were still alive") }
        let released = stages.log.aliveAtRelease
        #expect(released.count == 3)
        for (component, alive) in released { #expect(!alive, "\(component) was still alive when .released fired") }
    }

    /// The same, when the run dies in the middle: the DiT is gone before the error leaves, and no VAE was built.
    @Test func aFailureInTheVAEStageStillReleasesTheDiT() async throws {
        let stages = Klein3hTinyStages()
        stages.failVAE = true
        await #expect(throws: Klein3hError.self) { _ = try await tinyEngine(stages).generate(tinyRequest) }
        #expect(stages.log.aliveAtRelease.map(\.0) == [.textEncoder, .transformer])
        #expect(stages.log.aliveAtRelease.allSatisfy { !$0.1 })
    }

    @Test @MainActor func theEngineRunsOffTheMainThreadEvenWhenCalledFromTheMainActor() async throws {
        let stages = Klein3hTinyStages()
        let result = try await tinyEngine(stages).generate(tinyRequest)
        #expect(result.image.width == 32)
        #expect(stages.log.mainThreadAtMake == [false, false, false], "component loads ran on the main thread")
    }
}

@Suite(.serialized) struct Flux2KleinCacheLimitTests {
    static let found = 3_000_000_000

    @Test func everyStageIsCappedAndTheFoundValueComesBackAfterADoneRun() async throws {
        let limit = Klein3hFakeLimit(Self.found)
        let stages = Klein3hTinyStages(limit: limit)
        _ = try await tinyEngine(stages, limit: limit).generate(tinyRequest)
        let cap = Flux2KleinCacheLimits.standard
        #expect(stages.log.limitAtMake.map(\.0) == [.textEncoder, .transformer, .vae])
        #expect(stages.log.limitAtMake.map(\.1) == [cap.textEncoder, cap.transformer, cap.vae], "each stage must see its own cap in force when its component is built")
        #expect(limit.value == Self.found, "the limit must be put back")
    }

    @Test func aLowerLimitSomeoneElseSetIsNeverRaised() async throws {
        let limit = Klein3hFakeLimit(64 * 1_048_576)
        let stages = Klein3hTinyStages(limit: limit)
        _ = try await tinyEngine(stages, limit: limit).generate(tinyRequest)
        #expect(stages.log.limitAtMake.map(\.1) == [64 * 1_048_576, 64 * 1_048_576, 64 * 1_048_576])
        #expect(limit.value == 64 * 1_048_576)
    }

    @Test func aThrownRunRestoresIt() async throws {
        let limit = Klein3hFakeLimit(Self.found)
        let stages = Klein3hTinyStages(limit: limit)
        stages.failVAE = true
        await #expect(throws: Klein3hError.self) { _ = try await tinyEngine(stages, limit: limit).generate(tinyRequest) }
        #expect(stages.log.limitAtMake.count == 3, "the run reached the VAE stage with the cap in force: \(stages.log.limitAtMake)")
        #expect(limit.value == Self.found)
    }

    final class TaskBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<Flux2KleinResult, Error>?
        func set(_ t: Task<Flux2KleinResult, Error>) { lock.lock(); task = t; lock.unlock() }
        func cancel() { lock.lock(); let t = task; lock.unlock(); t?.cancel() }
    }

    @Test func aCancelledRunRestoresItAndStopsBetweenSteps() async throws {
        let limit = Klein3hFakeLimit(Self.found)
        let stages = Klein3hTinyStages(limit: limit)
        let engine = tinyEngine(stages, limit: limit)
        let box = TaskBox()
        var request = tinyRequest
        request.steps = 4
        let task = Task { try await engine.generate(request, progress: { p in if p.step == 1 { box.cancel() } }) }
        box.set(task)
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        let steps = stages.log.events.filter { if case .denoiseStep = $0 { return true } else { return false } }.count
        #expect(steps >= 1 && steps < 4, "cancellation is checked every step; ran \(steps) of 4")
        #expect(!stages.log.events.contains(.loaded(.vae)), "no VAE once cancelled")
        #expect(stages.log.aliveAtRelease.contains { $0.0 == .transformer && !$0.1 }, "the DiT is released on the cancelled path")
        #expect(limit.value == Self.found)
    }

    @Test func aRunCancelledBeforeItStartsChangesNothing() async throws {
        let limit = Klein3hFakeLimit(Self.found)
        let stages = Klein3hTinyStages(limit: limit)
        let engine = tinyEngine(stages, limit: limit)
        let task = Task { () -> Flux2KleinResult in
            withUnsafeCurrentTask { $0?.cancel() }
            return try await engine.generate(tinyRequest)
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(stages.log.events.isEmpty)
        #expect(limit.value == Self.found)
    }

    /// The Local Server (or EngineCache) changing the limit while a Klein run holds it: their value wins, ours is not written over it.
    @Test func aLimitChangedByAnotherHolderMeanwhileIsLeftAlone() async throws {
        let limit = Klein3hFakeLimit(Self.found)
        let stages = Klein3hTinyStages(limit: limit)
        stages.onMakeVAE = { limit.value = 777_000_000 }
        _ = try await tinyEngine(stages, limit: limit).generate(tinyRequest)
        #expect(limit.value == 777_000_000)
    }

    @Test func theGuardIsIdempotentAndNeverWritesBeforeAnApply() {
        let limit = Klein3hFakeLimit(500)
        let g = Flux2KleinCacheLimitGuard(read: { limit.value }, write: { limit.value = $0 })
        g.restore()                     // nothing applied: no write
        #expect(limit.value == 500)
        let g2 = Flux2KleinCacheLimitGuard(read: { limit.value }, write: { limit.value = $0 })
        #expect(g2.apply(cap: 100) == 100 && limit.value == 100)
        g2.restore(); g2.restore()
        #expect(limit.value == 500)
    }

    /// The real global, through the production closures: a tiny run leaves `Memory.cacheLimit` as it found it.
    @Test func theRealMemoryCacheLimitIsPutBack() async throws {
        let previous = Memory.cacheLimit
        defer { Memory.cacheLimit = previous }
        Memory.cacheLimit = 2_000_000_000
        let stages = Klein3hTinyStages()
        let engine = Flux2KleinEngine(stages: stages, budgetBytes: { 40_000_000_000 })
        _ = try await engine.generate(tinyRequest)
        #expect(Memory.cacheLimit == 2_000_000_000)
    }
}

// MARK: - quantize and size (done-when c, g)

@Suite struct Flux2KleinEngineValidationTests {
    @Test(arguments: [4, 8]) func supportedWidthsPass(q: Int) throws {
        try Flux2KleinEngine.validate(quantize: q)
        var r = tinyRequest; r.quantize = q
        try Flux2KleinEngine.validate(r, budgetBytes: 40_000_000_000)
    }

    /// Owner ruling 2026-11-01: the engine takes {4, 8} only, even though the loader still knows 3, 5, 6.
    @Test(arguments: [3, 5, 6, 7, 0, 1, 2, 9, 16, -4]) func everythingButFourAndEightIsRefusedInOneSentenceNamingTheRowAndBothValues(q: Int) {
        #expect(throws: Flux2KleinEngineError.unsupportedQuantize(row: "Row 3", quantize: q)) { try Flux2KleinEngine.validate(quantize: q, row: "Row 3") }
        let sentence = Flux2KleinEngineError.unsupportedQuantize(row: "Row 3", quantize: q).errorDescription ?? ""
        #expect(sentence == "Row 3: FLUX.2 Klein 4B runs at quantize 4 or 8 in this build, not \(q).")
        #expect(!sentence.contains("text encoder"))
        var r = tinyRequest; r.quantize = q; r.row = "Row 3"
        #expect(throws: Flux2KleinEngineError.unsupportedQuantize(row: "Row 3", quantize: q)) { try Flux2KleinEngine.validate(r, budgetBytes: 40_000_000_000) }
    }

    @Test func theLoadersOwnRangeIsUntouched() {
        #expect(Flux2KleinQuantization.ditSupportedBits == [3, 4, 5, 6, 8])
        #expect(Flux2KleinQuantization.acceptedRowQuantize == [4, 8])
    }

    @Test func aBadQuantizeIsRefusedBeforeAnythingLoads() async {
        let stages = Klein3hTinyStages()
        var r = tinyRequest; r.quantize = 5
        await #expect(throws: Flux2KleinEngineError.unsupportedQuantize(row: r.row, quantize: 5)) { _ = try await tinyEngine(stages).generate(r) }
        #expect(stages.log.events.isEmpty && stages.log.residentAtMake.isEmpty, "nothing may be loaded or capped for a refused row")
    }

    @Test func theEncodersOwnSentenceMentionsOnlyItsEightBitFloor() {
        let s = Flux2KleinTextEncoderError.unsupportedBits(7).errorDescription ?? ""
        #expect(s.contains("8 bits or more") && !s.contains("image model") && !s.contains("3, 4, 5, 6") && !s.contains("quantize 4 or 8"))
    }

    @Test func stepsMustBePositive() {
        var r = tinyRequest; r.steps = 0
        #expect(throws: Flux2KleinEngineError.invalidSteps(0)) { try Flux2KleinEngine.validate(r, budgetBytes: 40_000_000_000) }
    }

    /// Multiples of 16 only (SPEC-Q242): refused with the existing sentence, and before the memory check.
    @Test func sizesNotAMultipleOfSixteenAreRefusedBeforeTheMemoryCheck() {
        var r = tinyRequest; r.width = 1000; r.height = 1020
        #expect(throws: Flux2KleinSizeError.notMultipleOf16(width: 1000, height: 1020)) { try Flux2KleinEngine.validate(r, budgetBytes: 1) }
        r.width = 0
        #expect(throws: Flux2KleinSizeError.self) { try Flux2KleinEngine.validate(r, budgetBytes: 1) }
    }

    /// 60% of a 19.3 GB Mac: `Flux2KleinMemoryPlan.budgetBytes(physicalMemory:)`, Klein's own budget (owner ruling 2026-11-01).
    static let thisMac = Flux2KleinMemoryPlan.budgetBytes(physicalMemory: 19_300_000_000)

    @Test func twelveEightyBySquareFitsOnThisMacAtEveryQuantize() throws {
        for q in [4, 8] {
            var r = tinyRequest; r.width = 1280; r.height = 1280; r.quantize = q
            try Flux2KleinEngine.validate(r, budgetBytes: Self.thisMac)
        }
    }

    @Test func aSizeAboveTheCapIsRefusedNamingTheRowTheSizeAskedAndTheLargestThatFits() throws {
        var r = tinyRequest; r.width = 2048; r.height = 2048
        r.row = "FLUX.2 Klein 4B"
        let error = try #require({ () -> Flux2KleinEngineError? in
            do { try Flux2KleinEngine.validate(r, budgetBytes: Self.thisMac); return nil } catch let e as Flux2KleinEngineError { return e } catch { return nil }
        }())
        guard case let .sizeExceedsMemory(row, w, h, q, needed, available, largest) = error else { Issue.record("wrong error \(error)"); return }
        #expect(row == "FLUX.2 Klein 4B" && w == 2048 && h == 2048 && q == 8)
        #expect(needed * 1e9 > Double(Self.thisMac) && abs(available - Double(Self.thisMac) / 1e9) < 1e-9)
        #expect(largest % 16 == 0 && largest >= 1280 && largest < 2048)
        // the claim is exact: `largest` fits, one multiple of 16 more does not
        #expect(Flux2KleinMemoryPlan.peaks(width: largest, height: largest, quantize: 8).busiest * 1.15 * 1e9 <= Double(Self.thisMac))
        #expect(Flux2KleinMemoryPlan.peaks(width: largest + 16, height: largest + 16, quantize: 8).busiest * 1.15 * 1e9 > Double(Self.thisMac))
        let sentence = error.errorDescription ?? ""
        #expect(sentence.contains("FLUX.2 Klein 4B") && sentence.contains("2048 x 2048") && sentence.contains("\(largest) x \(largest)") && sentence.contains("quantize 8"))
    }

    @Test func aBiggerMacGetsABiggerCapAndASmallOneRefusesEverything() throws {
        let big = Int64(64e9 * 0.6)
        var r = tinyRequest; r.width = 2048; r.height = 2048
        try Flux2KleinEngine.validate(r, budgetBytes: big)            // "a 64 GB Mac still gets 2048 x 2048"
        #expect(Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: 1_000_000_000) == 0)
        r.width = 512; r.height = 512
        let error = try #require({ () -> Flux2KleinEngineError? in
            do { try Flux2KleinEngine.validate(r, budgetBytes: 1_000_000_000); return nil } catch let e as Flux2KleinEngineError { return e } catch { return nil }
        }())
        #expect(error.errorDescription?.contains("any size") == true)
    }

    /// Boundary: the budget exactly at the margin-scaled peak admits the size; one byte under refuses it.
    @Test(arguments: [4, 8]) func theBudgetBoundaryIsTheBusiestPeakTimesOnePointOneFive(q: Int) throws {
        var r = tinyRequest; r.width = 1024; r.height = 1024; r.quantize = q
        let need = Flux2KleinMemoryPlan.peaks(width: 1024, height: 1024, quantize: q).busiest * 1.15 * 1e9
        try Flux2KleinEngine.validate(r, budgetBytes: Int64(need.rounded(.up)))
        #expect(throws: Flux2KleinEngineError.self) { try Flux2KleinEngine.validate(r, budgetBytes: Int64(need.rounded(.down)) - 1) }
    }

    @Test func theDefaultBudgetIsSixtyPercentOfPhysicalMemoryNotTheSharedLedger() {
        #expect(Flux2KleinMemoryPlan.budgetBytes(physicalMemory: 16_000_000_000) == 9_600_000_000)
        #expect(Flux2KleinMemoryPlan.defaultBudgetBytes == Int64(Double(ProcessInfo.processInfo.physicalMemory) * 0.6))
    }

    @Test func capsOnThisMacAndOnCommonMemorySizes() {
        for gb in [16.0, 19.3, 32, 64] {
            let budget = Flux2KleinMemoryPlan.budgetBytes(physicalMemory: UInt64(gb * 1e9))
            let q8 = Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: budget)
            let q4 = Flux2KleinMemoryPlan.largestSquareSide(quantize: 4, budgetBytes: budget)
            print("KLEINCAP \(gb) GB budget \(budget): q8 \(q8), q4 \(q4)")
            #expect(q8 % 16 == 0 && q4 >= q8)
        }
        let here = Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: Self.thisMac)
        #expect(here >= 1024 && here <= 1360)
    }

    @Test func theCapUsesTheBitsActuallyChosen() {
        // fewer bits -> smaller DiT -> the DiT stage is cheaper; the weights model is the measured int8 / int4 pair
        #expect(abs(Flux2KleinMemoryPlan.ditWeightsGB(quantize: 8) - 4.128) < 1e-9)
        #expect(abs(Flux2KleinMemoryPlan.ditWeightsGB(quantize: 4) - 2.196) < 1e-9)
        let at8 = Flux2KleinMemoryPlan.peaks(width: 1024, height: 1024, quantize: 8)
        let at4 = Flux2KleinMemoryPlan.peaks(width: 1024, height: 1024, quantize: 4)
        #expect(at4.transformer < at8.transformer && at4.vae == at8.vae && at4.encoder == at8.encoder)
        #expect(Flux2KleinMemoryPlan.largestSquareSide(quantize: 4, budgetBytes: Self.thisMac) >= Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: Self.thisMac))
    }
}

/// The cost model is never optimistic about a size it was measured at: every measured peak (MLX `Memory.peakMemory`, decimal GB, this Mac, cache capped at 256 MB) is at or under the
/// plan's figure for that stage, and the plan is not wildly pessimistic either. Measured in MP7-3c / 3f / 3g / 3h (MP7-3h-report.md).
@Suite struct Flux2KleinMemoryPlanMeasuredTests {
    /// (side, measured DiT working GB = MLX peak minus the 4.128 GB of int8 weights) with per-block evaluation, and without.
    static let ditWithEval: [(Int, Double)] = [(512, 0.38), (768, 0.69), (1024, 1.09), (1280, 1.59), (2048, 3.46)]
    static let ditNoEval: [(Int, Double)] = [(512, 1.42), (768, 1.46), (1024, 1.54), (1280, 1.97), (1536, 2.29), (2048, 3.97)]
    static let vae: [(Int, Double)] = [(512, 1.66), (1024, 5.74), (1280, 8.92)]

    @Test func theDiTFigureIsAtOrOverEveryMeasuredPointAndNotWildlyOver() {
        for (flag, points) in [(true, Self.ditWithEval), (false, Self.ditNoEval)] {
            for (side, working) in points {
                let plan = Flux2KleinMemoryPlan.peaks(width: side, height: side, quantize: 8, evaluatesEachBlock: flag).transformer - Flux2KleinMemoryPlan.ditWeightsGB(quantize: 8)
                #expect(plan >= working, "eval-each-block \(flag) \(side): plan \(plan) under measured \(working)")
                #expect(plan <= working + 1.3, "eval-each-block \(flag) \(side): plan \(plan) far over measured \(working)")
            }
        }
    }

    @Test func theVAEFigureIsAtOrOverEveryMeasuredPoint() {
        for (side, peak) in Self.vae {
            let plan = Flux2KleinMemoryPlan.peaks(width: side, height: side, quantize: 8).vae
            #expect(plan >= peak && plan <= peak + 0.4, "\(side): plan \(plan) vs measured \(peak)")
        }
    }

    @Test func theDiTStagePeaksAtItsMeasuredLoadPeakAtTheSmallestSizes() {
        // streamed load peaks (3f): int8 4.50, int4 2.56 GB
        #expect(Flux2KleinMemoryPlan.peaks(width: 16, height: 16, quantize: 8).transformer >= 4.50)
        #expect(Flux2KleinMemoryPlan.peaks(width: 16, height: 16, quantize: 4).transformer >= 2.56)
        #expect(Flux2KleinMemoryPlan.peaks(width: 16, height: 16, quantize: 8).encoder == 3.83)
    }

    @Test func turningPerBlockEvaluationOffNeverLowersTheDiTFigure() {
        for side in [256, 512, 1024, 1536, 2048] {
            let on = Flux2KleinMemoryPlan.peaks(width: side, height: side, quantize: 8, evaluatesEachBlock: true).transformer
            let off = Flux2KleinMemoryPlan.peaks(width: side, height: side, quantize: 8, evaluatesEachBlock: false).transformer
            #expect(off >= on, "\(side)")
        }
    }
}

// MARK: - One conditioning source (done-when f)

@Suite struct Flux2KleinConditioningSourceTests {
    /// The transformer and the 3d embedders must not each carry their own copy of "temb + modulation chunks": both call `Conditioning.make`.
    @Test func noCopyOfTheConditioningLinesRemainsInTheTransformer() throws {
        let dir = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("MLXUI/Modules/Flux2Klein")
        let transformer = try String(contentsOf: dir.appendingPathComponent("Flux2KleinTransformer.swift"), encoding: .utf8)
        let embedders = try String(contentsOf: dir.appendingPathComponent("Flux2KleinEmbedders.swift"), encoding: .utf8)
        #expect(transformer.contains("Conditioning.make(") && embedders.contains("Conditioning.make("))
        for needle in ["doubleModulationImg.chunks(", "doubleModulationTxt.chunks(", "singleModulation.chunks(", "timeGuidanceEmbed(sigma:"] {
            #expect(!transformer.contains(needle), "the transformer duplicates the conditioning again: \(needle)")
        }
        // exactly one definition of the computation
        #expect(embedders.components(separatedBy: "static func make(sigma:").count == 2)
    }

    @Test func embeddersAndTransformerProduceTheSameConditioning() throws {
        let emb = try Klein3d.embedders()
        let sigma = MLXArray([0.7] as [Float])
        let c = emb.conditioning(sigma: sigma)
        let again = Flux2KleinEmbedders.Conditioning.make(sigma: sigma, dtype: .float32, time: emb.timeGuidanceEmbed, doubleImg: emb.doubleModulationImg,
                                                          doubleTxt: emb.doubleModulationTxt, single: emb.singleModulation)
        #expect(try Klein3fTiny.maxAbs(c.temb, again.temb) == 0)
        #expect(try Klein3fTiny.maxAbs(c.single[0].gate, again.single[0].gate) == 0)
    }
}

// MARK: - Per-block evaluation is the same arithmetic

@Suite(.serialized) struct Flux2KleinPerBlockEvalTests {
    @Test func evaluatingEveryBlockGivesTheSameResult() throws {
        let m = try Klein3f.model()
        let plain = try Klein3f.run(m)
        let each = try m(hidden: try #require(Klein3f.g3f["in.hidden_states"]), encoder: try #require(Klein3f.g3f["in.encoder_hidden_states"]),
                         sigma: try #require(Klein3f.g3f["in.timestep"]), imgIDs: try #require(Klein3f.g3f["in.img_ids"]), txtIDs: try #require(Klein3f.g3f["in.txt_ids"]),
                         evaluatesEachBlock: true)
        #expect(try Klein3fTiny.maxAbs(plain, each) == 0)
    }
}

// MARK: - The whole tiny run against diffusers (g3g glue golden)

@Suite(.serialized) struct Flux2KleinEngineGoldenTests {
    /// Floors set AFTER the first measurement (2026-11-01, MP7-3h-report.md): per-step latents max-abs 2.4e-7 / 6.0e-7, image max-abs 2.4e-6, 1 - cosine <= 1.6e-14, PSNR 130.8 dB.
    /// Pinned about 4-8x above the measured values (pins of this port's behaviour, not quality claims); the owner's targets for this slice were final latent cosine >= 0.99999
    /// and per-step latents max-abs <= 1e-3, which these are far inside. The margins are the implementer's call, pending owner confirmation.
    static let stepMaxAbs: Float = 4e-6
    static let latentCosine = 0.999999
    static let imageMaxAbs = 2e-5
    static let imageCosine = 0.999999

    final class StepCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var items: [(Int, MLXArray)] = []
        func add(_ i: Int, _ a: MLXArray) { lock.lock(); items.append((i, a)); lock.unlock() }
        var all: [(Int, MLXArray)] { lock.lock(); defer { lock.unlock() }; return items }
    }

    @Test func theTwoStepTinyRunMatchesDiffusersStepByStepAndInThePicture() async throws {
        let glue = try loadArrays(url: Klein3d.fixtures.appendingPathComponent("g3g_glue.safetensors"))
        let collector = StepCollector()
        let stages = Klein3hTinyStages()
        let engine = Flux2KleinEngine(stages: stages, budgetBytes: { 40_000_000_000 })
        var injection = Flux2KleinInjection(noise: try #require(glue["in.latents"]), embeddings: try #require(glue["in.txt"]))
        injection.stepProbe = { i, x in collector.add(i, x) }
        let result = try await engine.generate(tinyRequest, injection: injection)

        let steps = collector.all.sorted { $0.0 < $1.0 }
        #expect(steps.map(\.0) == [0, 1])
        for (i, x) in steps {
            let want = try #require(glue["step\(i).latents"])
            let maxAbs = try Klein3fTiny.maxAbs(x, want)
            let cos = try KleinCosine.of(x, want)
            print("ENGINEGOLDEN step \(i) latents: max-abs \(maxAbs), 1-cos \(1 - cos)")
            #expect(maxAbs <= Self.stepMaxAbs, "step \(i)")
            #expect(cos >= Self.latentCosine, "step \(i)")
        }
        let image = try Klein3gFix.compare(result.pixels, try #require(glue["out.image"]))
        print("ENGINEGOLDEN image: max-abs \(image.maxAbs), cos \(image.cosine), psnr \(image.psnr) dB")
        #expect(image.maxAbs <= Self.imageMaxAbs && image.cosine >= Self.imageCosine)
        #expect(result.image.width == 32 && result.image.height == 32)
        // the encoder stage was skipped (embeddings injected): only the DiT and the VAE were ever built
        #expect(stages.log.residentAtMake.map(\.0) == [.transformer, .vae])
    }

    @Test func theInjectedNoiseMustHaveTheRightShape() async throws {
        let stages = Klein3hTinyStages()
        let engine = Flux2KleinEngine(stages: stages, budgetBytes: { 40_000_000_000 })
        let bad = Flux2KleinInjection(noise: MLXArray.zeros([1, 128, 2, 2]))
        await #expect(throws: Flux2KleinEngineError.self) { _ = try await engine.generate(tinyRequest, injection: bad) }
    }

    /// Without injection the seed decides the picture: same seed, same pixels; another seed, other pixels.
    @Test func theSeedIsReproducibleThroughTheWholePipeline() async throws {
        let engine = Flux2KleinEngine(stages: Klein3hTinyStages(), budgetBytes: { 40_000_000_000 })
        var a = tinyRequest; a.seed = 11
        let first = try await engine.generate(a), second = try await engine.generate(a)
        a.seed = 12
        let other = try await engine.generate(a)
        #expect(try Klein3fTiny.maxAbs(first.pixels, second.pixels) == 0)
        #expect(try Klein3fTiny.maxAbs(first.pixels, other.pixels) > 1e-4)
    }
}

// MARK: - Main-actor probe (KICKOFF 17c), tiny

@MainActor
struct Flux2KleinEngineMainActorProbeTests {
    /// Run the way the Run UI calls it: from the main actor, inside `guardedRun`. The real-weight twin is in Flux2KleinEngineRealTests.
    @Test func runsFromTheMainActorInsideTheErrorGuard() async throws {
        let stages = Klein3hTinyStages()
        let engine = Flux2KleinEngine(stages: stages, budgetBytes: { 40_000_000_000 })
        let result = try await MLXErrorGuard.guardedRun { try await engine.generate(tinyRequest) }
        #expect(result.pixels.shape == [1, 3, 32, 32])
        #expect(stages.log.mainThreadAtMake == [false, false, false])
    }

    /// An MLX error recorded during a readback becomes a thrown error, not a crash (the injected failure fires on the pixel readback's `MLXErrorGuard.read`).
    @Test func anMLXErrorOnTheReadbackIsAThrownError() async throws {
        let engine = Flux2KleinEngine(stages: Klein3hTinyStages(), budgetBytes: { 40_000_000_000 })
        MLXErrorGuard.injectFailureForTesting("injected for the Klein engine")
        defer { MLXErrorGuard.clearInjectedFailureForTesting() }
        await #expect(throws: (any Error).self) { _ = try await MLXErrorGuard.guardedRun { try await engine.generate(tinyRequest) } }
        #expect(MLXErrorGuard.injectedFailureWasConsumedForTesting())
    }
}

// MARK: - Registry (spike 7.3)

@MainActor
struct Flux2KleinRegistryRoutingTests {
    /// A stand-in for the Klein module MP7-4 will register (this slice registers nothing): claims Klein exactly, nothing else.
    struct StandInKleinSDK: ModelSDK {
        let id = "flux2-klein-standin"
        func claim(_ model: ModelEntry) -> ClaimScore {
            (model.id + " " + model.hfModelId).lowercased().contains("klein") ? .exact : .no
        }
        func makeStage(for model: ModelEntry, config: StageConfig) throws -> any PipelineStage { throw StageError.unsupportedSetting(setting: "standin") }
    }

    private static let descriptor = ModelModuleDescriptor(id: "flux2-klein-standin", displayName: "stand-in", modalities: [], modelTypes: [.image], backingPackage: "",
                                                           packageLicense: "", maintainers: [], notes: "")

    @Test(arguments: [("mlx-community--FLUX.2-klein-4B-bf16", "mlx-community/FLUX.2-klein-4B-bf16"),
                      ("mlx-community--flux2-klein-9b-8bit", "mlx-community/flux2-klein-9b-8bit"),
                      ("x--FLUX-2-dev", "x/FLUX-2-dev"),
                      ("mlx-community--Klein-future", "mlx-community/Klein-future")])
    func theFlux1EngineNeverClaimsAFlux2FamilyId(id: String, hf: String) {
        let e = makeEntry(id: id, displayName: "FLUX.2", modelType: .image, source: .mlx, hfRepo: hf, hfModelId: hf)
        #expect(FluxSDK().claim(e) == .no)
    }

    @Test func klein_goesToTheKleinModuleWhicheverOrderTheyRegisterAndFlux1StaysWithFlux1() {
        let klein = makeEntry(id: "mlx-community--FLUX.2-klein-4B-bf16", displayName: "FLUX.2 Klein 4B", modelType: .image, source: .mlx,
                              hfRepo: "mlx-community/FLUX.2-klein-4B-bf16", hfModelId: "mlx-community/FLUX.2-klein-4B-bf16")
        let flux1 = makeEntry(id: "mlx-community--Flux-1.lite-8B-MLX-Q4", displayName: "FLUX.1 lite", modelType: .image, source: .mlx,
                              hfRepo: "mlx-community/Flux-1.lite-8B-MLX-Q4", hfModelId: "mlx-community/Flux-1.lite-8B-MLX-Q4")
        for fluxFirst in [true, false] {
            let registry = ModelRegistry()
            let adds: [() -> Void] = [
                { registry.add(sdk: FluxSDK(), ui: FluxUI(), descriptor: FluxModule.descriptor) },
                { registry.add(sdk: StandInKleinSDK(), ui: FluxUI(), descriptor: Self.descriptor) },
            ]
            for add in (fluxFirst ? adds : adds.reversed()) { add() }
            #expect(registry.bestModule(for: klein)?.descriptor.id == "flux2-klein-standin", "fluxFirst \(fluxFirst)")
            #expect(registry.bestModule(for: flux1)?.descriptor.id == "flux", "fluxFirst \(fluxFirst)")
        }
    }

    /// MP7-4 registers the Klein module (the Local Server scan lives in Flux2KleinRegistryClaimTests).
    @Test func theKleinModuleIsRegistered() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("MLXUI")
        let modules = try String(contentsOf: root.appendingPathComponent("App/ModelModules.swift"), encoding: .utf8)
        #expect(modules.contains("Flux2KleinModule.self"))
    }
}
