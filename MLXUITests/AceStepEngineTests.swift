import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3i: the engine end to end on a TINY random checkpoint folder built from the existing tiny fixtures (DiT, condition encoder, VAE), plus the
/// stage, the SDK claim and refusals, the module registration and the cache-limit guard. No real weights; the parts' parity is in their own suites.
@Suite(.serialized)
struct AceStepEngineTests {

    // MARK: Tiny model folder

    /// The tiny text side: fixed random tensors (the real one is Qwen3, covered by 3d).
    struct TinyText: AceStepTextSource {
        var fail = false
        func inputs(for request: AceStepRequest) async throws -> AceStepTextInputs {
            if fail { throw AceStepEngineError.emptyPrompt }
            let key = MLXRandom.key(99)
            let hidden = MLXRandom.normal([1, 6, 16], key: key) * 0.5, lyric = MLXRandom.normal([1, 5, 16], key: key) * 0.5
            return AceStepTextInputs(textIds: [], textMask: MLXArray.ones([1, 6], dtype: .int32), lyricIds: [],
                                     lyricMask: MLXArray.ones([1, 5], dtype: .int32), textHidden: hidden, lyricEmbeds: lyric)
        }
    }

    /// Writes `acestep-v15-turbo/{config.json,model.safetensors}` and `vae/{config.json,diffusion_pytorch_model.safetensors}` under a temp folder.
    static func makeFolder() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-tiny-\(UUID().uuidString)")
        let turbo = dir.appendingPathComponent("acestep-v15-turbo"), vae = dir.appendingPathComponent("vae")
        try FileManager.default.createDirectory(at: turbo, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: vae, withIntermediateDirectories: true)

        let dit = try SafetensorsIndex(url: AceStepDiTTests.fixtureURL)
        let ditAll = try dit.read(dit.names)
        let cond = try SafetensorsIndex(url: AceStepConditionEncoderTests.fixtureURL)
        let condAll = try cond.read(cond.names)

        var arrays: [String: MLXArray] = [:]
        for (k, v) in ditAll where k.hasPrefix("w.") { arrays["decoder." + k.dropFirst(2)] = v }
        for (k, v) in condAll where k.hasPrefix("w.") { arrays["encoder." + k.dropFirst(2)] = v }
        // The no-reference timbre input is silence[:750] with 64 channels, so the tiny timbre encoder takes 64, not the fixture's 8.
        arrays["encoder.timbre_encoder.embed_tokens.weight"] = MLXRandom.normal([32, 64], key: MLXRandom.key(5)) * 0.1
        try MLX.save(arrays: arrays, url: turbo.appendingPathComponent("model.safetensors"))

        let configText = try #require(dit.metadata["config"])
        var config = try #require(JSONSerialization.jsonObject(with: Data(configText.utf8)) as? [String: Any])
        config["timbre_hidden_dim"] = 64
        try JSONSerialization.data(withJSONObject: config).write(to: turbo.appendingPathComponent("config.json"))

        let raw = try AceStepVAETests.loadRaw()
        var vaeArrays: [String: MLXArray] = [:]
        for (k, v) in raw.tensors where k.hasPrefix("w.") { vaeArrays[String(k.dropFirst(2))] = v }
        try MLX.save(arrays: vaeArrays, url: vae.appendingPathComponent("diffusion_pytorch_model.safetensors"))
        try raw.config.write(to: vae.appendingPathComponent("config.json"))
        return dir
    }

    static func silence() -> MLXArray { MLXRandom.normal([15_000, 64], key: MLXRandom.key(3)) * 0.1 }

    static func engine(_ dir: URL, text: any AceStepTextSource = TinyText(), observer: (@Sendable (AceStepEvent) -> Void)? = nil,
                       read: @escaping @Sendable () -> Int = { Memory.cacheLimit },
                       write: @escaping @Sendable (Int) -> Void = { Memory.cacheLimit = $0 }) -> AceStepEngine {
        var cfg = AceStepEngine.Configuration()
        cfg.minDuration = 1
        cfg.vaeDType = .float32
        return AceStepEngine(directory: dir, configuration: cfg, textSource: text, silenceLatent: { silence() },
                             readCacheLimit: read, writeCacheLimit: write, observer: observer)
    }

    /// A cache-limit cell standing in for the process-global `Memory.cacheLimit`.
    final class LimitCell: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Int
        private(set) var writes: [Int] = []
        init(_ v: Int) { value = v }
        func read() -> Int { lock.withLock { value } }
        func write(_ v: Int) { lock.withLock { value = v; writes.append(v) } }
    }

    // MARK: Whole run

    @Test func promptToWavHasTheRightShapeIsNormalisedAndDeterministic() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let req = AceStepRequest(caption: "calm piano", duration: 2, seed: 7)
        let a = try await Self.engine(dir).generate(req)
        #expect(a.audio.frameCount == 50 * 1920 && a.audio.frameCount == Int(2.0 * 48_000), "duration x 48000 frames")
        #expect(a.audio.samples.allSatisfy { $0.isFinite })
        let peak = a.audio.samples.map { abs($0) }.max() ?? 0
        #expect(abs(peak - 0.891_25) < 1e-4, "peak-normalised to -1 dBFS, got \(peak)")
        #expect(a.latents.shape == [1, 50, 64])

        let wav = a.audio.wavData()
        func u32(_ o: Int) -> UInt32 { wav.subdata(in: o ..< o + 4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } }
        func u16(_ o: Int) -> UInt16 { wav.subdata(in: o ..< o + 2).withUnsafeBytes { $0.loadUnaligned(as: UInt16.self) } }
        #expect(String(decoding: wav.prefix(4), as: UTF8.self) == "RIFF" && String(decoding: wav.subdata(in: 8 ..< 12), as: UTF8.self) == "WAVE")
        #expect(u16(22) == 2 && u32(24) == 48_000 && u16(34) == 16, "stereo, 48 kHz, 16-bit PCM")
        #expect(Int(u32(40)) == a.audio.samples.count * 2 && wav.count == 44 + a.audio.samples.count * 2)

        let same = try await Self.engine(dir).generate(req)
        #expect(same.audio == a.audio, "the same seed gives the same audio")
        let other = try await Self.engine(dir).generate(AceStepRequest(caption: "calm piano", duration: 2, seed: 8))
        #expect(other.audio != a.audio, "another seed gives different audio")
        #expect(a.audio.monoMixdown.sampleRate == 48_000 && a.audio.monoMixdown.samples.count == a.audio.frameCount)
    }

    @Test func injectedNoiseIsUsedAndTheSeedIsThenIgnored() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let noise = MLXRandom.normal([1, 50, 64], key: MLXRandom.key(1))
        let a = try await Self.engine(dir).generate(AceStepRequest(caption: "x", duration: 2, seed: 1), noise: noise)
        let b = try await Self.engine(dir).generate(AceStepRequest(caption: "x", duration: 2, seed: 2), noise: noise)
        #expect(a.audio == b.audio)
    }

    @Test func componentsAreResidentOneAtATimeInOrder() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let log = EventLog()
        _ = try await Self.engine(dir, observer: { log.add($0) }).generate(AceStepRequest(caption: "x", duration: 2))
        let events = log.events
        func index(_ e: AceStepEvent) -> Int? { events.firstIndex(of: e) }
        let order: [AceStepEvent] = [.loaded(.text), .loaded(.condition), .released(.condition), .loaded(.dit), .released(.dit), .loaded(.vae), .released(.vae)]
        let indices = order.compactMap(index)
        #expect(indices.count == order.count && indices == indices.sorted(), "\(events)")
        #expect(events.contains(.phase(.denoise(step: 8, of: 8))))
        #expect(events.contains { if case .phase(.decode) = $0 { return true } else { return false } })
        let fractions = events.compactMap { e -> Double? in if case .phase(let p) = e { return p.fraction } else { return nil } }
        #expect(fractions == fractions.sorted(), "progress only goes forward")
    }

    final class EventLog: @unchecked Sendable {
        private let lock = NSLock()
        private var all: [AceStepEvent] = []
        func add(_ e: AceStepEvent) { lock.withLock { all.append(e) } }
        var events: [AceStepEvent] { lock.withLock { all } }
    }

    // MARK: Refusals, cancel, cache limit

    @Test func durationAndPromptAreValidatedBeforeAnythingLoads() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        var real = AceStepEngine(directory: dir, textSource: TinyText())
        real.silenceLatent = { Self.silence() }
        await #expect(throws: AceStepEngineError.durationOutOfRange(9, min: 10, max: 240)) { _ = try await real.generate(AceStepRequest(caption: "x", duration: 9)) }
        await #expect(throws: AceStepEngineError.durationOutOfRange(241, min: 10, max: 240)) { _ = try await real.generate(AceStepRequest(caption: "x", duration: 241)) }
        await #expect(throws: AceStepEngineError.emptyPrompt) { _ = try await real.generate(AceStepRequest(caption: "  ", duration: 30)) }
    }

    @Test(arguments: ["acestep-v15-turbo/model.safetensors", "acestep-v15-turbo/config.json", "vae/diffusion_pytorch_model.safetensors", "vae/config.json"])
    func aMissingFileThrowsNamingItAndTouchesNoLimit(_ file: String) async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.removeItem(at: dir.appendingPathComponent(file))
        let cell = LimitCell(1 << 40)
        await #expect(throws: AceStepWeightsError.missingFile(file)) {
            _ = try await Self.engine(dir, read: cell.read, write: cell.write).generate(AceStepRequest(caption: "x", duration: 2))
        }
        #expect(cell.writes.isEmpty)
    }

    @Test func theInstalledTextSideNeedsItsQwenFilesAndNamesTheMissingOne() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        var cfg = AceStepEngine.Configuration(); cfg.minDuration = 1
        let engine = AceStepEngine(directory: dir, configuration: cfg)
        await #expect(throws: AceStepWeightsError.missingFile("Qwen3-Embedding-0.6B/tokenizer.json")) {
            _ = try await engine.generate(AceStepRequest(caption: "x", duration: 2))
        }
    }

    @Test func cancelAfterTheDiTLoadsThrowsCancellationAndRestoresTheLimit() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let cell = LimitCell(1 << 40)
        let engine = Self.engine(dir, observer: { e in if e == .loaded(.dit) { withUnsafeCurrentTask { $0?.cancel() } } }, read: cell.read, write: cell.write)
        let task = Task { try await engine.generate(AceStepRequest(caption: "x", duration: 2)) }
        do { _ = try await task.value; Issue.record("expected CancellationError") } catch is CancellationError {} catch { Issue.record("\(error)") }
        #expect(cell.read() == 1 << 40, "the cache limit is back")
        #expect(cell.writes.count == 2 && cell.writes.first == 1 << 30)
    }

    @Test func cacheLimitIsCappedOnEntryAndRestoredOnSuccessAndOnThrow() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let ok = LimitCell(1 << 40)
        _ = try await Self.engine(dir, read: ok.read, write: ok.write).generate(AceStepRequest(caption: "x", duration: 2))
        #expect(ok.writes == [1 << 30, 1 << 40])
        let low = LimitCell(1 << 20)
        _ = try await Self.engine(dir, read: low.read, write: low.write).generate(AceStepRequest(caption: "x", duration: 2))
        #expect(low.writes == [1 << 20, 1 << 20], "a lower limit someone set is never raised")

        let bad = LimitCell(1 << 40)
        await #expect(throws: AceStepEngineError.emptyPrompt) {
            _ = try await Self.engine(dir, text: TinyText(fail: true), read: bad.read, write: bad.write).generate(AceStepRequest(caption: "x", duration: 2))
        }
        #expect(bad.read() == 1 << 40 && bad.writes.count == 2, "thrown mid-run: still restored")
    }

    // MARK: Pieces

    @Test func peakNormalisationFollowsUpstream() throws {
        let loud = MLXArray([Float(0.5), -0.25, 0.1, 0.4]).reshaped(1, 2, 2)      // channels-first: L = [0.5, -0.25], R = [0.1, 0.4]
        let out = try AceStepEngine.finish(loud)
        #expect(out.count == 4)
        #expect(abs((out.map { abs($0) }.max() ?? 0) - 0.891_25) < 1e-5)
        #expect(abs(out[0] - 0.891_25) < 1e-5 && abs(out[1] - 0.1 * 0.891_25 / 0.5) < 1e-5, "interleaved L R L R: \(out)")
        let silent = try AceStepEngine.finish(MLXArray.zeros([1, 2, 4]))
        #expect(silent.allSatisfy { $0 == 0 }, "near-silence is left alone")
        let hot = try AceStepEngine.finish(MLXArray([Float(2), 0, 0, 0]).reshaped(1, 2, 2))
        #expect(abs(hot[0] - 0.891_25) < 1e-5, "too-loud audio is scaled down too")
    }
}

// MARK: - MC5-T-fix: a cancelled run releases everything, in every phase

extension AceStepEngineTests {
    final class TaskBox: @unchecked Sendable {
        private let lock = NSLock()
        private var task: Task<AceStepResult, Error>?
        func set(_ t: Task<AceStepResult, Error>) { lock.withLock { task = t } }
        func cancel() { lock.withLock { task }?.cancel() }
    }

    /// Cancels the run when `trigger` is first observed, waits for it to unwind, clears the cache, and returns MLX's (active, cache) bytes.
    static func cancelled(at trigger: @escaping @Sendable (AceStepEvent) -> Bool) async throws -> (before: Int, active: Int, cache: Int, error: Error?) {
        let dir = try makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        Memory.clearCache()
        let before = Memory.snapshot().activeMemory
        let box = TaskBox()
        let fired = LimitCell(0)
        let engine = Self.engine(dir, observer: { event in
            if fired.read() == 0, trigger(event) { fired.write(1); box.cancel() }
        })
        let task = Task { try await engine.generate(AceStepRequest(caption: "calm piano", duration: 2, seed: 7)) }
        box.set(task)
        var error: Error?
        do { _ = try await task.value } catch let e { error = e }
        Memory.clearCache()
        let s = Memory.snapshot()
        return (before, s.activeMemory, s.cacheMemory, error)
    }

    @Test(arguments: ["text", "condition", "denoise", "decode"])
    func cancellingInAnyPhaseThrowsCancellationAndReleasesAllMLXMemory(phase: String) async throws {
        let trigger: @Sendable (AceStepEvent) -> Bool = { event in
            switch (phase, event) {
            case ("text", .phase(.text)): return true
            case ("condition", .loaded(.condition)): return true
            case ("denoise", .phase(.denoise(let step, _))): return step == 1
            case ("decode", .loaded(.vae)): return true
            default: return false
            }
        }
        let r = try await Self.cancelled(at: trigger)
        #expect(r.error is CancellationError, "\(phase): \(String(describing: r.error))")
        #expect(r.active == r.before, "\(phase): MLX active memory \(r.active) after the cancelled run, \(r.before) before")
        #expect(r.cache == 0, "\(phase): MLX cache \(r.cache) after clearCache")
    }

    // The teardown (GPU synchronize + cache clear) runs exactly once on every way out of `generate`, and the cache limit is restored on every one.
    final class Counter: @unchecked Sendable { private let l = NSLock(); private var n = 0; func bump() { l.withLock { n += 1 } }; var value: Int { l.withLock { n } } }

    @Test(arguments: ["done", "text", "condition", "denoise", "decode", "failed-text-source"])
    func teardownRunsOnceOnEveryExitAndTheCacheLimitIsRestored(how: String) async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let counter = Counter()
        let cell = LimitCell(7 << 30)
        let box = TaskBox()
        let fired = LimitCell(0)
        var engine = Self.engine(dir, text: TinyText(fail: how == "failed-text-source"), observer: { event in
            let hit: Bool
            switch (how, event) {
            case ("text", .phase(.text)), ("condition", .loaded(.condition)), ("decode", .loaded(.vae)): hit = true
            case ("denoise", .phase(.denoise(let step, _))): hit = step == 1
            default: hit = false
            }
            if hit, fired.read() == 0 { fired.write(1); box.cancel() }
        }, read: { cell.read() }, write: { cell.write($0) })
        engine.teardown = { counter.bump() }
        let task = Task { try await engine.generate(AceStepRequest(caption: "calm piano", duration: 2, seed: 7)) }
        box.set(task)
        do { _ = try await task.value } catch { }
        #expect(counter.value == 1, "\(how): teardown ran \(counter.value) times")
        #expect(cell.read() == 7 << 30, "\(how): cache limit \(cell.read()) not restored")
    }

    // MC5-T-mem: syncing the GPU stream at the stage boundaries changes memory only, never a value.
    @Test func syncingAtStageBoundariesGivesIdenticalAudioToNotSyncing() async throws {
        let dir = try Self.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let request = AceStepRequest(caption: "calm piano", lyrics: "la la", duration: 2, seed: 7)
        var off = Self.engine(dir); off.configuration.syncAtStageBoundaries = false
        var on = Self.engine(dir); on.configuration.syncAtStageBoundaries = true
        #expect(AceStepEngine.Configuration.standard.syncAtStageBoundaries, "on by default")
        let a = try await off.generate(request), b = try await on.generate(request)
        #expect(a.audio == b.audio, "same samples")
        #expect(a.latents.shape == b.latents.shape && Self.maxAbsDiff(a.latents, b.latents) == 0, "same latents")
    }

    static func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        (try? MLXErrorGuard.item(abs(a - b).max(), as: Float.self)) ?? .infinity
    }
}
