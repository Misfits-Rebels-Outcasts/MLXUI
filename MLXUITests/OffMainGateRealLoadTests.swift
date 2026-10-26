import Foundation
import MLX
import Testing
@testable import MLXUI

/// MP5-3g off-main gate (owner ruling 2026-10-26). `PipelineStage.run` became `nonisolated` (49a891c), so a flow step, which used to
/// hop to the main actor to start, now runs where `FlowRunner`'s task runs: the pool. Qwen3-ASR, Sortformer and Parakeet were only
/// ever cleared by main-actor probes, and their packages use the thread-bound-stream pattern (KICKOFF rule 17). This is the
/// evidence from the other side: each stage called the way a flow calls it -- from a task with no actor, inside
/// `MLXErrorGuard.guardedRun`, through a `LeasingStage` -- twice in one process, asserting no thrown error, no MLX error recorded by
/// the process-wide fallback, output equal to the main-actor probe's, and `Thread.isMainThread` false at the engine call.
///
/// "In the engine body" is read at the engine's call site, in the stage's injected closure, one line before the engine call (the
/// engines are `nonisolated` and run on their caller's thread; none of them hops to the main actor), and at every progress call.
/// Owner/lead-run, gated on each model's weights (`MLXUI_TEST_<ENGINE>_DIR`, passed as `TEST_RUNNER_…`); a skipped run does not
/// count. Run each suite on its own, scoped at the type: a single-method `-only-testing:` runs 0 tests and still says SUCCEEDED.
nonisolated final class GateThreadLog: @unchecked Sendable {
    private let lock = NSLock()
    private var total = 0, main = 0
    func record() { lock.lock(); total += 1; if Thread.isMainThread { main += 1 }; lock.unlock() }
    var counts: (calls: Int, onMain: Int) { lock.lock(); defer { lock.unlock() }; return (total, main) }
}

/// Records the thread at entry and at every progress call, then forwards.
nonisolated struct GateStage: PipelineStage {
    let inner: any PipelineStage
    let log: GateThreadLog
    var id: String { inner.id }
    var name: String { inner.name }
    var accepts: MediaKind { inner.accepts }
    var produces: MediaKind { inner.produces }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        log.record()
        return try await inner.run(input) { [log] fraction in log.record(); progress(fraction) }
    }
}

nonisolated enum OffMainGate {
    /// Runs `stage` `times` times the way a flow does: each pass from a task that has no actor (`@concurrent` here, so the enclosing
    /// context is the pool), inside `guardedRun`, through a `LeasingStage` on `entry`. Returns each pass's result.
    @concurrent static func flowShaped(_ stage: any PipelineStage, entry: ModelEntry, input: Media, times: Int) async -> [Result<Media, Error>] {
        let leased = LeasingStage(model: entry, config: .default, probe: stage)
        var results: [Result<Media, Error>] = []
        for _ in 0..<times {
            let result = await Task { () -> Result<Media, Error> in
                do { return .success(try await MLXErrorGuard.guardedRun { try await leased.run(input) { _ in } }) } catch { return .failure(error) }
            }.value
            results.append(result)
        }
        return results
    }

    nonisolated static func text(_ result: Result<Media, Error>) -> String? {
        if case .success(.text(let value)) = result { return value }
        return nil
    }

    nonisolated static func describe(_ result: Result<Media, Error>) -> String {
        switch result {
        case .success(let media): return "ok \(media.kind)"
        case .failure(let error): return "FAILED \(MLXErrorGuard.sentence(for: error))"
        }
    }
}

@MainActor
struct OffMainGateRealLoadTests {

    // MARK: Qwen3-ASR

    @Test(.enabled(if: Qwen3ASRRealLoad.directory != nil))
    func qwen3ASRAsAFlowCallsIt() async throws {
        let copy = try Qwen3ASRRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let samples = try Qwen3ASRRealLoad.sampleSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        // The main-actor probe's figure, in this process.
        let baseline = try await MLXErrorGuard.guardedRun { try await Qwen3ASREngine.transcribe(samples, modelDirectory: dir) }
        #expect(Qwen3ASRRealLoad.readsTheMemo(baseline), "\(baseline)")
        let engineLog = GateThreadLog(), stageLog = GateThreadLog()
        let asr = ASRStage(id: "gate.qwen3-asr", name: "Qwen3-ASR") { s in
            engineLog.record()
            return try await Qwen3ASREngine.transcribe(s, modelDirectory: dir)
        }
        let entry = makeEntry(id: "gate-qwen3-asr", modelType: .asr, ramGB: 3)
        let results = await OffMainGate.flowShaped(GateStage(inner: asr, log: stageLog), entry: entry,
                                                   input: .audio(AudioBuffer(samples: samples, sampleRate: 16_000)), times: 2)
        report("QWEN3ASR", results, engineLog, stageLog, expected: baseline)
        check(results, engineLog, stageLog, expected: baseline)
    }

    // MARK: Parakeet

    @Test(.enabled(if: ParakeetRealLoad.directory != nil))
    func parakeetAsAFlowCallsIt() async throws {
        let dir = try #require(ParakeetRealLoad.directory)
        let samples = try ParakeetRealLoad.sampleSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let baseline = try await MLXErrorGuard.guardedRun { try await ParakeetEngine.transcribe(samples, modelDirectory: dir) }
        #expect(ParakeetRealLoad.readsTheMemo(baseline), "\(baseline)")
        let engineLog = GateThreadLog(), stageLog = GateThreadLog()
        let asr = ASRStage(id: "gate.parakeet", name: "Parakeet") { s in
            engineLog.record()
            return try await ParakeetEngine.transcribe(s, modelDirectory: dir)
        }
        let entry = makeEntry(id: "gate-parakeet", modelType: .asr, ramGB: 3)
        let results = await OffMainGate.flowShaped(GateStage(inner: asr, log: stageLog), entry: entry,
                                                   input: .audio(AudioBuffer(samples: samples, sampleRate: 16_000)), times: 2)
        report("PARAKEET", results, engineLog, stageLog, expected: baseline)
        check(results, engineLog, stageLog, expected: baseline)
    }

    // MARK: Sortformer

    @Test(.enabled(if: SortformerRealLoad.directory != nil))
    func sortformerAsAFlowCallsIt() async throws {
        let copy = try SortformerRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let samples = try SortformerRealLoad.clipSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        func stage(_ log: GateThreadLog?) -> SortformerStage {
            SortformerStage(id: "gate.sortformer", name: "Sortformer") { s in
                log?.record()
                return try await SortformerEngine.diarize(s, modelDirectory: dir)
            }
        }
        // The main-actor probe's figure, in this process: the same stage, awaited from the main actor.
        let mainOutput = try await MLXErrorGuard.guardedRun { try await stage(nil).run(.audio(AudioBuffer(samples: samples, sampleRate: 16_000))) { _ in } }
        guard case let .text(baseline) = mainOutput else { Issue.record("not text"); return }
        #expect(baseline.split(separator: "\n").count > 1, "segments: \(baseline)")
        let engineLog = GateThreadLog(), stageLog = GateThreadLog()
        let entry = makeEntry(id: "gate-sortformer", modelType: .diarization, ramGB: 3)
        let results = await OffMainGate.flowShaped(GateStage(inner: stage(engineLog), log: stageLog), entry: entry,
                                                   input: .audio(AudioBuffer(samples: samples, sampleRate: 16_000)), times: 2)
        report("SORTFORMER", results, engineLog, stageLog, expected: baseline)
        check(results, engineLog, stageLog, expected: baseline)
    }

    // MARK: DeepFilterNet

    @Test(.enabled(if: DeepFilterNetRealLoad.directory != nil))
    func deepFilterNetAsAFlowCallsIt() async throws {
        let copy = try DeepFilterNetRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        try await DeepFilterNetRealLoad.withClip { clip in
            MLXErrorGuard.installFallback()
            MLXErrorGuard.resetFallbackMessage()
            let engineLog = GateThreadLog(), stageLog = GateThreadLog()
            let stage = DeepFilterNetStage(id: "gate.deepfilternet", name: "DeepFilterNet") { samples, progress in
                engineLog.record()
                return try await DeepFilterNetEngine.enhance(samples, modelDirectory: dir, progress: progress)
            }
            let entry = makeEntry(id: "gate-deepfilternet", modelType: .enhance, ramGB: 2.35)
            let results = await OffMainGate.flowShaped(GateStage(inner: stage, log: stageLog), entry: entry,
                                                       input: .audio(AudioBuffer(samples: clip.noisy, sampleRate: 48_000)), times: 2)
            for (pass, result) in results.enumerated() {
                #expect({ if case .success = result { return true } else { return false } }(), "pass \(pass + 1): \(OffMainGate.describe(result))")
                guard case .success(.audio(let out)) = result else { continue }
                let r = DeepFilterNetRealLoad.report("gate-pass\(pass + 1)", noisy: clip.noisy, cleaned: out.samples, clean: clip.clean)
                // The main-actor figure recorded 2026-10-25 / 26: 13.10 dB, RMS 0.959.
                #expect(abs(r.improvement - 13.10) < 0.15, "SNR gain \(r.improvement) dB")
                #expect(DeepFilterNetRealLoad.rmsRatioRange.contains(r.rmsRatio))
            }
            print("OFFMAINGATE DEEPFILTERNET passes=\(results.count) engineCalls=\(engineLog.counts.calls) engineOnMain=\(engineLog.counts.onMain) stageCalls=\(stageLog.counts.calls) stageOnMain=\(stageLog.counts.onMain)")
            #expect(engineLog.counts.calls == 2 && engineLog.counts.onMain == 0)
            #expect(stageLog.counts.calls > 2 && stageLog.counts.onMain == 0)
            #expect(MLXErrorGuard.lastFallbackMessage() == nil)
        }
    }

    // MARK: Shared assertions

    private func report(_ label: String, _ results: [Result<Media, Error>], _ engine: GateThreadLog, _ stage: GateThreadLog, expected: String) {
        let outputs = results.map { OffMainGate.text($0) == expected ? "equal" : OffMainGate.describe($0) }
        print("OFFMAINGATE \(label) passes=\(results.count) outputs=\(outputs) engineCalls=\(engine.counts.calls) engineOnMain=\(engine.counts.onMain) stageCalls=\(stage.counts.calls) stageOnMain=\(stage.counts.onMain) fallback=\(MLXErrorGuard.lastFallbackMessage() ?? "none")")
    }

    private func check(_ results: [Result<Media, Error>], _ engine: GateThreadLog, _ stage: GateThreadLog, expected: String) {
        #expect(results.count == 2)
        for (pass, result) in results.enumerated() {
            #expect(OffMainGate.text(result) == expected, "pass \(pass + 1): \(OffMainGate.describe(result))")
        }
        #expect(engine.counts.calls == 2, "the engine was reached \(engine.counts.calls) times")
        #expect(engine.counts.onMain == 0, "the engine call ran on the main thread \(engine.counts.onMain) times")
        #expect(stage.counts.onMain == 0, "the stage ran on the main thread \(stage.counts.onMain) times")
        #expect(MLXErrorGuard.lastFallbackMessage() == nil, "an MLX error reached the process-wide fallback")
    }
}
