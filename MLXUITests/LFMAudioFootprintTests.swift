import Darwin
import Foundation
import MLX
import Testing
@testable import MLXUI

/// MP6-3c -- the owner-run memory and latency harness for LFM2.5-Audio 1.5B. It measures; it changes no `ramGB`, limit or
/// budget (MP6-3d does, from the numbers this prints).
///
/// Skipped unless the weights are installed (see `LFMAudioRealLoad.directory`) **and** `MLXUI_TEST_LFMAUDIO_FOOTPRINT=1`
/// (`TEST_RUNNER_MLXUI_TEST_LFMAUDIO_FOOTPRINT=1` on the xcodebuild line). `MLXUI_TEST_LFMAUDIO_TURNS=<n>` overrides the 24 turns.
///
/// Method is MP4-3d's: `phys_footprint` (`task_info(TASK_VM_INFO)`, what Activity Monitor and jetsam count) via the shared
/// `FootprintTracker`, sampled every 20 ms while the work runs, plus `Memory.snapshot()` active/cache. The voice turns go through
/// `LFMVoiceRunModel` -- the object the Run UI drives -- so the path is the UI's: `LFMVoiceTurnStage` built by `LeasingStage.perRun`
/// per turn, `MLXErrorGuard.guardedRun`, the `@concurrent` hop, the audio thread, the model loaded per turn. Only the microphone
/// (the clip is `say` output, 16 kHz) and the loudspeaker (a null player) are replaced. The real backend is wrapped by a pass-through
/// recorder that sees token counts, whether the cap was hit, and load time; it alters nothing.
nonisolated final class LFMTurnProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var _loadSeconds = 0.0, _tokens = 0, _capHit = false, _textAt: Date?, _lastAudioAt: Date?, _audioFrames = 0
    func reset() { lock.withLock { _loadSeconds = 0; _tokens = 0; _capHit = false; _textAt = nil; _lastAudioAt = nil; _audioFrames = 0 } }
    func loaded(_ s: Double) { lock.withLock { _loadSeconds += s } }
    func generated(_ tokens: Int, cap: Int) { lock.withLock { _tokens += tokens; _capHit = _capHit || tokens >= cap } }
    func decoded(frames: Int) { lock.withLock { _audioFrames += frames } }
    func event(_ e: LFMVoiceEvent) {
        lock.withLock {
            switch e {
            case .text: if _textAt == nil { _textAt = Date() }
            case .audio: _lastAudioAt = Date()
            }
        }
    }
    var snapshot: (load: Double, tokens: Int, cap: Bool, textAt: Date?, lastAudioAt: Date?, frames: Int) {
        lock.withLock { (_loadSeconds, _tokens, _capHit, _textAt, _lastAudioAt, _audioFrames) }
    }
}

nonisolated final class LFMMeasuringBackend: LFMAudioBackend {
    private let inner: any LFMAudioBackend
    private let probe: LFMTurnProbe
    init(_ inner: any LFMAudioBackend, probe: LFMTurnProbe) { self.inner = inner; self.probe = probe }
    func generate(_ prompt: LFMPrompt, mode: LFMGenerationMode, sampling: LFMSampling) async throws -> LFMGenerated {
        let result = try await inner.generate(prompt, mode: mode, sampling: sampling)
        probe.generated(result.tokens, cap: sampling.maxNewTokens)
        return result
    }
    func decode(frames: [[Int32]]) async throws -> [Float] {
        probe.decoded(frames: frames.count)
        return try await inner.decode(frames: frames)
    }
    static func factory(_ probe: LFMTurnProbe) -> @Sendable (URL) async throws -> any LFMAudioBackend {
        { url in
            let started = Date()
            let real = try await LFMAudioRealBackend.load(url)
            probe.loaded(Date().timeIntervalSince(started))
            return LFMMeasuringBackend(real, probe: probe)
        }
    }
}

@MainActor
struct LFMAudioFootprintTests {
    private static let enabled = LFMAudioRealLoad.directory != nil && LFMAudioRealLoad.env["MLXUI_TEST_LFMAUDIO_FOOTPRINT"] == "1"
    private static let turns = max(LFMAudioRealLoad.env["MLXUI_TEST_LFMAUDIO_TURNS"].flatMap(Int.init) ?? 24, 1)

    private static let script = [
        "Hello! Please tell me a short fact about the moon.", "What is the capital of France?",
        "Can you explain what a rainbow is in one sentence?", "My favourite colour is green. Remember that.",
        "What is seven times eight?", "Tell me a very short joke.", "What did I say my favourite colour was?",
        "Name three kinds of fruit.", "How many days are in a leap year?", "Describe the ocean in one sentence.",
        "What is the opposite of hot?", "Say something kind about the morning.",
    ]

    private static func mib(_ bytes: Int64) -> Double { Double(bytes) / 1_048_576 }
    private static func mib(_ bytes: Int) -> Double { Double(bytes) / 1_048_576 }

    /// Every file under the model store, so a before/after comparison shows anything the harness wrote there.
    private static func storeListing() -> Set<String> {
        let base = ModelStore.shared.baseDirectory
        guard let walker = FileManager.default.enumerator(atPath: base.path) else { return [] }
        return Set(walker.compactMap { $0 as? String })
    }

    /// Tiles `clip` to at least `seconds` of 16 kHz audio (exactly, trimmed).
    private static func tiled(_ clip: [Float], seconds: Double) -> [Float] {
        let want = Int(seconds * 16_000)
        var out: [Float] = []
        out.reserveCapacity(want)
        while out.count < want { out += clip.prefix(want - out.count) }
        return out
    }

    @Test(.enabled(if: enabled))
    func aLongVoiceSessionAndTheLimitCases() async throws {
        let dir = try LFMAudioRealLoad.modelDirectory()
        let before = Self.storeListing()
        let tracker = FootprintTracker()
        let sampler = Task.detached {
            while !Task.isCancelled { tracker.note(FootprintTracker.phys()); try? await Task.sleep(for: .milliseconds(20)) }
        }
        defer { sampler.cancel() }
        func line(_ tag: String, _ text: String) {
            print(String(format: "LFMFOOTPRINT %@ now %.0f MiB, MLX active %.0f cache %.0f MiB, %@", tag,
                         Self.mib(FootprintTracker.phys()), Self.mib(Memory.snapshot().activeMemory), Self.mib(Memory.snapshot().cacheMemory), text))
        }
        tracker.note(FootprintTracker.phys()); line("start", "(before any model)")
        let baseline = FootprintTracker.phys()
        let probe = LFMTurnProbe()
        let factory = LFMMeasuringBackend.factory(probe)

        // ---- A: the scripted voice session, through the Run UI's view-model path.
        var clips: [[Float]] = []
        for sentence in Self.script { clips.append(try LFMAudioRealLoad.speech(sentence)) }
        let placeholder = LFMAudioStages.asr(modelDirectory: dir)
        let player = FakeReplyPlayer()
        let model = LFMVoiceRunModel(player: player, makeStage: { conversation, onEvent in
            LeasingStage.perRun(placeholder, config: .default) { _ in
                LFMVoiceTurnStage(modelDirectory: dir, conversation: conversation, factory: factory,
                                  onEvent: { probe.event($0); onEvent($0) })
            }
        })
        struct Row { var n: Int; var phys: Int64; var peak: Int64; var active: Int; var cache: Int; var textS: Double?; var audioS: Double?
                     var wall: Double; var load: Double; var tokens: Int; var frames: Int; var cap: Bool; var history: Int }
        var rows: [Row] = []
        tracker.takePhasePeak()
        for n in 1...Self.turns {
            probe.reset()
            let samples = clips[(n - 1) % clips.count]
            let started = Date()
            model.send(samples)
            while model.phase != .idle {
                try await Task.sleep(for: .milliseconds(50))
                if Date().timeIntervalSince(started) > 600 { Issue.record("turn \(n) did not finish in 10 minutes"); model.stop(); break }
            }
            let wall = Date().timeIntervalSince(started)
            if let message = model.errorText { Issue.record("turn \(n): \(message)"); break }
            let snap = probe.snapshot
            let snapshot = Memory.snapshot()
            let row = Row(n: n, phys: FootprintTracker.phys(), peak: tracker.takePhasePeak(), active: snapshot.activeMemory, cache: snapshot.cacheMemory,
                          textS: snap.textAt.map { $0.timeIntervalSince(started) }, audioS: snap.lastAudioAt.map { $0.timeIntervalSince(started) },
                          wall: wall, load: snap.load, tokens: snap.tokens, frames: snap.frames, cap: snap.cap, history: model.conversation.turns.count)
            rows.append(row)
        }
        print("LFMFOOTPRINT TABLE turn | after MiB | turn peak MiB | MLX active MiB | MLX cache MiB | to text s | to last audio s | wall s | load s | tokens | audio frames | cap hit | memory turns")
        for r in rows {
            print(String(format: "LFMFOOTPRINT TURN %2d | %7.0f | %7.0f | %7.0f | %7.0f | %@ | %@ | %6.1f | %5.1f | %4d | %4d | %@ | %d", r.n,
                         Self.mib(r.phys), Self.mib(r.peak), Self.mib(r.active), Self.mib(r.cache),
                         r.textS.map { String(format: "%6.1f", $0) } ?? "   n/a", r.audioS.map { String(format: "%6.1f", $0) } ?? "   n/a",
                         r.wall, r.load, r.tokens, r.frames, r.cap ? "YES" : "no", r.history))
        }
        let sessionPeak = rows.map(\.peak).max() ?? 0
        let growth = (rows.last?.phys ?? 0) - (rows.first?.phys ?? 0)
        print(String(format: "LFMFOOTPRINT SESSION %d turns: peak %.0f MiB (%.2f GiB), baseline before model %.0f MiB, after turn 1 %.0f MiB, after last %.0f MiB, drift %+.0f MiB; reload mean %.1f s; cap hit on %d turns; memory turns held at most %d (cap %d)",
                     rows.count, Self.mib(sessionPeak), Self.mib(sessionPeak) / 1024, Self.mib(baseline), Self.mib(rows.first?.phys ?? 0),
                     Self.mib(rows.last?.phys ?? 0), Self.mib(growth), rows.isEmpty ? 0 : rows.map(\.load).reduce(0, +) / Double(rows.count),
                     rows.filter(\.cap).count, rows.map(\.history).max() ?? 0, LFMConversation.maxTurns))
        #expect(rows.count == Self.turns, "the session stopped early")
        #expect((rows.map(\.history).max() ?? 0) <= LFMConversation.maxTurns)
        #expect(model.conversation.turns.count <= LFMConversation.maxTurns)

        // ---- B: the limit cases. One turn at the 120 s recording limit, one 20+ s STT file, one ~2,000-character TTS.
        let base = clips[0]
        let long = Self.tiled(base, seconds: 119.5)
        probe.reset(); tracker.takePhasePeak()
        model.newConversation()
        var started = Date()
        model.send(long)
        while model.phase != .idle { try await Task.sleep(for: .milliseconds(50)) }
        if let message = model.errorText { Issue.record("120 s turn: \(message)") }
        var snap = probe.snapshot
        print(String(format: "LFMFOOTPRINT LIMIT recording %.1f s: peak %.0f MiB, wall %.1f s, to text %@ s, tokens %d, cap hit %@, audio frames %d",
                     Double(long.count) / 16_000, Self.mib(tracker.takePhasePeak()), Date().timeIntervalSince(started),
                     snap.textAt.map { String(format: "%.1f", $0.timeIntervalSince(started)) } ?? "n/a", snap.tokens, snap.cap ? "YES" : "no", snap.frames))

        let stt = Self.tiled(base, seconds: 24)
        probe.reset(); tracker.takePhasePeak(); started = Date()
        let transcript = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.transcribe(stt, modelDirectory: dir, factory: factory) }
        snap = probe.snapshot
        print(String(format: "LFMFOOTPRINT LIMIT stt %.1f s of audio: peak %.0f MiB, wall %.1f s, tokens %d, cap hit %@, %d characters",
                     Double(stt.count) / 16_000, Self.mib(tracker.takePhasePeak()), Date().timeIntervalSince(started), snap.tokens, snap.cap ? "YES" : "no", transcript.count))

        var essay = ""
        while essay.count < 2_000 { essay += "The harbour was quiet that morning, and the fishing boats rocked slowly against the old stone wall. " }
        essay = String(essay.prefix(2_000))
        probe.reset(); tracker.takePhasePeak(); started = Date()
        let spoken = try await MLXErrorGuard.guardedRun { try await LFMAudioEngine.synthesize(essay, modelDirectory: dir, factory: factory) }
        snap = probe.snapshot
        print(String(format: "LFMFOOTPRINT LIMIT tts %d characters: peak %.0f MiB, wall %.1f s, %.1f s of audio, tokens %d, a chunk hit the cap %@, frames %d",
                     essay.count, Self.mib(tracker.takePhasePeak()), Date().timeIntervalSince(started), Double(spoken.count) / 24_000,
                     snap.tokens, snap.cap ? "YES" : "no", snap.frames))

        Memory.clearCache()
        line("end", "(after clearCache)")
        print(String(format: "LFMFOOTPRINT PEAK over the whole run %.0f MiB (%.2f GiB); download-size x 1.5 estimate to compare: 1.73 GB x 1.5 = %.2f GB",
                     Self.mib(tracker.runPeak), Self.mib(tracker.runPeak) / 1024, 1.73 * 1.5))
        model.shutdown()
        #expect(Self.storeListing() == before, "the harness wrote into the model store; a voice session persists nothing")
    }
}
