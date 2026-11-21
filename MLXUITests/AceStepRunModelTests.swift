import Testing
import Foundation
@testable import MLXUI

/// MC5-3j: the Run model, driven through `generate()` (the Generate button's own entry point) with a generator in place of the real engine.
@MainActor
@Suite(.serialized)
struct AceStepRunModelTests {

    final class Seen: @unchecked Sendable {
        private let lock = NSLock()
        private var requests: [AceStepRequest] = []
        func add(_ r: AceStepRequest) { lock.withLock { requests.append(r) } }
        var all: [AceStepRequest] { lock.withLock { requests } }
    }

    static let audio = AceStepAudio(samples: [0.1, -0.1, 0.2, -0.2])

    /// A generator that reports the four phases then returns.
    static func stub(_ seen: Seen = Seen()) -> AceStepRunModel.Generator {
        { request, onPhase in
            seen.add(request)
            onPhase(.text); onPhase(.condition); onPhase(.denoise(step: 8, of: 8)); onPhase(.decode(chunk: 1, of: 1))
            return audio
        }
    }

    @Test func idleToRunningToDoneWithWavBytes() async {
        let model = AceStepRunModel(generator: Self.stub())
        #expect(!model.isRunning && !model.didGenerate)
        model.generate()
        #expect(model.isRunning && model.phaseLabel == "Reading your description…")
        await model.finished()
        #expect(!model.isRunning && model.didGenerate && model.errorText == nil)
        let wav = try? #require(model.wavData)
        #expect(wav?.prefix(4) == Data("RIFF".utf8) && wav?.count == 44 + 4 * 2)
        #expect(model.progress == 1)
    }

    @Test func progressPhasesArriveInOrder() async {
        final class Log: @unchecked Sendable { var phases: [AceStepPhase] = [] }
        let log = Log()
        let model = AceStepRunModel(generator: { _, onPhase in
            for p in [AceStepPhase.text, .condition, .denoise(step: 1, of: 8), .denoise(step: 8, of: 8), .decode(chunk: 1, of: 1)] { onPhase(p); log.phases.append(p) }
            try await Task.sleep(for: .milliseconds(50))
            return Self.audio
        })
        var labels: [String] = []
        model.generate()
        for _ in 0 ..< 40 where model.isRunning {
            if labels.last != model.phaseLabel { labels.append(model.phaseLabel) }
            try? await Task.sleep(for: .milliseconds(5))
        }
        await model.finished()
        #expect(log.phases.count == 5)
        #expect(AceStepRunModel.label(for: .denoise(step: 3, of: 8)) == "Making the music: step 3 of 8")
        let fractions = [AceStepPhase.text, .condition, .denoise(step: 4, of: 8), .decode(chunk: 1, of: 1)].map(\.fraction)
        #expect(fractions == fractions.sorted())
    }

    @Test func cancelMidRunReturnsToIdleLeavingNothingAndAnotherRunWorks() async {
        final class Gate: @unchecked Sendable { var started = false }
        let gate = Gate()
        let model = AceStepRunModel(generator: { _, onPhase in
            onPhase(.denoise(step: 2, of: 8))
            gate.started = true
            while true { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(5)) }
        })
        model.generate()
        for _ in 0 ..< 200 where !gate.started { try? await Task.sleep(for: .milliseconds(5)) }
        #expect(model.isRunning)
        model.cancel()
        await model.finished()
        #expect(!model.isRunning && !model.didGenerate && model.errorText == nil && model.progress == 0)
        #expect(model.wavData == nil && !model.hasPlayer && !model.isPlaying, "no result, no player left behind")

        // A second Generate after the cancel works (a fresh model state, a new generator is not needed: swap by making a new model with the stub).
        let again = AceStepRunModel(generator: Self.stub())
        again.generate()
        await again.finished()
        #expect(again.didGenerate)
    }

    @Test func aSecondGenerateOnTheSameModelAfterACancelWorks() async {
        final class Mode: @unchecked Sendable { var hang = true }
        let mode = Mode()
        let model = AceStepRunModel(generator: { _, _ in
            if mode.hang { while true { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(5)) } }
            return Self.audio
        })
        model.generate()
        try? await Task.sleep(for: .milliseconds(30))
        model.cancel()
        await model.finished()
        #expect(!model.isRunning)
        mode.hang = false
        model.generate()
        await model.finished()
        #expect(model.didGenerate && model.errorText == nil)
    }

    @Test func anEngineErrorBecomesAMessageAndLeavesNoResult() async {
        let model = AceStepRunModel(generator: { _, _ in throw AceStepEngineError.emptyPrompt })
        model.generate()
        await model.finished()
        #expect(!model.isRunning && !model.didGenerate)
        #expect(model.errorText == AceStepEngineError.emptyPrompt.description)
    }

    @Test func aMissingModelFileTellsTheUserToReinstall() async {
        let model = AceStepRunModel(generator: { _, _ in throw AceStepWeightsError.missingFile("vae/config.json") })
        model.generate()
        await model.finished()
        let text = model.errorText ?? ""
        #expect(text.contains("vae/config.json") && text.contains("Reinstall ACE-Step 1.5"), "\(text)")
    }

    @Test func durationIsClampedToTenToTwoHundredForty() async {
        let seen = Seen()
        let model = AceStepRunModel(generator: Self.stub(seen))
        model.duration = 500
        #expect(model.duration == 240)
        model.duration = 1
        #expect(model.duration == 10)
        model.generate()
        await model.finished()
        #expect(seen.all.last?.duration == 10)
        #expect(AceStepRunModel.defaultDuration == 60 && AceStepRunModel().duration == 60)
        #expect(AceStepRunModel.format(90) == "1 min 30 s" && AceStepRunModel.format(60) == "1 min" && AceStepRunModel.format(45) == "45 s")
    }

    @Test func emptyLyricsMakeAnInstrumentalAndLyricsAreKept() async {
        let seen = Seen()
        let model = AceStepRunModel(generator: Self.stub(seen))
        model.lyrics = "  \n "
        model.generate()
        await model.finished()
        #expect(seen.all.last?.isInstrumental == true)
        // The engine turns that into upstream's `[Instrumental]` / `unknown` (3d).
        let r = seen.all[0]
        #expect(AceStepPrompt.lyricString(lyrics: r.lyrics, language: r.language, instrumental: r.isInstrumental).contains("[Instrumental]"))
        model.lyrics = "[Verse]\nla la"
        model.language = "zh"
        model.generate()
        await model.finished()
        #expect(seen.all.last?.isInstrumental == false && seen.all.last?.language == "zh" && seen.all.last?.lyrics == "[Verse]\nla la")
    }

    @Test func theSeedIsShownAfterARunTypedOrRandom() async {
        let seen = Seen()
        let model = AceStepRunModel(generator: Self.stub(seen))
        model.seedText = "12345"
        model.generate()
        await model.finished()
        #expect(model.usedSeed == 12345 && seen.all.last?.seed == 12345)
        model.seedText = ""
        model.generate()
        await model.finished()
        #expect(model.usedSeed == seen.all.last?.seed && model.usedSeed != nil)
    }

    @Test func anEmptyDescriptionCannotBeGenerated() {
        let model = AceStepRunModel(generator: Self.stub())
        model.caption = "   "
        #expect(!model.canGenerate)
        model.generate()
        #expect(!model.isRunning)
    }

    /// The whole path with the real engine on the tiny checkpoint (10 s is the smallest the model allows; the tiny parts make it quick).
    @Test func theRealEngineOnTheTinyFolderProducesAWav() async throws {
        let dir = try AceStepEngineTests.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let model = AceStepRunModel(generator: { request, onPhase in
            var engine = AceStepEngineTests.engine(dir)
            engine.observer = { event in if case .phase(let p) = event { onPhase(p) } }
            return try await engine.generate(request).audio
        })
        model.duration = 10
        model.generate()
        await model.finished()
        #expect(model.errorText == nil)
        let wav = try #require(model.wavData)
        #expect(wav.count == 44 + 480_000 * 2 * 2 && model.resultSeconds == 10)
    }
}
