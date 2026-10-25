import Testing
import Foundation
import AVFoundation
@testable import MLXUI

/// MP5-3a: the DeepFilterNet module's resolution, claim, stage, resampling, chunk plan, WAV output, limit hook and error paths,
/// with no weights needed. The real load, the main-actor probe and the measurements are in `DeepFilterNetRealLoadTests.swift`
/// (gated on the install; the lead runs them).
@MainActor
struct DeepFilterNetModuleTests {

    private func entry() -> ModelEntry {
        makeEntry(id: "mlx-community--DeepFilterNet-mlx", family: "DeepFilterNet", modelType: .enhance, source: .mlx)
    }

    private func sine(count: Int, rate: Int, hz: Double = 440, amplitude: Float = 0.5) -> [Float] {
        (0..<count).map { amplitude * Float(sin(2 * Double.pi * hz * Double($0) / Double(rate))) }
    }

    private func tempURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("dfn-\(UUID().uuidString)-\(name)")
    }

    // MARK: Claim

    @Test func resolvesToTheDeepFilterNetModule() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: entry()))
        #expect(resolved.descriptor.id == "deepfilternet")
        #expect(resolved.sdk.claim(entry()) == .exact)
        #expect(resolved.ui.claim(entry()) == .exact)
    }

    @Test func claimsOnlyDeepFilterNetFamilyMLXEnhance() {
        let sdk = DeepFilterNetSDK()
        #expect(sdk.claim(makeEntry(family: "DeepFilterNet", modelType: .enhance, source: .coreml)) == .no)
        #expect(sdk.claim(makeEntry(id: "o", family: "Other", modelType: .enhance, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "a", family: "DeepFilterNet", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "d", family: "DeepFilterNet", modelType: .diarization, source: .mlx)) == .no)
        #expect(DeepFilterNetUI().claim(makeEntry(id: "a", family: "DeepFilterNet", modelType: .asr, source: .mlx)) == .no)
        for other: any ModelSDK in [WhisperSDK(), MLXWhisperSDK(), VoxtralSDK(), ParakeetSDK(), Qwen3ASRSDK(), SortformerSDK()] {
            #expect(other.claim(entry()) == .no, "\(other.id) must decline DeepFilterNet")
        }
    }

    /// Only the DeepFilterNet entry resolves to the new module, and every other bundled entry resolves exactly as without it.
    @Test func onlyTheDeepFilterNetEntryResolvesToItsModule() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        let without = ModelRegistry()
        for module in installedModules where module.descriptor.id != "deepfilternet" { module.register(into: without) }
        var enhanceEntries = 0
        for e in all {
            let after = full.bestModule(for: e)?.descriptor.id
            if e.modelType == .enhance {
                #expect(after == "deepfilternet", "\(e.id)")
                enhanceEntries += 1
            } else {
                #expect(after == without.bestModule(for: e)?.descriptor.id, "\(e.id): resolution changed")
            }
        }
        #expect(enhanceEntries == 1)
    }

    // MARK: Stage and resampling

    private actor Seen {
        var count = 0
        func note(_ n: Int) { count = n }
    }

    @Test(arguments: [44_100, 16_000, 48_000, 8_000])
    func stageResamplesAnyInputTo48kHzAndOutputsTheEnginesSamples(inputRate: Int) async throws {
        let seen = Seen()
        let stage = DeepFilterNetStage(id: "t", name: "t") { samples, progress in
            await seen.note(samples.count)
            progress(1)
            return samples.map { $0 * 0.5 }
        }
        let seconds = 2.0
        let input = AudioBuffer(samples: sine(count: Int(seconds * Double(inputRate)), rate: inputRate), sampleRate: inputRate)
        let output = try await stage.run(.audio(input)) { _ in }
        guard case let .audio(buffer) = output else { Issue.record("not audio"); return }
        #expect(buffer.sampleRate == 48_000)
        let expected = Int(seconds * 48_000)
        let fed = await seen.count
        #expect(abs(fed - expected) <= 2_000, "engine got \(fed) samples, expected about \(expected) at 48 kHz")
        #expect(buffer.samples.count == fed)
        // Resampling keeps the tone: RMS of a 0.5-amplitude sine is 0.354; the stage halves it.
        let tail = Array(buffer.samples.dropFirst(4_800).dropLast(4_800))
        #expect(abs(AudioMetrics.rms(tail) - 0.5 * 0.5 / 2.0.squareRoot()) < 0.01)
    }

    @Test func resamplingHelperMakes48kFrom44k1AndFrom16k() throws {
        for rate in [44_100, 16_000] {
            let input = AudioBuffer(samples: sine(count: rate, rate: rate), sampleRate: rate)
            let out = try AudioResampler.resample(input, to: DeepFilterNetEngine.sampleRate)
            #expect(out.sampleRate == 48_000)
            #expect(abs(out.samples.count - 48_000) <= 1_100, "\(rate): \(out.samples.count)")
        }
    }

    @Test func stageProgressRisesAndRefusesNonAudio() async throws {
        let stage = DeepFilterNetStage(id: "t", name: "t") { samples, progress in progress(0.5); return samples }
        let recorded = ProgressLog()
        _ = try await stage.run(.audio(AudioBuffer(samples: sine(count: 4_800, rate: 48_000), sampleRate: 48_000))) { recorded.add($0) }
        let values = recorded.values
        #expect(values == values.sorted() && values.last == 1.0)
        await #expect(throws: StageError.self) { _ = try await stage.run(.text("hi")) { _ in } }
    }

    final class ProgressLog: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [Double] = []
        func add(_ v: Double) { lock.lock(); stored.append(v); lock.unlock() }
        var values: [Double] { lock.lock(); defer { lock.unlock() }; return stored }
    }

    // MARK: Chunk plan and crossfade

    @Test func noChunkLengthIsOneWindow() {
        let plan = DeepFilterNetEngine.chunkPlan(sampleCount: 1_000, chunkSamples: nil, overlapSamples: 100)
        #expect(plan.windows == [0..<1_000] && plan.overlap == 0)
        #expect(DeepFilterNetEngine.chunkPlan(sampleCount: 1_000, chunkSamples: 2_000, overlapSamples: 100).windows == [0..<1_000])
        #expect(DeepFilterNetEngine.chunkPlan(sampleCount: 0, chunkSamples: 100, overlapSamples: 10).windows.isEmpty)
    }

    @Test func windowsOverlapCoverTheClipAndEndOnTheLastSample() {
        let plan = DeepFilterNetEngine.chunkPlan(sampleCount: 1_050, chunkSamples: 400, overlapSamples: 100)
        #expect(plan.overlap == 100)
        #expect(plan.windows.first?.lowerBound == 0 && plan.windows.last?.upperBound == 1_050)
        for (a, b) in zip(plan.windows, plan.windows.dropFirst()) {
            #expect(a.upperBound - b.lowerBound == 100, "consecutive windows share exactly the overlap")
        }
        #expect(plan.windows.last.map { $0.count > 100 } == true)
    }

    @Test func overlapIsClampedUnderHalfAChunk() {
        let plan = DeepFilterNetEngine.chunkPlan(sampleCount: 1_000, chunkSamples: 100, overlapSamples: 90)
        #expect(plan.overlap < 50)
        #expect(plan.windows.last?.upperBound == 1_000)
    }

    @Test(arguments: [(1_050, 400, 100), (1_000, 300, 0), (777, 200, 50), (10_001, 1_000, 499)])
    func crossfadeOfAPassThroughReproducesTheInput(count: Int, chunk: Int, overlap: Int) {
        let input = sine(count: count, rate: 8_000, hz: 97)
        let plan = DeepFilterNetEngine.chunkPlan(sampleCount: count, chunkSamples: chunk, overlapSamples: overlap)
        var out = [Float](repeating: 0, count: count)
        for (i, w) in plan.windows.enumerated() {
            DeepFilterNetEngine.accumulate(into: &out, window: w, values: Array(input[w]), overlap: plan.overlap,
                                           isFirst: i == 0, isLast: i == plan.windows.count - 1)
        }
        let worst = zip(out, input).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 1e-5, "max error \(worst)")
    }

    // MARK: Limit hook

    /// MP5-3c: the measured defaults (journal 2026-598): 8 s windows, 1 s overlap, refuse past 60 minutes.
    @Test func theDefaultsAreTheMeasuredOnes() {
        #expect(DeepFilterNetEngine.defaultChunkSeconds == 8)
        #expect(DeepFilterNetEngine.defaultOverlapSeconds == 1)
        #expect(DeepFilterNetEngine.maxDurationSeconds == 3_600)
    }

    @Test func theLimitIsInclusiveAt3600AndRefusesPastIt() throws {
        try DeepFilterNetEngine.requireWithinLimit(seconds: 3_600)
        try DeepFilterNetEngine.requireWithinLimit(seconds: 3_600 + 2.0 / 48_000)   // resampling can add a sample or two
        #expect(throws: DeepFilterNetEngine.LimitError.self) { try DeepFilterNetEngine.requireWithinLimit(seconds: 3_600.5) }
        #expect(throws: DeepFilterNetEngine.LimitError.self) { try DeepFilterNetEngine.requireWithinLimit(seconds: 3_601) }
        try DeepFilterNetEngine.requireWithinLimit(seconds: 60, limit: 60)
        #expect(throws: DeepFilterNetEngine.LimitError.self) { try DeepFilterNetEngine.requireWithinLimit(seconds: 61, limit: 60) }
        try DeepFilterNetEngine.requireWithinLimit(seconds: 1_000_000, limit: nil)
    }

    @Test func theRefusalSentenceReadsWell() {
        let message = DeepFilterNetEngine.LimitError(seconds: 3_601, limit: 3_600).errorDescription
        #expect(message == "This recording is 60 minutes 1 second long. Audio cleanup handles up to 60 minutes at a time; split the recording into shorter parts and run each one.")
    }

    @Test func theChunkedDefaultPlansAnHourAsAboutFiveHundredWindows() {
        let samples = 3_600 * 48_000
        let plan = DeepFilterNetEngine.chunkPlan(sampleCount: samples, chunkSamples: Int(8 * 48_000), overlapSamples: 48_000)
        #expect(plan.windows.count == 515 && plan.overlap == 48_000)   // step 7 s: ceil((3600 - 8) / 7) + 1
        #expect(plan.windows.last?.upperBound == samples)
    }

    /// The Run UI refuses an over-long file from its header, before decoding, and never reaches the stage.
    @Test func theRunViewModelRefusesAnOverLongFileBeforeDecoding() async throws {
        let url = tempURL("two-seconds.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: sine(count: 96_000, rate: 48_000), sampleRate: 48_000), to: url)
        #expect(abs((DeepFilterNetInput.durationSeconds(of: url) ?? 0) - 2.0) < 1e-6)
        let stage = DeepFilterNetStage(id: "t", name: "t") { _, _ in
            Issue.record("the stage must not run for an over-long file")
            return []
        }
        let model = DeepFilterNetRunModel()
        model.start(audioURL: url, stage: stage, maxDurationSeconds: 1)
        for _ in 0..<200 where model.isRunning { try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.errorText == "This recording is 2 seconds long. Audio cleanup handles up to 1 second at a time; split the recording into shorter parts and run each one.")
        #expect(model.original == nil && !model.hasResult)
    }

    @Test func aFileThatIsNotAudioHasNoHeaderDuration() throws {
        let url = tempURL("text.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("not audio".utf8).write(to: url)
        #expect(DeepFilterNetInput.durationSeconds(of: url) == nil)
    }

    // MARK: WAV output

    @Test func savedWAVIsA48kHzMonoFileThatReadsBack() throws {
        let samples = sine(count: 48_000, rate: 48_000, hz: 1_000, amplitude: 0.5)
        let url = tempURL("out.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: 48_000), to: url)
        let file = try AVAudioFile(forReading: url)
        #expect(file.fileFormat.sampleRate == 48_000)
        #expect(file.fileFormat.channelCount == 1)
        let back = try AudioFileReader.read(url)
        #expect(back.sampleRate == 48_000 && back.samples.count == 48_000)
        #expect(AudioMetrics.snrDB(estimate: back.samples, reference: samples) > 60, "16-bit quantisation only")
    }

    @Test func cleanedFileNameKeepsTheStem() {
        #expect(DeepFilterNetRunModel.cleanedFileName(for: URL(fileURLWithPath: "/tmp/interview.m4a")) == "interview-cleaned.wav")
        #expect(DeepFilterNetRunModel.cleanedFileName(for: nil) == "audio-cleaned.wav")
    }

    // MARK: Error paths

    @Test func aNonAudioFileRenamedWavGivesAReadableError() throws {
        let url = tempURL("not-audio.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("This is a text file, not a recording.".utf8).write(to: url)
        let error = #expect(throws: DeepFilterNetInput.ReadError.self) { _ = try DeepFilterNetInput.read(url) }
        let text = error?.errorDescription ?? ""
        #expect(text.contains("couldn't be read as audio") && text.contains("not-audio.wav"))
        #expect(!text.contains("1685348671") && !text.contains("com.apple"))
    }

    @Test func aMissingFileAndAnEmptyWavGiveReadableErrors() throws {
        #expect(throws: DeepFilterNetInput.ReadError.self) { _ = try DeepFilterNetInput.read(tempURL("missing.wav")) }
        let url = tempURL("empty.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: [], sampleRate: 48_000), to: url)
        let error = #expect(throws: DeepFilterNetInput.ReadError.self) { _ = try DeepFilterNetInput.read(url) }
        #expect(error?.reason == .empty)
    }

    @Test func theRunViewModelShowsTheReadErrorAndNeverCallsTheStage() async throws {
        let url = tempURL("fake.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("RIFF but not really".utf8).write(to: url)
        let stage = DeepFilterNetStage(id: "t", name: "t") { _, _ in
            Issue.record("the stage must not run for an unreadable file")
            return []
        }
        let model = DeepFilterNetRunModel()
        model.start(audioURL: url, stage: stage)
        for _ in 0..<200 where model.isRunning { try await Task.sleep(for: .milliseconds(20)) }
        #expect(!model.isRunning)
        #expect(model.errorText?.contains("couldn't be read as audio") == true)
        #expect(!model.hasResult)
    }

    @Test func theRunViewModelHoldsTheCleanedAudioAfterASuccessfulRun() async throws {
        let url = tempURL("ok.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: sine(count: 22_050, rate: 22_050), sampleRate: 22_050), to: url)
        let stage = DeepFilterNetStage(id: "t", name: "t") { samples, _ in samples }
        let model = DeepFilterNetRunModel()
        model.start(audioURL: url, stage: stage)
        for _ in 0..<400 where model.isRunning { try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.errorText == nil)
        #expect(model.original?.sampleRate == 22_050)
        #expect(model.cleaned?.sampleRate == 48_000)
        #expect(model.source == .cleaned && model.hasResult)
    }

    @Test func missingWeightsAndEmptyAudioAreReadableEngineErrors() async throws {
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent("dfn-empty-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        do {
            _ = try await DeepFilterNetEngine.enhance([0.1, 0.2], modelDirectory: empty)
            Issue.record("expected noWeights")
        } catch let error as DeepFilterNetEngine.EngineError {
            guard case .noWeights = error else { Issue.record("\(error)"); return }
            #expect(error.errorDescription?.contains("Re-install") == true)
        }
        do {
            _ = try await DeepFilterNetEngine.enhance([], modelDirectory: empty)
            Issue.record("expected emptyAudio")
        } catch let error as DeepFilterNetEngine.EngineError {
            #expect(error == .emptyAudio)
        }
    }
}

/// MP5-3a: the measuring helpers and the deterministic noise.
struct AudioMetricsTests {
    @Test func rmsOfASineIsAmplitudeOverRootTwo() {
        let s = (0..<48_000).map { 0.5 * Float(sin(2 * Double.pi * 100 * Double($0) / 48_000)) }
        #expect(abs(AudioMetrics.rms(s) - 0.5 / 2.0.squareRoot()) < 1e-4)
        #expect(AudioMetrics.rms([]) == 0)
    }

    @Test func snrIsExactForAKnownMix() {
        let reference = (0..<48_000).map { 0.3 * Float(sin(2 * Double.pi * 200 * Double($0) / 48_000)) }
        let noise = AudioMetrics.noise(count: 48_000, sampleRate: 48_000, seed: 7)
        for target in [-5.0, 0, 10, 20] {
            let noisy = AudioMetrics.mix(reference: reference, noise: noise, snrDB: target)
            #expect(abs(AudioMetrics.snrDB(estimate: noisy, reference: reference) - target) < 0.01, "\(target) dB")
        }
        #expect(AudioMetrics.snrDB(estimate: reference, reference: reference) == .infinity)
    }

    @Test func improvementIsTheDifferenceOfTwoSNRs() {
        let reference = (0..<10_000).map { 0.3 * Float(sin(Double($0) * 0.05)) }
        let noise = AudioMetrics.noise(count: 10_000, sampleRate: 48_000, seed: 3)
        let noisy = AudioMetrics.mix(reference: reference, noise: noise, snrDB: 5)
        let cleaned = AudioMetrics.mix(reference: reference, noise: noise, snrDB: 15)
        #expect(abs(AudioMetrics.improvementDB(noisy: noisy, cleaned: cleaned, reference: reference) - 10) < 0.05)
    }

    @Test func bestLagFindsAKnownShift() {
        let reference = AudioMetrics.noise(count: 6_000, sampleRate: 48_000, seed: 11)
        let late = [Float](repeating: 0, count: 37) + reference
        #expect(AudioMetrics.bestLag(estimate: late, reference: reference, maxLag: 100, window: 4_000) == 37)
        #expect(AudioMetrics.bestLag(estimate: reference, reference: reference, maxLag: 100, window: 4_000) == 0)
    }

    @Test func noiseIsDeterministicPerSeedAndDiffersAcrossSeeds() {
        let a = AudioMetrics.noise(count: 1_000, sampleRate: 48_000, seed: 42)
        let b = AudioMetrics.noise(count: 1_000, sampleRate: 48_000, seed: 42)
        let c = AudioMetrics.noise(count: 1_000, sampleRate: 48_000, seed: 43)
        #expect(a == b)
        #expect(a != c)
        #expect(a.contains { abs($0) > 0.5 } && AudioMetrics.rms(a) > 0.1)
    }
}
