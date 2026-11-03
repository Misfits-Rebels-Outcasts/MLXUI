import Testing
import Foundation
import AVFoundation
import MLXAudioCodecs
import MLXAudioSTS
@testable import MLXUI

/// MC1-4: the SAM-Audio module's resolution, claim, stage, resampling, window plan, crossfade, limit, load path (Trap 1)
/// and error paths, with no weights. The real load, the main-actor probe and the footprint measurement are in
/// `SAMAudioRealLoadTests.swift` / `SAMAudioFootprintTests.swift` (gated on the install; skipped unless the weights are
/// pointed at).
@MainActor
struct SAMAudioModuleTests {

    private func entry() -> ModelEntry {
        makeEntry(id: "mlx-community--sam-audio-base", family: "sam-audio", modelType: .separate, source: .mlx)
    }

    static func sine(count: Int, rate: Int, hz: Double = 440, amplitude: Float = 0.5) -> [Float] {
        (0..<count).map { amplitude * Float(sin(2 * Double.pi * hz * Double($0) / Double(rate))) }
    }

    private func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("sam-\(UUID().uuidString)")
    }

    private func tempURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("sam-\(UUID().uuidString)-\(name)")
    }

    private actor Seen {
        var count = 0
        func note(_ n: Int) { count = n }
    }

    final class ProgressLog: @unchecked Sendable {
        private let lock = NSLock()
        private var stored: [Double] = []
        func add(_ v: Double) { lock.lock(); stored.append(v); lock.unlock() }
        var values: [Double] { lock.lock(); defer { lock.unlock() }; return stored }
    }

    // MARK: Claim

    @Test func resolvesToTheSAMAudioModule() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: entry()))
        #expect(resolved.descriptor.id == "samaudio")
        #expect(resolved.sdk.claim(entry()) == .exact)
        #expect(resolved.ui.claim(entry()) == .exact)
    }

    /// Trap 6: by family, never by architecture. The family override in `runnerKind` routes any `sam-audio` entry to
    /// `.separate` whatever its `modelType`, so the family is the claim that matters; a different family with a
    /// `.separate` kind is still not ours, and a non-MLX source never is.
    @Test func claimsOnlyTheSAMAudioFamilyMLXSeparate() {
        let sdk = SAMAudioSDK()
        let ui = SAMAudioUI()
        #expect(sdk.claim(makeEntry(family: "sam-audio", modelType: .separate, source: .coreml)) == .no)
        #expect(sdk.claim(makeEntry(family: "sam-audio", modelType: .asr, source: .mlx)) == .exact,
                "the family override wins, by design")
        #expect(sdk.claim(makeEntry(id: "o", family: "Other", modelType: .separate, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "o", family: "Other", modelType: .enhance, source: .mlx)) == .no)
        #expect(ui.claim(makeEntry(id: "o", family: "Other", modelType: .separate, source: .mlx)) == .no)
        #expect(ui.claim(makeEntry(family: "sam-audio", modelType: .separate, source: .coreml)) == .no)
        #expect(ui.claim(makeEntry(family: "sam-audio", modelType: .separate, source: .mlx)) == .exact)
    }

    /// Only the SAM-Audio entry resolves to the new module, and every other bundled entry resolves exactly as without it.
    @Test func onlyTheSAMAudioEntryResolvesToItsModule() throws {
        let all = try SAMAudioCatalogTests.catalog().domains.flatMap { $0.allModels }
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        let without = ModelRegistry()
        for module in installedModules where module.descriptor.id != "samaudio" { module.register(into: without) }
        var separateEntries = 0
        for e in all {
            let after = full.bestModule(for: e)?.descriptor.id
            if e.modelType == .separate {
                #expect(after == "samaudio", "\(e.id)")
                separateEntries += 1
            } else {
                #expect(after == without.bestModule(for: e)?.descriptor.id, "\(e.id): resolution changed")
            }
        }
        #expect(separateEntries == 1)
    }

    @Test func theModelHasNoRunnerGapAndStageFactoryRefusesTheKindExplicitly() throws {
        let e = try SAMAudioCatalogTests.samAudio()
        #expect(ModelSupport.unsupportedReason(for: e) == nil)
        // Run UI models go through the registry (LeasingStage); StageFactory is the flow/stage seam, which has no
        // `.separate` stage. The explicit case keeps that a compile-time decision.
        #expect(throws: StageError.self) {
            _ = try StageFactory.make(for: e, installedModelIDs: [e.id], availableRAMGB: 32)
        }
    }

    // MARK: Stage and resampling

    @Test(arguments: [44_100, 16_000, 48_000, 8_000])
    func stageResamplesAnyInputToTheModelsRateAndHandsOutBothTracks(inputRate: Int) async throws {
        let seen = Seen()
        let box = SAMAudioTrackBox()
        let stage = SAMAudioStage(id: "t", name: "t", sampleRate: 48_000, output: box) { samples, progress in
            await seen.note(samples.count)
            progress(1)
            return SAMAudioEngine.Result(isolated: samples.map { $0 * 0.5 },
                                         other: samples.map { -$0 },
                                         sampleRate: 48_000)
        }
        let seconds = 2.0
        let input = AudioBuffer(samples: Self.sine(count: Int(seconds * Double(inputRate)), rate: inputRate), sampleRate: inputRate)
        let output = try await stage.run(.audio(input)) { _ in }
        guard case let .audio(isolated) = output else { Issue.record("not audio"); return }
        #expect(isolated.sampleRate == 48_000)
        let fed = await seen.count
        #expect(abs(fed - Int(seconds * 48_000)) <= 2_000, "engine got \(fed) samples, expected about 48000 x 2")
        #expect(isolated.samples.count == fed)
        #expect(box.other?.sampleRate == 48_000)
        #expect(box.other?.samples.count == fed, "both tracks are the recording's length")
        // Resampling keeps the tone: RMS of a 0.5-amplitude sine is 0.354; the stage halves the isolated track.
        let tail = Array(isolated.samples.dropFirst(4_800).dropLast(4_800))
        #expect(abs(AudioMetrics.rms(tail) - 0.5 * 0.5 / 2.0.squareRoot()) < 0.01)
        #expect(AudioMetrics.rms(box.other?.samples ?? []) > 0)
    }

    @Test func stageProgressRisesAndRefusesNonAudio() async throws {
        let box = SAMAudioTrackBox()
        let stage = SAMAudioStage(id: "t", name: "t", output: box) { samples, progress in
            progress(0.5)
            return SAMAudioEngine.Result(isolated: samples, other: samples, sampleRate: 48_000)
        }
        let recorded = ProgressLog()
        _ = try await stage.run(.audio(AudioBuffer(samples: Self.sine(count: 4_800, rate: 48_000), sampleRate: 48_000))) { recorded.add($0) }
        let values = recorded.values
        #expect(values == values.sorted() && values.last == 1.0)
        await #expect(throws: StageError.self) { _ = try await stage.run(.text("hi")) { _ in } }
    }

    @Test func defaultStageDescriptionIsSpeechForAGenericCaller() throws {
        #expect(SAMAudioSDK().promptSupport == .freeText(default: "speech"))
        let stage = try #require(try? SAMAudioSDK().makeStage(for: entry(), config: .default))
        #expect(stage.accepts == .audio && stage.produces == .audio)
    }

    // MARK: Window plan and crossfade

    @Test func aShortClipIsOneWindowAndLongerOnesOverlap() {
        #expect(SAMAudioEngine.chunkPlan(sampleCount: 480_000, chunkSamples: 480_000, overlapSamples: 144_000).windows == [0..<480_000])
        let plan = SAMAudioEngine.chunkPlan(sampleCount: 3_000, chunkSamples: 1_000, overlapSamples: 300)
        #expect(plan.overlap == 300)
        #expect(plan.windows.first?.lowerBound == 0 && plan.windows.last?.upperBound == 3_000)
        for (a, b) in zip(plan.windows, plan.windows.dropFirst()) {
            #expect(a.upperBound - b.lowerBound == 300, "consecutive windows share exactly the overlap")
        }
        #expect(SAMAudioEngine.chunkPlan(sampleCount: 0, chunkSamples: 100, overlapSamples: 10).windows.isEmpty)
    }

    @Test func theLibrarysWindowShapeIsTheOneWePlan() {
        // A 30-minute clip at 48 kHz with the engine's defaults: 30 s windows every 27 s (MC1-10; was 10 s every 7 s).
        let samples = SAMAudioEngine.maxDurationSeconds.secondsToSamples(rate: 48_000)
        let plan = SAMAudioEngine.chunkPlan(sampleCount: samples,
                                            chunkSamples: Int(48_000 * SAMAudioEngine.chunkSeconds),
                                            overlapSamples: Int(48_000 * SAMAudioEngine.overlapSeconds))
        #expect(plan.overlap == 144_000)
        #expect(plan.windows.count == 67, "ceil((1800 - 30) / 27) + 1")
        #expect(plan.windows.last?.upperBound == samples)
    }

    @Test func cosineFadeRunsFromZeroToOneAndPassesASignalThrough() {
        let fade = SAMAudioEngine.cosineFade(count: 100)
        #expect(fade.first == 0)
        #expect(abs((fade.last ?? 0) - 1) < 1e-6)
        #expect(zip(fade, fade.dropFirst()).allSatisfy { $0 <= $1 })
        // A pass-through through the plan reproduces the input (the two ramps sum to 1 at every seam sample).
        let input = Self.sine(count: 1_050, rate: 8_000, hz: 97)
        let plan = SAMAudioEngine.chunkPlan(sampleCount: input.count, chunkSamples: 400, overlapSamples: 100)
        var out = [Float](repeating: 0, count: input.count)
        for (i, w) in plan.windows.enumerated() {
            SAMAudioEngine.accumulate(into: &out, window: w, values: Array(input[w]), overlap: plan.overlap,
                                      isFirst: i == 0, isLast: i == plan.windows.count - 1)
        }
        let worst = zip(out, input).map { abs($0 - $1) }.max() ?? 0
        #expect(worst < 1e-5, "max error \(worst)")
    }

    @Test func crossfadeKeepsTheJumpAtASeamUnderTheSignalsOwn() {
        let rate = 8_000
        let samples = Self.sine(count: 20_000, rate: rate, hz: 200, amplitude: 0.5)
        func run(overlap: Int) -> (out: [Float], plan: (windows: [Range<Int>], overlap: Int)) {
            let plan = SAMAudioEngine.chunkPlan(sampleCount: samples.count, chunkSamples: 4_010, overlapSamples: overlap)
            var call = 0
            var out = [Float](repeating: 0, count: samples.count)
            for (i, w) in plan.windows.enumerated() {
                let gain: Float = call == 0 ? 1.0 : 0.5   // as if two cold starts disagreed
                call += 1
                SAMAudioEngine.accumulate(into: &out, window: w, values: Array(samples[w]).map { $0 * gain },
                                          overlap: plan.overlap, isFirst: i == 0, isLast: i == plan.windows.count - 1)
            }
            return (out, plan)
        }
        func seamAndElsewhere(_ r: (out: [Float], plan: (windows: [Range<Int>], overlap: Int))) -> (seam: Float, elsewhere: Float) {
            let near = 40
            var seam: Float = 0, elsewhere: Float = 0
            let seams = zip(r.plan.windows, r.plan.windows.dropFirst()).map { ($0.1.lowerBound - near)..<($0.0.upperBound + near) }
            for i in 1..<r.out.count {
                let jump = abs(r.out[i] - r.out[i - 1])
                if seams.contains(where: { $0.contains(i) }) { seam = max(seam, jump) } else { elsewhere = max(elsewhere, jump) }
            }
            return (seam, elsewhere)
        }
        let faded = seamAndElsewhere(run(overlap: 300))
        #expect(faded.seam <= faded.elsewhere * 1.05, "seam \(faded.seam) vs elsewhere \(faded.elsewhere)")
        let cut = seamAndElsewhere(run(overlap: 0))
        #expect(cut.seam > cut.elsewhere * 1.5, "the control must show a click: seam \(cut.seam) vs elsewhere \(cut.elsewhere)")
    }

    // MARK: Limit and estimate

    @Test func theDefaultsAreTheOnesTheIssueNames() {
        #expect(SAMAudioEngine.wholeClipLimitSeconds == 30)
        #expect(SAMAudioEngine.chunkSeconds == 30, "owner ruling 2026-11-03 (MC1-10)")
        #expect(Double(SAMAudioEngine.chunkSeconds) == SAMAudioEngine.wholeClipLimitSeconds,
                "a clip that fits one pass and one window of a longer clip are the same size")
        #expect(SAMAudioEngine.overlapSeconds == 3)
        #expect(SAMAudioEngine.maxDurationSeconds == 1_800)
    }

    @Test func theLimitIsInclusiveAt30MinutesAndRefusesPastIt() throws {
        try SAMAudioEngine.requireWithinLimit(seconds: 1_800)
        try SAMAudioEngine.requireWithinLimit(seconds: 1_800 + 2.0 / 48_000)
        #expect(throws: SAMAudioEngine.LimitError.self) { try SAMAudioEngine.requireWithinLimit(seconds: 1_800.5) }
        #expect(throws: SAMAudioEngine.LimitError.self) { try SAMAudioEngine.requireWithinLimit(seconds: 1_801) }
        try SAMAudioEngine.requireWithinLimit(seconds: 60, limit: 60)
        #expect(throws: SAMAudioEngine.LimitError.self) { try SAMAudioEngine.requireWithinLimit(seconds: 61, limit: 60) }
        try SAMAudioEngine.requireWithinLimit(seconds: 1_000_000, limit: nil)
    }

    @Test func theRefusalSentenceReadsWell() {
        let message = SAMAudioEngine.LimitError(seconds: 1_801, limit: 1_800).errorDescription
        #expect(message == "This recording is 30 minutes 1 second long. Sound isolation handles up to 30 minutes at a time; split the recording into shorter parts and run each one.")
        #expect(SAMAudioEngine.spoken(1_800) == "30 minutes")
        #expect(SAMAudioEngine.spoken(43) == "43 seconds")
        #expect(SAMAudioEngine.spoken(97) == "1 minute 37 seconds")
    }

    @Test func theEstimatesReadAsEstimates() {
        #expect(SAMAudioEngine.estimate(seconds: 0) == SAMAudioEngine.estimatedLoadSeconds)
        #expect(SAMAudioEngine.estimate(seconds: 60) > 60)
        // MC1-12 (R6): 60 s fixed + 1.3 s per audio second (65 s clip: 144.5 s against ~66 s measured; 30 min: 2,400 s against 1,675 s).
        #expect(SAMAudioEngine.estimatedLoadSeconds == 60)
        #expect(SAMAudioEngine.estimatedProcessingSecondsPerAudioSecond == 1.3)
        #expect(abs(SAMAudioEngine.estimate(seconds: 1_800) - 2_400) < 1e-9)
        #expect(SAMAudioEngine.spokenEstimate(SAMAudioEngine.estimate(seconds: 1_800)) == "40 minutes")
        #expect(SAMAudioEngine.spokenEstimate(45) == "45 seconds")
        #expect(SAMAudioEngine.spokenEstimate(966) == "16 minutes")
        #expect(!SAMAudioEngine.spokenEstimate(966).contains("second"))
    }

    // MARK: Trap 1 — the installed load path

    /// The pinned `config.json` of `mlx-community/sam-audio-base` (sha `614abc87`).
    static let repoConfig = """
    {
      "model_type": "sam_audio",
      "model_size": "base",
      "in_channels": 768,
      "audio_codec": {
        "encoder_dim": 64,
        "encoder_rates": [2, 8, 10, 12],
        "latent_dim": 1024,
        "decoder_dim": 1536,
        "decoder_rates": [12, 10, 8, 2],
        "n_codebooks": 16,
        "codebook_size": 1024,
        "codebook_dim": 128,
        "sample_rate": 48000
      },
      "text_encoder": { "name": "t5-base", "max_length": 512, "dim": 768 },
      "transformer": {
        "dim": 2048, "n_heads": 16, "n_layers": 16, "dropout": 0.1, "qk_norm": true,
        "fc_bias": false, "ffn_exp": 4, "context_dim": 2048, "out_channels": 256
      },
      "num_anchors": 3,
      "anchor_embedding_dim": 128
    }
    """

    private func installedFolder(withTextEncoder: Bool) throws -> URL {
        let dir = tempDirectory()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(Self.repoConfig.utf8).write(to: dir.appendingPathComponent("config.json"))
        if withTextEncoder {
            let t5 = dir.appendingPathComponent("t5", isDirectory: true)
            try FileManager.default.createDirectory(at: t5, withIntermediateDirectories: true)
            try Data(#"{"vocab_size": 32128}"#.utf8).write(to: t5.appendingPathComponent("config.json"))
            try Data(count: 8).write(to: t5.appendingPathComponent("model.safetensors"))
        }
        return dir
    }

    /// Trap 1: the config the model is built from points `textEncoder.name` at the installed folder's **absolute**
    /// path. `T5TextEncoder.resolveModelFolder` returns an existing path as-is, so it never calls `HubApi`.
    @Test func thePatchedConfigPointsTheTextEncoderAtTheInstalledFolder() throws {
        let dir = try installedFolder(withTextEncoder: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let config = try SAMAudioEngine.installedConfig(modelDirectory: dir)
        let expected = dir.appendingPathComponent("t5", isDirectory: true).path
        #expect(config.textEncoder.name == expected)
        #expect(config.textEncoder.name.hasPrefix("/"), "an absolute path, not a repo id")
        #expect(FileManager.default.fileExists(atPath: config.textEncoder.name))
        #expect(config.audioCodec.sampleRate == 48_000)
        #expect(SAMAudioEngine.sampleRate(in: dir) == 48_000)
    }

    @Test func aMissingTextEncoderThrowsBeforeAnyWeightOrNetwork() throws {
        let dir = try installedFolder(withTextEncoder: false)
        defer { try? FileManager.default.removeItem(at: dir) }
        let error = #expect(throws: SAMAudioEngine.EngineError.self) {
            _ = try SAMAudioEngine.installedConfig(modelDirectory: dir)
        }
        #expect(error == .noTextEncoder(dir.lastPathComponent))
        let sentence = error?.errorDescription ?? ""
        #expect(sentence.contains("t5/") && sentence.contains("Re-install"))
        #expect(SAMAudioEngine.sampleRate(in: dir) == 48_000, "the rate is readable before the encoder check")
    }

    @Test func anEmptyOrPartlyInstalledFolderGivesAReadableError() throws {
        let empty = tempDirectory()
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: empty) }
        let error = #expect(throws: SAMAudioEngine.EngineError.self) {
            _ = try SAMAudioEngine.load(modelDirectory: empty)
        }
        #expect(error == .noWeights(empty.lastPathComponent))
        #expect(error?.errorDescription?.contains("Re-install") == true)
        // The real folder, minus its weights: the encoder is present, no .safetensors → noWeights, not a crash.
        let dir = try installedFolder(withTextEncoder: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(throws: SAMAudioEngine.EngineError.self) { _ = try SAMAudioEngine.load(modelDirectory: dir) }
        #expect(SAMAudioEngine.sampleRate(in: tempDirectory()) == SAMAudioEngine.fallbackSampleRate)
    }

    /// No code path in the module can download weights: the library entry points that reach `HubApi` /
    /// `resolveOrDownloadModel` are absent from every file of `MLXUI/Modules/SAMAudio/`.
    @Test(.enabled(if: SAMAudioNoNetworkScan.canReadSources))
    func noModuleFileCanDownloadWeights() {
        let hits = SAMAudioNoNetworkScan.findings(forbidden: [
            "fromPretrained", "fromDirectory", "HubApi", "resolveOrDownloadModel", "ModelUtils", "Hub.Repo",
            "snapshot(", "Downloader", "URLSession",
        ])
        #expect(hits.isEmpty, "a run must only read installed files:\n\(hits.joined(separator: "\n"))")
    }

    /// The load path is reachable without a download precisely because it calls `SAMAudio(config:)` +
    /// `loadConvertedWeights` — the two library entry points that take arrays, not a repo.
    @Test(.enabled(if: SAMAudioNoNetworkScan.canReadSources))
    func theLoadPathUsesTheInstalledEntryPoints() {
        let source = SAMAudioNoNetworkScan.combinedSource()
        #expect(source.contains("SAMAudio(config:"))
        #expect(source.contains("loadConvertedWeights("))
        #expect(source.contains("installedConfig(modelDirectory:"))
    }

    // MARK: Error paths without weights

    @Test func emptyAudioAndAnEmptyDescriptionAreReadableEngineErrors() async throws {
        let dir = tempDirectory()
        do {
            _ = try await SAMAudioEngine.separate([], description: "speech", modelDirectory: dir)
            Issue.record("expected emptyAudio")
        } catch let error as SAMAudioEngine.EngineError {
            #expect(error == .emptyAudio)
        }
        do {
            _ = try await SAMAudioEngine.separate([0.1, 0.2], description: "   ", modelDirectory: dir)
            Issue.record("expected missingDescription")
        } catch let error as SAMAudioEngine.EngineError {
            #expect(error == .missingDescription)
            #expect(error.errorDescription?.contains("Type what to isolate") == true)
        }
    }

    @Test func aNonAudioFileRenamedWavGivesAReadableError() throws {
        let url = tempURL("not-audio.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("This is a text file, not a recording.".utf8).write(to: url)
        let error = #expect(throws: SAMAudioInput.ReadError.self) { _ = try SAMAudioInput.read(url) }
        let text = error?.errorDescription ?? ""
        #expect(text.contains("couldn't be read as audio") && text.contains("not-audio.wav"))
        #expect(!text.contains("1685348671") && !text.contains("com.apple"))
    }

    // MARK: Run model

    /// A stub stage that fills the second-track box and returns a halved isolated track, so the run model's two-track
    /// handoff is testable without weights.
    private func stubStage(box: SAMAudioTrackBox, rate: Int = 48_000) -> SAMAudioStage {
        SAMAudioStage(id: "t", name: "t", sampleRate: rate, output: box) { samples, progress in
            progress(0.5)
            box.store(AudioBuffer(samples: samples.map { -$0 }, sampleRate: rate))
            return SAMAudioEngine.Result(isolated: samples.map { $0 * 0.5 }, other: samples.map { -$0 }, sampleRate: rate)
        }
    }

    @Test func theRunModelHoldsBothTracksAfterASuccessfulRun() async throws {
        let url = tempURL("ok.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: Self.sine(count: 22_050, rate: 22_050), sampleRate: 22_050), to: url)
        let box = SAMAudioTrackBox()
        let model = SAMAudioRunModel()
        model.start(audioURL: url, stage: stubStage(box: box), output: box)
        for _ in 0..<400 where model.isRunning { try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.errorText == nil)
        #expect(model.isolated?.sampleRate == 48_000)
        #expect(model.other?.sampleRate == 48_000)
        #expect(model.hasResult)
        #expect(model.buffer(.isolated)?.samples.count == model.buffer(.other)?.samples.count)
    }

    /// A file over the limit is refused from its header, before decoding, and the stage never runs.
    @Test func theRunModelRefusesAnOverLongFileBeforeDecoding() async throws {
        let url = tempURL("two-seconds.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioWriter.writeWAV(AudioBuffer(samples: Self.sine(count: 8_000, rate: 4_000), sampleRate: 4_000), to: url)
        #expect(abs((SAMAudioInput.durationSeconds(of: url) ?? 0) - 2.0) < 1e-6)
        let box = SAMAudioTrackBox()
        let stage = SAMAudioStage(id: "t", name: "t") { _, _ in
            Issue.record("the stage must not run for an over-long file")
            return SAMAudioEngine.Result(isolated: [], other: [], sampleRate: 48_000)
        }
        let model = SAMAudioRunModel()
        model.start(audioURL: url, stage: stage, output: box, maxDurationSeconds: 1)
        for _ in 0..<200 where model.isRunning { try await Task.sleep(for: .milliseconds(20)) }
        #expect(model.errorText == "This recording is 2 seconds long. Sound isolation handles up to 1 second at a time; split the recording into shorter parts and run each one.")
        #expect(!model.hasResult)
    }

    @Test func theRunModelShowsAReadErrorAndNeverCallsTheStage() async throws {
        let url = tempURL("fake.wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try Data("RIFF but not really".utf8).write(to: url)
        let box = SAMAudioTrackBox()
        let stage = SAMAudioStage(id: "t", name: "t") { _, _ in
            Issue.record("the stage must not run for an unreadable file")
            return SAMAudioEngine.Result(isolated: [], other: [], sampleRate: 48_000)
        }
        let model = SAMAudioRunModel()
        model.start(audioURL: url, stage: stage, output: box)
        for _ in 0..<200 where model.isRunning { try await Task.sleep(for: .milliseconds(20)) }
        #expect(!model.isRunning)
        #expect(model.errorText?.contains("couldn't be read as audio") == true)
        #expect(!model.hasResult)
    }

    @Test func savedWAVsKeepTheModelsRateAndTheSuggestedNamesDiffer() throws {
        let samples = Self.sine(count: 48_000, rate: 48_000, hz: 1_000, amplitude: 0.5)
        for track in SAMAudioRunModel.Track.allCases {
            let url = tempURL("out-\(track.fileSuffix).wav")
            defer { try? FileManager.default.removeItem(at: url) }
            try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: 48_000), to: url)
            let file = try AVAudioFile(forReading: url)
            #expect(file.fileFormat.sampleRate == 48_000)
            #expect(file.fileFormat.channelCount == 1)
        }
        #expect(SAMAudioRunModel.fileName(for: URL(fileURLWithPath: "/tmp/interview.m4a"), track: .isolated) == "interview-isolated.wav")
        #expect(SAMAudioRunModel.fileName(for: URL(fileURLWithPath: "/tmp/interview.m4a"), track: .other) == "interview-everything-else.wav")
        #expect(SAMAudioRunModel.fileName(for: nil, track: .isolated) == "audio-isolated.wav")
    }
}

/// Reads the module's own source to prove the load path cannot download (the `BareMLXReadScanTests` pattern). Skipped
/// where the sources aren't readable (a sandboxed test host).
enum SAMAudioNoNetworkScan {
    static let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()

    static var canReadSources: Bool {
        FileManager.default.isReadableFile(atPath: repoRoot.appendingPathComponent("MLXUI/Modules/SAMAudio").path)
    }

    static func files() -> [URL] {
        let dir = repoRoot.appendingPathComponent("MLXUI/Modules/SAMAudio")
        return ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "swift" }
    }

    static func combinedSource() -> String {
        files().compactMap { try? String(contentsOf: $0, encoding: .utf8) }.joined(separator: "\n")
    }

    /// One hit per forbidden token, with its file and line. Comments are stripped: the type comment names the
    /// `fromPretrained` / `HubApi` path it deliberately does **not** use.
    static func findings(forbidden: [String]) -> [String] {
        var hits: [String] = []
        for file in files() {
            guard let text = try? String(contentsOf: file, encoding: .utf8) else { continue }
            for (index, line) in text.components(separatedBy: "\n").enumerated() {
                let code = line.components(separatedBy: "//").first ?? line
                for token in forbidden where code.contains(token) {
                    hits.append("\(file.lastPathComponent):\(index + 1): \(token) — \(code.trimmingCharacters(in: .whitespaces))")
                }
            }
        }
        return hits
    }
}

private extension Double {
    /// Seconds x rate as a sample count, without a force-unwrap or an overflow.
    func secondsToSamples(rate: Int) -> Int {
        Int((self * Double(rate)).rounded())
    }
}
