import AVFoundation
import Foundation
import MLX
import Testing
@testable import MLXUI

/// AM-T, MC5-T part 1: ACE-Step 1.5 turbo (`ACE-Step/Ace-Step1.5`), smoke row 155 (AUTO; the REVIEW row is 156). OPT-IN, real weights, run only through the runner:
///
///     bash RSI/evals/model-smoke.sh run ACE-Step/Ace-Step1.5 --suite AceStepSmokeTests \
///         --weights /Volumes/Ovid/Users/chinpinboo/Documents/PipelineStudio/RSI/notes/modelcandidates/mc5/oracle/Ace-Step1.5 --worktree <worktree>
///
/// The runner sets `MLXUI_SMOKE_REPO` / `MLXUI_SMOKE_WEIGHTS` / `MLXUI_SMOKE_OUT` and builds the host unsandboxed; without them the whole suite is skipped (never a failure), which
/// the runner reports as NOT-RUN. The weights folder is READ ONLY here (this suite never downloads and never writes there); WAVs go to `MLXUI_SMOKE_OUT`, never into the repo.
/// `model-smoke.sh` turns the `REVIEW|156|...` lines into `review.html` (an audio player per WAV, the judging text, a Copy verdict button), so there is no separate page generator.
///
/// Every run goes through `AceStepRunModel.generate()`, the Generate button's own entry point, with the production generator (`AceStepEngine.generator`) pointed at the checkpoint
/// folder instead of `ModelStore.shared` (a scratch/oracle folder is not the app's store, the Klein / Parakeet precedent). The oracle is readme T23.
///
/// Not checked here (REVIEW): musicality, artifacts at chunk joins, vocal intelligibility, that the output matches the caption; the SwiftUI window itself.
@Suite(.enabled(if: ModelSmoke.enabled(AceStepSmoke.hfID), "run through RSI/evals/model-smoke.sh with --weights"), .serialized)
@MainActor
struct AceStepSmokeTests {

    // MARK: A. install layout (read only)

    @Test(.enabled(if: AceStepSmoke.selected("a")))
    func a_installLayoutIsTheSelectorsFourteenFilesWithExactSizes() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let fm = FileManager.default
        let wanted = AceStepInstallTests.selected()
        ModelSmoke.check(AceStepSmoke.auto, "install.selector-picks-the-14-files", Set(wanted) == AceStepInstallTests.expected && wanted.count == 14, "\(wanted.count) files")
        let sizes = Dictionary(uniqueKeysWithValues: AceStepInstallTests.siblings)
        var missing: [String] = [], wrong: [String] = []
        var bytes = 0
        for file in wanted {
            let url = dir.appendingPathComponent(file)
            guard let attrs = try? fm.attributesOfItem(atPath: url.path), let size = (attrs[.size] as? NSNumber)?.intValue else { missing.append(file); continue }
            bytes += size
            if size != sizes[file] { wrong.append("\(file) \(size) != \(sizes[file] ?? -1)") }
        }
        ModelSmoke.check(AceStepSmoke.auto, "install.all-14-present", missing.isEmpty, missing.isEmpty ? "all present" : "missing \(missing)")
        ModelSmoke.check(AceStepSmoke.auto, "install.exact-sizes", wrong.isEmpty, wrong.isEmpty ? "all 14 match the pinned listing (19671f4)" : wrong.joined(separator: "; "))
        ModelSmoke.check(AceStepSmoke.auto, "install.no-lm-folder", !fm.fileExists(atPath: dir.appendingPathComponent("acestep-5Hz-lm-1.7B").path), "the 1.7B LM folder is not part of the install")
        let card = try ModelSmoke.catalogEntry(AceStepSmoke.hfID)
        let gb = Double(bytes) / 1e9
        ModelSmoke.check(AceStepSmoke.auto, "install.size-within-5pct-of-card", abs(gb - card.downloadSizeGB) / card.downloadSizeGB <= 0.05,
                         String(format: "%.2f GB selected vs card %.2f GB", gb, card.downloadSizeGB))
        let marker = fm.fileExists(atPath: dir.appendingPathComponent(".installed").path)
        ModelSmoke.info(AceStepSmoke.auto, "install.marker", "\(marker ? ".installed present" : "no .installed marker (the folder is the oracle checkout, not an InstallManager install: a scratch install is not possible offline)")")
        print("INSTALL|\(AceStepSmoke.hfID)|files=\(wanted.count)|bytes=\(bytes)|seconds=0|state=read-only layout check of \(dir.lastPathComponent)")
    }

    // MARK: B. 10 s: the oracle, determinism, the module is registered

    @Test(.enabled(if: AceStepSmoke.selected("b")))
    func b_tenSecondsPassesTheOracleAndIsDeterministic() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let first = try await AceStepSmoke.run(dir: dir, label: "10 s", seconds: 10, seed: 7)
        AceStepSmoke.oracle("10s", first.wav, requested: 10, channels: 2)
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-10s-seed7.wav", data: first.wav,
                                judge: "Caption: \(AceStepSmoke.caption). Is it music, in that style? Any hiss, buzzing, clicks or a stuck loop? Does it sound finished rather than cut off?")
        if let other = ModelSmoke.value("MLXUI_SMOKE_MUSICGEN_WAV") {
            try ModelSmoke.artifact(AceStepSmoke.review, "musicgen-same-caption.wav", copying: URL(fileURLWithPath: other),
                                    judge: "MusicGen on the same caption, for comparison with ace-step-10s-seed7.wav: which sounds more musical and closer to the caption?")
        }
        let again = try await AceStepSmoke.run(dir: dir, label: "10 s repeat", seconds: 10, seed: 7)
        ModelSmoke.check(AceStepSmoke.auto, "determinism.same-seed-identical", first.wav == again.wav, "\(first.wav.count) vs \(again.wav.count) bytes")
        let different = try await AceStepSmoke.run(dir: dir, label: "10 s seed 8", seconds: 10, seed: 8)
        ModelSmoke.check(AceStepSmoke.auto, "determinism.different-seed-differs", different.wav != first.wav && different.wav.count == first.wav.count,
                         "seed 8 vs seed 7, same length \(different.wav.count == first.wav.count)")
        ModelSmoke.check(AceStepSmoke.auto, "run.seed-shown", first.usedSeed == 7 && different.usedSeed == 8, "used seeds \(first.usedSeed.map(String.init) ?? "nil"), \(different.usedSeed.map(String.init) ?? "nil")")
        ModelSmoke.check(AceStepSmoke.auto, "run.instrumental-request", first.request?.isInstrumental == true && first.request?.steps == 8, "lyrics empty, steps \(first.request?.steps ?? -1)")
    }

    // MARK: C / D. 60 s and 240 s: the oracle and the memory phases

    @Test(.enabled(if: AceStepSmoke.selected("c")))
    func c_sixtySecondsPassesTheOracle() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let run = try await AceStepSmoke.run(dir: dir, label: "60 s", seconds: 60, seed: 7)
        AceStepSmoke.oracle("60s", run.wav, requested: 60, channels: 2)
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-60s-seed7.wav", data: run.wav,
                                judge: "Same caption at the default length (60 s). Does the piece develop over a minute (not one bar repeated)? Any artifacts at the chunk joins, drift in tempo or pitch, or a thin or washed-out tail?")
    }

    @Test(.enabled(if: AceStepSmoke.selected("d")))
    func d_twoHundredFortySecondsPassesTheOracleAndTheMemoryGate() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let run = try await AceStepSmoke.run(dir: dir, label: "240 s", seconds: 240, seed: 7)
        AceStepSmoke.oracle("240s", run.wav, requested: 240, channels: 2)
        AceStepSmoke.chunkJoins("240s", run.wav, seconds: 240)
        let peakGB = AceStepSmoke.gb(run.peakMiB)
        ModelSmoke.check(AceStepSmoke.auto, "memory.240s-peak-at-most-12.0-GB", peakGB <= AceStepSmoke.ramGateGB && !run.hitStopLine,
                         String(format: "whole-process peak %.2f GB (R5 gate %.1f GB, stop line %.0f GB), base %.2f GB", peakGB, AceStepSmoke.ramGateGB, AceStepSmoke.stopLineGB, AceStepSmoke.gb(run.baseMiB)))
        // The 240 s WAV is large (about 46 MB); it is judged by the oracle only, and kept in the out folder for the owner to open if wanted.
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-240s-seed7.wav", data: run.wav,
                                judge: "The longest length (4 min). Does it hold together to the end with no long silences, loops or noise bursts? LISTEN AROUND THE CHUNK JOINS: the VAE decodes in tiles and joins them without a crossfade, a join every 128 latent frames = every 5.12 s (5.12, 10.24, 15.36 ... s). Seek to a few of them (for example 5.1 s, 10.2 s, 15.4 s, 61.4 s, 122.9 s, 230.4 s) and listen for a click, a pop, a step in level or a change of tone right at the join. The automated check (smoke line 240s.chunk-joins-no-discontinuity) looks within +-5 ms of every join for a sample-to-sample jump over 3x the file's own 99th-percentile step; a pass does not replace listening.")
    }

    // MARK: E. one vocal sample (English), 30 s

    @Test(.enabled(if: AceStepSmoke.selected("e")))
    func e_englishVocalThirtySeconds() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let run = try await AceStepSmoke.run(dir: dir, label: "30 s vocal", seconds: 30, seed: 7, caption: AceStepSmoke.vocalCaption, lyrics: AceStepSmoke.vocalLyrics)
        AceStepSmoke.oracle("vocal30s", run.wav, requested: 30, channels: 2)
        ModelSmoke.check(AceStepSmoke.auto, "vocal.request-carried-the-lyrics", run.request?.isInstrumental == false && run.request?.language == "en", "language \(run.request?.language ?? "nil")")
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-vocal-30s-seed7.wav", data: run.wav,
                                judge: "English lyrics (verse then chorus: 'The city lights are fading slow ... Calling out your name into the blue'). Is there a sung voice? How much of the lyric is intelligible? Does the style match a warm acoustic ballad?")
    }

    // MARK: E2. Mandarin and French vocal samples, 30 s (REVIEW-only)

    @Test(.enabled(if: AceStepSmoke.selected("e")))
    func e2_mandarinVocalThirtySeconds() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let run = try await AceStepSmoke.run(dir: dir, label: "30 s Mandarin vocal", seconds: 30, seed: 7, caption: AceStepSmoke.mandarinCaption,
                                             lyrics: AceStepSmoke.mandarinLyrics, language: "zh")
        AceStepSmoke.oracle("mandarin30s", run.wav, requested: 30, channels: 2)
        ModelSmoke.check(AceStepSmoke.auto, "mandarin.request-carried-the-lyrics", run.request?.isInstrumental == false && run.request?.language == "zh", "language \(run.request?.language ?? "nil")")
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-vocal-mandarin-30s-seed7.wav", data: run.wav,
                                judge: "Mandarin lyrics (the Run screen's Chinese setting; verse then chorus: '窗外的雨还在下 ... 你的名字还在心上'). Is there a sung voice, and is it Mandarin? How much is intelligible? Does it sound like a gentle female pop ballad with piano and strings?")
    }

    @Test(.enabled(if: AceStepSmoke.selected("french")))
    func e3_frenchVocalThirtySeconds() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        // Same seed, caption style, length and steps as the English clip (e), so the two can be compared; the language is the picker's French ("fr"), not Other.
        let run = try await AceStepSmoke.run(dir: dir, label: "30 s French vocal", seconds: 30, seed: 7, caption: AceStepSmoke.vocalCaption,
                                             lyrics: AceStepSmoke.frenchLyrics, language: "fr")
        AceStepSmoke.oracle("french30s", run.wav, requested: 30, channels: 2)
        ModelSmoke.check(AceStepSmoke.auto, "french.request-carried-the-lyrics", run.request?.isInstrumental == false && run.request?.language == "fr", "language \(run.request?.language ?? "nil")")
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-vocal-french-30s-seed7.wav", data: run.wav,
                                judge: "French vocal: is it sung in French and understandable? Same settings as the English clip for comparison.")
    }

    // MARK: E4. the English vocal clip at other seeds (REVIEW-only; `MLXUI_SMOKE_ONLY=seeds`)

    /// Does the engine ever sing without the planner LM? The English clip exactly as (e) (caption, lyrics, language en, 30 s, default 8 steps, DCW on) at four more seeds.
    @Test(.enabled(if: AceStepSmoke.only == "seeds"))
    func f_englishVocalSeeds() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        for seed in [UInt64(1), 2, 3, 20_251_106] {
            let run = try await AceStepSmoke.run(dir: dir, label: "30 s English vocal seed \(seed)", seconds: 30, seed: seed, caption: AceStepSmoke.vocalCaption,
                                                 lyrics: AceStepSmoke.vocalLyrics, language: "en")
            AceStepSmoke.oracle("english-seed\(seed)", run.wav, requested: 30, channels: 2)
            try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-vocal-english-seed\(seed)-30s.wav", data: run.wav,
                                    judge: "English vocal, no LM, seed \(seed): is there a sung voice, and from when?")
        }
    }

    // MARK: E5. the French vocal clip at other seeds (REVIEW-only; `MLXUI_SMOKE_ONLY=frseeds`)

    /// The French clip exactly as `e3` (caption, French lyrics, language fr, 30 s, default 8 steps, DCW on; `run` opens the Vocals section because lyrics are given) at seeds 1, 2 and 3, the same
    /// settings as the English seeds set.
    @Test(.enabled(if: AceStepSmoke.only == "frseeds"))
    func g_frenchVocalSeeds() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        for seed in [UInt64(1), 2, 3] {
            let run = try await AceStepSmoke.run(dir: dir, label: "30 s French vocal seed \(seed)", seconds: 30, seed: seed, caption: AceStepSmoke.vocalCaption,
                                                 lyrics: AceStepSmoke.frenchLyrics, language: "fr")
            AceStepSmoke.oracle("french-seed\(seed)", run.wav, requested: 30, channels: 2)
            ModelSmoke.check(AceStepSmoke.auto, "french-seed\(seed).request-was-a-french-vocal", run.request?.isInstrumental == false && run.request?.language == "fr", "language \(run.request?.language ?? "nil")")
            try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-vocal-french-seed\(seed)-30s.wav", data: run.wav,
                                    judge: "French vocal, no LM, seed \(seed): is there a sung voice, and from when?")
        }
    }

    // MARK: F. a one-row flow through RealExecutor

    @Test(.enabled(if: AceStepSmoke.selected("f")))
    func f_flowRowThroughRealExecutorIsCachedAndPassesTheOracle() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        let out = try #require(ModelSmoke.out)
        let blobs = out.appendingPathComponent("flow-blobs", isDirectory: true)
        let cacheRoot = out.appendingPathComponent("flow-cache", isDirectory: true)
        try? FileManager.default.removeItem(at: cacheRoot)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let catalog = try AceStepSmoke.catalog()
        let entry = try #require(catalog.first { $0.hfModelId == AceStepSmoke.hfID })
        let make: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            // the production SDK decides the settings (and refuses what the row can't have); only the weights folder is the checkpoint's
            guard let base = try AceStepSDK().makeStage(for: model, config: config) as? AceStepStage else { throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind) }
            return AceStepStage(modelID: model.id, settings: base.settings, directory: dir)
        }
        let real = RealExecutor(workspace: FlowWorkspace(root: blobs), flowID: "acestep-smoke", blobDirectory: blobs, makeModelStage: make,
                                installedModelIDs: [entry.id], catalog: catalog)
        let caching = CachingExecutor(inner: real, store: FlowCacheStore(root: cacheRoot), cacheTier: "real", catalog: catalog, runSeed: 1,
                                      workspace: FlowWorkspace(root: blobs), flowID: "acestep-smoke")
        let flow = "mlxflow 0.8\n1. Generate Sound   ACE-Step 1.5; steps=8; seed=7 \"\(AceStepSmoke.caption)\"\n\nmodels:\n  ACE-Step 1.5 = ACE-Step/Ace-Step1.5\n"
        let doc = try CatParser.parse(flow)
        ModelSmoke.check(AceStepSmoke.auto, "flow.row-names-ace-step", doc.rows.first?.model == "ACE-Step 1.5", doc.rows.first?.model ?? "nil")

        let started = Date()
        let events = try await ModelSmoke.measure("flow row Generate Sound (30 s, steps=8)", detail: "seed 7") {
            try await FlowInterpreter.run(doc, executor: caching, definitions: doc.definitions, presets: doc.presets)
        }
        let firstSeconds = Date().timeIntervalSince(started)
        let failed = events.filter { $0.kind == .rowFailed }
        ModelSmoke.check(AceStepSmoke.auto, "flow.row-completes", failed.isEmpty && events.contains { $0.kind == .rowCompleted && $0.path == "1" },
                         failed.compactMap { $0.error ?? $0.message }.joined(separator: "; "))
        let first = try AceStepSmoke.flowWAV(events)
        // The flow-facing result is the repo's mono mixdown (`AudioBuffer` is mono), at the fixed 30 s.
        AceStepSmoke.oracle("flow30s", first, requested: 30, channels: 1)
        try ModelSmoke.artifact(AceStepSmoke.review, "ace-step-flow-30s-seed7.wav", data: first,
                                judge: "The flow row's output (a fixed 30 s, mono mixdown, same caption, steps=8, seed 7). Same musical quality as the Run screen's 10 s sample, no thinning from the mono mixdown?")

        let secondStarted = Date()
        let second = try await FlowInterpreter.run(doc, executor: caching, definitions: doc.definitions, presets: doc.presets)
        let secondSeconds = Date().timeIntervalSince(secondStarted)
        ModelSmoke.check(AceStepSmoke.auto, "flow.second-run-is-a-cache-hit", second.contains { $0.kind == .cacheHit }, String(format: "first %.1f s, second %.2f s", firstSeconds, secondSeconds))
        let again = try AceStepSmoke.flowWAV(second)
        ModelSmoke.check(AceStepSmoke.auto, "flow.cached-output-identical", first == again, "\(first.count) vs \(again.count) bytes")

        // A different `steps=` is a different row: it must run again (not a hit) and still pass, proving `steps` reaches the sampler.
        let steps4 = try CatParser.parse(flow.replacingOccurrences(of: "steps=8", with: "steps=4"))
        let fourEvents = try await FlowInterpreter.run(steps4, executor: caching, definitions: steps4.definitions, presets: steps4.presets)
        let four = try AceStepSmoke.flowWAV(fourEvents)
        ModelSmoke.check(AceStepSmoke.auto, "flow.steps-4-runs-and-differs", !fourEvents.contains { $0.kind == .cacheHit } && four != first, "\(four.count) bytes")
        AceStepSmoke.oracle("flow30s-steps4", four, requested: 30, channels: 1)
    }

    // MARK: G. cancel

    @Test(.enabled(if: AceStepSmoke.selected("g")))
    func g_cancelAfterTheFirstDenoiseStepReturnsToIdleAndTheNextRunWorks() async throws {
        let dir = try await ModelSmoke.checkpoint(for: AceStepSmoke.hfID)
        MLX.Memory.clearCache()
        let base = ModelSmoke.physMiB()
        let model = AceStepRunModel(modelID: AceStepSmoke.hfID, directory: dir)
        model.caption = AceStepSmoke.caption; model.seedText = "7"; model.duration = 60
        model.generate()
        let started = Date()
        var reached = false
        while model.isRunning, Date().timeIntervalSince(started) < 900 {
            if model.phaseLabel.hasPrefix("Making the music: step 1 of") { reached = true; break }
            try await Task.sleep(for: .milliseconds(50))
        }
        ModelSmoke.check(AceStepSmoke.auto, "cancel.first-denoise-step-reached", reached, model.phaseLabel)
        let cancelAt = Date()
        model.cancel()
        await model.finished()
        let toIdle = Date().timeIntervalSince(cancelAt)
        ModelSmoke.check(AceStepSmoke.auto, "cancel.back-to-idle-without-result", !model.isRunning && model.wavData == nil && model.errorText == nil && !model.didGenerate,
                         "running \(model.isRunning), wav \(model.wavData != nil), error \(model.errorText ?? "none"), \(String(format: "%.1f", toIdle)) s to idle")
        // MC5-T-fix: idle now means the engine has unwound AND run its teardown (GPU synchronize + cache clear), so memory must come back promptly. The kernel hands pages back
        // a moment after the free, so poll for up to 10 s; the limit is unchanged (base + 1500 MiB) and the time it took is reported.
        let right = ModelSmoke.physMiB()
        let limitBefore = MLX.Memory.cacheLimit
        MLX.Memory.clearCache()
        var after = ModelSmoke.physMiB()
        let settleStart = Date()
        while after >= base + AceStepSmoke.cancelSlackMiB, Date().timeIntervalSince(settleStart) < 10 {
            try await Task.sleep(for: .milliseconds(100))
            after = ModelSmoke.physMiB()
        }
        ModelSmoke.info(AceStepSmoke.auto, "cancel.footprint-timeline", String(format: "idle %.0f MiB, after clearCache+settle %.0f MiB (%.1f s), cacheLimit %d MiB", right, after, Date().timeIntervalSince(settleStart), limitBefore >> 20))
        ModelSmoke.check(AceStepSmoke.auto, "cancel.memory-returns-near-base", after < base + AceStepSmoke.cancelSlackMiB,
                         String(format: "base %.0f MiB, after %.0f MiB, slack %.0f MiB", base, after, AceStepSmoke.cancelSlackMiB))
        print("FOOTPRINT|cancel after first denoise step|\(Int(base))|\(Int(after))|\(Int(after))|60 s run cancelled, idle = engine torn down, then settled")
        let next = try await AceStepSmoke.run(dir: dir, label: "10 s after cancel", seconds: 10, seed: 7)
        AceStepSmoke.oracle("after-cancel-10s", next.wav, requested: 10, channels: 2)
    }

    // MARK: H. both editions carry the module (compile-time reference + registration)

    @Test(.enabled(if: AceStepSmoke.selected("h")))
    func h_theModuleIsRegisteredAndCompiledIntoThisScheme() throws {
        // `AceStepModule`, `AceStepSDK` and `AceStepRunView` are referenced here, so this scheme's target must compile them. The Direct scheme is the same synchronized
        // folder (no per-target membership, no `#if`): it is proven by building both schemes, which every MC5 task does.
        _ = AceStepModule.self; _ = AceStepSDK.self; _ = AceStepRunView.self
        let registered = installedModules.contains { $0.descriptor.id == "acestep" }
        ModelSmoke.check(AceStepSmoke.auto, "editions.module-registered", registered, "acestep in installedModules")
        let entry = try ModelSmoke.catalogEntry(AceStepSmoke.hfID)
        ModelSmoke.check(AceStepSmoke.auto, "editions.sdk-claims-the-card", AceStepSDK().claim(entry) == .exact, "claim \(AceStepSDK().claim(entry))")
    }
}

// MARK: - Harness

nonisolated enum AceStepSmoke {
    static let hfID = "ACE-Step/Ace-Step1.5"
    static let auto = "155"
    static let review = "156"
    static let caption = "warm lo-fi hip hop, mellow piano, vinyl crackle, 85 bpm"
    // The oracle's english_vocal prompt (RSI/notes/modelcandidates/mc5/oracle/english_vocal/tokens_only.json).
    /// MC5-T review items (owner, 2026-11-21): more vocal samples, REVIEW-only (the oracle checks still run on them). Mandarin is the oracle's own `chinese_vocal` prompt (caption and
    /// lyrics from `oracle/chinese_vocal/tokens_only.json`), language "zh" = the Run picker's "Chinese". French (owner ruling 2026-11-21: German and French are the non-English languages): the
    /// picker's "fr", the English clip's caption, seed and length.
    static let mandarinCaption = "gentle Mandarin pop ballad, female vocal, piano and strings, emotional, 72 BPM"
    static let mandarinLyrics = "[Verse 1]\n窗外的雨还在下\n回忆在房间慢慢发芽\n那些没说完的话\n\n[Chorus]\n我还站在原地等你回答\n夜色把街灯都点亮\n你的名字还在心上"
    static let frenchLyrics = "[Verse 1]\nLe soleil se lève sur la mer\nLes vagues chantent doucement\nUn petit bateau part au vent\n\n[Chorus]\nEmporte-moi vers le matin\nLa mer m'appelle, le jour revient\nChante avec moi jusqu'au matin"
    /// `MLXUI_SMOKE_ONLY=french` (as `TEST_RUNNER_MLXUI_SMOKE_ONLY` through `xcodebuild`, which `model-smoke.sh` inherits) runs only the named clip's test; every other test of this suite skips.
    static let only: String? = { let v = ProcessInfo.processInfo.environment["MLXUI_SMOKE_ONLY"]; return (v?.isEmpty ?? true) ? nil : v }()
    static func selected(_ name: String) -> Bool { only == nil || only == name }
    static let vocalCaption = "warm acoustic singer-songwriter ballad, intimate male vocal, fingerpicked guitar, soft strings, 84 BPM"
    static let vocalLyrics = "[Verse 1]\nThe city lights are fading slow\nI trace the roads we used to know\nEvery window holds a ghost of you\n\n[Chorus]\nBut I keep walking through the rain\nHolding on to what remains\nCalling out your name into the blue"
    /// R5: whole-process peak at 240 s, decimal GB. The stop line aborts the run (the Klein sampler's rule).
    static let ramGateGB = 12.0
    static let stopLineGB = 14.0
    /// Footprint after a cancel must be under base + this (MiB): the allocator does not hand every page back, and the test host keeps its own small residue.
    static let cancelSlackMiB = 1500.0
    /// T23 floors. RMS over the whole file; windows are 1 s; a silent gap is a run of 0.5 s windows under -60 dBFS.
    static let rmsFloor: Float = 0.01
    static let activeWindowRMS: Float = 0.01
    static let activeWindowsNeeded = 3
    static let silentWindowRMS: Float = 0.001
    static let maxSilentGapSeconds = 2.0
    static let peakCeiling: Float = 0.95
    static let clipFractionCeiling = 0.0005
    static let dcCeiling: Float = 0.01

    static func gb(_ mib: Double) -> Double { mib * 1_048_576 / 1e9 }

    static func catalog() throws -> [ModelEntry] {
        let data = try Data(contentsOf: ModelSmoke.repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    // MARK: A run through the button's own path, with the footprint tracked per phase

    final class PhaseTracker: @unchecked Sendable {
        private let lock = NSLock()
        private var current = "load"
        private var peaks: [String: Double] = [:]
        private var ends: [String: Double] = [:]
        private var order: [String] = ["load"]
        private var stopped = false
        func enter(_ phase: AceStepPhase) {
            let name: String
            switch phase { case .text, .condition: name = "load"; case .denoise: name = "denoise"; case .decode: name = "decode" }
            lock.lock(); defer { lock.unlock() }
            guard name != current else { return }
            ends[current] = ModelSmoke.physMiB(); current = name; if !order.contains(name) { order.append(name) }
        }
        func note(_ mib: Double) { lock.lock(); peaks[current] = max(peaks[current] ?? 0, mib); lock.unlock() }
        func stop() { lock.lock(); stopped = true; lock.unlock() }
        var hitStopLine: Bool { lock.lock(); defer { lock.unlock() }; return stopped }
        func finish() { lock.lock(); ends[current] = ModelSmoke.physMiB(); lock.unlock() }
        var summary: [(name: String, peak: Double, end: Double)] {
            lock.lock(); defer { lock.unlock() }
            return order.map { ($0, peaks[$0] ?? 0, ends[$0] ?? 0) }
        }
    }

    struct RunResult {
        var wav: Data
        var usedSeed: UInt64?
        var request: AceStepRequest?
        var baseMiB: Double
        var peakMiB: Double
        var hitStopLine: Bool
    }

    /// One Generate press: the run model on the checkpoint folder, the whole process sampled every 20 ms, FOOTPRINT lines per phase and for the run. Aborts at the stop line.
    @MainActor
    static func run(dir: URL, label: String, seconds: Double, seed: UInt64, caption: String = AceStepSmoke.caption, lyrics: String = "", language: String = "en") async throws -> RunResult {
        MLX.Memory.clearCache()
        let tracker = PhaseTracker()
        let inner = AceStepEngine.generator(modelID: hfID, directory: dir)
        let model = AceStepRunModel(modelID: hfID, generator: { request, onPhase in
            try await inner(request) { phase in tracker.enter(phase); onPhase(phase) }
        })
        model.caption = caption; model.vocalsEnabled = !lyrics.isEmpty; model.lyrics = lyrics; model.language = language; model.seedText = String(seed); model.duration = seconds
        let base = ModelSmoke.physMiB()
        tracker.note(base)
        let limitMiB = stopLineGB * 1e9 / 1_048_576
        let sampler = Task.detached {
            while !Task.isCancelled {
                let now = ModelSmoke.physMiB()
                tracker.note(now)
                if now > limitMiB { tracker.stop(); await MainActor.run { model.cancel() }; break }
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        try await ModelSmoke.measure("run \(label) (whole process)", detail: "seed \(seed), \(lyrics.isEmpty ? "instrumental" : "with lyrics")") {
            model.generate()
            await model.finished()
        }
        sampler.cancel()
        tracker.finish()
        var peak = base
        for phase in tracker.summary {
            peak = max(peak, phase.peak)
            print("FOOTPRINT|run \(label) \(phase.name)|\(Int(base))|\(Int(phase.peak))|\(Int(phase.end))|phase peak, whole process")
        }
        if tracker.hitStopLine { ModelSmoke.check(auto, "memory.\(label)-stop-line", false, String(format: "footprint passed the %.0f GB stop line, run aborted", stopLineGB)) }
        try #require(model.errorText == nil, "\(label): the run failed: \(model.errorText ?? "")")
        let wav = try #require(model.wavData, "\(label): no WAV")
        return RunResult(wav: wav, usedSeed: model.usedSeed, request: model.request, baseMiB: base, peakMiB: peak, hitStopLine: tracker.hitStopLine)
    }

    // MARK: Chunk-join check (MC5-T review item)

    /// The VAE's tiling (`AceStepVAE.plan`, chunk 256, overlap 64 each side) keeps each window's core and joins the cores with no crossfade, so the joins are at the end of every core but the last:
    /// every 128 latent frames, x 1920 samples per frame = 245,760 samples = 5.12 s at 48 kHz. Derived from the plan itself, never hard-coded.
    static func joinSamples(seconds: Double) -> [Int] {
        let frames = Int((seconds * 25).rounded())
        return AceStepVAE.plan(frames: frames).dropLast().map { $0.coreEnd * 1920 }
    }

    static let joinHalfWindow = 240        // 5 ms at 48 kHz
    static let joinFactor = 3.0

    /// Threshold, derived from the signal's own step size: for each channel, the largest sample-to-sample step within +-5 ms of a join must be at most `factor` x the 99th percentile of the
    /// same statistic (largest step in a 10 ms window) taken over every window of the file that is NOT near a join. Returns the worst ratio and where it was.
    static func chunkJoinRatio(_ d: Decoded, joins: [Int]) -> (ratio: Double, join: Int, p99: Double) {
        var worst = (ratio: 0.0, join: 0, p99: 0.0)
        let window = 2 * joinHalfWindow
        for channel in d.samples {
            let n = channel.count
            guard n > window + 1 else { continue }
            func maxStep(_ lo: Int, _ hi: Int) -> Double {
                var m = 0.0
                var i = max(lo, 0)
                while i < min(hi, n - 1) { m = max(m, Double(abs(channel[i + 1] - channel[i]))); i += 1 }
                return m
            }
            var others: [Double] = []
            var start = 0
            while start + window <= n {
                if !joins.contains(where: { abs($0 - (start + joinHalfWindow)) < window }) { others.append(maxStep(start, start + window)) }
                start += window
            }
            guard !others.isEmpty else { continue }
            others.sort()
            let p99 = max(others[min(others.count - 1, Int(Double(others.count) * 0.99))], 1e-4)
            for j in joins {
                let ratio = maxStep(j - joinHalfWindow, j + joinHalfWindow) / p99
                if ratio > worst.ratio { worst = (ratio, j, p99) }
            }
        }
        return worst
    }

    @discardableResult
    static func chunkJoins(_ name: String, _ wav: Data, seconds: Double) -> Bool {
        guard let decoded = decode(wav) else { ModelSmoke.check(auto, "\(name).chunk-joins-no-discontinuity", false, "WAV did not decode"); return false }
        let joins = joinSamples(seconds: seconds)
        let r = chunkJoinRatio(decoded, joins: joins)
        let ok = !joins.isEmpty && r.ratio <= joinFactor
        ModelSmoke.check(auto, "\(name).chunk-joins-no-discontinuity", ok,
                         String(format: "%d joins every %.2f s; worst step near a join is %.2fx the file's 99th-percentile step (limit %.1fx), at %.3f s", joins.count,
                                Double(joins.first ?? 0) / 48_000, r.ratio, joinFactor, Double(r.join) / 48_000))
        return ok
    }

    // MARK: The T23 oracle

    struct Decoded { var rate: Double; var channels: Int; var frames: Int; var samples: [[Float]] }

    static func decode(_ data: Data) -> Decoded? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-smoke-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: url) }
        guard (try? data.write(to: url)) != nil, let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else { return nil }
        do { try file.read(into: buffer) } catch { return nil }
        guard let channelData = buffer.floatChannelData else { return nil }
        let n = Int(buffer.frameLength)
        let channels = (0 ..< Int(format.channelCount)).map { Array(UnsafeBufferPointer(start: channelData[$0], count: n)) }
        return Decoded(rate: format.sampleRate, channels: channels.count, frames: n, samples: channels)
    }

    /// Prints one SMOKE line per check (and records an issue per failure). `requested` is the asked-for length in seconds.
    @discardableResult
    static func oracle(_ name: String, _ wav: Data, requested: Double, channels: Int) -> Bool {
        let results = oracleResults(name, wav, requested: requested, channels: channels)
        for r in results { ModelSmoke.check(auto, r.name, r.ok, r.detail) }
        return results.allSatisfy(\.ok)
    }

    /// The T23 checks as data (no printing), so the oracle itself can be unit-tested on synthetic audio.
    static func oracleResults(_ name: String, _ wav: Data, requested: Double, channels: Int) -> [(name: String, ok: Bool, detail: String)] {
        var results: [(name: String, ok: Bool, detail: String)] = []
        @discardableResult func add(_ check: String, _ ok: Bool, _ detail: String) -> Bool { results.append(("\(name).\(check)", ok, detail)); return ok }
        guard let d = decode(wav) else { add("decodes", false, "\(wav.count) bytes did not decode"); return results }
        add("decodes", true, "\(wav.count) bytes")
        add("48kHz", d.rate == 48_000, "\(d.rate) Hz")
        add("channels", d.channels == channels, "\(d.channels) channels (expected \(channels))")
        let seconds = Double(d.frames) / max(d.rate, 1)
        add("duration", abs(seconds - requested) <= 0.1, String(format: "%.3f s for %.0f s requested", seconds, requested))
        let all = d.samples.flatMap { $0 }
        add("finite", all.allSatisfy { $0.isFinite }, "\(all.count) samples")
        var sum: Float = 0, peak: Float = 0, clipped = 0
        for s in all { sum += s * s; peak = max(peak, abs(s)); if abs(s) >= 0.999 { clipped += 1 } }
        let rms = all.isEmpty ? 0 : (sum / Float(all.count)).squareRoot()
        add("not-silent-rms", rms > rmsFloor, String(format: "rms %.4f (floor %.2f)", rms, rmsFloor))
        let win = Int(d.rate)
        let half = max(win / 2, 1)
        func windowRMS(_ size: Int) -> [Float] {
            stride(from: 0, to: max(d.frames - size + 1, 1), by: size).map { start in
                var s: Float = 0, n = 0
                for ch in d.samples { for i in start ..< min(start + size, d.frames) { s += ch[i] * ch[i]; n += 1 } }
                return n == 0 ? 0 : (s / Float(n)).squareRoot()
            }
        }
        let active = windowRMS(win).filter { $0 > activeWindowRMS }.count
        add("seconds-of-sound", active >= min(activeWindowsNeeded, Int(requested)), "\(active) one-second windows above rms \(activeWindowRMS)")
        // No silent gap over 2 s, except trailing silence at the end (an instrumental may end on a tail).
        var windows = windowRMS(half)
        while let last = windows.last, last < silentWindowRMS { windows.removeLast() }
        var run = 0, longest = 0
        for w in windows { if w < silentWindowRMS { run += 1; longest = max(longest, run) } else { run = 0 } }
        let gap = Double(longest) * 0.5
        add("no-silent-gap", gap <= maxSilentGapSeconds, String(format: "longest internal silence %.1f s (limit %.0f s)", gap, maxSilentGapSeconds))
        add("peak", peak <= peakCeiling, String(format: "peak %.3f (ceiling %.2f)", peak, peakCeiling))
        let clipFraction = all.isEmpty ? 0 : Double(clipped) / Double(all.count)
        add("not-hard-clipped", clipFraction <= clipFractionCeiling, String(format: "%.5f of samples at full scale (limit %.4f)", clipFraction, clipFractionCeiling))
        let dc = d.samples.map { ch in ch.isEmpty ? 0 : abs(ch.reduce(0, +) / Float(ch.count)) }.max() ?? 0
        add("no-dc-offset", dc <= dcCeiling, String(format: "largest channel mean %.5f (ceiling %.2f)", dc, dcCeiling))
        return results
    }

    /// The WAV bytes of the flow row's output (row 1), read from the blob path the executor wrote.
    static func flowWAV(_ events: [FlowInterpreter.PathEvent]) throws -> Data {
        let path = try #require(events.first { $0.kind == .rowCompleted && $0.path == "1" }?.output?.items.first?.path, "row 1 produced no audio file")
        return try Data(contentsOf: path)
    }
}

/// The oracle's own tests: no weights, so they run in every suite. They exist so a wrong threshold or a decode bug shows up here, not during the lead's 20-minute real run.
struct AceStepSmokeOracleTests {
    private func wav(seconds: Double, channels: Int = 2, amplitude: Float = 0.5, dc: Float = 0, hz: Double = 220, silenceFrom: Double? = nil) -> Data {
        let frames = Int(seconds * 48_000)
        var samples = [Float](repeating: 0, count: frames * 2)
        for i in 0 ..< frames {
            let t = Double(i) / 48_000
            let on = silenceFrom.map { t < $0 } ?? true
            let v = on ? Float(sin(2 * Double.pi * hz * t)) * amplitude + dc : 0
            samples[2 * i] = v; samples[2 * i + 1] = v
        }
        return AceStepAudio(samples: samples).wavData()
    }

    @Test func theJoinPositionsComeFromTheTilingPlan() {
        let joins = AceStepSmoke.joinSamples(seconds: 240)
        #expect(joins.first == 128 * 1920 && joins.count == 6000 / 128, "\(joins.count) joins, first \(joins.first ?? -1)")
        #expect(AceStepSmoke.joinSamples(seconds: 10).isEmpty, "250 frames fit one chunk: no joins")
    }

    @Test func aStepAtAJoinIsCaughtAndACleanSignalOrALoudTransientElsewhereIsNot() throws {
        let rate = 48_000.0, seconds = 30.0
        let joins = AceStepSmoke.joinSamples(seconds: seconds)
        func decoded(_ tweak: (inout [Float]) -> Void) throws -> AceStepSmoke.Decoded {
            var x = (0 ..< Int(seconds * rate)).map { Float(sin(2 * Double.pi * 220 * Double($0) / rate)) * 0.3 + Float(sin(2 * Double.pi * 3_100 * Double($0) / rate)) * 0.05 }
            tweak(&x)
            return try #require(AceStepSmoke.decode(AceStepAudio(samples: x.flatMap { [$0, $0] }).wavData()))
        }
        let clean = try decoded { _ in }
        #expect(AceStepSmoke.chunkJoinRatio(clean, joins: joins).ratio <= AceStepSmoke.joinFactor)
        let clicked = try decoded { x in for i in joins[1]...(joins[1] + 40) { x[i] += 0.5 } }
        #expect(AceStepSmoke.chunkJoinRatio(clicked, joins: joins).ratio > AceStepSmoke.joinFactor, "a 0.5 jump at a join")
        let elsewhere = try decoded { x in for i in (joins[1] + 40_000)...(joins[1] + 40_040) { x[i] += 0.5 } }
        #expect(AceStepSmoke.chunkJoinRatio(elsewhere, joins: joins).ratio <= AceStepSmoke.joinFactor, "the same jump away from a join is not the join check's business")
    }

    private func failing(_ results: [(name: String, ok: Bool, detail: String)]) -> [String] { results.filter { !$0.ok }.map(\.name) }

    @Test func aHealthyClipPassesEveryCheck() {
        let r = AceStepSmoke.oracleResults("t", wav(seconds: 10), requested: 10, channels: 2)
        #expect(failing(r).isEmpty, "\(failing(r))")
        #expect(r.count == 11)
    }

    @Test func wrongLengthSilenceClippingDCAndGapsAreCaught() {
        #expect(failing(AceStepSmoke.oracleResults("t", wav(seconds: 9), requested: 10, channels: 2)) == ["t.duration"])
        #expect(failing(AceStepSmoke.oracleResults("t", wav(seconds: 10, amplitude: 0), requested: 10, channels: 2)).contains("t.not-silent-rms"))
        #expect(failing(AceStepSmoke.oracleResults("t", wav(seconds: 10, amplitude: 1.4), requested: 10, channels: 2)).contains("t.not-hard-clipped"))
        #expect(failing(AceStepSmoke.oracleResults("t", wav(seconds: 10, amplitude: 0.3, dc: 0.2), requested: 10, channels: 2)).contains("t.no-dc-offset"))
        #expect(failing(AceStepSmoke.oracleResults("t", wav(seconds: 10), requested: 10, channels: 1)) == ["t.channels"])
        #expect(failing(AceStepSmoke.oracleResults("t", Data([1, 2, 3]), requested: 10, channels: 2)) == ["t.decodes"])
    }

    @Test func aTailOfSilenceIsAllowedButASilentGapInTheMiddleIsNot() {
        // sound for the first 3 s then silence to 10 s: trailing silence only
        #expect(failing(AceStepSmoke.oracleResults("t", wav(seconds: 10, silenceFrom: 3), requested: 10, channels: 2)).isEmpty)
        // sound, then 4 s of silence, then sound again
        let frames = 10 * 48_000
        var samples = [Float](repeating: 0, count: frames * 2)
        for i in 0 ..< frames where i < 2 * 48_000 || i > 6 * 48_000 { let v = Float(sin(2 * Double.pi * 220 * Double(i) / 48_000)) * 0.5; samples[2 * i] = v; samples[2 * i + 1] = v }
        #expect(failing(AceStepSmoke.oracleResults("t", AceStepAudio(samples: samples).wavData(), requested: 10, channels: 2)) == ["t.no-silent-gap"])
    }
}
