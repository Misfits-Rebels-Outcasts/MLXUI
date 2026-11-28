import Testing
import Foundation
import Observation
@testable import MLXUI

/// FA-5 (issue #13 decision 7): the Run screen's view-model — Stop, and the two refusals it makes before any work.
/// No weights: the stage is a stub.
@MainActor
struct AlignRunModelTests {

    private func wavURL(seconds: Double = 1.0) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fa-model-\(UUID().uuidString).wav")
        let samples = (0..<Int(16_000 * seconds)).map { Float(sin(Double($0) / 16_000.0 * 2 * .pi * 440) * 0.2) }
        try AudioWriter.writeWAV(AudioBuffer(samples: samples, sampleRate: 16_000), to: url)
        return url
    }

    /// A sheet's model with its own step tracker, so tests running together never see each other's steps.
    private func newModel(_ modelID: String = "align-test", tracker: AlignStepTracker? = nil) -> AlignRunModel {
        AlignRunModel(modelID: modelID, tracker: tracker ?? AlignStepTracker())
    }

    /// A stage that takes long enough to be stopped, and records that it was entered.
    private struct SlowStage: PipelineStage, @unchecked Sendable {
        let calls = AlignRunCallLog()
        var id: String { "slow-align-stub" }
        var name: String { "Qwen3 Aligner 0.6B" }
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            calls.enter()
            try await Task.sleep(for: .seconds(30))
            return .text("text,start,end\nhello,0.000,0.500\n")
        }
    }

    @Test func stopCancelsTheRunWithoutAnError() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = SlowStage()
        let model = newModel()
        model.start(audioURL: url, transcript: "hello", language: "en", stage: stage)
        // Wait until the stage is actually running, then press Stop (the button's call).
        for _ in 0..<200 where stage.calls.count == 0 {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(stage.calls.count == 1, "the run reached the stage")
        model.stop()
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(!model.isRunning, "Stop ends the run")
        #expect(model.errorText == nil, "a stopped run is never an error")
        #expect(model.result == nil)
    }

    @Test func anEmptyTranscriptIsRefusedBeforeTheStageIsBuilt() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = SlowStage()
        let model = newModel()
        model.start(audioURL: url, transcript: " \n\t ", language: "en", stage: stage)
        #expect(!model.isRunning)
        #expect(model.errorText == ForcedAlignerEngine.EngineError.emptyTranscript.errorDescription)
        #expect(stage.calls.count == 0)
    }

    @Test func anUnsupportedLanguageIsRefusedBeforeTheStageIsBuilt() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = SlowStage()
        let model = newModel()
        model.start(audioURL: url, transcript: "hello", language: "xx", stage: stage)
        #expect(!model.isRunning)
        #expect(model.errorText?.contains("xx") == true)
        #expect(stage.calls.count == 0)
    }

    /// An audio-in, fixed-CSV-out stage (the real stage is `ForcedAlignerStage`).
    private struct TextStubStage: PipelineStage, @unchecked Sendable {
        let csv: String
        var id: String { "align-text-stub" }
        var name: String { "Qwen3 Aligner 0.6B" }
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            progress(1)
            return .text(csv)
        }
    }

    @Test func aSuccessfulRunFoldsTheCSVBackIntoRows() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = TextStubStage(csv: "text,start,end\nHello,0.100,0.500\nworld,0.500,1.000\n")
        let model = newModel()
        model.start(audioURL: url, transcript: "Hello world", language: "en", stage: stage)
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(25))
        }
        let result = try #require(model.result)
        #expect(result.units.map(\.text) == ["Hello", "world"])
        #expect(abs(result.duration - 1.0) < 0.01)
        #expect(result.transcript == "Hello world")
        #expect(model.errorText == nil)
    }

    /// The Export SRT/VTT buttons end cues at the sentences of the transcript the run aligned.
    @Test func theExportsUseTheRunsTranscript() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = TextStubStage(csv: "text,start,end\nBonjour,0.100,0.400\nÀ,0.400,0.600\ndemain,0.600,0.900\n")
        let model = newModel()
        model.start(audioURL: url, transcript: " Bonjour.\nÀ demain ! ", language: "fr", stage: stage)
        for _ in 0..<200 where model.isRunning {
            try await Task.sleep(for: .milliseconds(25))
        }
        let result = try #require(model.result)
        #expect(result.transcript == "Bonjour. À demain !", "the text the run aligned")
        #expect(ForcedAlignerRunView.ExportFormat.srt.render(result)
                == "1\n00:00:00,100 --> 00:00:00,400\nBonjour\n\n2\n00:00:00,400 --> 00:00:00,900\nÀ demain\n")
        #expect(ForcedAlignerRunView.ExportFormat.vtt.render(result)
                == "WEBVTT\n\n00:00:00.100 --> 00:00:00.400\nBonjour\n\n00:00:00.400 --> 00:00:00.900\nÀ demain\n")
        #expect(ForcedAlignerRunView.ExportFormat.csv.render(result) == AlignmentExport.csv(result.units))
    }

    // MARK: - A late Stop, a replaced run, repeated rows

    /// A stage that ignores cancellation, as the engine's synchronous `generate` does: it waits for the test to release
    /// it, reports progress, then returns its CSV (or fails).
    private struct GatedStage: PipelineStage, @unchecked Sendable {
        let csv: String
        var fails = false
        let gate = AlignRunGate()
        var id: String { "align-gated-stub" }
        var name: String { "Qwen3 Aligner 0.6B" }
        var accepts: MediaKind { .audio }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            await gate.wait()
            progress(0.4)
            if fails { throw StageError.engineFailure(stage: "Forced aligner", underlying: CocoaError(.fileNoSuchFile)) }
            return .text(csv)
        }
    }

    private func waitUntilEntered(_ gate: AlignRunGate) async throws {
        for _ in 0..<200 where !gate.isEntered {
            try await Task.sleep(for: .milliseconds(25))
        }
        #expect(gate.isEntered, "the run reached the stage")
    }

    /// Releases a gated run and waits for its task to end, then for any progress hop it queued.
    private func release(_ gate: AlignRunGate, _ task: Task<Void, Never>) async throws {
        gate.open()
        await task.value
        try await Task.sleep(for: .milliseconds(100))
    }

    private static let oldCSV = "text,start,end\nalt,0.100,0.500\n"
    private static let newCSV = "text,start,end\nneu,0.200,0.600\n"

    @Test(arguments: [false, true])
    func aStopDuringTheForwardPassDropsTheResult(stepFails: Bool) async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = GatedStage(csv: Self.oldCSV, fails: stepFails)
        let model = newModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: stage)
        try await waitUntilEntered(stage.gate)
        let task = try #require(model.task)
        model.stop()
        try await release(stage.gate, task)
        #expect(model.result == nil, "a stopped run shows no result")
        #expect(model.errorText == nil, "a stopped run is never an error")
        #expect(!model.isRunning)
        #expect(model.progress == 0)
    }

    // MARK: - Stop, Done and the memory lease (owner rulings 11 and 12, 2026-11-27)

    /// A test-owned budget and engine cache, and the stage the Run view builds on them: `LeasingStage.perRun` over the
    /// sheet's `LeasingStage`. The ledger is what says whether the lease is still held.
    private struct LeaseRig {
        let budget = MemoryBudget(capacityBytes: 20 * 1_073_741_824,
                                  cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
        let cache: EngineCache
        let model = makeEntry(id: "align-lease", modelType: .align, ramGB: 3)

        init() {
            cache = EngineCache(budget: budget, countsLLMWeights: true)
        }

        func stage(_ probe: any PipelineStage, language: String) -> any PipelineStage {
            let handed = LeasingStage(model: model, config: .default, probe: probe, cache: cache, maxWait: .milliseconds(300))
            return LeasingStage.perRun(handed, config: StageConfig(language: language)) { _ in probe }
        }
    }

    /// Long enough for a cancellation to unwind anything that would give up early.
    private func pause() async throws { try await Task.sleep(for: .milliseconds(200)) }

    @Test func stopShowsStoppingAtOnceAndKeepsTheLeaseUntilTheStepReturns() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let rig = LeaseRig()
        let step = GatedStage(csv: Self.oldCSV)
        let model = newModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: rig.stage(step, language: "de"))
        try await waitUntilEntered(step.gate)
        let task = try #require(model.task)
        let booked = rig.budget.usedBytes
        #expect(booked > 0 && rig.budget.inUseCount == 1, "leased while the step runs")
        #expect(!model.isStopping)
        #expect(model.runButtonTitle == "Aligning…")

        model.stop()
        // At once, before the step has had any chance to return.
        #expect(model.isStopping)
        #expect(model.runButtonTitle == "Stopping…")
        #expect(model.progressText == "Stopping…")
        #expect(rig.budget.usedBytes == booked && rig.budget.inUseCount == 1)
        try await pause()
        #expect(model.isStopping && model.isRunning, "the step is still in flight")
        #expect(rig.budget.usedBytes == booked && rig.budget.inUseCount == 1, "Stop releases nothing")

        try await release(step.gate, task)
        #expect(rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0 && rig.cache.count == 0)
        #expect(model.result == nil)
        #expect(model.errorText == nil)
        #expect(!model.isRunning && !model.isStopping)
        #expect(model.runButtonTitle == "Align")
    }

    /// Done calls `stop()` and dismisses, and the sheet disappearing calls it again.
    @Test func doneDuringARunDiscardsTheResultAndReleasesOnlyAfterTheStep() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let rig = LeaseRig()
        let step = GatedStage(csv: Self.oldCSV)
        let model = newModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: rig.stage(step, language: "de"))
        try await waitUntilEntered(step.gate)
        let task = try #require(model.task)
        let booked = rig.budget.usedBytes
        #expect(booked > 0 && rig.budget.inUseCount == 1)

        model.stop()   // the Done button
        model.stop()   // onDisappear
        #expect(model.isStopping)
        try await pause()
        #expect(rig.budget.usedBytes == booked && rig.budget.inUseCount == 1, "held while the step runs")

        try await release(step.gate, task)
        #expect(rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0 && rig.cache.count == 0)
        #expect(model.result == nil, "the result is discarded")
        #expect(model.errorText == nil)
        #expect(!model.isRunning && !model.isStopping)
    }

    @Test func stopWithNothingRunningDoesNothing() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let model = newModel()
        model.stop()
        #expect(!model.isStopping && model.runButtonTitle == "Align")
        model.start(audioURL: url, transcript: "neu", language: "de", stage: TextStubStage(csv: Self.newCSV))
        await model.task?.value
        model.stop()
        #expect(!model.isStopping)
        #expect(model.result?.units.map(\.text) == ["neu"], "Done after a run keeps nothing from being exported")
    }

    // MARK: - No input change, and no second run, until the step returns (owner ruling 14, 2026-11-27)

    private func attempt(_ path: AlignInputPath, _ model: AlignRunModel, audio: URL, transcript: URL,
                         stage: any PipelineStage) -> Bool {
        switch path {
        case .chooseAudio: return model.chooseAudio(audio)
        case .dropAudio: return model.dropAudio([audio])
        case .openTranscript: return model.openTranscript(transcript)
        case .start:
            let before = model.currentRun
            model.start(audioURL: audio, transcript: "neu", language: "de", stage: stage)
            return model.currentRun != before
        }
    }

    @Test(arguments: AlignInputPath.allCases, [false, true])
    func everyInputPathIsRefusedUntilTheStepReturns(path: AlignInputPath, stopping: Bool) async throws {
        let url = try wavURL()
        let other = try wavURL()
        let text = FileManager.default.temporaryDirectory.appendingPathComponent("fa-model-\(UUID().uuidString).txt")
        try "neu".write(to: text, atomically: true, encoding: .utf8)
        defer { for file in [url, other, text] { try? FileManager.default.removeItem(at: file) } }
        let first = GatedStage(csv: Self.oldCSV)
        let second = GatedStage(csv: Self.newCSV)
        let model = newModel()
        #expect(model.canChangeInputs)
        #expect(model.chooseAudio(url))
        model.transcript = "alt"
        model.start(audioURL: url, transcript: model.transcript, language: "de", stage: first)
        try await waitUntilEntered(first.gate)
        if stopping { model.stop() }
        model.errorText = "unchanged"   // a reset would clear it
        let task = try #require(model.task)
        let run = try #require(model.currentRun)

        #expect(!model.canChangeInputs)
        #expect(!attempt(path, model, audio: other, transcript: text, stage: second), "refused")
        #expect(model.task == task && model.currentRun == run)
        #expect(model.isRunning && model.isStopping == stopping)
        #expect(model.errorText == "unchanged")
        #expect(model.audioURL == url && model.transcript == "alt")
        try await pause()
        #expect(!second.gate.isEntered, "no second step")

        try await release(first.gate, task)
        #expect(model.canChangeInputs)
        #expect(attempt(path, model, audio: other, transcript: text, stage: second), "accepted again")
        #expect(model.errorText == nil)
        switch path {
        case .chooseAudio, .dropAudio:
            #expect(model.audioURL == other && model.result == nil)
        case .openTranscript:
            #expect(model.transcript == "neu" && model.result == nil)
        case .start:
            try await waitUntilEntered(second.gate)
            try await release(second.gate, try #require(model.task))
            #expect(model.result?.units.map(\.text) == ["neu"])
        }
    }

    @Test func noSecondStepStartsUntilTheFirstHasReturned() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let rig = LeaseRig()
        let step = GatedStage(csv: Self.oldCSV)
        let next = GatedStage(csv: Self.newCSV)
        let model = newModel()
        model.start(audioURL: url, transcript: "alt", language: "de", stage: rig.stage(step, language: "de"))
        try await waitUntilEntered(step.gate)
        let task = try #require(model.task)
        let one = rig.budget.usedBytes
        model.start(audioURL: url, transcript: "neu", language: "fr", stage: rig.stage(next, language: "fr"))
        model.stop()
        model.start(audioURL: url, transcript: "neu", language: "fr", stage: rig.stage(next, language: "fr"))
        try await pause()
        #expect(step.gate.entries == 1 && next.gate.entries == 0)
        #expect(rig.budget.usedBytes == one && rig.budget.inUseCount == 1, "one run, one booking")

        try await release(step.gate, task)
        #expect(rig.budget.usedBytes == 0)
        model.start(audioURL: url, transcript: "neu", language: "fr", stage: rig.stage(next, language: "fr"))
        try await waitUntilEntered(next.gate)
        #expect(next.gate.entries == 1 && rig.budget.inUseCount == 1)
        try await release(next.gate, try #require(model.task))
        #expect(model.result?.units.map(\.text) == ["neu"])
        #expect(rig.budget.usedBytes == 0 && rig.budget.inUseCount == 0)
    }

    // MARK: - One step per model, across sheets (owner ruling 15, 2026-11-27)

    @Test func doneDuringARunKeepsAReopenedSheetDisabledUntilTheStepReturns() async throws {
        let url = try wavURL()
        let text = FileManager.default.temporaryDirectory.appendingPathComponent("fa-model-\(UUID().uuidString).txt")
        try "neu".write(to: text, atomically: true, encoding: .utf8)
        defer { for file in [url, text] { try? FileManager.default.removeItem(at: file) } }
        let tracker = AlignStepTracker()
        let step = GatedStage(csv: Self.oldCSV)
        let next = GatedStage(csv: Self.newCSV)
        let first = newModel(tracker: tracker)
        first.start(audioURL: url, transcript: "alt", language: "de", stage: step)
        try await waitUntilEntered(step.gate)
        let task = try #require(first.task)
        first.stop()   // Done

        let reopened = newModel(tracker: tracker)
        #expect(!reopened.canChangeInputs, "Align is disabled")
        #expect(!reopened.isRunning && !reopened.isStopping)
        #expect(!reopened.chooseAudio(url) && !reopened.dropAudio([url]) && !reopened.openTranscript(text))
        #expect(reopened.audioURL == nil && reopened.transcript.isEmpty)
        reopened.start(audioURL: url, transcript: "neu", language: "de", stage: next)
        #expect(reopened.task == nil && !reopened.isRunning, "start is refused")

        let otherModel = newModel("another-aligner", tracker: tracker)
        #expect(otherModel.canChangeInputs, "a different model is not blocked")
        otherModel.start(audioURL: url, transcript: "neu", language: "de", stage: TextStubStage(csv: Self.newCSV))
        await otherModel.task?.value
        #expect(otherModel.result?.units.map(\.text) == ["neu"])

        try await pause()
        // Watch what an open sheet watches (registered after the other model's run, which changes the same set).
        let changed = AlignRunCallLog()
        withObservationTracking {
            #expect(!reopened.canChangeInputs && next.gate.entries == 0, "still waiting for the first step")
        } onChange: { changed.enter() }
        #expect(changed.count == 0)
        try await release(step.gate, task)
        #expect(changed.count == 1, "an open sheet is told")
        #expect(reopened.canChangeInputs)
        #expect(reopened.chooseAudio(url))
        reopened.start(audioURL: url, transcript: "neu", language: "de", stage: next)
        try await waitUntilEntered(next.gate)
        try await release(next.gate, try #require(reopened.task))
        #expect(reopened.result?.units.map(\.text) == ["neu"])
    }

    @Test(arguments: [false, true], [false, true])
    func aReturnedStepFreesTheModelWhateverItsOutcome(stopped: Bool, stepFails: Bool) async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let tracker = AlignStepTracker()
        let step = GatedStage(csv: Self.oldCSV, fails: stepFails)
        let model = newModel(tracker: tracker)
        model.start(audioURL: url, transcript: "alt", language: "de", stage: step)
        try await waitUntilEntered(step.gate)
        if stopped { model.stop() }
        #expect(tracker.isBusy("align-test"))
        try await release(step.gate, try #require(model.task))
        #expect(!tracker.isBusy("align-test"))
        #expect((model.errorText != nil) == (stepFails && !stopped))
        #expect(newModel(tracker: tracker).canChangeInputs)
    }

    @Test func aRunThatNeverReachesTheStageLeavesTheModelFree() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let tracker = AlignStepTracker()
        let stage = GatedStage(csv: Self.oldCSV)
        let model = newModel(tracker: tracker)
        model.start(audioURL: url, transcript: "  ", language: "de", stage: stage)
        #expect(!tracker.isBusy("align-test"), "refused before any work")
        model.start(audioURL: url.appendingPathExtension("gone"), transcript: "alt", language: "de", stage: stage)
        await model.task?.value
        #expect(model.errorText != nil && stage.gate.entries == 0)
        #expect(!tracker.isBusy("align-test") && model.canChangeInputs, "an unreadable file frees it too")
    }

    /// Two sheets built as the app builds them (no tracker passed) see each other's steps.
    @Test func sheetsShareOneTrackerByDefault() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let modelID = "align-shared-\(UUID().uuidString)"
        let step = GatedStage(csv: Self.oldCSV)
        let first = AlignRunModel(modelID: modelID)
        first.start(audioURL: url, transcript: "alt", language: "de", stage: step)
        try await waitUntilEntered(step.gate)
        let second = AlignRunModel(modelID: modelID)
        #expect(!second.canChangeInputs)
        try await release(step.gate, try #require(first.task))
        #expect(second.canChangeInputs)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("MLXUI/Modules/ForcedAligner/ForcedAlignerUI.swift"),
                                encoding: .utf8)
        #expect(source.contains("AlignRunModel(modelID: modelID)"), "the view builds its model that way")
    }

    // MARK: - The boxes stay editable (owner ruling 16, 2026-11-27)

    @Test func editingTheInputsLeavesTheRunsResultAndExportsAlone() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let csv = "text,start,end\nGuten,0.100,0.300\nTag,0.300,0.500\nBis,0.500,0.700\nbald,0.700,0.900\n"
        let step = GatedStage(csv: csv)
        let model = newModel()
        model.transcript = "Guten Tag. Bis bald."
        model.languageCode = "de"
        model.start(audioURL: url, transcript: model.transcript, language: model.languageCode, stage: step)
        try await waitUntilEntered(step.gate)
        model.transcript = "Etwas ganz anderes"
        model.languageCode = "fr"
        try await release(step.gate, try #require(model.task))
        let result = try #require(model.result)
        #expect(result.transcript == "Guten Tag. Bis bald.")
        #expect(result.units == (try AlignmentExport.parseCSV(csv)))
        let srt = "1\n00:00:00,100 --> 00:00:00,500\nGuten Tag\n\n2\n00:00:00,500 --> 00:00:00,900\nBis bald\n"
        let vtt = "WEBVTT\n\n00:00:00.100 --> 00:00:00.500\nGuten Tag\n\n00:00:00.500 --> 00:00:00.900\nBis bald\n"
        for edit in ["Etwas ganz anderes", "Guten Tag Bis bald"] {
            model.transcript = edit
            model.languageCode = "en"
            let shown = try #require(model.result)
            #expect(shown == result)
            #expect(ForcedAlignerRunView.ExportFormat.srt.render(shown) == srt)
            #expect(ForcedAlignerRunView.ExportFormat.vtt.render(shown) == vtt)
            #expect(ForcedAlignerRunView.ExportFormat.csv.render(shown) == csv)
        }
    }

    /// The view's half of ruling 14, source-level like the test below.
    @Test func theViewChangesInputsOnlyThroughTheModel() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("MLXUI/Modules/ForcedAligner/ForcedAlignerUI.swift"),
                                encoding: .utf8)
        #expect(source.contains("model.chooseAudio(url)"))
        #expect(source.contains("model.dropAudio(urls)"))
        #expect(source.contains("model.openTranscript(url)"))
        #expect(source.contains("isDropTargeted = $0 && model.canChangeInputs"))
        #expect(source.contains("if isDropTargeted && model.canChangeInputs"))
        #expect(source.components(separatedBy: ".disabled(!model.canChangeInputs").count == 4,
                "Choose Audio, Open and Align")
        #expect(!source.contains("model.reset()"))
        #expect(source.contains("private func reset()"))
    }

    /// The view's half of rulings 11-13. Source-level, as `noRunViewBuildsItsOwnStage`: the view's model is private
    /// SwiftUI state.
    @Test func theViewSendsDoneAndDisappearThroughStop() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("MLXUI/Modules/ForcedAligner/ForcedAlignerUI.swift"),
                                encoding: .utf8)
        #expect(source.contains(#"Button("Done") { model.stop(); dismiss() }"#))
        #expect(source.contains(".onDisappear { model.stop() }"))
        // Rulings 13 and 44: one cancel key in the sheet, which asks the model as it is pressed (Done and Stop hold none).
        #expect(source.components(separatedBy: ".keyboardShortcut(.cancelAction)").count == 2)
        #expect(source.contains("private func escape() { if model.isRunning { model.stop() } else { dismiss() } }"))
        #expect(source.contains("Button(model.runButtonTitle)"))
        #expect(source.contains("Text(model.progressText)"))
    }

    /// The library's neighbour fill can give a repeated word equal times; the table's rows still need their own ids.
    @Test func repeatedUnitsGetDistinctRowIDs() async throws {
        let url = try wavURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let stage = TextStubStage(csv: "text,start,end\nthe,0.080,0.080\nthe,0.080,0.080\n")
        let model = newModel()
        model.start(audioURL: url, transcript: "the the", language: "en", stage: stage)
        await model.task?.value
        let result = try #require(model.result)
        #expect(result.units.count == 2)
        #expect(Set(result.rows.map(\.id)).count == 2, "the Table's ids are unique")
        #expect(result.rows.map(\.unit) == result.units)
    }
}

/// Holds a stub stage until the test opens it; deliberately deaf to cancellation.
nonisolated final class AlignRunGate: @unchecked Sendable {
    private let lock = NSLock()
    private var opened = false
    private var entered = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []

    var isEntered: Bool { entries > 0 }
    /// How many runs have reached the stage.
    var entries: Int { lock.lock(); defer { lock.unlock() }; return entered }

    func wait() async {
        await withCheckedContinuation { continuation in
            lock.lock()
            entered += 1
            let resumeNow = opened
            if !resumeNow { waiters.append(continuation) }
            lock.unlock()
            if resumeNow { continuation.resume() }
        }
    }

    func open() {
        lock.lock()
        opened = true
        let pending = waiters
        waiters = []
        lock.unlock()
        for continuation in pending { continuation.resume() }
    }
}

/// The ways a run's inputs can change, or a run can start.
nonisolated enum AlignInputPath: CaseIterable, Sendable {
    case chooseAudio, dropAudio, openTranscript, start
}

nonisolated final class AlignRunCallLog: @unchecked Sendable {
    private let lock = NSLock()
    private var entered = 0
    func enter() { lock.lock(); entered += 1; lock.unlock() }
    var count: Int { lock.lock(); defer { lock.unlock() }; return entered }
}
