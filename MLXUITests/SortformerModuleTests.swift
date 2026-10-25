import Testing
import Foundation
@testable import MLXUI

/// MP4-3a: the Sortformer module's resolution, claim, stage and engine parameters, with no weights needed.
/// The real load, the main-actor probe and the measurement are in `SortformerRealLoadTests.swift` (gated on the install).
@MainActor
struct SortformerModuleTests {

    private func entry() -> ModelEntry {
        makeEntry(id: "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16", family: "Sortformer",
                  modelType: .diarization, source: .mlx)
    }

    @Test func resolvesToTheSortformerModule() throws {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let resolved = try #require(registry.bestModule(for: entry()))
        #expect(resolved.descriptor.id == "sortformer")
        #expect(resolved.sdk.claim(entry()) == .exact)
        #expect(resolved.ui.claim(entry()) == .exact)
    }

    @Test func claimsOnlySortformerFamilyMLXDiarization() {
        let sdk = SortformerSDK()
        #expect(sdk.claim(makeEntry(family: "Sortformer", modelType: .diarization, source: .coreml)) == .no)
        #expect(sdk.claim(makeEntry(id: "o", family: "Other", modelType: .diarization, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "a", family: "Sortformer", modelType: .asr, source: .mlx)) == .no)
        #expect(sdk.claim(makeEntry(id: "l", family: "Sortformer", modelType: .llm, source: .mlx)) == .no)
        #expect(SortformerUI().claim(makeEntry(id: "a", family: "Sortformer", modelType: .asr, source: .mlx)) == .no)
        for other: any ModelSDK in [WhisperSDK(), MLXWhisperSDK(), VoxtralSDK(), ParakeetSDK(), Qwen3ASRSDK()] {
            #expect(other.claim(entry()) == .no, "\(other.id) must decline Sortformer")
        }
    }

    /// Every bundled entry resolves exactly as before, except the Sortformer entry, which resolves to this module; and only that
    /// entry resolves to it (T3 for the ASR entries: each still has its own single module).
    @Test func onlyTheSortformerEntryResolvesToTheSortformerModuleAndEveryASREntryHasOneModule() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: data)
        let all = catalog.domains.flatMap { $0.allModels }
        let full = ModelRegistry()
        for module in installedModules { module.register(into: full) }
        let withoutSortformer = ModelRegistry()
        for module in installedModules where module.descriptor.id != "sortformer" { module.register(into: withoutSortformer) }
        var sortformerEntries = 0
        for e in all {
            let after = full.bestModule(for: e)?.descriptor.id
            if e.modelType == .diarization {
                #expect(after == "sortformer", "\(e.id)")
                sortformerEntries += 1
            } else {
                #expect(after == withoutSortformer.bestModule(for: e)?.descriptor.id, "\(e.id): resolution changed")
                #expect(after != "sortformer", "\(e.id) must not resolve to sortformer")
            }
        }
        #expect(sortformerEntries == 1)

        let asr = try #require(catalog.domains.first { $0.id == "asr" }).allModels
        for e in asr {
            var claimers: Set<String> = []
            for module in installedModules {
                let registry = ModelRegistry()
                module.register(into: registry)
                if registry.bestModule(for: e) != nil { claimers.insert(module.descriptor.id) }
            }
            #expect(!claimers.contains("sortformer"), "\(e.id)")
            if !e.family.lowercased().contains("whisper") { #expect(claimers.count == 1, "\(e.id): \(claimers)") }
        }
    }

    @Test func makeStageProducesAnAudioToTextStage() throws {
        let stage = try SortformerSDK().makeStage(for: entry(), config: .default)
        #expect(stage is SortformerStage)
        #expect(stage.accepts == .audio && stage.produces == .text)
        #expect(stage.id == "sortformer.mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16")
    }

    @Test func descriptorIsDiarizationOnlyAndDeclaresNoTaskFamily() {
        #expect(SortformerModule.descriptor.modelTypes == [.diarization])
        #expect(SortformerModule.descriptor.id == "sortformer")
        #expect(TaskModels.runnerKind(forTask: "Transcribe") == .asr)
        for task in TaskCatalog.allTasks().map(\.name) { #expect(TaskModels.runnerKind(forTask: task) != .diarization) }
    }

    // MARK: stage

    @Test func stageResamplesNormalizesAndReturnsCSV() async throws {
        let seen = Locked<Int>(0)
        let stage = SortformerStage(id: "t", name: "t") { samples in
            seen.set(samples.count)
            // Out of order, one past the end of the clip, one empty: the stage cleans them.
            return [SpeakerSegment(speaker: 1, start: 1.0, end: 9.0),
                    SpeakerSegment(speaker: 0, start: 0.0, end: 0.5),
                    SpeakerSegment(speaker: 0, start: 3.0, end: 3.0)]
        }
        // 2 s at 8 kHz -> about 32,000 samples at 16 kHz (AVAudioConverter may drop a few at the edge).
        let out = try await stage.run(.audio(AudioBuffer(samples: [Float](repeating: 0.1, count: 16_000), sampleRate: 8_000)), progress: { _ in })
        guard case let .text(csv) = out else { Issue.record("expected text"); return }
        let count = seen.get()
        #expect(abs(count - 32_000) <= 64, "resampled to 16 kHz: \(count)")
        // The second segment is clamped to the resampled clip's own length.
        let end = SpeakerSegments.seconds(Double(count) / 16_000)
        #expect(csv == "speaker,start,end\n1,0.000,0.500\n2,1.000,\(end)\n")
    }

    @Test func stageRejectsNonAudio() async {
        let stage = SortformerStage(id: "t", name: "t") { _ in [] }
        await #expect(throws: StageError.self) { _ = try await stage.run(.text("hi"), progress: { _ in }) }
    }

    @Test func stagePropagatesTheEngineError() async {
        struct Boom: Error {}
        let stage = SortformerStage(id: "t", name: "t") { _ in throw Boom() }
        await #expect(throws: Boom.self) {
            _ = try await stage.run(.audio(AudioBuffer(samples: [0.1], sampleRate: 16_000)), progress: { _ in })
        }
    }

    // MARK: engine

    @Test func thresholdIsClampedAndNaNFallsBackToTheDefault() {
        #expect(SortformerEngine.defaultThreshold == 0.5)
        #expect(SortformerEngine.clampedThreshold(0.5) == 0.5)
        #expect(SortformerEngine.clampedThreshold(0) == 0.05)
        #expect(SortformerEngine.clampedThreshold(7) == 0.95)
        #expect(SortformerEngine.clampedThreshold(.nan) == 0.5)
        #expect(SortformerEngine.clampedThreshold(.infinity) == 0.5)
    }

    @Test func engineRefusesEmptyAudioBeforeTouchingTheModel() async {
        await #expect(throws: StageError.self) {
            _ = try await SortformerEngine.diarize([], modelDirectory: URL(fileURLWithPath: "/nonexistent"))
        }
    }

    @Test func engineFailsCleanlyOnAMissingModelDirectory() async {
        await #expect(throws: StageError.self) {
            _ = try await SortformerEngine.diarize([0.1, 0.2], modelDirectory: URL(fileURLWithPath: "/nonexistent-sortformer"))
        }
    }

    /// Rule 17's one-line switch stays off until the owner-run probe says otherwise.
    @Test func theAudioThreadSwitchIsOffAndTheSampleRateIs16k() {
        #expect(SortformerEngine.usesAudioThread == false)
        #expect(SortformerEngine.sampleRate == 16_000)
    }

    @Test func installedDirectoryIsTheSlugFolder() {
        let dir = SortformerEngine.installedModelDirectory(id: "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16")
        #expect(dir.lastPathComponent == "mlx-community--diar_streaming_sortformer_4spk-v2.1-fp16")
        #expect(dir.deletingLastPathComponent().lastPathComponent == "models")
    }

    // MARK: load profile

    @Test func loadProfileRowIsTheUnmeasuredDefault() {
        let row = ModelLoadProfile.entry(for: .diarization)
        #expect(row.measured == false)
        #expect(row.workingSetBytes == 0)
        #expect(row.factor == ModelWeightCharge.loadFactor)
    }
}

/// A tiny lock-protected box for a `@Sendable` closure to report into.
private final class Locked<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: T
    init(_ value: T) { self.value = value }
    func set(_ new: T) { lock.lock(); value = new; lock.unlock() }
    func get() -> T { lock.lock(); defer { lock.unlock() }; return value }
}
