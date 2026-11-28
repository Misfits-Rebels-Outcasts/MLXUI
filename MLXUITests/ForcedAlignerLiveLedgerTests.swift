import Testing
import Foundation
import AVFoundation
import MLX
@testable import MLXUI

/// FA-R23: the app's own ledger, `MemoryBudget.shared`, read during real alignments on both app paths and held against
/// the process footprint measured over the same run.
///
/// Owner ruling 31 (Boo, 2026-11-28): "Gate 2 must now include a real-weights test that reads the live MemoryBudget
/// ledger during a real Run screen run and during a real Align row, and asserts the booking is at or above the measured
/// phys_footprint peak for the 5 s, 60 s and 300 s clips. 'Checked against ModelLoadProfile' doesn't count. Correct the
/// earlier claim in the journal and on the PR, and list the PR's own tests whose expected values FA-R20 changed."
///
/// Owner ruling 27 (Boo, 2026-11-28): "Accepted: ruling 17's 'one booking' means one booking while each run is active,
/// with nothing held between runs, because the aligner unloads the model after every run."
///
/// **What is real.** The weights, the engine, `MemoryBudget.shared` and `EngineCache.shared`. The Run screen path is an
/// `AlignRunModel` started as `ForcedAlignerRunView.run()` starts it, on the per-run rebuild of the `LeasingStage` the
/// Run sheet makes (`RunSheetLease`). The row path is a `RealExecutor` wired as `AppFlowExecutorFactory.make` wires it,
/// running an `Align (1,2)` row's two inputs. The one difference from the app is the model folder: the stage is given a
/// writable copy of the pointed-at install (`ForcedAlignerRealLoad.workingCopy`).
///
/// **How the peak is caught.** FA-7's method (`ForcedAlignerFootprintTests`): `phys_footprint`
/// (`task_info(TASK_VM_INFO)`, the whole test host process, not a difference from the baseline) polled every 20 ms by a
/// detached task for the whole run, plus one sample right after it returns. The ledger's `usedBytes` is read at the same
/// instants and its highest value kept. **The limit:** a spike that rises and falls between two polls (shorter than
/// about 20 ms) could be missed. `bookedGB` is the highest booking seen during the run.
///
/// Gated like `ForcedAlignerFootprintTests` (`MLXUI_TEST_ALIGN=1`, `MLXUI_TEST_ALIGN_DIR`,
/// `MLXUI_TEST_ALIGN_FOOTPRINT=1`). Run this suite on its own: it expects an empty ledger. One line per case:
///
/// `FALEDGER path=<runscreen|alignrow> clip=<5s|60s|300s> seconds=… units=… bookedGB=… peakGB=… baselineGB=… covered=<yes|no> wall=…s`
@Suite(.serialized)
@MainActor
struct ForcedAlignerLiveLedgerTests {

    /// A clip: `repeats` copies of FA-7's sentence (2, 22 and 110 are about 5, 60 and 297 seconds).
    nonisolated struct Clip: Sendable, CustomTestStringConvertible {
        let label: String
        let repeats: Int
        var testDescription: String { label }
    }

    nonisolated static let clips = [Clip(label: "5s", repeats: 2), Clip(label: "60s", repeats: 22),
                                    Clip(label: "300s", repeats: 110)]

    private nonisolated static let cardID = "mlx-community--Qwen3-ForcedAligner-0.6B-8bit"

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    private func card() throws -> ModelEntry {
        try #require(try loadCatalog().first { $0.id == Self.cardID })
    }

    private func seconds(of url: URL) throws -> Double {
        let file = try AVAudioFile(forReading: url)
        return Double(file.length) / file.processingFormat.sampleRate
    }

    /// Runs `run` with the ledger and the footprint polled beside it, prints the case's line and makes the assertions.
    private func measure(path: String, clip: Clip, seconds: Double,
                         run: () async throws -> [AlignedUnit]) async throws {
        let budget = MemoryBudget.shared
        let usedBefore = budget.usedBytes
        let heldBefore = budget.heldCount
        #expect(usedBefore == 0, "something else is booked, so the reading would not be this run's alone")

        Memory.clearCache()
        Memory.peakMemory = 0
        let baseline = FootprintTracker.phys()
        let footprint = FootprintTracker()
        let booked = FootprintTracker()   // the same running maximum, over the ledger's bytes
        footprint.note(baseline)
        let sampler = Task.detached {
            while !Task.isCancelled {
                booked.note(MemoryBudget.shared.usedBytes)
                footprint.note(FootprintTracker.phys())
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        defer { sampler.cancel() }
        let started = Date()
        let units = try await run()
        let wall = Date().timeIntervalSince(started)
        booked.note(budget.usedBytes)
        footprint.note(FootprintTracker.phys())
        sampler.cancel()
        await sampler.value   // no later sample can move the figures between the line and the assertions

        let covered = booked.runPeak >= footprint.runPeak
        let line = String(format: "FALEDGER path=%@ clip=%@ seconds=%.1f units=%d bookedGB=%.3f peakGB=%.3f baselineGB=%.3f covered=%@ wall=%.1fs",
                          path, clip.label, seconds, units.count, Double(booked.runPeak) / 1e9,
                          Double(footprint.runPeak) / 1e9, Double(baseline) / 1e9, covered ? "yes" : "no", wall)
        print(line)
        // As FA-7 does: keep the evidence in a file too, for a harness that cannot read the test host's stdout.
        let log = NSTemporaryDirectory() + "forcedaligner-live-ledger.txt"
        let previous = (try? String(contentsOfFile: log, encoding: .utf8)) ?? ""
        try? (previous + line + "\n").write(toFile: log, atomically: true, encoding: .utf8)

        #expect(!units.isEmpty)
        #expect(units.map(\.text).joined(separator: " ").contains("quick"), "the transcript is timed")
        #expect(baseline > 0 && footprint.runPeak > baseline, "the footprint was read")
        #expect(booked.runPeak >= footprint.runPeak,
                "\(path) \(clip.label): booked \(booked.runPeak) bytes, footprint peak \(footprint.runPeak) bytes")
        #expect(budget.usedBytes == usedBefore && budget.heldCount == heldBefore, "nothing is held between runs")
    }

    @Test(.enabled(if: ForcedAlignerFootprintTests.enabled), arguments: ForcedAlignerLiveLedgerTests.clips)
    func aRunScreenRunBooksAtOrAboveItsPeak(clip: Clip) async throws {
        let card = try card()
        let copy = try ForcedAlignerRealLoad.workingCopy()
        defer { copy.remove() }
        let audio = try ForcedAlignerFootprintTests.sayClip(repeats: clip.repeats)
        defer { try? FileManager.default.removeItem(at: audio.url) }
        let seconds = try seconds(of: audio.url)

        let sdk: any ModelSDK = ForcedAlignerSDK()
        let modelID = card.id
        let footprintBytes = ForcedAlignerSDK.footprintBytes(for: card)
        let directory = copy.url
        // The sheet's stage, as `RunSheetLease` makes it, on `EngineCache.shared`.
        let handed = LeasingStage(model: card, config: .default,
                                  probe: ForcedAlignerStage(modelID: modelID, footprintBytes: footprintBytes,
                                                            modelDirectory: directory),
                                  booksWeights: !(sdk is StageBooksOwnWeights), confirmer: ServedLoadConfirmer())
        let model = AlignRunModel(modelID: modelID)
        #expect(model.chooseAudio(audio.url))
        model.transcript = audio.text

        try await measure(path: "runscreen", clip: clip, seconds: seconds) {
            // `ForcedAlignerRunView.run()`.
            let language = model.languageCode
            let stage = LeasingStage.perRun(handed, config: StageConfig(language: language)) { config in
                ForcedAlignerStage(modelID: modelID, language: config.language ?? language,
                                   footprintBytes: footprintBytes, modelDirectory: directory)
            }
            model.start(audioURL: audio.url, transcript: model.transcript, language: language, stage: stage)
            await model.task?.value
            #expect(model.errorText == nil)
            return try #require(model.result, "the run returned no result: \(model.errorText ?? "no error")").units
        }
    }

    @Test(.enabled(if: ForcedAlignerFootprintTests.enabled), arguments: ForcedAlignerLiveLedgerTests.clips)
    func anAlignRowBooksAtOrAboveItsPeak(clip: Clip) async throws {
        let catalog = try loadCatalog()
        let card = try card()
        let copy = try ForcedAlignerRealLoad.workingCopy()
        defer { copy.remove() }
        let audio = try ForcedAlignerFootprintTests.sayClip(repeats: clip.repeats)
        defer { try? FileManager.default.removeItem(at: audio.url) }
        let seconds = try seconds(of: audio.url)
        let workspace = FlowWorkspace(root: FileManager.default.temporaryDirectory
            .appendingPathComponent("fa-ledger-ws-\(UUID().uuidString)"))
        try FileManager.default.createDirectory(at: workspace.root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: workspace.root) }

        // `AppFlowExecutorFactory.make`: the SDK from a registry, `booksWeights` derived from it, `EngineCache.shared`'s
        // `leasedStage` and the live served-load gate.
        let registry = ModelRegistry()
        ForcedAlignerModule.register(into: registry)
        let sdk = try #require(registry.bestModule(for: card)?.sdk)
        let books = !(sdk is StageBooksOwnWeights)
        let cache = EngineCache.shared
        let config = StageConfig(language: "en")
        // A stage cached by an earlier case would point at that case's removed copy.
        cache.discard(card, config: config)
        defer { cache.discard(card, config: config) }
        let directory = copy.url
        let makeStage: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            try await cache.leasedStage(for: model, config: config, booksWeights: books,
                                        footprintOverride: Flux2KleinBooking.leaseOverride(for: model, config: config)) { model, config in
                ForcedAlignerStage(modelID: model.id,
                                   language: config.language ?? ForcedAlignerEngine.defaultLanguageCode,
                                   footprintBytes: ForcedAlignerSDK.footprintBytes(for: model),
                                   modelDirectory: directory)
            }
        }
        var executor = RealExecutor(workspace: workspace, flowID: "fa-live-ledger",
                                    blobDirectory: workspace.root.appendingPathComponent("blobs"),
                                    makeModelStage: makeStage, installedModelIDs: [card.id], catalog: catalog)
        let gate = FlowServedLoadGate.live(flowID: "fa-live-ledger", cache: cache) { _ in books }
        executor.servedLoadGate = { path, model in try await gate(path: path, model: model) }
        let row = Row(task: "Align", model: "Qwen3 Aligner 0.6B", settings: "lang=en")
        let inputs = [Asset(items: [Item(kind: .audio, value: nil, path: audio.url, sourceText: nil)]),
                      Asset(items: [Item(kind: .text, value: audio.text, path: nil, sourceText: nil)])]
        let runner = executor

        try await measure(path: "alignrow", clip: clip, seconds: seconds) {
            let output = try await runner.execute(path: "3", row: row, inputs: inputs, transcript: nil,
                                                  context: nil, usedFlowContent: nil)
            #expect(cache.isCached(card, config: config), "the row ran on the shared cache's stage")
            let csv = try #require(output.items.first?.value, "the row returned no table")
            return try AlignmentExport.parseCSV(csv)
        }
    }
}
