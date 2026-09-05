import Testing
import Foundation
import CoreGraphics
@testable import MLXUI

/// CFM-R16-1 — the diffusion settings reach the engine.
///
/// `StageConfig` gains `seed`/`width`/`height`/`steps`; `RealExecutor.stageConfig` reads them
/// off an `engines.diffusion.*` row's settings; `FluxSDK.makeStage` hands `seed` to the
/// `FluxStage` and **refuses** `width`/`height`/`steps` the FLUX engine cannot honour — a row
/// naming one must fail with a sentence naming the setting, never silently drop it (the whole
/// lesson of the phase, and the reason `66-SeedSweep` would otherwise produce four unseeded
/// images that a smoke row signs off as a pass). See `RSI/DelegateMergeBacklog.md` CFM-R16-1.
struct CatFlowR16DiffusionSettingsTests {

    // MARK: - Fixtures

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    private func fluxEntry() throws -> ModelEntry {
        let catalog = try loadCatalog()
        return try #require(catalog.first { $0.runnerKind == .image && $0.id.lowercased().contains("flux") })
    }

    private func makeExecutor() -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "r16",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in StubStage() },
            installedModelIDs: [],
            catalog: [])
    }

    // MARK: - stageConfig reads diffusion settings

    @Test func stageConfigReadsSeedWidthHeightSteps() throws {
        let desc = try #require(TaskCatalog.get("Generate Image"))
        let row = Row(task: "Generate Image", model: "Flux-1.lite-8B",
                      settings: "seed=7; width=1024; height=1024; steps=25")
        let config = try makeExecutor().stageConfig(for: desc, row: row)
        #expect(config.seed == 7)
        #expect(config.width == 1024)
        #expect(config.height == 1024)
        #expect(config.steps == 25)
    }

    @Test func nonDiffusionRowsDoNotPickUpDiffusionSettings() throws {
        // The diffusion branch must only fire for `engines.diffusion.*` ref names — an ASR row
        // carrying `seed=`/`width=` keys must not smuggle them into a config the ASR stage sees.
        let desc = try #require(TaskCatalog.get("Transcribe"))
        let row = Row(task: "Transcribe", model: "Whisper Small", settings: "lang=en; seed=7; width=1024")
        let config = try makeExecutor().stageConfig(for: desc, row: row)
        #expect(config.seed == nil)
        #expect(config.width == nil)
        #expect(config.height == nil)
        #expect(config.steps == nil)
    }

    // MARK: - 66-SeedSweep: four `<each>` items, four different concrete seeds

    @Test func seedSweepItemsGetFourDifferentConcreteSeeds() throws {
        // 66-SeedSweep: `Range 3..6` yields items 3,4,5,6; the `<each>` fan-out substitutes
        // `{item}` into the Generate Image row's settings (`seed={item}`) before dispatch, so
        // at the stageConfig boundary the four rows must carry four different concrete seeds.
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/66-SeedSweep.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        let block = try #require(doc.rows.first { $0.blockKind != nil })
        let generate = try #require(block.children.first { $0.task == "Generate Image" })
        let settings = try #require(generate.settings)
        #expect(settings.contains("{item}"))

        let bounds = try RangeTool.parseBounds("3..6")
        let desc = try #require(TaskCatalog.get("Generate Image"))
        let executor = makeExecutor()
        let seeds = try (bounds.start...bounds.stop).map { item -> UInt64? in
            let substituted = settings.replacingOccurrences(of: "{item}", with: String(item))
            return try executor.stageConfig(
                for: desc,
                row: Row(task: "Generate Image", model: generate.model, settings: substituted)
            ).seed
        }
        #expect(seeds.count == 4)
        #expect(seeds.allSatisfy { $0 != nil })
        #expect(Set(seeds.compactMap { $0 }).count == 4)
    }

    // MARK: - seed drives deterministic output on the real seam

    @Test func generateImageSeedDrivesDeterministicOutput() async throws {
        // The real seam: row settings → stageConfig (the real function) → FluxSDK.makeStage →
        // FluxStage (the real stage, whose `run` hands the seed to the generator). The
        // generator consumes the seed the *stage* passes it — standing in for FluxEngine's
        // `MLXRandom.seed` determinism — so the test is not a mock echoing the config back.
        let desc = try #require(TaskCatalog.get("Generate Image"))
        let executor = makeExecutor()
        let config7 = try executor.stageConfig(for: desc, row: Row(task: "Generate Image", model: "Flux-1.lite-8B", settings: "seed=7"))
        let config8 = try executor.stageConfig(for: desc, row: Row(task: "Generate Image", model: "Flux-1.lite-8B", settings: "seed=8"))
        #expect(config7.seed == 7)
        #expect(config8.seed == 8)

        // The SDK passes config.seed into the stage.
        let entry = try fluxEntry()
        let stage = try #require(try FluxSDK().makeStage(for: entry, config: config7) as? FluxStage)
        #expect(stage.seed == 7)

        // The stage honors its seed: two runs at 7 are byte-identical, 8 differs.
        let run7a = try await runFluxStage(seed: 7)
        let run7b = try await runFluxStage(seed: 7)
        let run8 = try await runFluxStage(seed: 8)
        #expect(run7a == run7b)
        #expect(run7a != run8)
    }

    private func runFluxStage(seed: UInt64) async throws -> Data {
        let stage = FluxStage(id: "flux.test", name: "Flux (test)", seed: seed, generate: deterministicGenerator())
        let result = try await stage.run(.text("a lighthouse"), progress: { _ in })
        guard case .image(let media) = result else {
            throw ImageConversionFailed()
        }
        return PNGEncoder.pngData(from: media.cgImage) ?? Data()
    }

    /// A seed-driven deterministic generator standing in for FluxEngine's `MLXRandom.seed`
    /// contract: same seed → identical bytes, different seed → different bytes. It uses the
    /// `effectiveSeed` the **stage** hands it, so the test proves the seed is plumbed through.
    private func deterministicGenerator() -> @Sendable (String, UInt64?, @Sendable (Double) -> Void) async throws -> CGImage {
        { _, effectiveSeed, _ in
            var state: UInt64 = (effectiveSeed ?? 0) &+ 0x9E37_79B9_7F4A_7C15
            let w = 8, h = 8
            var rgba = [UInt8](repeating: 0, count: w * h * 4)
            for i in 0..<rgba.count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                rgba[i] = UInt8(truncatingIfNeeded: state >> 32)
            }
            let colorSpace = CGColorSpaceCreateDeviceRGB()
            let info = CGImageAlphaInfo.premultipliedLast.rawValue
            guard let ctx = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8,
                                      bytesPerRow: w * 4, space: colorSpace, bitmapInfo: info),
                  let image = ctx.makeImage() else {
                throw ImageConversionFailed()
            }
            return image
        }
    }

    // MARK: - width/height/steps refuse with a named sentence

    @Test func fluxRefusesSizeAndStepSettingsItCannotHonour() throws {
        let entry = try fluxEntry()
        let sdk = FluxSDK()

        func setting(of error: Error) -> String? {
            if case .unsupportedSetting(let setting)? = error as? StageError { return setting }
            return nil
        }

        for (config, expected) in [
            (StageConfig(seed: 7, width: 1024), "width"),
            (StageConfig(seed: 7, height: 1024), "height"),
            (StageConfig(seed: 7, steps: 25), "steps"),
        ] {
            do {
                _ = try sdk.makeStage(for: entry, config: config)
                Issue.record("expected a refusal for \(expected)")
            } catch {
                #expect(setting(of: error) == expected)
            }
        }

        // A seed-only row builds fine — the FLUX engine honours the seed.
        _ = try sdk.makeStage(for: entry, config: StageConfig(seed: 7))
    }

    @Test func unsupportedSettingSentenceNamesTheSetting() {
        let sentence = String(describing: StageError.unsupportedSetting(setting: "width"))
        #expect(sentence.contains("width"))
        #expect(sentence.contains("remove it"))
        // The run-view mapper says the same thing (the switch has no default).
        #expect(FlowErrorDisplay.sentence(for: StageError.unsupportedSetting(setting: "steps")).contains("steps"))
    }

    @Test func rowNamingAnUnsupportedSettingFailsThroughTheExecutor() async throws {
        // The real executor path end to end: a Generate Image row with width=1024 goes through
        // resolveModel → stageConfig → the registry's makeStage (FluxSDK), which refuses. The
        // surfaced sentence names the setting.
        let catalog = try loadCatalog()
        let flux = try #require(catalog.first { $0.hfModelId == "mlx-community/Flux-1.lite-8B-MLX-Q4" })
        let executor = RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "r16",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { model, config in
                try FluxSDK().makeStage(for: model, config: config)
            },
            installedModelIDs: [flux.id],
            catalog: catalog)
        let row = Row(task: "Generate Image", model: flux.displayName, settings: "seed=7; width=1024")
        let input = Asset(items: [Item(kind: .text, value: "a mug", path: nil, sourceText: nil)])
        do {
            _ = try await executor.execute(path: "2", row: row, inputs: [input])
            Issue.record("expected the row to refuse width=1024")
        } catch {
            let message = String(describing: error)
            #expect(message.contains("width"))
            #expect(message.contains("isn't supported"))
        }
    }
}

/// A no-op stage for `RealExecutor` config tests (no model needed).
private struct StubStage: PipelineStage {
    let id = "stub.r16"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .text)
        progress(1.0)
        return input
    }
}

private struct ImageConversionFailed: Error {}
