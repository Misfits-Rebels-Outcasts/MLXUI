import CoreGraphics
import Foundation
import MLX
import Testing
@testable import MLXUI

// MP7-4: the Klein stage, SDK, Run UI logic, registry claim and memory lease. No real weights: the stage runs the REAL engine over the tiny golden components
// (`Klein3hTinyStages`, MP7-3h), so the settings really reach the engine and a cancellation really stops it.

private func kleinEntry() -> ModelEntry {
    makeEntry(id: "mlx-community--FLUX.2-klein-4B-bf16", displayName: "FLUX.2 Klein 4B", modelType: .image, source: .mlx,
              hfRepo: "mlx-community/FLUX.2-klein-4B-bf16", hfModelId: "mlx-community/FLUX.2-klein-4B-bf16")
}

private func flux1Entry() -> ModelEntry {
    makeEntry(id: "mlx-community--Flux-1.lite-8B-MLX-Q4", displayName: "FLUX.1 lite", modelType: .image, source: .mlx,
              hfRepo: "mlx-community/Flux-1.lite-8B-MLX-Q4", hfModelId: "mlx-community/Flux-1.lite-8B-MLX-Q4")
}

/// A stage over the tiny engine; `events` collects the engine's lifecycle in order.
private final class TinyKlein: @unchecked Sendable {
    let stages = Klein3hTinyStages()
    private let lock = NSLock()
    private var _phases: [Flux2KleinPhase] = []
    var phases: [Flux2KleinPhase] { lock.lock(); defer { lock.unlock() }; return _phases }

    func stage(_ settings: Flux2KleinRunSettings, rowLabel: String = Flux2KleinStage.runWindowLabel, onPhase: (@Sendable (Flux2KleinPhase) -> Void)? = nil) -> Flux2KleinStage {
        let stages = self.stages
        return Flux2KleinStage(id: "tiny", name: "tiny", settings: settings, rowLabel: rowLabel, onPhase: { [self] p in lock.lock(); _phases.append(p); lock.unlock(); onPhase?(p) }) { observer in
            Flux2KleinEngine(stages: stages, budgetBytes: { 40_000_000_000 },
                             observer: { e in stages.log.record(e); if case let .released(c) = e { stages.checkReleased(c) }; observer(e) })
        }
    }
}

private func tinySettings(width: Int = 32, height: Int = 32, steps: Int = 2, seed: UInt64? = 7, quantize: Int = 8) -> Flux2KleinRunSettings {
    Flux2KleinRunSettings(width: width, height: height, steps: steps, seed: seed, quantize: quantize)
}

// MARK: - Stage honours seed / width / height / steps

@Suite(.serialized) struct Flux2KleinStageSettingsTests {
    @Test func theManifestDefaultsAre1024x1024FourStepsQuantize8() {
        let s = Flux2KleinRunSettings()
        #expect(s.width == 1024 && s.height == 1024 && s.steps == 4 && s.quantize == 8 && s.seed == nil)
    }

    @Test func widthHeightAndStepsReachTheEngine() async throws {
        let tiny = TinyKlein()
        let out = try await tiny.stage(tinySettings(width: 48, height: 32, steps: 3)).run(.text("a tiny prompt")) { _ in }
        guard case let .image(media) = out else { Issue.record("not an image"); return }
        #expect(media.cgImage.width == 48 && media.cgImage.height == 32)
        let steps = tiny.stages.log.events.filter { if case .denoiseStep = $0 { return true } else { return false } }
        #expect(steps.count == 3)
    }

    @Test func theSeedIsHonouredSameSeedSamePictureDifferentSeedDifferent() async throws {
        func pixels(_ seed: UInt64) async throws -> [UInt8] {
            let tiny = TinyKlein()
            let out = try await tiny.stage(tinySettings(seed: seed)).run(.text("p")) { _ in }
            guard case let .image(m) = out else { return [] }
            return KleinPNG.rgbaBytes(m.cgImage)
        }
        let a = try await pixels(7), b = try await pixels(7), c = try await pixels(8)
        #expect(!a.isEmpty && a == b)
        #expect(a != c)
    }

    @Test func aMissingSeedPicksARandomOneThatFitsAKey() {
        for _ in 0 ..< 50 { #expect(Flux2KleinRunSettings.randomSeed() < (1 << 32)) }
    }

    @Test func progressIsMonotoneAndEndsAtOne() async throws {
        let tiny = TinyKlein()
        let fractions = KleinBox<[Double]>([])
        _ = try await tiny.stage(tinySettings(steps: 3)).run(.text("p")) { f in fractions.mutate { $0.append(f) } }
        let f = fractions.value
        #expect(f.count >= 8)
        #expect(zip(f, f.dropFirst()).allSatisfy { $0 <= $1 }, "progress went backwards: \(f)")
        #expect(f.last == 1.0)
    }

    @Test func phasesComeInStageOrderWithStepNumbers() async throws {
        let tiny = TinyKlein()
        _ = try await tiny.stage(tinySettings(steps: 2)).run(.text("p")) { _ in }
        #expect(tiny.phases == [.loadingTextEncoder, .encoding, .loadingImageModel, .denoising(step: 0, of: 2), .denoising(step: 1, of: 2), .denoising(step: 2, of: 2),
                                .loadingDecoder, .decoding])
    }

    @Test func phaseFractionsAreIncreasingAndBounded() {
        let order: [Flux2KleinPhase] = [.loadingTextEncoder, .encoding, .loadingImageModel, .denoising(step: 0, of: 4), .denoising(step: 1, of: 4), .denoising(step: 4, of: 4),
                                        .loadingDecoder, .decoding]
        let f = order.map(\.fraction)
        #expect(zip(f, f.dropFirst()).allSatisfy { $0 < $1 }, "\(f)")
        #expect(f.allSatisfy { $0 > 0 && $0 < 1 })
        #expect(Flux2KleinPhase.denoising(step: 9, of: 4).fraction == Flux2KleinPhase.denoising(step: 4, of: 4).fraction, "clamped")
        #expect(Flux2KleinPhase.denoising(step: 2, of: 4).label == "Denoising: step 2 of 4")
    }

    @Test func nonTextInputIsAKindMismatch() async {
        let tiny = TinyKlein()
        let stage = tiny.stage(tinySettings())
        await #expect(throws: StageError.self) { _ = try await stage.run(.audio(AudioBuffer(samples: [], sampleRate: 16000))) { _ in } }
        #expect(tiny.stages.log.events.isEmpty)
    }
}

// MARK: - Refusals (engine validate through the stage; real row label)

@Suite struct Flux2KleinStageRefusalTests {
    private func refusal(_ settings: Flux2KleinRunSettings, row: String = "Row 3") async -> String? {
        let tiny = TinyKlein()
        do { _ = try await tiny.stage(settings, rowLabel: row).run(.text("p")) { _ in }; return nil } catch { return (error as? LocalizedError)?.errorDescription ?? "\(error)" }
    }

    @Test(arguments: [3, 5, 6, 7, 0, 16])
    func quantizeOutsideFourOrEightIsRefusedBeforeAnythingLoadsWithTheRealRowLabel(_ q: Int) async throws {
        let tiny = TinyKlein()
        let s = tiny.stage(tinySettings(quantize: q), rowLabel: "Row 3")
        do { _ = try await s.run(.text("p")) { _ in }; Issue.record("quantize \(q) ran") } catch {
            let text = (error as? LocalizedError)?.errorDescription ?? ""
            #expect(text == "Row 3: FLUX.2 Klein 4B runs at quantize 4 or 8 in this build, not \(q).")
            #expect(!text.contains("FLUX.2 Klein 4B: FLUX.2 Klein 4B"), "the model name must not be doubled")
        }
        #expect(tiny.stages.log.events.isEmpty, "nothing may load for a refused row")
    }

    @Test func aSizeThatIsNotAMultipleOfSixteenIsRefusedNotRounded() async throws {
        let text = try #require(await refusal(tinySettings(width: 1000, height: 1024)))
        #expect(text.contains("multiple of 16") || text.contains("16"), Comment(rawValue: text))
        #expect(text.contains("1000"), Comment(rawValue: text))
    }

    @Test func aSizeOverThisMacsCapNamesTheRowTheSizeAskedAndTheLargestPossible() async throws {
        let text = try #require(await refusal(tinySettings(width: 4096, height: 4096), row: "Row 9"))
        #expect(text.hasPrefix("Row 9 can't make a 4096 x 4096 image at quantize 8 on this Mac"), Comment(rawValue: text))
        #expect(text.contains("The largest it can make here is"), Comment(rawValue: text))
        let largest = Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: Flux2KleinMemoryPlan.defaultBudgetBytes)
        #expect(text.contains("\(largest) x \(largest)"), Comment(rawValue: text))
    }

    @Test func zeroStepsIsRefused() async throws {
        let text = try #require(await refusal(tinySettings(steps: 0)))
        #expect(text.contains("at least 1 step"), Comment(rawValue: text))
    }
}

// MARK: - SDK: claim, makeStage, row settings

@MainActor
struct Flux2KleinSDKTests {
    @Test func claimsOnlyTheKlein4BEntryExactly() {
        #expect(Flux2KleinSDK().claim(kleinEntry()) == .exact)
        #expect(Flux2KleinUI().claim(kleinEntry()) == .exact)
        #expect(Flux2KleinSDK().claim(flux1Entry()) == .no)
        for hf in ["mlx-community/flux2-klein-9b-8bit", "black-forest-labs/FLUX.2-klein-4B", "x/FLUX.2-dev"] {
            let e = makeEntry(id: hf.replacingOccurrences(of: "/", with: "--"), modelType: .image, source: .mlx, hfRepo: hf, hfModelId: hf)
            // the 4B id under another namespace is still the 4B architecture; 9B / dev are not this engine
            #expect((Flux2KleinSDK().claim(e) == .exact) == hf.lowercased().contains("klein-4b"), Comment(rawValue: hf))
        }
        let notMLX = makeEntry(id: kleinEntry().id, modelType: .image, source: .coreml, hfRepo: "x", hfModelId: kleinEntry().hfModelId)
        #expect(Flux2KleinSDK().claim(notMLX) == .no)
    }

    @Test func makeStageWithDefaultConfigGivesTheManifestDefaults() throws {
        let stage = try #require(try Flux2KleinSDK().makeStage(for: kleinEntry(), config: .default) as? Flux2KleinStage)
        #expect(stage.settings == Flux2KleinRunSettings())
        #expect(stage.accepts == .text && stage.produces == .image)
        #expect(stage.rowLabel == Flux2KleinStage.runWindowLabel)
    }

    @Test func makeStageCarriesSeedWidthHeightAndSteps() throws {
        let cfg = StageConfig(seed: 99, width: 768, height: 512, steps: 6)
        let stage = try #require(try Flux2KleinSDK().makeStage(for: kleinEntry(), config: cfg, quantize: 4, rowLabel: "Row 2") as? Flux2KleinStage)
        #expect(stage.settings == Flux2KleinRunSettings(width: 768, height: 512, steps: 6, seed: 99, quantize: 4))
        #expect(stage.rowLabel == "Row 2")
    }

    @Test func makeStageRefusesWhatTheEngineWouldRefuseNamingTheRow() {
        #expect(throws: Flux2KleinEngineError.unsupportedQuantize(row: "Row 5", quantize: 6)) {
            _ = try Flux2KleinSDK().makeStage(for: kleinEntry(), config: .default, quantize: 6, rowLabel: "Row 5")
        }
        #expect(throws: (any Error).self) { _ = try Flux2KleinSDK().makeStage(for: kleinEntry(), config: StageConfig(width: 1001, height: 1024), quantize: 8, rowLabel: "Row 5") }
        #expect(throws: (any Error).self) { _ = try Flux2KleinSDK().makeStage(for: kleinEntry(), config: StageConfig(width: 8192, height: 8192), quantize: 8, rowLabel: "Row 5") }
        #expect(throws: StageError.self) { _ = try Flux2KleinSDK().makeStage(for: kleinEntry(), config: StageConfig(scale: 2)) }
    }

    @Test func rowSettingsTheStageCannotHonourAreRefusedWithASentenceNamingTheSetting() throws {
        func refusal(_ pairs: [String: String]) -> String? { Flux2KleinSDK.refusal(forRowSetting: { pairs[$0] }, rowLabel: "Row 4") }
        #expect(refusal([:]) == nil)
        #expect(refusal(["guidance": "2.0"]) == nil && refusal(["guidance": "2"]) == nil, "the manifest default is accepted")
        #expect(refusal(["quantize": "8"]) == nil && refusal(["quantize": "4"]) == nil)
        #expect(refusal(["preview": "false"]) == nil && refusal(["negative": ""]) == nil)
        let guidance = try #require(refusal(["guidance": "4.5"]))
        #expect(guidance.hasPrefix("Row 4: ") && guidance.contains("guidance") && guidance.contains("4.5"))
        for key in ["negative", "lora", "controlnet"] {
            let text = try #require(refusal([key: "x"]))
            #expect(text.hasPrefix("Row 4: ") && text.contains(key), Comment(rawValue: text))
        }
        #expect(try #require(refusal(["preview": "true"])).contains("preview"))
        #expect(try #require(refusal(["quantize": "5"])) == "Row 4: FLUX.2 Klein 4B runs at quantize 4 or 8 in this build, not 5.")
        #expect(try #require(refusal(["quantize": "eight"])).contains("whole number"))
    }
}

// MARK: - Registry

@MainActor
struct Flux2KleinRegistryClaimTests {
    private func shipping() -> ModelRegistry {
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        return registry
    }

    @Test func theShippingRegistryPicksKleinForKleinAndFluxForFlux1() {
        let registry = shipping()
        #expect(registry.bestModule(for: kleinEntry())?.descriptor.id == "flux2-klein")
        #expect(registry.bestModule(for: flux1Entry())?.descriptor.id == "flux")
    }

    @Test func registrationOrderDoesNotMatter() {
        for kleinFirst in [true, false] {
            let registry = ModelRegistry()
            let adds: [() -> Void] = [
                { Flux2KleinModule.register(into: registry) },
                { FluxModule.register(into: registry) },
            ]
            for add in (kleinFirst ? adds : adds.reversed()) { add() }
            #expect(registry.bestModule(for: kleinEntry())?.descriptor.id == "flux2-klein", "kleinFirst \(kleinFirst)")
            #expect(registry.bestModule(for: flux1Entry())?.descriptor.id == "flux", "kleinFirst \(kleinFirst)")
        }
    }

    @Test func theKleinModuleResolvesAUIAndAnSDKThatAgree() throws {
        let resolved = try #require(shipping().bestModule(for: kleinEntry()))
        #expect(resolved.sdk is Flux2KleinSDK)
        #expect(resolved.ui is Flux2KleinUI)
        #expect(Flux2KleinModule.descriptor.modelTypes == [.image])
    }

    @Test func klein_isNotInTheLocalServersSurface() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("MLXUI")
        let serverDir = root.appendingPathComponent("Services/Server")
        let files = (FileManager.default.enumerator(at: serverDir, includingPropertiesForKeys: nil)?.allObjects as? [URL] ?? []).filter { $0.pathExtension == "swift" }
        #expect(!files.isEmpty)
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            #expect(!text.contains("Flux2Klein"), Comment(rawValue: "\(file.lastPathComponent) mentions Klein: the Local Server never serves this model"))
        }
    }
}

// MARK: - Memory lease (LeasingStage / EngineCache / MemoryBudget)

@Suite(.serialized)
struct Flux2KleinLeaseTests {
    private static let gib = Int64(1_073_741_824)

    private func makeBudget(_ gbs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gbs * Self.gib, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }
    private func makeCache(_ budget: MemoryBudget) -> EngineCache {
        EngineCache(budget: budget, countsLLMWeights: false, modelDirectory: { _ in FileManager.default.temporaryDirectory.appendingPathComponent("no-klein-here") })
    }

    @Test func kleinBooksTheStagedPeakNotItsSixteenGigabytesOfFiles() throws {
        let cache = makeCache(makeBudget(16))
        let booked = cache.footprint(of: kleinEntry())
        #expect(booked == Flux2KleinBooking.bookedBytes)
        #expect(booked > 5_500_000_000 && booked < 6_500_000_000, "\(booked): the largest measured whole-run footprint of the default run")
        let plan = Flux2KleinMemoryPlan.peaks(width: 1024, height: 1024, quantize: 8).busiest * 1e9
        #expect(Double(booked) >= plan, "the lease must not book less than the plan's busiest stage at the default size")
        // another .image model is untouched by the Klein rule
        #expect(cache.footprint(of: flux1Entry()) == Int64(flux1Entry().ramGB * 1_073_741_824))
        #expect(Flux2KleinBooking.bytes(for: flux1Entry()) == nil)
    }

    @Test func kleinFitsAFourteenGigabyteBudgetThatItsFilesWouldNeverFit() async throws {
        let budget = makeBudget(10)
        let cache = makeCache(budget)
        let tiny = TinyKlein()
        let stage = LeasingStage(model: kleinEntry(), config: .default, probe: tiny.stage(tinySettings()), cache: cache, maxWait: .seconds(2))
        _ = try await stage.run(.text("p")) { _ in }
        #expect(budget.inUseCount == 0 && !stage.isRunning)
    }

    @Test func theLeaseIsHeldDuringTheRunAndReturnedWhenItEnds() async throws {
        let budget = makeBudget(10)
        let cache = makeCache(budget)
        let seenInUse = KleinBox(-1), seenUsed = KleinBox(Int64(-1))
        let probe = KleinProbeStage(hook: { seenInUse.set(budget.inUseCount); seenUsed.set(budget.usedBytes) })
        let stage = LeasingStage(model: kleinEntry(), config: .default, probe: probe, cache: cache, maxWait: .seconds(2))
        _ = try await stage.run(.text("p")) { _ in }
        #expect(seenInUse.value == 1)
        #expect(seenUsed.value == Flux2KleinBooking.bookedBytes)
        #expect(budget.inUseCount == 0)
    }

    @Test func theLeaseIsReturnedWhenTheStageThrows() async {
        let budget = makeBudget(10)
        let stage = LeasingStage(model: kleinEntry(), config: .default, probe: KleinProbeStage(failure: Flux2KleinEngineError.imageConversionFailed), cache: makeCache(budget), maxWait: .seconds(2))
        await #expect(throws: Flux2KleinEngineError.self) { _ = try await stage.run(.text("p")) { _ in } }
        #expect(budget.inUseCount == 0 && !stage.isRunning)
    }

    @Test func theLeaseIsReturnedAndTheComponentReleasedWhenTheRunIsCancelledMidDenoise() async throws {
        let budget = makeBudget(10)
        let tiny = TinyKlein()
        let started = KleinBox(false)
        // cancel as soon as the first denoise step lands
        let task = KleinBox<Task<Media, Error>?>(nil)
        let stage = LeasingStage(model: kleinEntry(), config: .default,
                                 probe: tiny.stage(tinySettings(steps: 6), onPhase: { phase in
                                     if case .denoising(step: 1, _) = phase { started.set(true); task.value?.cancel() }
                                 }), cache: makeCache(budget), maxWait: .seconds(2))
        let run = Task { try await stage.run(.text("p")) { _ in } }
        task.set(run)
        await #expect(throws: CancellationError.self) { _ = try await run.value }
        #expect(started.value)
        #expect(budget.inUseCount == 0 && !stage.isRunning, "the lease must come back on cancel")
        let steps = tiny.stages.log.events.filter { if case .denoiseStep = $0 { return true } else { return false } }
        #expect(steps.count < 6, "cancel must stop the denoise loop")
        #expect(tiny.stages.log.events.contains(.released(.transformer)))
        #expect(tiny.stages.log.aliveAtRelease.allSatisfy { !$0.1 }, "released components must really be gone")
    }
}

// MARK: - Run UI view-model logic

@MainActor
struct Flux2KleinRunModelTests {
    private let big: Int64 = 40_000_000_000

    @Test func defaultsAreQuantize8AndTheManifestSize() throws {
        let m = Flux2KleinRunModel()
        #expect(m.quantize == 8 && m.widthText == "1024" && m.heightText == "1024" && m.stepsText == "4" && m.seedText.isEmpty)
        #expect(Flux2KleinRunModel.quantizeChoices.map(\.bits) == [8, 4])
        #expect(Flux2KleinRunModel.quantizeChoices[1].label.contains("smaller, lower quality"))
        #expect(Flux2KleinRunModel.quantizeChoices[0].label == "8-bit")
        m.budgetBytes = { big }
        #expect(m.settingsRefusal == nil)
        #expect(try m.resolve().get() == Flux2KleinRunSettings())
    }

    @Test func theEncoderNoteShowsOnlyBelowEight() {
        let m = Flux2KleinRunModel()
        #expect(m.encoderNote == nil)
        m.quantize = 4
        #expect(m.encoderNote?.contains("The text encoder stays 8-bit") == true)
        m.quantize = 8
        #expect(m.encoderNote == nil)
    }

    @Test func fieldsParseToSettingsAndABlankSeedStaysNil() throws {
        let r = Flux2KleinRunModel.resolve(width: " 768 ", height: "512", steps: "6", seed: " 123 ", quantize: 4, budgetBytes: big)
        #expect(try r.get() == Flux2KleinRunSettings(width: 768, height: 512, steps: 6, seed: 123, quantize: 4))
        #expect(try Flux2KleinRunModel.resolve(width: "512", height: "512", steps: "4", seed: "  ", quantize: 8, budgetBytes: big).get().seed == nil)
    }

    @Test func badFieldsGiveOnePlainSentenceEach() {
        func message(w: String = "1024", h: String = "1024", s: String = "4", seed: String = "", q: Int = 8, budget: Int64? = nil) -> String? {
            if case let .failure(f) = Flux2KleinRunModel.resolve(width: w, height: h, steps: s, seed: seed, quantize: q, budgetBytes: budget ?? big) { return f.message }
            return nil
        }
        #expect(message(w: "wide") == "Width must be a whole number of pixels.")
        #expect(message(h: "") == "Height must be a whole number of pixels.")
        #expect(message(s: "0") == "Steps must be a whole number, 1 or more.")
        #expect(message(s: "x") == "Steps must be a whole number, 1 or more.")
        #expect(message(seed: "-3") == "Seed must be a whole number, or blank for a random one.")
        #expect(message(q: 6) == "This image: FLUX.2 Klein 4B runs at quantize 4 or 8 in this build, not 6.")
    }

    @Test func aSizeThatIsNotAMultipleOfSixteenIsRefusedInTheSPECQ242Wording() throws {
        let text = try #require(Flux2KleinRunModel.resolve(width: "1000", height: "1024", steps: "4", seed: "", quantize: 8, budgetBytes: big).failureMessage)
        let expected = (try? Flux2KleinLatents.validateSize(width: 1000, height: 1024)) == nil
        #expect(expected)
        do { _ = try Flux2KleinLatents.validateSize(width: 1000, height: 1024) } catch { #expect(text == (error as? LocalizedError)?.errorDescription, Comment(rawValue: text)) }
    }

    @Test func aSizeOverTheCapIsRefusedNamingTheLargestAndRunIsDisabled() throws {
        let m = Flux2KleinRunModel()
        m.budgetBytes = { 12_000_000_000 }
        m.widthText = "2048"; m.heightText = "2048"
        let text = try #require(m.settingsRefusal)
        #expect(text.hasPrefix("This image can't make a 2048 x 2048 image at quantize 8 on this Mac"), Comment(rawValue: text))
        #expect(text.contains("largest it can make here is"))
        #expect(!m.canRun)
        m.applySizePreset(1024)
        #expect(m.widthText == "1024" && m.heightText == "1024" && m.canRun)
        #expect(m.largestSquare == Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: 12_000_000_000))
    }

    @Test func runNeedsAPrompt() {
        let m = Flux2KleinRunModel()
        m.budgetBytes = { big }
        m.prompt = "   \n"
        #expect(m.promptIsEmpty && !m.canRun)
        m.prompt = "a fox"
        #expect(m.canRun)
    }

    @Test func pngDataRoundTripsAndTheFileNameCarriesTheSeed() throws {
        let image = try #require(KleinPNG.solid(width: 4, height: 3))
        let data = try #require(Flux2KleinRunModel.pngData(for: image))
        #expect(data.starts(with: [0x89, 0x50, 0x4E, 0x47]))
        #expect(Flux2KleinRunModel.suggestedFileName(seed: 42) == "klein-42.png")
        #expect(Flux2KleinRunModel.suggestedFileName(seed: nil) == "klein-image.png")
    }

    // MARK: start / cancel, with the stage run through a real LeasingStage

    private func leasing(_ budget: MemoryBudget) -> LeasingStage {
        let cache = EngineCache(budget: budget, countsLLMWeights: false, modelDirectory: { _ in FileManager.default.temporaryDirectory.appendingPathComponent("no-klein-here") })
        return LeasingStage(model: kleinEntry(), config: .default, probe: KleinProbeStage(), cache: cache, maxWait: .seconds(2))
    }
    private func budget() -> MemoryBudget {
        MemoryBudget(capacityBytes: 10 * 1_073_741_824, cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    /// The tiny engine's size rules: 32 x 32 is a valid Klein size, and its busiest stage fits the real budget on any Mac.
    private func tinyModel() -> (Flux2KleinRunModel, TinyKlein) {
        let m = Flux2KleinRunModel()
        m.widthText = "32"; m.heightText = "32"; m.stepsText = "2"; m.seedText = "7"
        return (m, TinyKlein())
    }

    @Test func startRunsTheStageOffTheMainActorAndFillsTheResult() async throws {
        let (m, tiny) = tinyModel()
        let b = budget()
        m.start(modelID: kleinEntry().id, handedStage: leasing(b)) { settings, onPhase in tiny.stage(settings, onPhase: onPhase) }
        #expect(m.isRunning && m.usedSeed == 7)
        await m.waitForRun()
        #expect(!m.isRunning && m.errorText == nil && !m.wasStopped)
        #expect(m.image?.width == 32 && m.image?.height == 32)
        #expect(m.progress == 1)
        // KICKOFF 17c: started from the main actor (this test is @MainActor), every component still loaded off it
        #expect(tiny.stages.log.mainThreadAtMake == [false, false, false])
        #expect(b.inUseCount == 0)
        #expect(m.usedSettings == Flux2KleinRunSettings(width: 32, height: 32, steps: 2, seed: 7, quantize: 8))
    }

    @Test func aBlankSeedIsReplacedByARandomOneThatIsShown() async throws {
        let (m, tiny) = tinyModel()
        m.seedText = ""
        m.start(modelID: kleinEntry().id, handedStage: leasing(budget())) { s, p in tiny.stage(s, onPhase: p) }
        let seed = try #require(m.usedSeed)
        #expect(m.usedSettings?.seed == seed)
        await m.waitForRun()
        #expect(m.image != nil)
    }

    @Test func phasesReachTheViewModelWhileItRuns() async throws {
        let (m, tiny) = tinyModel()
        m.start(modelID: kleinEntry().id, handedStage: leasing(budget())) { s, p in tiny.stage(s, onPhase: p) }
        await m.waitForRun()
        // after the run the last phase hop has landed and progress never exceeded 1
        for _ in 0 ..< 100 where m.phase != .decoding { try await Task.sleep(for: .milliseconds(10)) }
        #expect(m.phase == .decoding)
    }

    @Test func startDoesNothingWhenTheFieldsAreInvalidOrARunIsGoing() async throws {
        let (m, tiny) = tinyModel()
        m.widthText = "33"
        m.start(modelID: kleinEntry().id, handedStage: leasing(budget())) { s, p in tiny.stage(s, onPhase: p) }
        #expect(!m.isRunning && m.usedSeed == nil)
        m.widthText = "32"
        m.start(modelID: kleinEntry().id, handedStage: leasing(budget())) { s, p in tiny.stage(s, onPhase: p) }
        let first = m.usedSeed
        m.seedText = "99"
        m.start(modelID: kleinEntry().id, handedStage: leasing(budget())) { s, p in tiny.stage(s, onPhase: p) }
        #expect(m.usedSeed == first, "a second start while running must be ignored")
        await m.waitForRun()
    }

    @Test func cancelStopsTheRunWithoutAnErrorAndReturnsTheLease() async throws {
        let m = Flux2KleinRunModel()
        let b = budget()
        let gate = KleinBox(false)
        m.start(modelID: kleinEntry().id, handedStage: leasing(b)) { _, _ in KleinProbeStage(untilCancelled: { gate.set(true) }) }
        for _ in 0 ..< 200 where !gate.value { try await Task.sleep(for: .milliseconds(10)) }
        #expect(gate.value && m.isRunning && b.inUseCount == 1)
        m.cancel()
        await m.waitForRun()
        #expect(!m.isRunning && m.wasStopped && m.errorText == nil && m.image == nil)
        #expect(b.inUseCount == 0)
    }

    @Test func aFailureShowsOneSentenceAndEndsTheRun() async {
        let m = Flux2KleinRunModel()
        let b = budget()
        m.start(modelID: kleinEntry().id, handedStage: leasing(b)) { _, _ in KleinProbeStage(failure: Flux2KleinEngineError.imageConversionFailed) }
        await m.waitForRun()
        #expect(!m.isRunning && !m.wasStopped)
        #expect(m.errorText == "FLUX.2 Klein could not turn the decoded picture into an image.")
        #expect(b.inUseCount == 0)
    }

    @Test func aMissingInstallIsAReadableError() async throws {
        let (m, _) = tinyModel()
        m.start(modelID: "no-such--Klein", handedStage: KleinProbeStage())     // production stage, no model folder
        await m.waitForRun()
        #expect(m.errorText?.contains("isn't installed") == true, Comment(rawValue: m.errorText ?? "nil"))
    }
}

// MARK: - Helpers

private extension Result where Failure == Flux2KleinSettingsRefusal {
    var failureMessage: String? { if case let .failure(f) = self { return f.message } else { return nil } }
}

final class KleinBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T
    init(_ v: T) { _value = v }
    var value: T { get { lock.lock(); defer { lock.unlock() }; return _value } set { lock.lock(); _value = newValue; lock.unlock() } }
    func set(_ v: T) { value = v }
    func mutate(_ f: (inout T) -> Void) { lock.lock(); f(&_value); lock.unlock() }
}

/// A stage that does nothing real: runs a hook, optionally fails, optionally waits until cancelled.
nonisolated final class KleinProbeStage: PipelineStage, @unchecked Sendable {
    var id: String { "probe" }
    var name: String { "probe" }
    var accepts: MediaKind { .text }
    var produces: MediaKind { .image }
    private let hook: (@Sendable () -> Void)?
    private let failure: Error?
    private let waitsForCancel: Bool
    init(hook: (@Sendable () -> Void)? = nil, failure: Error? = nil) { self.hook = hook; self.failure = failure; self.waitsForCancel = false }
    init(untilCancelled hook: @escaping @Sendable () -> Void) { self.hook = hook; self.failure = nil; self.waitsForCancel = true }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        hook?()
        if let failure { throw failure }
        if waitsForCancel { while true { try await Task.sleep(for: .milliseconds(10)) } }
        return .image(ImageMedia(cgImage: KleinPNG.solid(width: 2, height: 2)!))
    }
}

enum KleinPNG {
    static func solid(width: Int, height: Int) -> CGImage? {
        var rgba = [UInt8](repeating: 200, count: width * height * 4)
        return CGContext(data: &rgba, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)?.makeImage()
    }

    static func rgbaBytes(_ image: CGImage) -> [UInt8] {
        var rgba = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let ctx = CGContext(data: &rgba, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        ctx?.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return rgba
    }
}
