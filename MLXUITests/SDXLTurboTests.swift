import Foundation
import Testing
@testable import MLXUI

/// SD-CAT1 — verifies the `stabilityai/sdxl-turbo` catalog entry (fp16 imagegen model)
/// decodes and resolves to the `.image` run path with **no** `ModelSupport` gap. The
/// engine tasks (SD-ENG1…SD-UI1) add the SDXL-Turbo module; until then the resolution is
/// asserted directly (runnerKind + no gap). Uses an inline fixture (the bundled catalog
/// isn't reachable from the test bundle) mirroring the `browser.json` entry.
struct SDXLTurboTests {

    private func makeEntry() throws -> ModelEntry {
        let json = """
        {
            "id": "stabilityai--sdxl-turbo",
            "family": "SDXL-Turbo",
            "displayName": "SDXL-Turbo",
            "paramSize": "3.5B",
            "paramCountB": 3.5,
            "modelType": "image",
            "source": "mlx",
            "format": "mlx-fp16",
            "platforms": ["macOS 13+"],
            "minMacOSVersion": "13.0",
            "hfRepo": "stabilityai",
            "hfModelId": "stabilityai/sdxl-turbo",
            "ramGB": 10.5,
            "downloadSizeGB": 7.0,
            "contextWindow": null,
            "license": "stability-ai-community-license",
            "licenseUrl": "https://huggingface.co/stabilityai/sdxl-turbo/blob/main/LICENSE.md",
            "architecture": "SDXL-Turbo",
            "languages": ["en"],
            "lastUpdated": "2024-07-10",
            "taskTags": ["text-to-image", "image-generation"],
            "descriptionSource": "stabilityai/sdxl-turbo",
            "variants": [
                {
                    "quantization": "fp16",
                    "format": "mlx-fp16",
                    "ramGB": 10.5,
                    "downloadSizeGB": 7.0,
                    "qualityPercent": 85,
                    "hfModelId": "stabilityai/sdxl-turbo",
                    "recommended": true
                }
            ]
        }
        """
        return try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
    }

    /// The catalog entry decodes with the expected fp16 / image / mlx shape and RAM
    /// (`3.5B × 2 bytes × 1.5 ≈ 10.5 GB` per pipedesign6 §10).
    @Test func sdxlTurboEntryDecodes() throws {
        let entry = try makeEntry()
        #expect(entry.modelType == .image)
        #expect(entry.source == .mlx)
        #expect(entry.format == "mlx-fp16")
        #expect(entry.ramGB == 10.5)
        #expect(entry.downloadSizeGB == 7.0)
        #expect(entry.taskTags?.contains("text-to-image") == true)
        #expect(entry.variants?.first?.quantization == "fp16")
    }

    /// An `image` model with `source == .mlx` routes to the `.image` runner kind
    /// (no family override mislabels it).
    @Test func sdxlTurboResolvesToImageRunnerKind() throws {
        let entry = try makeEntry()
        #expect(entry.runnerKind == .image)
    }

    /// The SDXL-Turbo architecture is being ported (SD-ENG1…SD-ENG4), so the entry must
    /// carry **no** `ModelSupport` gap — the model is runnable once the module lands, not
    /// flagged for the unsupported view.
    @Test func sdxlTurboHasNoSupportGap() throws {
        let entry = try makeEntry()
        #expect(ModelSupport.unsupportedReason(for: entry) == nil)
    }

    /// SD-MOD1 — the SDXL-Turbo SDK claims the entry `.exact`, and the Flux SDK's disjoint
    /// `"flux"` predicate does not (registration order is not load-bearing).
    @Test func sdxlTurboResolvesToItsOwnSDK() throws {
        let entry = try makeEntry()
        #expect(SDXLTurboSDK().claim(entry) == .exact)
        #expect(FluxSDK().claim(entry) == .no)
    }
}

/// SD-ENG4 — DDIM scheduler contract: 4 steps → 5 decreasing timesteps from ~999 to 0, and the
/// classic epsilon-DDIM step with the computed `alphas_cumprod`.
struct SDSchedulerTests {

    @Test func timestepsAreMonotonicFrom999To0() {
        let scheduler = SDScheduler(alphasCumprod: [Float](repeating: 1.0, count: 1000))
        let steps = scheduler.timesteps(numSteps: 4)
        #expect(steps.count == 5)
        #expect(abs(steps[0] - 999.0) < 0.01)
        #expect(steps.last == 0)
        for i in 1 ..< steps.count {
            #expect(steps[i] < steps[i - 1])
        }
    }

    @Test func scaledLinearAlphasCumprodStartsNearOneAndFalls() {
        let cumprod = SDScheduler.scaledLinearAlphasCumprod()
        #expect(cumprod.count == 1000)
        #expect(abs(cumprod[0] - 1.0) < 0.01)       // alpha_cumprod[0] ≈ 1
        #expect(cumprod[cumprod.count - 1] < 0.05)   // deep into the schedule (beta_end 0.012)
        #expect(cumprod[500] < cumprod[100])
    }

    @Test func alphaInterpolatesAtFractionalTimestep() {
        // With all-ones cumprod, alpha(t) == 1 everywhere (synthetic).
        let scheduler = SDScheduler(alphasCumprod: [Float](repeating: 1.0, count: 1000))
        #expect(abs(scheduler.alpha(at: 749.25) - 1.0) < 0.0001)
        #expect(abs(scheduler.alpha(at: 0) - 1.0) < 0.0001)
        // A clamped-out-of-range timestep returns the last value.
        let ramp = SDScheduler(alphasCumprod: (0 ... 9).map { Float($0) / 9.0 })
        #expect(abs(ramp.alpha(at: 500) - 1.0) < 0.001)
        #expect(abs(ramp.alpha(at: -1) - 0.0) < 0.001)
    }
}
