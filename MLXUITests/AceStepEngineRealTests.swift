import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3i: the whole engine on the REAL installed folder. Owner-run, skipped by default (`TEST_RUNNER_MLXUI_TEST_ACESTEP=1`, the model folder with
/// `acestep-v15-turbo/`, `Qwen3-Embedding-0.6B/`, `vae/`; env `TEST_RUNNER_MLXUI_TEST_ACESTEP_DIR`, defaulting to the oracle copy, same as 3c-3h).
///
/// 1. `goldenWithInjectedInputs`: the engine's own noise cannot equal the oracle's torch noise, so this runs the real engine with the oracle's
///    stage-a `noise` and `text.hidden` / `text.lyric_embed` injected (the text source seam and the `noise:` parameter) and compares the final latents
///    with stage-b `sampler.latents_dcw_on` (cos >= 0.99) and the audio with stage-c `audio.final_peak_normalised` (cos >= 0.99). Instrumental prompt.
/// 2. `freeNoiseRunReportsTheFootprint`: a real 10 s instrumental run, seed 12345, the installed Qwen3 text side. Finite, non-silent, sane rms; prints
///    `ACESTEP-ENGINE phase=<event> peakPhysMB=<x> mlxPeakMB=<x> seconds=<x>` for each stage boundary (peakPhys is the process phys_footprint peak since
///    the previous line) and a final `phase=total` line, plus `ACESTEP-ENGINE audioRms=<x> audioPeak=<x>`.
@Suite(.serialized, .enabled(if: AceStepRealLoad.canRun && AceStepOracle.haveDumps, "needs MLXUI_TEST_ACESTEP=1, the model folder and the oracle dumps"))
struct AceStepEngineRealTests {

    /// Returns the oracle's text tensors as the text side.
    struct OracleText: AceStepTextSource {
        let hidden: MLXArray, lyric: MLXArray
        func inputs(for request: AceStepRequest) async throws -> AceStepTextInputs {
            AceStepTextInputs(textIds: [], textMask: MLXArray.ones([1, hidden.dim(1)], dtype: .int32), lyricIds: [],
                              lyricMask: MLXArray.ones([1, lyric.dim(1)], dtype: .int32), textHidden: hidden, lyricEmbeds: lyric)
        }
    }

    private func read(_ url: URL, _ names: [String]) throws -> [String: MLXArray] { try SafetensorsIndex(url: url).read(names) }

    @Test func goldenWithInjectedInputs() async throws {
        let dir = try #require(AceStepRealLoad.directory)
        let id = "instrumental"
        let a = try read(AceStepOracle.dump(id), ["noise", "text.hidden", "text.lyric_embed"])
        let b = try read(AceStepDiTReal.dump(id), ["sampler.latents_dcw_on"])
        let c = try read(AceStepVAEReal.dump(id), ["audio.final_peak_normalised"])
        let hidden = try #require(a["text.hidden"]).asType(.bfloat16), lyric = try #require(a["text.lyric_embed"]).asType(.bfloat16)
        let engine = AceStepEngine(directory: dir, textSource: OracleText(hidden: hidden, lyric: lyric))
        let result = try await engine.generate(AceStepRequest(caption: "x", duration: 10, seed: 12345), noise: try #require(a["noise"]))

        let wantLatents = try #require(b["sampler.latents_dcw_on"])
        let cosL = Self.cosine(result.latents, wantLatents)
        #expect(result.latents.shape == wantLatents.shape)
        #expect(cosL >= 0.99, "final latents cosine \(cosL)")

        let want = try #require(c["audio.final_peak_normalised"])                  // [2, 480000]
        let got = MLXArray(result.audio.samples, [want.dim(1), 2]).transposed()      // interleaved -> [2, S]
        #expect(got.shape == want.shape)
        let cosA = Self.cosine(got, want)
        let maxAbs = abs(got - want).max().item(Float.self)
        print(String(format: "ACESTEP-ENGINE golden latentsCos=%.5f audioCos=%.5f audioMaxAbs=%.5f", cosL, cosA, maxAbs))
        #expect(cosA >= 0.99, "audio cosine \(cosA)")
    }

    /// MC5-T-vocal: the engine golden for BOTH oracle prompts that have stage-a/b/c dumps, in two variants, each with the oracle's injected noise (seed 12345, 10 s, 8 steps):
    /// V1 injects the oracle's text tensors (as `goldenWithInjectedInputs` does); V2 injects NOTHING of the text side: the engine's own tokenizer + Qwen + text_projector run on the
    /// request's caption / lyrics / language. Prints `ACESTEP-VOCAL prompt=.. variant=V1|V2 latentsCos=.. audioCos=.. audioMaxAbs=..`. For english_vocal V2 it also prints the request and the
    /// prompt strings the Qwen sees, and asserts them (strings and token ids) against `Fixtures/MC5/mc5_tokens.json`.
    @Test(arguments: ["instrumental", "english_vocal"])
    func engineGoldenBothTextVariants(id: String) async throws {
        let dir = try #require(AceStepRealLoad.directory)
        let p = try #require(AceStepTextFixtures.prompts[id])
        let a = try read(AceStepOracle.dump(id), ["noise", "text.hidden", "text.lyric_embed"])
        let b = try read(AceStepDiTReal.dump(id), ["sampler.latents_dcw_on"])
        let c = try read(AceStepVAEReal.dump(id), ["audio.final_peak_normalised"])
        let noise = try #require(a["noise"])
        let wantLatents = try #require(b["sampler.latents_dcw_on"])
        let want = try #require(c["audio.final_peak_normalised"])
        // Exactly what the Run screen builds for these inputs: lyrics as typed (empty = instrumental), the picker's language code.
        let request = AceStepRequest(caption: p.caption, lyrics: p.instrumental ? "" : p.lyrics, language: p.language, duration: 10, seed: 12345)

        if id == "english_vocal" {
            let e = try AceStepTextFixtures.entry(id)
            let textStr = AceStepPrompt.textString(caption: request.caption, duration: request.duration)
            let lyricStr = AceStepPrompt.lyricString(lyrics: request.lyrics, language: request.language, instrumental: request.isInstrumental)
            func esc(_ s: String) -> String { s.replacingOccurrences(of: "\n", with: "\\n") }
            print("ACESTEP-VOCAL request caption=\(esc(request.caption)) | language=\(request.language) | isInstrumental=\(request.isInstrumental) | duration=\(request.duration) | seed=\(request.seed) | steps=\(request.steps)")
            print("ACESTEP-VOCAL request lyrics=\(esc(request.lyrics))")
            print("ACESTEP-VOCAL prompt text_str=\(esc(textStr))")
            print("ACESTEP-VOCAL prompt lyric_str=\(esc(lyricStr))")
            #expect(textStr == (e["text_str"] as? String), "text prompt string differs from the oracle's")
            #expect(lyricStr == (e["lyric_str"] as? String), "lyric prompt string differs from the oracle's")
            let tokenizer = try await AceStepTokenizer(directory: dir)
            let textIds = tokenizer.encodeText(caption: request.caption, duration: request.duration)
            let lyricIds = tokenizer.encodeLyrics(lyrics: request.lyrics, language: request.language, instrumental: request.isInstrumental)
            print("ACESTEP-VOCAL tokens text=\(textIds.count) lyric=\(lyricIds.count) (oracle \(AceStepTextFixtures.ints(e["text_ids"]).count) / \(AceStepTextFixtures.ints(e["lyric_ids"]).count))")
            #expect(textIds == AceStepTextFixtures.ints(e["text_ids"]), "text token ids differ from mc5_tokens.json")
            #expect(lyricIds == AceStepTextFixtures.ints(e["lyric_ids"]), "lyric token ids differ from mc5_tokens.json")
        }

        for variant in ["V1", "V2"] {
            let engine: AceStepEngine
            if variant == "V1" {
                let hidden = try #require(a["text.hidden"]).asType(.bfloat16), lyric = try #require(a["text.lyric_embed"]).asType(.bfloat16)
                engine = AceStepEngine(directory: dir, textSource: OracleText(hidden: hidden, lyric: lyric))
            } else {
                engine = AceStepEngine(directory: dir)           // the engine's own text side
            }
            let result = try await engine.generate(request, noise: noise)
            let cosL = Self.cosine(result.latents, wantLatents)
            let got = MLXArray(result.audio.samples, [want.dim(1), 2]).transposed()
            let cosA = Self.cosine(got, want)
            let maxAbs = abs(got - want).max().item(Float.self)
            print(String(format: "ACESTEP-VOCAL prompt=%@ variant=%@ latentsCos=%.5f audioCos=%.5f audioMaxAbs=%.5f", id, variant, cosL, cosA, maxAbs))
            #expect(result.latents.shape == wantLatents.shape && got.shape == want.shape)
            #expect(cosL >= 0.99, "\(id) \(variant) final latents cosine \(cosL)")
            #expect(cosA >= 0.99, "\(id) \(variant) audio cosine \(cosA)")
        }
    }

    @Test func freeNoiseRunReportsTheFootprint() async throws {
        let dir = try #require(AceStepRealLoad.directory)
        Memory.clearCache()
        Memory.peakMemory = 0
        let tracker = FootprintTracker()
        let started = Date()
        var last = started
        let sampler = Task.detached {
            while !Task.isCancelled { tracker.note(FootprintTracker.phys()); try? await Task.sleep(for: .milliseconds(20)) }
        }
        let lock = NSLock()
        var engine = AceStepEngine(directory: dir)
        engine.observer = { event in
            var label: String?
            switch event {
            case .loaded(let c): label = "loaded-\(c.rawValue)"
            case .released(let c): label = "released-\(c.rawValue)"
            default: break
            }
            guard let label else { return }
            tracker.note(FootprintTracker.phys())
            lock.lock(); defer { lock.unlock() }
            let now = Date()
            print(String(format: "ACESTEP-ENGINE phase=%@ peakPhysMB=%.0f mlxPeakMB=%.0f seconds=%.1f", label,
                         Double(tracker.takePhasePeak()) / 1_048_576, Double(Memory.snapshot().peakMemory) / 1_048_576, now.timeIntervalSince(last)))
            last = now
        }
        let result = try await engine.generate(AceStepRequest(caption: "warm lofi piano and soft drums, relaxed, instrumental", duration: 10, seed: 12345))
        sampler.cancel()
        print(String(format: "ACESTEP-ENGINE phase=total peakPhysMB=%.0f mlxPeakMB=%.0f seconds=%.1f", Double(tracker.runPeak) / 1_048_576,
                     Double(Memory.snapshot().peakMemory) / 1_048_576, Date().timeIntervalSince(started)))

        let samples = result.audio.samples
        #expect(samples.count == 480_000 * 2)
        #expect(samples.allSatisfy { $0.isFinite })
        let peak = samples.map { abs($0) }.max() ?? 0
        let rms = (samples.reduce(0) { $0 + $1 * $1 } / Float(samples.count)).squareRoot()
        print(String(format: "ACESTEP-ENGINE audioRms=%.4f audioPeak=%.4f", rms, peak))
        #expect(abs(peak - 0.891_25) < 1e-3, "peak-normalised")
        #expect(rms > 0.01 && rms < 0.6, "non-silent with a sane level, rms \(rms)")
    }

    static func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.asType(.float32), y = b.asType(.float32)
        return ((x * y).sum() / (sqrt(x.square().sum()) * sqrt(y.square().sum()))).item(Float.self)
    }
}
