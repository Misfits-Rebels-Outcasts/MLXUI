import Foundation
import Testing
import CoreGraphics
@testable import MLXUI

// WW2-3-8 — "a recorded MLX error becomes a row error, never a read", probed per engine against the real weights
// (owner ruling 2026-10-23, journal 2026-560).
//
// Each probe injects one MLX failure (`MLXErrorGuard.injectFailureForTesting`) and runs the engine the way the Run UI does
// (`@MainActor`, inside `MLXErrorGuard.guardedRun`). It must (1) throw, not crash, (2) have reached a guarded evaluate/read
// site (the injected failure was consumed), and, for the engines cheap enough to repeat, (3) run normally right after.
// Gated on the installed weights (`TEST_RUNNER_MLXUI_TEST_MODELS_DIR`); an engine whose weights aren't installed prints
// "not installed — skipped". Not covered here (their entry points need structured inputs): Laya, CLM, Clef, Rerank — the
// bare-read scan (`BareMLXReadScanTests`) covers their read sites. The vendored PaddleOCR loop is a known gap.

@MainActor
struct RecordedErrorRowProbeTests {
    nonisolated private static let modelsDir = ProcessInfo.processInfo.environment["MLXUI_TEST_MODELS_DIR"]

    nonisolated static func installed(_ slug: String) -> URL? {
        guard let root = modelsDir else { return nil }
        let url = URL(fileURLWithPath: root).appendingPathComponent(slug)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    private func image(_ w: Int, _ h: Int) -> CGImage {
        let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(CGColor(red: 0.8, green: 0.3, blue: 0.2, alpha: 1))
        ctx.fillEllipse(in: CGRect(x: w / 4, y: h / 4, width: w / 2, height: h / 2))
        return ctx.makeImage()!
    }

    /// The probe body: inject, run, expect a thrown error and a consumed injection; optionally run again normally.
    private func probe(_ name: String, recover: Bool, _ run: @escaping @Sendable () async throws -> Void) async {
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        MLXErrorGuard.injectFailureForTesting("injected MLX error (\(name))")
        defer { MLXErrorGuard.clearInjectedFailureForTesting() }
        do {
            try await MLXErrorGuard.guardedRun { try await run() }
            Issue.record("\(name): the injected MLX error was swallowed — the engine returned normally")
        } catch {
            #expect(MLXErrorGuard.injectedFailureWasConsumedForTesting(), "\(name): the engine never reached a guarded site (\(error))")
        }
        guard recover else { return }
        do { try await MLXErrorGuard.guardedRun { try await run() } }
        catch { Issue.record("\(name): did not recover after the injected failure: \(error)") }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil, "\(name)")
    }

    @Test func kokoro() async throws {
        guard let dir = Self.installed("mlx-community--Kokoro-82M-bf16") else { print("ROWPROBE kokoro: not installed — skipped"); return }
        await probe("Kokoro", recover: true) { _ = try await KokoroEngine.synthesize("Hello there.", voice: "af_heart", speed: 1.0, modelDirectory: dir) }
    }

    @Test func mlxAudioTTS() async throws {
        guard let dir = Self.installed("mlx-community--Qwen3-TTS-12Hz-0.6B-Base-bf16") else { print("ROWPROBE tts: not installed — skipped"); return }
        await probe("Qwen3-TTS", recover: true) { _ = try await MLXAudioTTSEngine.synthesize("Hello there.", voice: nil, modelDirectory: dir) }
    }

    @Test func seedVR2() async throws {
        guard Self.installed("mlx-community--SeedVR2-3B-mlx-int8") != nil else { print("ROWPROBE seedvr2: not installed — skipped"); return }
        let img = image(128, 128)
        await probe("SeedVR2", recover: true) {
            _ = try await SeedVR2Engine.upscale(image: img, scale: 2, seed: 7, modelID: "mlx-community--SeedVR2-3B-mlx-int8") { _ in }
        }
    }

    /// The real B18 scenario, no injection: with the pre-flight off, a 640×480 photo (1280×960 out) runs the VAE decode out of GPU
    /// memory on an 18 GiB Mac. It used to segfault in `tensorToCGImage`; it must now be a thrown error. (A bigger Mac may simply
    /// succeed — both are fine; only a crash fails this test, because the host would die.)
    @Test func seedVR2RealOutOfMemoryIsAThrownErrorNotACrash() async throws {
        guard Self.installed("mlx-community--SeedVR2-3B-mlx-int8") != nil else { print("ROWPROBE seedvr2-oom: not installed — skipped"); return }
        SeedVR2Preflight.disabledForTesting = true
        defer { SeedVR2Preflight.disabledForTesting = false }
        MLXErrorGuard.installFallback()
        let img = image(640, 480)
        do {
            let out = try await MLXErrorGuard.guardedRun {
                try await SeedVR2Engine.upscale(image: img, scale: 2, seed: 42, modelID: "mlx-community--SeedVR2-3B-mlx-int8") { _ in }
            }
            print("ROWPROBE seedvr2-oom: completed \(out.width)×\(out.height) (enough memory on this Mac)")
        } catch {
            print("ROWPROBE seedvr2-oom: thrown as expected: \(MLXErrorGuard.sentence(for: error).prefix(160))")
        }
    }

    @Test func sdxlTurbo() async throws {
        guard Self.installed("stabilityai--sdxl-turbo") != nil else { print("ROWPROBE sdxl: not installed — skipped"); return }
        await probe("SDXL-Turbo", recover: false) {
            _ = try await SDXLTurboEngine.generate(prompt: "a red ball", modelID: "stabilityai--sdxl-turbo", seed: 1) { _ in }
        }
    }

    @Test func flux() async throws {
        guard Self.installed("mlx-community--Flux-1.lite-8B-MLX-Q4") != nil else { print("ROWPROBE flux: not installed — skipped"); return }
        await probe("Flux", recover: false) {
            _ = try await FluxEngine.generate(prompt: "a red ball", modelID: "mlx-community--Flux-1.lite-8B-MLX-Q4", seed: 1) { _ in }
        }
    }

    @Test func wan() async throws {
        guard Self.installed("Wan-AI--Wan2.1-T2V-1.3B-Diffusers") != nil else { print("ROWPROBE wan: not installed — skipped"); return }
        await probe("Wan", recover: false) {
            _ = try await WanVideoEngine.generate(prompt: "a cat", modelID: "Wan-AI--Wan2.1-T2V-1.3B-Diffusers", seed: 1) { _ in }
        }
    }

    @Test func segmentAnything() async throws {
        guard Self.installed("mlx-community--sam3-4bit") != nil else { print("ROWPROBE sam: not installed — skipped"); return }
        let img = image(256, 256)
        await probe("SAM3", recover: true) { _ = try await SegmentAnythingEngine.encodeImage(img, modelID: "mlx-community--sam3-4bit") }
    }

    @Test func musicGen() async throws {
        guard Self.installed("jasonvassallo--mlx-musicgen-small") != nil else { print("ROWPROBE musicgen: not installed — skipped"); return }
        await probe("MusicGen", recover: false) {
            _ = try await MusicGenEngine.generate(prompt: "happy rock", modelID: "jasonvassallo--mlx-musicgen-small", maxSteps: 8) { _ in }
        }
    }

    @Test(arguments: ["mlx-community--all-MiniLM-L6-v2-4bit", "mlx-community--nomicai-modernbert-embed-base-bf16",
                      "mlx-community--embeddinggemma-300m-4bit"])
    func embedding(slug: String) async throws {
        guard let dir = Self.installed(slug) else { print("ROWPROBE \(slug): not installed — skipped"); return }
        await probe(slug, recover: true) { _ = try await EmbeddingEngine.embed(["a dog", "a puppy"], modelDirectory: dir) }
    }
}
