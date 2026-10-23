import Foundation
import CoreImage
import ImageIO
import MLXLMCommon
import Testing
@testable import MLXUI

/// WW3-1 — the real load of "Gemma 4 E4B Vision" on mlx-swift-lm 3.32.3. 3.31.4's MLXVLM `Gemma4`
/// declared `k_proj`/`v_proj` on the KV-shared layers (24-41) the checkpoint does not ship, so the
/// load died with `keyNotFound(layers.24.self_attn.v_proj.weight)` (smoke row 111, MP1-R-FIX-1).
/// 3.32.3 skips them (`if !kvSharedOnly && !isKVSharedLayer`). This proves it on the real weights:
/// load through the VLM module's own loader (`VLMEngine.loadContainer` → `VLMModelFactory` from
/// `ModelStore`'s directory — never `fromPretrained`), then one greedy generation on the sample
/// image, which must read "Trailhead".
///
/// **Owner-run, gated on the weights** (5.18 GB; one real-model test at a time, no other model
/// loaded). Skipped unless `mlx-community--gemma-4-e4b-it-4bit/.installed` exists under the
/// directory this process resolves: `ModelStore`'s, or `MLXUI_TEST_GEMMA4_DIR=<folder>` (passed to
/// the host as `TEST_RUNNER_MLXUI_TEST_GEMMA4_DIR`). Run the host unsandboxed so it can read it.
/// The exact command is in the WW3-1 journal. A skipped run does not count.
struct Gemma4VisionRealLoadTests {
    nonisolated static let directory: URL? = {
        let url: URL
        if let override = ProcessInfo.processInfo.environment["MLXUI_TEST_GEMMA4_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/gemma-4-e4b-it-4bit")
        }
        let marker = url.appendingPathComponent(".installed")
        return FileManager.default.fileExists(atPath: marker.path) ? url : nil
    }()

    nonisolated static func sampleImage() throws -> CGImage {
        // <repo>/MLXUITests/<this file> → <repo>/MLXUI/Resources/Samples/sample-image.png
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Samples/sample-image.png")
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil), "cannot open \(url.path)")
        return try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    @Test(.enabled(if: directory != nil))
    func theCheckpointLoadsAndReadsTheSampleImage() async throws {
        let dir = try #require(Self.directory)
        let image = try Self.sampleImage()

        // The loader the VLM pool uses — a load error (the old keyNotFound) fails the test here.
        let container = try await MLXErrorGuard.guarded {
            try await VLMEngine.loadContainer(directory: dir, architecture: nil)
        }

        let bounded = ImageLoader.downscaled(image, maxPixels: VLMEngine.maxInputPixels)
        let text = try await MLXErrorGuard.guarded {
            try await container.perform { context in
                let input = try await context.processor.prepare(
                    input: UserInput(chat: [
                        .user("What does the text say?", images: [.ciImage(CIImage(cgImage: bounded))])
                    ]))
                // Greedy, raw token ids — the same stream the engine consumes (OCP-FIX-1-2).
                let stream = try MLXLMCommon.generateTokens(
                    input: input,
                    parameters: GenerateParameters(maxTokens: 128, temperature: 0),
                    context: context)
                var ids: [Int] = []
                for await generation in stream {
                    if case .token(let id) = generation { ids.append(id) }
                }
                return context.tokenizer.decode(tokenIds: ids)
            }
        }
        print("WW3-1 Gemma 4 E4B Vision: \(text.debugDescription)")
        #expect(text.localizedCaseInsensitiveContains("Trailhead"))
    }
}
