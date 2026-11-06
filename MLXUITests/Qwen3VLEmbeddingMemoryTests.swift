import Darwin
import Foundation
import Testing
import AppKit
import MLX
import MLXLMCommon
@testable import MLXUI

/// MC4-7 — the footprint measurement for Qwen3-VL Embedding 2B. It **measures**; it changes no
/// `ramGB`, limit or budget (decision 9: the maintainers set `ramGB` from their own smoke run, and no
/// `.embedding` row is added). Shape: `SAMAudioFootprintTests` — `phys_footprint`
/// (`task_info(TASK_VM_INFO)`, what Activity Monitor and jetsam count) through the shared
/// `FootprintTracker`, sampled every 20 ms, plus `Memory.snapshot()`.
///
/// Skipped unless the weights are installed (`Qwen3VLEmbedRealLoad`) **and**
/// `MLXUI_TEST_QWEN3VL_EMBED_FOOTPRINT=1` (`TEST_RUNNER_MLXUI_TEST_QWEN3VL_EMBED_FOOTPRINT=1`).
/// One line per phase: `QWEN3VLEMBEDFOOTPRINT phase=… peakGB=… baselineMiB=… mlxPeakGB=… wall=…`.
@Suite(.serialized)
struct Qwen3VLEmbeddingMemoryTests {

    static let env = ProcessInfo.processInfo.environment

    static var enabled: Bool {
        Qwen3VLEmbedRealLoad.canRun
            && (env["MLXUI_TEST_QWEN3VL_EMBED_FOOTPRINT"] ?? env["TEST_RUNNER_MLXUI_TEST_QWEN3VL_EMBED_FOOTPRINT"]) == "1"
    }

    struct PhaseResult {
        let peak: Int64
        let baseline: Int64
        let mlxPeak: Int64
        let wall: TimeInterval
    }

    /// Run `body`, tracking the whole-process peak. `Memory.clearCache()` starts each phase clean.
    private func measure(
        _ phase: String, _ body: @MainActor () async throws -> Int
    ) async throws -> PhaseResult {
        Memory.clearCache()
        Memory.peakMemory = 0
        let baseline = FootprintTracker.phys()
        let tracker = FootprintTracker()
        tracker.note(baseline)
        let sampler = Task.detached {
            while !Task.isCancelled {
                tracker.note(FootprintTracker.phys())
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        let started = Date()
        let count = try await body()
        let wall = Date().timeIntervalSince(started)
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let result = PhaseResult(
            peak: tracker.runPeak, baseline: baseline,
            mlxPeak: Int64(Memory.snapshot().peakMemory), wall: wall)
        let mib = 1_048_576.0
        print(String(
            format: "QWEN3VLEMBEDFOOTPRINT phase=%@ peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f mlxPeakGB=%.3f wall=%.1f s count=%d",
            phase, Double(result.peak) / 1e9, Double(result.peak) / mib,
            Double(baseline) / mib, Double(result.mlxPeak) / 1e9, wall, count))
        return result
    }

    /// 1. Load the model and embed one sentence.
    @MainActor
    @Test(.enabled(if: Qwen3VLEmbeddingMemoryTests.enabled))
    func phase1_loadAndOneSentence() async throws {
        let directory = try #require(Qwen3VLEmbedRealLoad.directory)
        let result = try await measure("load_and_sentence") {
            let session = try await Qwen3VLEmbeddingSession.load(from: directory)
            let vector = try await session.embedText("A woman playing with her dog on a beach at sunset.")
            #expect(vector.count == 2048)
            return 1
        }
        print(String(format: "QWEN3VLEMBEDFOOTPRINT summary phase=load_and_sentence peakGB=%.3f", Double(result.peak) / 1e9))
    }

    /// 2. One maximum-size image (1,310,720 px = the processor's `max_pixels`) with a caption.
    @MainActor
    @Test(.enabled(if: Qwen3VLEmbeddingMemoryTests.enabled))
    func phase2_maximumSizeImageWithCaption() async throws {
        let directory = try #require(Qwen3VLEmbedRealLoad.directory)
        let image = try Self.maximumSizeImage()
        let result = try await measure("max_image_caption") {
            let session = try await Qwen3VLEmbeddingSession.load(from: directory)
            let vector = try await session.embedImage(
                image, text: "A photograph of a busy street at dusk.")
            #expect(vector.count == 2048)
            return 1
        }
        print(String(format: "QWEN3VLEMBEDFOOTPRINT summary phase=max_image_caption peakGB=%.3f", Double(result.peak) / 1e9))
    }

    /// 3. Worst case: an ≈8,192-token text, then a 200-image ranking run.
    @MainActor
    @Test(.enabled(if: Qwen3VLEmbeddingMemoryTests.enabled))
    func phase3_worstCaseTextThenTwoHundredImages() async throws {
        let directory = try #require(Qwen3VLEmbedRealLoad.directory)
        let session = try await Qwen3VLEmbeddingSession.load(from: directory)
        let longText = try await Self.eightThousandTokenText(directory: directory)
        let images = try Self.twoHundredImages()
        defer { for url in images { try? FileManager.default.removeItem(at: url) } }

        let result = try await measure("worst_case_8192_text_then_200_images") {
            let textVector = try await session.embedText(longText)
            #expect(textVector.count == 2048)
            let model = EmbeddingImagesRunModel(
                embedImage: { image in try await session.embedImage(image) },
                embedText: { query in try await session.embedText(query) })
            await model.addImages(at: images)
            #expect(model.entries.count == 200)
            await model.rank(query: "A red bicycle leans against a brick wall.")
            #expect(model.ranked.count == 200)
            return 201
        }
        print(String(format: "QWEN3VLEMBEDFOOTPRINT summary phase=worst_case peakGB=%.3f", Double(result.peak) / 1e9))

        // Decision 9: the default `.embedding` row books max(files × 1.10, catalog ramGB). Compare
        // the peak with today's provisional ramGB (2.7 GB) — this test does not change it.
        let files = ModelWeightCharge.safetensorsBytes(in: directory)
        let charge = max(ModelWeightCharge.charge(onDiskBytes: files), Int64(2_700_000_000))
        print(String(
            format: "QWEN3VLEMBEDFOOTPRINT booking filesGB=%.3f chargeGB=%.3f ramGB=2.700 peakGB=%.3f peakOverBookingGB=%.3f",
            Double(files) / 1e9, Double(charge) / 1e9, Double(result.peak) / 1e9,
            Double(result.peak - charge) / 1e9))
    }

    // MARK: - Inputs

    /// 1,280 × 1,024 = 1,310,720 px, the processor's maximum.
    private static func maximumSizeImage() throws -> CGImage {
        guard let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 1_280, pixelsHigh: 1_024,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else {
            throw CocoaError(.fileWriteUnknown)
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor.systemOrange.setFill()
        NSRect(x: 0, y: 0, width: 1_280, height: 1_024).fill()
        NSGraphicsContext.restoreGraphicsState()
        return try #require(rep.cgImage)
    }

    /// A text whose rendered prompt is as close to the 8,192-token cap as this test can get without
    /// crossing it (one token count via the real processor, then repeat).
    private static func eightThousandTokenText(directory: URL) async throws -> String {
        let sentence = "The quick brown fox jumps over the lazy dog. "
        var repeats = 700
        for _ in 0..<6 {
            let candidate = String(repeating: sentence, count: repeats)
            let prepared = try await Qwen3VLEmbedder.preparedInput(
                text: candidate, image: nil, modelDirectory: directory)
            let count = prepared.tokenIDs.count
            if count > Qwen3VLEmbedder.maxInputTokens {
                repeats = Int(Double(repeats) * Double(Qwen3VLEmbedder.maxInputTokens) / Double(count)) - 1
                continue
            }
            if count < 7_900 {
                repeats = Int(Double(repeats) * 8_050.0 / Double(count))
                continue
            }
            print("QWEN3VLEMBEDFOOTPRINT text_tokens=\(count)")
            return candidate
        }
        throw Qwen3VLEmbedError.inputTooLong(tokens: Qwen3VLEmbedder.maxInputTokens)
    }

    private static func twoHundredImages() throws -> [URL] {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-footprint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var urls: [URL] = []
        for index in 0..<200 {
            let url = dir.appendingPathComponent("image-\(index).png")
            guard let rep = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 48,
                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0) else {
                throw CocoaError(.fileWriteUnknown)
            }
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
            NSColor(hue: CGFloat(index % 10) / 10, saturation: 0.5, brightness: 0.8, alpha: 1).setFill()
            NSRect(x: 0, y: 0, width: 64, height: 48).fill()
            NSGraphicsContext.restoreGraphicsState()
            guard let data = rep.representation(using: .png, properties: [:]) else {
                throw CocoaError(.fileWriteUnknown)
            }
            try data.write(to: url)
            urls.append(url)
        }
        return urls
    }
}
