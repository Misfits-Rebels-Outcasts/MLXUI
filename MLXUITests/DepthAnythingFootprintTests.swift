import Darwin
import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLX
import Testing
@testable import MLXUI

/// MC3-7 — the footprint measurement for Depth Anything V2 Small. It **measures**; it changes no `ramGB`
/// (the maintainers set that from their own run). Shape: `SAMAudioFootprintTests` / the DeepFilterNet
/// harness — `phys_footprint` (`task_info(TASK_VM_INFO)`, what Activity Monitor and jetsam count) sampled
/// every 20 ms through the whole Run path (decode, preprocess, model, bicubic upsample, PNG).
///
/// Skipped unless the weights are installed **and** `MLXUI_TEST_DEPTH_FOOTPRINT=1`
/// (`TEST_RUNNER_MLXUI_TEST_DEPTH_FOOTPRINT=1` on the xcodebuild line). `MLXUI_TEST_DEPTH_FOOTPRINT_PHASES`
/// limits the phases; the default is all three (sample photo, 12 MP photo, the 8,192 px / 3:1 worst case).
///
/// One line per phase: `DEPTHFOOTPRINT phase=… input=… peakGB=… baselineMiB=… mlxPeakGB=… wall=… pngMiB=…`.
/// The `.depth` row is booked from the measurement above. Implementer's call, pending owner confirmation: the
/// growth over the 99 MB of weights is image buffers, not weights, so it is a per-call working set (the `.enhance`
/// row's form), never a factor over the files.
struct DepthAnythingMemoryBookingTests {
    @Test func theLedgerBooksTheMeasuredWorkingSet() {
        let row = ModelLoadProfile.entry(for: .depth)
        #expect(row.measured)
        #expect(row.workingSetBytes == 2_900_000_000, "the booking is the measured working set (MC3-12, R15)")
        #expect(ModelLoadProfile.workingSetBytes(for: .depth) == row.workingSetBytes)
    }
}

@Suite(.serialized)
struct DepthAnythingFootprintTests {
    static let env = ProcessInfo.processInfo.environment

    static var enabled: Bool {
        DepthAnythingRealLoad.canRun
            && (env["MLXUI_TEST_DEPTH_FOOTPRINT"] ?? env["TEST_RUNNER_MLXUI_TEST_DEPTH_FOOTPRINT"]) == "1"
    }

    static var phases: [String] {
        guard let raw = env["MLXUI_TEST_DEPTH_FOOTPRINT_PHASES"] ?? env["TEST_RUNNER_MLXUI_TEST_DEPTH_FOOTPRINT_PHASES"] else {
            return ["sample", "12mp", "worst"]
        }
        let parsed = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
        return parsed.isEmpty ? ["sample", "12mp", "worst"] : parsed
    }

    @Test(.enabled(if: DepthAnythingFootprintTests.enabled), arguments: DepthAnythingFootprintTests.phases)
    func footprint(phase: String) throws {
        let directory = try #require(DepthAnythingRealLoad.directory)
        let source = try DepthAnythingFootprintTests.source(phase: phase)
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
        // The whole Run path: decode (the Run screen's `choose`), then the engine.
        let image = try #require(DepthAnythingImage.decoded(source))
        let result = try DepthAnythingEngine.estimateDepth(image, modelDirectory: directory)
        let wall = Date().timeIntervalSince(started)
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let mib = 1_048_576.0
        let line = String(format: "DEPTHFOOTPRINT phase=%@ input=%dx%d peakGB=%.3f peakMiB=%.0f baselineMiB=%.0f mlxPeakGB=%.3f wall=%.2f pngMiB=%.1f",
                          phase, image.width, image.height,
                          Double(tracker.runPeak) / 1e9, Double(tracker.runPeak) / mib, Double(baseline) / mib,
                          Double(Memory.snapshot().peakMemory) / 1e9, wall, Double(result.pngData.count) / mib)
        print(line)
        // The sandboxed test host cannot write the caller's folder; the line is also appended to the test host's
        // own temporary directory so the measurement can be collected (the maintainers run this on their own Mac).
        if let data = (line + "\n").data(using: .utf8) {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent("DEPTHFOOTPRINT.txt")
            if let handle = try? FileHandle(forWritingTo: url) {
                handle.seekToEndOfFile()
                handle.write(data)
                try? handle.close()
            } else {
                try? data.write(to: url)
            }
        }
        #expect(result.width == image.width && result.height == image.height)
        #expect(!result.pngData.isEmpty)
    }

    /// PNG bytes for one phase. The synthetic photos are deterministic gradients (not committed anywhere).
    nonisolated static func source(phase: String) throws -> Data {
        switch phase {
        case "sample":
            let root = URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()   // MLXUITests
                .deletingLastPathComponent()   // repo
            return try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/Samples/sample-image.png"))
        case "12mp":
            return try pngData(width: 4_000, height: 3_000)
        case "worst":
            // Owner decision 5's worst case: 8,192 px on the long side at 3 : 1.
            return try pngData(width: 8_192, height: 2_731)
        default:
            throw DepthAnythingError.badInput("unknown footprint phase \(phase)")
        }
    }

    private nonisolated static func pngData(width: Int, height: Int) throws -> Data {
        // A smooth grey gradient: every channel equal, so the encoder stays cheap and the content is harmless.
        var bytes = [UInt8](repeating: 0, count: width * height)
        for row in 0 ..< height {
            let level = UInt8((Double(row) / Double(max(height - 1, 1)) * 255).rounded())
            for column in 0 ..< width {
                bytes[row * width + column] = level
            }
        }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8,
                                  bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw DepthAnythingError.badInput("could not build the \(width)x\(height) test image")
        }
        let output = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(output, UTType.png.identifier as CFString, 1, nil) else {
            throw DepthAnythingError.badInput("could not encode the test image")
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw DepthAnythingError.badInput("could not encode the test image")
        }
        return output as Data
    }
}
