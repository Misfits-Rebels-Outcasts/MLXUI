import Testing
import Foundation
import CoreGraphics
import MLX
@testable import MLXUI

/// SC-2c step 1 — what each non-LLM load really adds to the process, against `ModelWeightCharge` of its directory.
/// **Gated on env**, one case per process so MLX starts clean (`TEST_RUNNER_<NAME>=…`, unsandboxed host):
///
///     MLXUI_TEST_SC2C_CASE=dots|vlm|voxtral|laya
///
/// Models are read from `~/Library/Application Support/AI Browser/models/` (installed there by hand). Each case runs
/// the engine's real path twice and prints one `SC2C` line: `phys_footprint` before, the **peak during the run**
/// (sampled every 25 ms), after the run (MLX's buffer cache still held), and after `Memory.clearCache()`.
@Suite(.serialized)
struct RealSizesMeasureTests {
    static let only = ProcessInfo.processInfo.environment["MLXUI_TEST_SC2C_CASE"]

    private static func phys() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int64(info.phys_footprint) : -1
    }
    private static func gib(_ b: Int64) -> String { String(format: "%.2f", Double(b) / 1_073_741_824) }

    private final class Peak: @unchecked Sendable {
        private let lock = NSLock(); private var p: Int64 = 0, a: Int = 0, c: Int = 0
        func note(_ phys: Int64, _ active: Int, _ cache: Int) {
            lock.lock(); p = max(p, phys); a = max(a, active); c = max(c, cache); phase = max(phase, phys); lock.unlock()
        }
        private var phase: Int64 = 0
        /// The highest sample since the last call (a per-step peak).
        func takePhasePeak() -> Int64 { lock.lock(); defer { lock.unlock() }; let v = phase; phase = 0; return v }
        var value: (phys: Int64, active: Int, cache: Int) { lock.lock(); defer { lock.unlock() }; return (p, a, c) }
    }

    private static func sample() -> CGImage {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Samples/sample-image.png")
        guard let data = try? Data(contentsOf: url), let image = ImageLoader.decodedCGImage(from: data) else {
            let ctx = CGContext(data: nil, width: 448, height: 448, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            return ctx.makeImage()!
        }
        return image
    }

    /// Run `body` twice; report the peak while it runs and what is left behind.
    private func measure(_ label: String, directory: URL, body: () async throws -> Void) async throws {
        let disk = ModelWeightCharge.safetensorsBytes(in: directory)
        let recursive = EngineLeaseCache<Int>.charge(directory: directory)
        MLX.Memory.clearCache()
        let base = Self.phys()
        let peak = Peak()
        let sampler = Task.detached {
            while !Task.isCancelled {
                let m = MLX.Memory.snapshot()
                peak.note(Self.phys(), m.activeMemory, m.cacheMemory)
                try? await Task.sleep(for: .milliseconds(25))
            }
        }
        let t0 = ContinuousClock.now
        try await body()
        let first = ContinuousClock.now - t0
        let afterFirst = Self.phys()
        let peakFirst = peak.value
        try await body()
        sampler.cancel()
        let after = Self.phys()
        let m = MLX.Memory.snapshot()
        MLX.Memory.clearCache()
        try await Task.sleep(for: .milliseconds(500))
        let cleared = Self.phys()
        let p = peak.value
        print("SC2C \(label): safetensorsTopLevel=\(Self.gib(disk)) GiB recursive×1.10=\(Self.gib(recursive)) GiB | base=\(Self.gib(base)) peakDuringRuns=\(Self.gib(p.phys)) (mlxActivePeak=\(Self.gib(Int64(p.active))) mlxCachePeak=\(Self.gib(Int64(p.cache)))) afterRun1=\(Self.gib(afterFirst)) afterRun2=\(Self.gib(after)) (mlxCache=\(Self.gib(Int64(m.cacheMemory)))) afterClearCache=\(Self.gib(cleared)) | peakAdded=\(Self.gib(p.phys - base)) GiB = \(String(format: "%.2f", Double(p.phys - base) / Double(max(disk, 1)))) × safetensors | firstRun=\(first) peakFirstRun=\(Self.gib(peakFirst.phys))")
    }

    private func installed(_ slug: String) throws -> URL {
        let dir = ModelStore.shared.directory(forModelID: slug)
        try #require(FileManager.default.fileExists(atPath: dir.path), "\(slug) is not installed")
        return dir
    }

    @Test(.enabled(if: RealSizesMeasureTests.only == "dots"))
    func dotsOCR() async throws {
        let dir = try installed("mlx-community--dots.ocr-4bit")
        let image = Self.sample()
        try await measure("dots.ocr-4bit (OCR, own engine)", directory: dir) {
            let text = try await DotsOCREngine.generate(image: image, modelDir: dir, maxTokens: 256)
            print("SC2C dots answer chars=\(text.count)")
        }
    }

    @Test(.enabled(if: RealSizesMeasureTests.only == "vlm"))
    func vlm() async throws {
        let dir = try installed("mlx-community--gemma-3-4b-it-4bit")
        let image = Self.sample()
        let pool = ModelContainerPool(budgetBytes: 64 << 30) { dir, arch in try await VLMEngine.loadContainer(directory: dir, architecture: arch) }
        try await measure("gemma-3-4b-it-4bit (VLM pool, resident between calls)", directory: dir) {
            let text = try await VLMEngine.generate(prompt: "Describe this image.", image: image, modelDir: dir,
                                                    maxTokens: 128, pool: pool)
            print("SC2C vlm answer chars=\(text.count)")
        }
    }

    @Test(.enabled(if: RealSizesMeasureTests.only == "voxtral"))
    func voxtral() async throws {
        let dir = try installed("mlx-community--Voxtral-Mini-4B-Realtime-2602-4bit")
        let samples = (0..<(16_000 * 6)).map { Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) * 0.2 }
        try await measure("Voxtral-Mini-4B (ASR, loads per call)", directory: dir) {
            let text = try await VoxtralEngine.transcribe(samples, modelDirectory: dir)
            print("SC2C voxtral transcript chars=\(text.count)")
        }
    }

    @Test(.enabled(if: RealSizesMeasureTests.only == "laya"))
    func laya() async throws {
        let dir = try installed("aac6fef--laya-mlx")
        let definition = LayaQuestionDefinition(
            type: .choice, instructions: .string("What does the customer want?"),
            criteria: .object([LayaJSONField(key: "refund", value: .string("wants money back")),
                               LayaJSONField(key: "replacement", value: .string("wants a new item"))]))
        let question = try definition.resolve()
        try await measure("laya (decision, resident engine)", directory: dir) {
            let engine = try await LayaEngine.load(modelDirectory: dir)
            let answers = try engine.predict(state: "The package arrived broken and I want my money back.", questions: [question])
            print("SC2C laya choice=\(answers.first?.choiceLabel ?? "-")")
        }
    }

    // MARK: step 5 — the OCR → VLM sequence under an 8 GiB budget

    /// SC-2b's live check reached 11 GB on an 8 GiB cap with this pair. Here the real code paths — a Run window's
    /// `LeasingStage` on an `EngineCache` for dots.ocr, then `VLMEngine.generate` on a VLM pool — share one 8 GiB
    /// `MemoryBudget`. Bound: capacity + 0.5 GiB at every step and under a 25 ms sampler.
    /// `MLXUI_TEST_SC2C_BOOK=ramgb` books dots.ocr at its catalog `ramGB` (the pre-SC-2c figure) to show the difference.
    @Test(.enabled(if: RealSizesMeasureTests.only == "seq"))
    func ocrThenVlmOnEightGiB() async throws {
        let ocrDir = try installed("mlx-community--dots.ocr-4bit")
        let vlmDir = try installed("mlx-community--gemma-3-4b-it-4bit")
        let image = Self.sample()
        let capacity = Int64(8) << 30
        let slack = Int64(512) << 20
        let oldBooking = ProcessInfo.processInfo.environment["MLXUI_TEST_SC2C_BOOK"] == "ramgb"
        let budget = MemoryBudget(capacityBytes: capacity)
        let missing = URL(fileURLWithPath: "/nonexistent-sc2c")
        let cache = EngineCache(budget: budget, countsLLMWeights: false, modelDirectory: { _ in oldBooking ? missing : ocrDir })
        let pool = ModelContainerPool(budget: budget) { dir, arch in try await VLMEngine.loadContainer(directory: dir, architecture: arch) }
        let ocrModel = makeEntry(id: "mlx-community--dots.ocr-4bit", modelType: .ocr, ramGB: 1.3,
                                 hfModelId: "mlx-community/dots.ocr-4bit")
        let peak = Peak()
        let sampler = Task.detached {
            while !Task.isCancelled { peak.note(Self.phys(), 0, 0); try? await Task.sleep(for: .milliseconds(25)) }
        }
        func step(_ label: String) {
            let p = Self.phys()
            let phasePeak = peak.takePhasePeak()
            print("SC2C seq \(oldBooking ? "[old ramGB booking] " : "")\(label): peak-in-step=\(Self.gib(phasePeak)) ledger=\(Self.gib(budget.usedBytes)) GiB in-use=\(budget.inUseCount) cached-stages=\(cache.count) phys=\(Self.gib(p)) GiB over-capacity=\(Self.gib(p - capacity)) GiB")
            #expect(p <= capacity + slack, "\(label): real memory \(Self.gib(p)) GiB exceeds capacity + 0.5 GiB")
        }
        step("0 before")
        let ocr = LeasingStage(model: ocrModel, config: .default,
                               probe: try DotsOCRSDK().makeStage(for: ocrModel, config: .default), cache: cache)
        _ = try await ocr.run(.image(ImageMedia(cgImage: image))) { _ in }
        step("1 OCR run finished (window idle, engine cached)")
        let text = try await VLMEngine.generate(prompt: "Describe this image.", image: image, modelDir: vlmDir,
                                                maxTokens: 128, pool: pool)
        print("SC2C seq vlm answer chars=\(text.count)")
        step("2 VLM run finished")
        try await Task.sleep(for: .seconds(1))
        step("3 +1 s (cache given back)")
        _ = try await ocr.run(.image(ImageMedia(cgImage: image))) { _ in }
        step("4 OCR again")
        sampler.cancel()
        print("SC2C seq PEAK phys over the whole sequence = \(Self.gib(peak.value.phys)) GiB (capacity 8 GiB)")
        #expect(peak.value.phys <= capacity + slack)
    }
}
