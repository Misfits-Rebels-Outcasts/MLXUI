import CoreGraphics
import Foundation
import ImageIO
import MLX
import Testing
@testable import MLXUI

// MP7-W, smoke row 139 (AUTO), OPT-IN, real weights. Every check prints one `SMOKE|139|<check>|PASS|FAIL|INFO|<detail>` line (the format `model-smoke.sh` counts), the footprints print
// `FOOTPRINT|<label>|<base MiB>|<peak MiB>|<after MiB>|<detail>`, and the pictures for the owner print `REVIEW|139|<file>|<what to judge>`.
//
// Weights: the checkpoint folder is `MLXUI_SMOKE_WEIGHTS` (model-smoke.sh `--weights`), else `MLXUI_TEST_KLEIN_DIR`, else the Direct store
// (`~/Library/Application Support/AI Browser/models/mlx-community--FLUX.2-klein-4B-bf16`), READ ONLY: this suite never downloads and never writes there. Pictures and the flow cache go to
// `MLXUI_SMOKE_OUT` (model-smoke.sh) or `MLXUI_TEST_KLEIN_SMOKE_OUT`. Unsandboxed host, app CLOSED (`pgrep -fl MLXUI` empty), ONE process, footprint stop line 14 GB.
//
//   TEST_RUNNER_MLXUI_TEST_KLEIN_SMOKE_OUT=<dir> xcodebuild test -project MLXUI.xcodeproj -scheme MLXUI -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO \
//       CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO -only-testing:MLXUITests/Flux2KleinSmokeTests
//
// What it does not check: that a picture depicts its prompt (REVIEW), the SwiftUI window (row 140).

enum KleinSmoke {
    nonisolated static let env = ProcessInfo.processInfo.environment
    nonisolated static let repo = "mlx-community/FLUX.2-klein-4B-bf16"
    nonisolated static let seed: UInt64 = 7
    nonisolated static let prompt = "a red fox in a snowy forest at sunrise"
    nonisolated static let stopLineGB = 14.0

    nonisolated static let out: URL? = (env["MLXUI_SMOKE_OUT"] ?? env["MLXUI_TEST_KLEIN_SMOKE_OUT"]).flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
    nonisolated static let directory: URL? = {
        let url: URL
        if let w = env["MLXUI_SMOKE_WEIGHTS"], !w.isEmpty { url = URL(fileURLWithPath: w) }
        else if let d = env["MLXUI_TEST_KLEIN_DIR"], !d.isEmpty { url = URL(fileURLWithPath: d) }
        else { url = ModelStore.shared.directory(forHFModelID: repo) }
        let fm = FileManager.default
        let have = ["text_encoder/config.json", "tokenizer/tokenizer.json", "transformer/config.json", "vae/config.json", "vae/diffusion_pytorch_model.safetensors"]
            .allSatisfy { fm.fileExists(atPath: url.appendingPathComponent($0).path) }
        return have ? url : nil
    }()
    nonisolated static let enabled = directory != nil && out != nil
    nonisolated static let skip: Comment = "needs the installed Klein model (MLXUI_SMOKE_WEIGHTS / TEST_RUNNER_MLXUI_TEST_KLEIN_DIR / the Direct store) and TEST_RUNNER_MLXUI_TEST_KLEIN_SMOKE_OUT=<dir>, an unsandboxed host, the app closed"

    static func line(_ s: String) { print(s); fflush(stdout) }

    @discardableResult
    static func check(_ name: String, _ ok: Bool, _ detail: String, info: Bool = false, sourceLocation: SourceLocation = #_sourceLocation) -> Bool {
        line("SMOKE|139|\(name)|\(ok ? "PASS" : (info ? "INFO" : "FAIL"))|\(detail.replacingOccurrences(of: "|", with: "/").replacingOccurrences(of: "\n", with: " "))")
        if !info { #expect(ok, Comment(rawValue: "\(name): \(detail)"), sourceLocation: sourceLocation) }
        return ok
    }

    static func mib(_ bytes: UInt64) -> Double { Double(bytes) / 1_048_576 }

    /// Runs `body` with the footprint sampled every 20 ms; prints the FOOTPRINT line; returns the peak in decimal GB.
    @discardableResult
    static func measured<T>(_ label: String, detail: String, _ body: () async throws -> T) async rethrows -> (value: T, peakGB: Double) {
        Memory.clearCache()
        let base = KleinFootprint.now()
        let sampler = KleinFootprint(); sampler.start()
        let value: T
        do { value = try await body() } catch { _ = sampler.stop(); throw error }
        let peak = sampler.stop()
        Memory.clearCache()
        line(String(format: "FOOTPRINT|%@|%.0f|%.0f|%.0f|%@", label, mib(base), mib(peak), mib(KleinFootprint.now()), detail))
        return (value, KleinFootprint.gb(peak))
    }

    static func pixelStd(_ image: CGImage) -> Double {
        let rgba = KleinPNG.rgbaBytes(image)
        var sum = 0.0, sq = 0.0, n = 0.0
        var i = 0
        while i < rgba.count { for c in 0 ..< 3 { let v = Double(rgba[i + c]); sum += v; sq += v * v; n += 1 }; i += 4 }
        let mean = sum / n
        return (sq / n - mean * mean).squareRoot()
    }

    static func loadPNG(_ url: URL) -> CGImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    static func ensureOut() throws -> URL {
        let dir = try #require(out)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func catalogKlein() throws -> ModelEntry {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let all = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == repo })
    }

    static func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    /// A Klein stage over the CHECKPOINT folder (not `ModelStore.shared`), phases recorded.
    static func stage(width: Int, height: Int, quantize: Int, seed: UInt64?, steps: Int = 4, rowLabel: String = Flux2KleinStage.runWindowLabel,
                      onPhase: (@Sendable (Flux2KleinPhase) -> Void)? = nil) throws -> Flux2KleinStage {
        let dir = try #require(directory)
        let settings = Flux2KleinRunSettings(width: width, height: height, steps: steps, seed: seed, quantize: quantize)
        return Flux2KleinStage(id: "flux2klein.smoke", name: "FLUX.2 Klein 4B (smoke)", settings: settings, rowLabel: rowLabel, onPhase: onPhase) { observer in
            Flux2KleinEngine(stages: Flux2KleinInstalledStages(modelDirectory: dir), observer: observer)
        }
    }

    static func image(_ media: Media) throws -> CGImage {
        guard case let .image(i) = media else { throw StageError.kindMismatch(expected: .image, got: media.kind) }
        return i.cgImage
    }

    final class PhaseLog: @unchecked Sendable {
        private let lock = NSLock()
        private var phases: [Flux2KleinPhase] = []
        func add(_ p: Flux2KleinPhase) { lock.lock(); phases.append(p); lock.unlock() }
        var all: [Flux2KleinPhase] { lock.lock(); defer { lock.unlock() }; return phases }
        var denoiseSteps: [Int] { all.compactMap { if case let .denoising(step, _) = $0, step > 0 { return step } else { return nil } } }
    }

    // MARK: A one-row flow through the real executor

    static func executor(blobs: URL) throws -> RealExecutor {
        let klein = try catalogKlein()
        guard let dir = directory else { throw StageError.modelNotInstalled(id: repo) }
        return RealExecutor(
            workspace: FlowWorkspace(root: blobs), flowID: "klein-smoke", blobDirectory: blobs,
            makeModelStage: { model, config in
                // the production SDK decides the settings (and refuses what the row can't have); only the weights folder is the checkpoint's
                guard let base = try Flux2KleinSDK().makeStage(for: model, config: config) as? Flux2KleinStage else { throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind) }
                return Flux2KleinStage(id: base.id, name: base.name, settings: base.settings, rowLabel: base.rowLabel) { observer in
                    Flux2KleinEngine(stages: Flux2KleinInstalledStages(modelDirectory: dir), observer: observer)
                }
            },
            installedModelIDs: [klein.id], catalog: try catalog())
    }

    struct FlowResult { let events: [FlowInterpreter.PathEvent]; let seconds: Double }

    static func runFlow(_ text: String, executor: any FlowExecutor) async throws -> FlowResult {
        let doc = try CatParser.parse(text)
        let started = Date()
        let events = try await FlowInterpreter.run(doc, executor: executor, definitions: doc.definitions, presets: doc.presets)
        return FlowResult(events: events, seconds: Date().timeIntervalSince(started))
    }

    static func flowImage(_ result: FlowResult, row: String = "1") -> CGImage? {
        guard let output = result.events.first(where: { $0.kind == .rowCompleted && $0.path == row })?.output,
              let path = output.items.first?.path else { return nil }
        return loadPNG(path)
    }
}

@Suite(.enabled(if: KleinSmoke.enabled, KleinSmoke.skip), .serialized)
struct Flux2KleinSmokeTests {

    // MARK: Install (read only)

    @Test @MainActor func theInstalledFolderIsTheSelectorsFilesAndTheRightSize() throws {
        let dir = try #require(KleinSmoke.directory)
        let fm = FileManager.default
        let marker = fm.fileExists(atPath: dir.appendingPathComponent(".installed").path)
        let rootSafetensors = (try? fm.contentsOfDirectory(atPath: dir.path))?.contains { $0.hasSuffix(".safetensors") } ?? false
        var bytes: Int64 = 0
        var files = 0
        if let e = fm.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey]) {
            for case let url as URL in e {
                let v = try? url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
                if v?.isRegularFile == true { bytes += Int64(v?.fileSize ?? 0); files += 1 }
            }
        }
        let gb = Double(bytes) / 1e9
        KleinSmoke.line("INSTALL|\(dir.path)|\(files) files|\(String(format: "%.2f", gb)) GB|marker \(marker)")
        KleinSmoke.check("install: .installed marker", marker, dir.path, info: !marker)
        KleinSmoke.check("install: no root flux-2-klein-4b.safetensors", !rootSafetensors, "root has no .safetensors")
        KleinSmoke.check("install: on-disk size within 5% of 15.98 GB", abs(gb - 15.98) / 15.98 <= 0.05, String(format: "%.2f GB, %d files", gb, files))
    }

    // MARK: Run: size, steps, not blank, determinism (the Run screen's stage, on the checkpoint)

    @Test @MainActor func requestedSizeFourStepsNotBlankAndTheDefaultRunStaysUnderRamGB() async throws {
        let out = try KleinSmoke.ensureOut()
        let log = KleinSmoke.PhaseLog()
        let stage = try KleinSmoke.stage(width: 1024, height: 1024, quantize: 8, seed: KleinSmoke.seed, onPhase: { log.add($0) })
        let (media, peak) = try await KleinSmoke.measured("1024 x 1024 q8 4 steps (default run)", detail: "seed 7") {
            try await stage.run(.text(KleinSmoke.prompt)) { _ in }
        }
        let image = try KleinSmoke.image(media)
        try Klein3hReal.writePNG(image, to: out.appendingPathComponent("klein-1024-q8-seed7.png"))
        KleinSmoke.check("size: 1024 x 1024 honoured", image.width == 1024 && image.height == 1024, "\(image.width) x \(image.height)")
        KleinSmoke.check("steps: exactly 4 denoise steps", log.denoiseSteps == [1, 2, 3, 4], "\(log.denoiseSteps)")
        let std = KleinSmoke.pixelStd(image)
        KleinSmoke.check("not blank: pixel std > 8", std > 8, String(format: "std %.1f", std))
        KleinSmoke.check("staged load: peak at or under ramGB 6.05", peak <= 6.05, String(format: "%.2f GB", peak))
        try #require(peak < KleinSmoke.stopLineGB, "STOP LINE")
        KleinSmoke.line("REVIEW|139|klein-1024-q8-seed7.png|does it depict '\(KleinSmoke.prompt)' (a red fox in a snowy forest at sunrise)")
    }

    @Test @MainActor func aNonSquareMultipleOfSixteenIsHonoured() async throws {
        let out = try KleinSmoke.ensureOut()
        let stage = try KleinSmoke.stage(width: 768, height: 512, quantize: 8, seed: KleinSmoke.seed)
        let image = try KleinSmoke.image(try await stage.run(.text(KleinSmoke.prompt)) { _ in })
        try Klein3hReal.writePNG(image, to: out.appendingPathComponent("klein-768x512-q8-seed7.png"))
        KleinSmoke.check("size: non-square 768 x 512 honoured", image.width == 768 && image.height == 512, "\(image.width) x \(image.height)")
        KleinSmoke.line("REVIEW|139|klein-768x512-q8-seed7.png|a wide 768 x 512 picture of the same prompt, not stretched or cropped")
    }

    @Test @MainActor func theSameSeedGivesAnIdenticalImageAndADifferentSeedADifferentOne() async throws {
        let out = try KleinSmoke.ensureOut()
        func render(_ seed: UInt64) async throws -> CGImage {
            try KleinSmoke.image(try await KleinSmoke.stage(width: 512, height: 512, quantize: 8, seed: seed).run(.text(KleinSmoke.prompt)) { _ in })
        }
        let a = try await render(7), b = try await render(7), c = try await render(8)
        try Klein3hReal.writePNG(c, to: out.appendingPathComponent("klein-512-q8-seed8.png"))
        KleinSmoke.check("determinism: same seed, identical image", KleinPNG.rgbaBytes(a) == KleinPNG.rgbaBytes(b), "two 512 x 512 runs, seed 7")
        KleinSmoke.check("determinism: different seed, different image", KleinPNG.rgbaBytes(a) != KleinPNG.rgbaBytes(c), "seed 7 vs seed 8")
    }

    // MARK: Flow: a one-row Generate Image through RealExecutor (quantize 4 and 8, cache hit, refusals)

    @Test @MainActor func aOneRowGenerateImageFlowRunsAtQuantize8AndTheSecondRunIsACacheHit() async throws {
        let out = try KleinSmoke.ensureOut()
        let blobs = out.appendingPathComponent("flow-blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let cacheRoot = out.appendingPathComponent("flow-cache", isDirectory: true)
        try? FileManager.default.removeItem(at: cacheRoot)
        let inner = try KleinSmoke.executor(blobs: blobs)
        let caching = CachingExecutor(inner: inner, store: FlowCacheStore(root: cacheRoot), cacheTier: "real", catalog: try KleinSmoke.catalog(), runSeed: 1,
                                      workspace: FlowWorkspace(root: blobs), flowID: "klein-smoke")
        let flow = "mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=7; width=512; height=512; quantize=8 \"\(KleinSmoke.prompt)\"\n"
        let (first, peak) = try await KleinSmoke.measured("flow row 512 x 512 q8", detail: "Generate Image on FLUX.2 Klein 4B") { try await KleinSmoke.runFlow(flow, executor: caching) }
        KleinSmoke.check("flow: row completes", first.events.contains { $0.kind == .rowCompleted && $0.path == "1" } && !first.events.contains { $0.kind == .rowFailed },
                         first.events.compactMap(\.error).joined(separator: "; "))
        let image = try #require(KleinSmoke.flowImage(first))
        KleinSmoke.check("flow: size honoured", image.width == 512 && image.height == 512, "\(image.width) x \(image.height)")
        KleinSmoke.check("flow: not blank", KleinSmoke.pixelStd(image) > 8, String(format: "std %.1f, %.1f s, peak %.2f GB", KleinSmoke.pixelStd(image), first.seconds, peak))
        try Klein3hReal.writePNG(image, to: out.appendingPathComponent("klein-flow-512-q8-seed7.png"))
        let second = try await KleinSmoke.runFlow(flow, executor: caching)
        KleinSmoke.check("flow: second run is a cache hit", second.events.contains { $0.kind == .cacheHit }, String(format: "%.2f s", second.seconds))
        let again = try #require(KleinSmoke.flowImage(second))
        KleinSmoke.check("flow: cached run is identical", KleinPNG.rgbaBytes(image) == KleinPNG.rgbaBytes(again), "same pixels")
    }

    @Test @MainActor func aFlowRowAtQuantize4RunsToo() async throws {
        let out = try KleinSmoke.ensureOut()
        let blobs = out.appendingPathComponent("flow-blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let exec = try KleinSmoke.executor(blobs: blobs)
        let flow = "mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=7; width=512; height=512; quantize=4 \"\(KleinSmoke.prompt)\"\n"
        let (result, peak) = try await KleinSmoke.measured("flow row 512 x 512 q4", detail: "quantize=4") { try await KleinSmoke.runFlow(flow, executor: exec) }
        let image = KleinSmoke.flowImage(result)
        KleinSmoke.check("flow: quantize=4 runs", image != nil && !result.events.contains { $0.kind == .rowFailed }, result.events.compactMap(\.error).joined(separator: "; "))
        if let image {
            try Klein3hReal.writePNG(image, to: out.appendingPathComponent("klein-flow-512-q4-seed7.png"))
            KleinSmoke.check("flow: quantize=4 not blank", KleinSmoke.pixelStd(image) > 8, String(format: "std %.1f, %.1f s, peak %.2f GB", KleinSmoke.pixelStd(image), result.seconds, peak))
        }
        KleinSmoke.line("REVIEW|139|klein-flow-512-q4-seed7.png|the 4-bit image model, same seed and prompt as the 8-bit flow picture: comparable subject, no heavy artifacts")
    }

    @Test @MainActor func settingsTheStageCannotHonourAreRefusedByNameAndNothingLoads() async throws {
        let out = try KleinSmoke.ensureOut()
        let exec = try KleinSmoke.executor(blobs: out.appendingPathComponent("flow-blobs", isDirectory: true))
        let cases: [(String, String)] = [
            ("negative=\"blurry\"", "negative"), ("lora=\"x.safetensors\"", "LoRA"), ("controlnet=\"canny\"", "ControlNet"), ("preview=true", "preview"),
            ("guidance=7", "guidance"), ("quantize=6", "quantize 4 or 8"),
        ]
        for (setting, word) in cases {
            let flow = "mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=7; width=512; height=512; \"a fox\"; \(setting)\n"   // lora=/controlnet= run greedy to `;`, so the prompt goes first
            let result = try await KleinSmoke.runFlow(flow, executor: exec)
            let message = result.events.first { $0.kind == .rowFailed }?.error ?? ""
            KleinSmoke.check("refused: \(setting)", message.contains(word) && message.contains("FLUX.2 Klein 4B") && result.seconds < 5, "\(message) (\(String(format: "%.2f", result.seconds)) s)")
        }
    }

    @Test @MainActor func aSizeAboveTheCapIsRefusedWithTheModelAsSubject() async throws {
        let out = try KleinSmoke.ensureOut()
        let exec = try KleinSmoke.executor(blobs: out.appendingPathComponent("flow-blobs", isDirectory: true))
        let flow = "mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=7; width=4096; height=4096 \"a fox\"\n"
        let result = try await KleinSmoke.runFlow(flow, executor: exec)
        let message = result.events.first { $0.kind == .rowFailed }?.error ?? ""
        let largest = Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: Flux2KleinMemoryPlan.defaultBudgetBytes)
        KleinSmoke.check("cap: refused with the model as subject, the size asked and the largest",
                         message.hasPrefix("FLUX.2 Klein can't make a 4096 \u{00D7} 4096 image at quantize 8 on this Mac") && message.contains("\(largest) \u{00D7} \(largest)"), message)
    }

    // MARK: The cap row: at (or near) the largest square this Mac can make

    @Test @MainActor func theLargestSquareThisMacCanMakeRunsAndItsFootprintMatchesTheBooking() async throws {
        let out = try KleinSmoke.ensureOut()
        let largest = Flux2KleinMemoryPlan.largestSquareSide(quantize: 8, budgetBytes: Flux2KleinMemoryPlan.defaultBudgetBytes)
        let side = min(largest, 1344)
        guard side >= 1344 else {
            KleinSmoke.check("cap row: 1344 x 1344 q8", false, "this Mac's cap is \(largest) x \(largest), below 1344: row not applicable here", info: true)
            return
        }
        let blobs = out.appendingPathComponent("flow-blobs", isDirectory: true)
        try FileManager.default.createDirectory(at: blobs, withIntermediateDirectories: true)
        let exec = try KleinSmoke.executor(blobs: blobs)
        let flow = "mlxflow 0.8\n1. Generate Image   FLUX.2 Klein 4B; seed=7; width=1344; height=1344; quantize=8 \"\(KleinSmoke.prompt)\"\n"
        let (result, peak) = try await KleinSmoke.measured("flow row 1344 x 1344 q8 (cap row)", detail: "largest square on a 19 GB Mac") { try await KleinSmoke.runFlow(flow, executor: exec) }
        let image = KleinSmoke.flowImage(result)
        KleinSmoke.check("cap row: 1344 x 1344 q8 completes", image?.width == 1344 && image?.height == 1344, result.events.compactMap(\.error).joined(separator: "; "))
        let booked = Double(Flux2KleinBooking.bytes(width: 1344, height: 1344, quantize: 8)) / 1e9
        KleinSmoke.check("cap row: footprint under the booked estimate and the stop line", peak <= booked && peak < KleinSmoke.stopLineGB,
                         String(format: "peak %.2f GB (measured 8.25 GB on 2026-11-01), booked %.2f GB, %.1f s", peak, booked, result.seconds))
        if let image { try Klein3hReal.writePNG(image, to: out.appendingPathComponent("klein-flow-1344-q8-seed7.png")) }
        KleinSmoke.line("REVIEW|139|klein-flow-1344-q8-seed7.png|1344 x 1344 at the cap: a coherent picture of the prompt")
    }
}
