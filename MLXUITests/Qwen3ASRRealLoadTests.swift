import Foundation
import MLX
import MLXAudioSTT
import Testing
@testable import MLXUI

/// MP3-3 -- the real load of Qwen3-ASR 1.7B (8-bit) (KICKOFF rule 13b) and the main-actor probe (rule 17c). Owner-run,
/// gated on the weights (2.47 GB; one real model at a time). Skipped unless
/// `mlx-community--Qwen3-ASR-1.7B-8bit/.installed` exists under the directory this process resolves: `ModelStore`'s, or
/// `MLXUI_TEST_QWEN3ASR_DIR=<model folder>` (passed to the host as `TEST_RUNNER_MLXUI_TEST_QWEN3ASR_DIR`). Run the host
/// unsandboxed. The exact command is in the MP3-3 journal. A skipped run does not count.
enum Qwen3ASRRealLoad {
    nonisolated static let directory: URL? = {
        let url: URL
        if let override = ProcessInfo.processInfo.environment["MLXUI_TEST_QWEN3ASR_DIR"] {
            url = URL(fileURLWithPath: override)
        } else {
            url = ModelStore.shared.directory(forHFModelID: "mlx-community/Qwen3-ASR-1.7B-8bit")
        }
        let marker = url.appendingPathComponent(".installed")
        return FileManager.default.fileExists(atPath: marker.path) ? url : nil
    }()

    /// A model folder the test may write into (MP3 T2). The sandboxed test host cannot write `tokenizer.json` into the Direct
    /// store (Cocoa 513), so with `MLXUI_TEST_QWEN3ASR_DIR` set the folder is cloned (`copyItem` is an APFS clone on the same
    /// volume) into a fresh temp directory, and `tokenizer.json` is deleted from the copy so the engine's write (T5) is
    /// genuinely exercised even when the source already has one. The source is never touched. Without the override the
    /// `ModelStore` folder is used as is. Call `remove()` (a `defer`) when done.
    struct WorkingCopy {
        nonisolated let url: URL
        nonisolated let root: URL?

        nonisolated func remove() {
            if let root { try? FileManager.default.removeItem(at: root) }
        }
    }

    nonisolated static func workingCopy() throws -> WorkingCopy {
        let source = try #require(directory)
        guard ProcessInfo.processInfo.environment["MLXUI_TEST_QWEN3ASR_DIR"] != nil else {
            return WorkingCopy(url: source, root: nil)
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("qwen3asr-model-\(UUID().uuidString)")
        let copy = root.appendingPathComponent(source.lastPathComponent)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        do {
            try FileManager.default.copyItem(at: source, to: copy)
            let tokenizer = copy.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: tokenizer.path) { try FileManager.default.removeItem(at: tokenizer) }
        } catch {
            try? FileManager.default.removeItem(at: root)
            throw error
        }
        return WorkingCopy(url: copy, root: root)
    }

    /// The bundled English sample, as the 16 kHz mono buffer `ASRStage` hands the engine.
    nonisolated static func sampleSamples() throws -> [Float] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Samples/sample-audio.m4a")
        let buffer = try AudioFileReader.read(url)
        return try AudioResampler.resample(buffer, to: 16_000).samples
    }

    /// The sample is the `say`-generated voice memo from catflow-mlx's `01-SpokenSummary`
    /// ("Quick voice memo. Follow up with the design team ...").
    nonisolated static func readsTheMemo(_ transcript: String) -> Bool {
        transcript.lowercased().contains("memo")
    }
}

struct Qwen3ASRRealLoadTests {
    /// Loads through `Qwen3ASREngine`'s own path (`Qwen3ASRModel.fromModelDirectory` on the installed folder). Success
    /// with only the files `InstallManager` fetched proves the install covers the loader (T5, selection half). Afterwards
    /// the folder holds the generated `tokenizer.json` and still its `.installed` marker (T5, the write).
    @Test(.enabled(if: Qwen3ASRRealLoad.directory != nil))
    func theCheckpointLoadsAndTranscribesTheSample() async throws {
        let copy = try Qwen3ASRRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        if copy.root != nil {
            #expect(!FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path), "the copy starts without one")
        }
        let samples = try Qwen3ASRRealLoad.sampleSamples()
        let text = try await Qwen3ASREngine.transcribe(samples, modelDirectory: dir)
        print("QWEN3ASR transcript: \(text)")
        #expect(!text.isEmpty)
        #expect(Qwen3ASRRealLoad.readsTheMemo(text), "\(text)")
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path))
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent(".installed").path))
    }
}

/// The shape the Run UI gives the engines: awaited from a main-actor `Task`, inside `MLXErrorGuard.guardedRun`
/// (`ASRRunView`). The library's repetition-penalty gather (`Qwen3ASR.swift:1326`, `:1528` @ 5c4c12b) is not built at the
/// default penalty of 1.0, so this is the evidence that the whole path -- load, `generate`, `item()` reads -- needs no
/// `MLXAudioThread.run`: real output, twice, and no MLX error recorded by the process-wide fallback.
@MainActor
struct Qwen3ASRMainActorProbeTests {
    @Test(.enabled(if: Qwen3ASRRealLoad.directory != nil))
    func transcribeFromTheMainActorReturnsTheTranscript() async throws {
        let copy = try Qwen3ASRRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let samples = try Qwen3ASRRealLoad.sampleSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        for _ in 0..<2 {   // twice: the second call reuses nothing but must not differ
            let text = try await MLXErrorGuard.guardedRun {
                try await Qwen3ASREngine.transcribe(samples, modelDirectory: dir)
            }
            print("QWEN3ASRPROBE transcript: \(text)")
            #expect(!text.isEmpty)
            #expect(Qwen3ASRRealLoad.readsTheMemo(text), "\(text)")
        }
        #expect(MLXErrorGuard.lastFallbackMessage() == nil)
    }

    /// Rule 17's other half, on purpose: with a repetition penalty switched on, the library builds the mixed-subscript
    /// gather. This runs it from the main actor and only reports (`QWEN3ASRPENALTY ...`); it never fails on the outcome,
    /// because it is a probe of a path the app does not use. If it reports a stream error, the engine must keep the
    /// penalty at 1.0 (the unit test pins that); if it works, `MLXAudioThread` is not needed even for that path.
    @Test(.enabled(if: Qwen3ASRRealLoad.directory != nil && ProcessInfo.processInfo.environment["MLXUI_TEST_QWEN3ASR_PENALTY_PROBE"] == "1"))
    func penaltyPathFromTheMainActorIsReportedNotAsserted() async throws {
        let copy = try Qwen3ASRRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let samples = try Qwen3ASRRealLoad.sampleSamples()
        MLXErrorGuard.installFallback()
        MLXErrorGuard.resetFallbackMessage()
        let model = try await Qwen3ASRModel.fromModelDirectory(dir)
        let base = model.defaultGenerationParameters
        let params = STTGenerateParameters(
            maxTokens: base.maxTokens, temperature: base.temperature, topP: base.topP, topK: base.topK,
            verbose: base.verbose, language: base.language, chunkDuration: Qwen3ASREngine.chunkSeconds,
            minChunkDuration: base.minChunkDuration, repetitionPenalty: 1.1, repetitionContextSize: 32)
        do {
            let text = try await MLXErrorGuard.guardedRun {
                model.generate(audio: MLXArray(samples), generationParameters: params).text
            }
            print("QWEN3ASRPENALTY ok: \(text)")
        } catch {
            print("QWEN3ASRPENALTY threw: \(error)")
        }
        print("QWEN3ASRPENALTY fallback message: \(MLXErrorGuard.lastFallbackMessage() ?? "none")")
    }
}

/// Long-clip measurement (KICKOFF; the MP2 Parakeet precedent, owner rulings X, Y, Z of 2026-10-24). Owner-run, one run
/// at a time, nothing else loaded. Gated on the installed weights and `MLXUI_TEST_QWEN3ASR_MEASURE=1` (host env:
/// `TEST_RUNNER_...`). Inputs: `MLXUI_TEST_QWEN3ASR_LONG_CLIP=<audio file>` (default: the bundled sample),
/// `MLXUI_TEST_QWEN3ASR_CHUNK=<seconds>` (default `Qwen3ASREngine.chunkSeconds`; a test-only override, so 120 and the
/// library's own 1200 can be compared on one clip) and `MLXUI_TEST_QWEN3ASR_REPEAT=<n>` (concatenate the clip in memory).
/// The host is sandboxed, so nothing is written to disk: everything is printed (`QWEN3ASRMEASURE ...`). MLX's own counters
/// are read after the load, after `generate`, and after `Memory.clearCache()`, with a peak reset between.
struct Qwen3ASRLongClipMeasurement {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var clip: URL {
        if let path = env["MLXUI_TEST_QWEN3ASR_LONG_CLIP"] { return URL(fileURLWithPath: path) }
        return URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/Samples/sample-audio.m4a")
    }

    nonisolated static func line(_ label: String, _ s: Memory.Snapshot) -> String {
        func gib(_ b: Int) -> String { String(format: "%.3f GiB", Double(b) / 1_073_741_824) }
        return "QWEN3ASRMEASURE \(label): active \(s.activeMemory) B (\(gib(s.activeMemory))), cache \(s.cacheMemory) B (\(gib(s.cacheMemory))), peak \(s.peakMemory) B (\(gib(s.peakMemory)))"
    }

    nonisolated static func directPhase(dir: URL, audio: MLXArray, chunk: Float, seconds: Int) async throws -> String {
        let t0 = Date()
        let model = try await Qwen3ASRModel.fromModelDirectory(dir)
        let loadSeconds = Date().timeIntervalSince(t0)
        print(line("afterLoad", Memory.snapshot()) + String(format: ", %.1f s", loadSeconds))

        let base = model.defaultGenerationParameters
        let params = STTGenerateParameters(
            maxTokens: base.maxTokens, temperature: base.temperature, topP: base.topP, topK: base.topK,
            verbose: base.verbose, language: base.language, chunkDuration: chunk,
            minChunkDuration: base.minChunkDuration, repetitionPenalty: base.repetitionPenalty,
            repetitionContextSize: base.repetitionContextSize, kvBits: base.kvBits,
            kvGroupSize: base.kvGroupSize, quantizedKVStart: base.quantizedKVStart)
        Memory.peakMemory = 0   // the setter ignores the value and resets the peak: the load floor is separate
        let t1 = Date()
        let output = model.generate(audio: audio, generationParameters: params)
        let text = output.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let generateSeconds = Date().timeIntervalSince(t1)
        print(line("afterGenerate", Memory.snapshot()) + String(format: ", %.1f s", generateSeconds))
        Memory.clearCache()
        print(line("afterClearCache", Memory.snapshot()))
        print("QWEN3ASRMEASURE summary: clip \(clip.lastPathComponent), \(seconds) s of audio, chunk \(chunk) s, "
              + String(format: "load %.1f s, generate %.1f s", loadSeconds, generateSeconds)
              + ", \(text.split(separator: " ").count) words, language \(output.language ?? "nil"), "
              + "prompt tokens \(output.promptTokens), generated tokens \(output.generationTokens)")
        for (i, seg) in (output.segments ?? []).enumerated() {
            print("QWEN3ASRMEASURE seg \(i): " + seg.map { "\($0.key)=\($0.value)" }.sorted().joined(separator: " | "))
        }
        print("QWEN3ASRMEASURE transcript-begin")
        print(text)
        print("QWEN3ASRMEASURE transcript-end")
        return text
    }

    @Test(.enabled(if: Qwen3ASRRealLoad.directory != nil && env["MLXUI_TEST_QWEN3ASR_MEASURE"] == "1"))
    func measureLoadAndTranscribe() async throws {
        let copy = try Qwen3ASRRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let buffer = try AudioFileReader.read(Self.clip)
        let once = try AudioResampler.resample(buffer, to: 16_000).samples
        let repeats = Self.env["MLXUI_TEST_QWEN3ASR_REPEAT"].flatMap(Int.init) ?? 1
        let samples = (1..<max(repeats, 1)).reduce(once) { acc, _ in acc + once }
        let audio = MLXArray(samples)
        let chunk = Self.env["MLXUI_TEST_QWEN3ASR_CHUNK"].flatMap(Float.init) ?? Qwen3ASREngine.chunkSeconds

        eval(audio)
        Memory.clearCache()
        Memory.peakMemory = 0
        print(Self.line("start", Memory.snapshot()))
        let text = try await Self.directPhase(dir: dir, audio: audio, chunk: chunk, seconds: samples.count / 16_000)
        #expect(!text.isEmpty)

        // The engine's own path (it returns the cache itself, after the model is released): what the app keeps after one
        // transcription.
        if Self.env["MLXUI_TEST_QWEN3ASR_ENGINE_PHASE"] == "1" {
            Memory.clearCache()
            Memory.peakMemory = 0
            print(Self.line("engineStart", Memory.snapshot()))
            let t2 = Date()
            let engineText = try await Qwen3ASREngine.transcribe(samples, modelDirectory: dir)
            print(Self.line("engineAfterTranscribe", Memory.snapshot()) + String(format: ", %.1f s", Date().timeIntervalSince(t2)))
            print("QWEN3ASRMEASURE engineWords \(engineText.split(separator: " ").count)")
            #expect(!engineText.isEmpty)
        }
    }
}

/// MP3-W — a real two-row flow through the real executor (the MP2-W precedent): `Read Audio` of the bundled
/// sample, then `Transcribe` with the model named exactly as a `.cat` writes it. Owner-run, gated on the installed
/// weights (the same `MLXUI_TEST_QWEN3ASR_DIR` as the other real-load tests); a skipped run does not count.
/// Real: the parser, `FlowRunner`, `RealExecutor`, the bridge resolution, `Read Audio`, `ASRStage`, `Qwen3ASREngine`. Not
/// the SDK's `makeStage` (it reads `ModelStore.shared`, the test host's container): see the factory comment.
struct Qwen3ASRFlowRealRunTests {
    @Test(.enabled(if: Qwen3ASRRealLoad.directory != nil))
    func twoRowFlowTranscribesTheSampleWithQwen3ASR() async throws {
        let copy = try Qwen3ASRRealLoad.workingCopy()
        defer { copy.remove() }
        let dir = copy.url
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let catalogData = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let catalog = try JSONDecoder().decode(BrowserData.self, from: catalogData).domains.flatMap { $0.allModels }
        let qwen = try #require(catalog.first { $0.id == "mlx-community--Qwen3-ASR-1.7B-8bit" })

        let base = FileManager.default.temporaryDirectory.appendingPathComponent("qwen3asr-flow-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let workspace = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let flowDir = workspace.directory(for: "qwen3asr-flow")
        try FileManager.default.createDirectory(at: flowDir, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: repoRoot.appendingPathComponent("MLXUI/Resources/Samples/sample-audio.m4a"),
                                         to: flowDir.appendingPathComponent("sample-audio.m4a"))

        let registry = await MainActor.run { () -> ModelRegistry in
            let registry = ModelRegistry()
            for module in installedModules { module.register(into: registry) }
            return registry
        }
        let blobs = base.appendingPathComponent("blobs")
        let executor = RealExecutor(
            workspace: workspace, flowID: "qwen3asr-flow", blobDirectory: blobs,
            makeModelStage: { model, _ in
                // The registry must hand this card to the Qwen3-ASR module. `Qwen3ASRSDK.makeStage` reads
                // `ModelStore.shared`, which in this sandboxed test host is the container, not the owner's install,
                // so the stage itself is the engine's own path on the override directory.
                let moduleID = await MainActor.run { registry.bestModule(for: model)?.descriptor.id }
                guard moduleID == "qwen3-asr" else { throw StageError.unsupportedModel(id: model.id, kind: model.runnerKind) }
                return ASRStage(id: "qwen3-asr.\(model.id)", name: "Qwen3-ASR") { samples in
                    try await Qwen3ASREngine.transcribe(samples, modelDirectory: dir)
                }
            },
            installedModelIDs: [qwen.id], catalog: catalog)

        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Read Audio   sample-audio.m4a
        2. Transcribe   Qwen3-ASR 1.7B
        """)
        let row2 = try #require(doc.rows.last)
        let context = FlowRunner.RunContext(flowID: "qwen3asr-flow", workspace: workspace, blobDirectory: blobs, executor: executor)

        let started = Date()
        var output: String?
        var failure: String?
        for await event in FlowRunner().run(doc, context: context) {
            switch event {
            case .finished(let id, let asset) where id == row2.id: output = asset.items.first?.value
            case .failed(_, let error): failure = "\(error)"
            default: break
            }
        }
        let elapsed = Date().timeIntervalSince(started)
        print("QWEN3ASRFLOW elapsed \(String(format: "%.1f", elapsed)) s, failure: \(failure ?? "none")")
        print("QWEN3ASRFLOW output: \(output ?? "nil")")
        #expect(failure == nil)
        #expect(Qwen3ASRRealLoad.readsTheMemo(output ?? ""), "\(output ?? "nil")")
    }
}
