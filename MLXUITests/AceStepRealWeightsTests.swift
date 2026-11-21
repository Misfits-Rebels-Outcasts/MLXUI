import Testing
import Foundation
import MLX
@testable import MLXUI

/// MC5-3c: the real-weights load of every group the port reads, with memory numbers. Owner-run, skipped by default.
/// Opt in with `MLXUI_TEST_ACESTEP=1` (the host sees it as `TEST_RUNNER_MLXUI_TEST_ACESTEP=1`); the folder defaults to the
/// lead's oracle copy of the install layout and is overridden by `MLXUI_TEST_ACESTEP_DIR`. Nothing is downloaded or written.
/// Run one component at a time for clean per-group numbers (the suite is serialized); each prints one line:
/// `ACESTEP-LOAD component=<name> bytes=<n> mlxActiveMB=<n> physFootprintMB=<n> peakPhysMB=<n> mlxPeakMB=<n>`.
enum AceStepRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var optedIn: Bool {
        (env["MLXUI_TEST_ACESTEP"] ?? env["TEST_RUNNER_MLXUI_TEST_ACESTEP"]) == "1"
    }

    nonisolated static var directory: URL? {
        let raw = env["MLXUI_TEST_ACESTEP_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_ACESTEP_DIR"]
            ?? "/Volumes/Ovid/Users/chinpinboo/Documents/PipelineStudio/RSI/notes/modelcandidates/mc5/oracle/Ace-Step1.5"
        let url = URL(fileURLWithPath: raw)
        let fm = FileManager.default
        for rel in [AceStepWeights.turboPath, AceStepWeights.qwenPath, AceStepWeights.vaePath]
        where !fm.fileExists(atPath: url.appendingPathComponent(rel).path) { return nil }
        return url
    }

    nonisolated static var canRun: Bool { optedIn && directory != nil }
}

@Suite(.serialized, .enabled(if: AceStepRealLoad.canRun, "needs MLXUI_TEST_ACESTEP=1 and an ACE-Step model folder"))
struct AceStepRealWeightsTests {

    private static let mib = 1_048_576.0

    /// Loads one group under a footprint sampler and prints the greppable line. The result stays alive until after the print.
    private func measure<T>(_ component: String, bytes: (T) -> Int, _ load: () throws -> T) throws -> T {
        Memory.clearCache()
        Memory.peakMemory = 0
        let tracker = FootprintTracker()
        tracker.note(FootprintTracker.phys())
        let sampler = Task.detached {
            while !Task.isCancelled {
                tracker.note(FootprintTracker.phys())
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        let result = try load()
        tracker.note(FootprintTracker.phys())
        sampler.cancel()
        let snap = Memory.snapshot()
        print(String(format: "ACESTEP-LOAD component=%@ bytes=%ld mlxActiveMB=%.0f physFootprintMB=%.0f peakPhysMB=%.0f mlxPeakMB=%.0f",
                     component, bytes(result), Double(snap.activeMemory) / Self.mib,
                     Double(FootprintTracker.phys()) / Self.mib, Double(tracker.runPeak) / Self.mib,
                     Double(snap.peakMemory) / Self.mib))
        return result
    }

    private func total(_ d: [String: MLXArray]) -> Int { d.values.reduce(0) { $0 + $1.nbytes } }

    private func finiteSample(_ a: MLXArray) -> Bool {
        let flat = a.asType(.float32).reshaped(-1)
        let head = flat[0 ..< min(4096, flat.size)]
        return abs(head).max().item(Float.self).isFinite
    }

    @Test func ditDecoder() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let dit = try measure("dit-decoder", bytes: total) { try AceStepWeights.loadDecoder(directory: dir) }
        #expect(dit.count == 476, "476 decoder tensors, none dropped (no rotary buffers in the file)")
        #expect(!dit.keys.contains { $0.hasPrefix("decoder.") || $0.contains("proj_in.1.") || $0.contains("rotary_emb") })
        #expect(dit["proj_in.weight"]?.shape == [2048, 2, 192])
        #expect(dit["proj_out.weight"]?.shape == [64, 2, 2048])
        #expect(dit.values.allSatisfy { $0.dtype == .bfloat16 }, "bf16 stays bf16 (R3)")
        #expect(dit.values.allSatisfy { finiteSample($0) })
        // ~3.15 GB (readme F-MC5).
        #expect(abs(Double(total(dit)) / 1e9 - 3.151) < 0.01)
    }

    @Test func conditionEncoderAndNullEmbedding() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let enc = try measure("condition-encoder", bytes: total) { try AceStepWeights.loadConditionEncoder(directory: dir) }
        eval(Array(enc.values))
        #expect(enc.count == 140)
        #expect(enc.keys.allSatisfy { $0.hasPrefix("encoder.") })
        #expect(enc.values.allSatisfy { $0.dtype == .bfloat16 && finiteSample($0) })
        #expect(abs(Double(total(enc)) / 1e9 - 1.217) < 0.01)
        let null = try AceStepWeights.loadNullConditionEmbedding(directory: dir)
        #expect(null.shape == [1, 1, 2048] && null.dtype == .bfloat16)
    }

    @Test func qwenEmbedTokensAlone() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let table = try measure("qwen-embed-tokens", bytes: { $0.nbytes }) { try AceStepWeights.loadQwenEmbedTokens(directory: dir) }
        #expect(table.shape == [151_669, 1024] && table.dtype == .bfloat16)
        #expect(abs(Double(table.nbytes) / 1e9 - 0.311) < 0.001)
        #expect(finiteSample(table))
    }

    @Test func qwenTrunkByPrefix() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let trunk = try measure("qwen-trunk", bytes: total) {
            try AceStepWeights.readQwen(directory: dir, prefixes: ["layers."], names: ["norm.weight"])
        }
        #expect(trunk.count == 309)
        #expect(!trunk.keys.contains(AceStepWeights.qwenEmbedTokensName))
        #expect(abs(Double(total(trunk)) / 1e9 - 0.881) < 0.01)
    }

    @Test func vaeDecoderOnly() throws {
        let dir = try #require(AceStepRealLoad.directory)
        let index = try SafetensorsIndex(url: dir.appendingPathComponent(AceStepWeights.vaePath))
        let rawNames = index.names(withPrefixes: AceStepWeights.vaeReadPrefixes)
        let pairs = rawNames.filter { $0.hasSuffix(".weight_g") }.count
        #expect(rawNames.count == 182, "the VAE file's decoder.* is 182 tensors; its 183 encoder.* tensors are not read")
        let vae = try measure("vae-decoder", bytes: total) { try AceStepWeights.loadVAEDecoder(directory: dir) }
        #expect(vae.count == 182 - pairs, "each weight_g + weight_v pair becomes one weight")
        #expect(vae.keys.allSatisfy { $0.hasPrefix("decoder.") })
        #expect(!vae.keys.contains { $0.hasSuffix(".weight_g") || $0.hasSuffix(".weight_v") })
        #expect(vae.values.allSatisfy { $0.dtype == .float32 && finiteSample($0) })
        #expect(vae["decoder.conv1.weight"]?.shape == [2048, 7, 64])
        #expect(vae["decoder.block.0.conv_t1.weight"]?.shape == [1024, 20, 2048])
        #expect(vae["decoder.snake1.alpha"]?.shape == [128])
    }

    /// The tokenizer/detokenizer groups sit in the same file and must stay untouched: reading the decoder alone costs no more
    /// than the decoder (phys footprint after the load, relative to before).
    @Test func readingOneGroupDoesNotPullTheWholeFile() throws {
        let dir = try #require(AceStepRealLoad.directory)
        Memory.clearCache()
        let activeBefore = Memory.snapshot().activeMemory
        let physBefore = FootprintTracker.phys()
        let enc = try AceStepWeights.loadConditionEncoder(directory: dir)
        eval(Array(enc.values))
        // Measure after the buffer cache is cleared: transient buffers freed by the load are not retained growth. The real guard is
        // MLX-active growth == the group's file bytes (a 1.217 GB group read, not the 4.79 GB file; lead's run: 1160 MiB active);
        // process phys is the noisier secondary check.
        Memory.clearCache()
        let activeGrown = Double(Memory.snapshot().activeMemory - activeBefore) / 1e9
        let physGrown = Double(FootprintTracker.phys() - physBefore) / 1e9
        #expect(abs(activeGrown - 1.217) < 0.05, "MLX active grew \(activeGrown) GB, expected the group's 1.217 GB")
        #expect(physGrown < 1.217 + 0.6, "condition encoder (1.22 GB) loaded without the 4.79 GB file: phys grew \(physGrown) GB")
        #expect(enc.count == 140)
    }
}
