import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXUI

/// BCC-3a (`RSI/DelegateBufferCacheBacklog.md`, owner ruling R22): a 20-turn scripted chat on the **real** `ModelRunner` path
/// (`prepare` -> `send` -> the streaming `AgentSession` the chat sheet uses; not the Local Server), to measure what MLX's
/// buffer cache does to a long in-app chat. Harness only: no production change, so running it before and after BCC-3b gives a
/// true baseline.
///
/// Runs only when BOTH hold:
///   1. `MLXUI_BCC3_SMOKE=1` is in the test process's environment (`TEST_RUNNER_MLXUI_BCC3_SMOKE=1` on the xcodebuild line), and
///   2. `mlx-community--Qwen3-8B-4bit` has its `.installed` marker in `ModelStore.shared` (the unsandboxed host reads the
///      Direct edition's store; the sandboxed host skips).
/// Otherwise it skips cleanly. Conversations are written to a temp folder, never the user's store.
///
/// Per turn: tokens (the reply re-encoded with the model's own tokenizer, so approximate), wall seconds from `send` to
/// `isRunning == false` (includes prefill of the whole history), tokens/s, `phys_footprint` at the end and its peak during the
/// turn, and MLX active / cache / peak / cacheLimit. No `clearCache` between turns (that is the point).
///
/// Notes: `ModelRunner` has one generation path -- chat and the agent tool loop are the same `AgentSession` (tools are on, as in the
/// UI; a tool approval, if one is requested, is declined so a turn cannot hang), so there is no separate agent path to script.
/// `maxTokens` is fixed at 2048 in production and temperature is 0.7, so the reply length is steered by the prompt
/// ("/no_think ... in about 80 words") and turns are comparable by tokens/s, not by identical text.
nonisolated enum BCC3Real {
    static let modelID = "mlx-community--Qwen3-8B-4bit"
    static var optedIn: Bool { ProcessInfo.processInfo.environment["MLXUI_BCC3_SMOKE"] == "1" }
    static var entry: ModelEntry? { BasicShowcaseRealLoad.catalog().first { $0.id == modelID } }
    static var ready: Bool {
        guard optedIn, let entry else { return false }
        return BasicShowcaseRealLoad.installedIDs([entry]).contains(entry.id)
    }
    static let topics = ["the Roman aqueducts", "how bread rises", "black holes", "the printing press", "coral reefs",
                         "the history of chess", "how vaccines work", "volcanoes", "the Silk Road", "honeybees",
                         "tides", "the invention of the bicycle", "glaciers", "how a refrigerator works", "the Great Library of Alexandria",
                         "migratory birds", "the periodic table", "monsoons", "the telephone", "tea"]
}

@Suite(.serialized)
@MainActor
struct ModelRunnerChatSmokeTests {
    static let row = "BCC3"

    private struct Turn {
        var n: Int
        var tokens: Int
        var seconds: Double
        var peakMiB: Double
        var afterMiB: Double
        var text: String
    }

    @Test(.enabled(if: BCC3Real.ready, "needs MLXUI_BCC3_SMOKE=1 and Qwen3-8B-4bit installed"))
    func twentyTurnChat() async throws {
        let entry = try #require(BCC3Real.entry)
        let dir = ModelStore.shared.directory(forModelID: entry.id)
        let tokenizer = try await HFTokenizerLoader().load(from: dir)
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("bcc3-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }

        let runner = ModelRunner()
        runner.prepare(for: entry, in: scratch)

        // Turn 0 (not counted): loads the weights. Then clear the pool once, so the next figure is the weights and nothing else.
        _ = try await turn(runner, entry, "/no_think Reply with the single word OK.", tokenizer: tokenizer, n: 0, report: false)
        MLX.Memory.clearCache()
        let weights = ModelSmoke.physMiB()
        let w = ModelSmoke.mlxMiB()
        ModelSmoke.info(Self.row, "weights-footprint", String(format: "%.0f MiB phys_footprint after load and one clearCache; mlx active %d cache %d MiB, cacheLimit %d MiB",
                                                             weights, w.active, w.cache, MLX.Memory.cacheLimit / 1_048_576))

        var turns: [Turn] = []
        for (i, topic) in BCC3Real.topics.enumerated() {
            let t = try await turn(runner, entry, "/no_think In about 80 words, tell me about \(topic).", tokenizer: tokenizer, n: i + 1, report: true)
            turns.append(t)
        }

        let after = ModelSmoke.physMiB()
        let last5 = turns.suffix(5)
        let meanTPS = last5.map { $0.tokens > 0 && $0.seconds > 0 ? Double($0.tokens) / $0.seconds : 0 }.reduce(0, +) / Double(max(last5.count, 1))
        let m = ModelSmoke.mlxMiB()
        ModelSmoke.info(Self.row, "footprint-after-turn-20", String(format: "%.0f MiB (weights %.0f, so +%.0f MiB); mlx active %d cache %d MiB; peak over all turns %.0f MiB",
                                                                    after, weights, after - weights, m.active, m.cache, turns.map(\.peakMiB).max() ?? 0))
        ModelSmoke.info(Self.row, "mean-tokens-per-second-last-5", String(format: "%.2f", meanTPS))
        ModelSmoke.info(Self.row, "mean-tokens-per-second-first-5", String(format: "%.2f",
            turns.prefix(5).map { $0.tokens > 0 && $0.seconds > 0 ? Double($0.tokens) / $0.seconds : 0 }.reduce(0, +) / Double(max(min(5, turns.count), 1))))
        ModelSmoke.check(Self.row, "20-turns-answered", turns.count == 20 && turns.allSatisfy { !$0.text.isEmpty },
                         "answered \(turns.filter { !$0.text.isEmpty }.count) of \(turns.count)")
    }

    private func turn(_ runner: ModelRunner, _ entry: ModelEntry, _ prompt: String, tokenizer: any MLXLMCommon.Tokenizer,
                      n: Int, report: Bool) async throws -> Turn {
        let sampler = ModelSmoke.PeakSampler()
        sampler.note(ModelSmoke.physMiB())
        let watcher = Task.detached {
            while !Task.isCancelled {
                sampler.note(ModelSmoke.physMiB())
                try? await Task.sleep(for: .milliseconds(20))
            }
        }
        MLX.Memory.peakMemory = 0
        let before = ModelSmoke.mlxMiB()
        let t0 = ContinuousClock.now
        let accepted = await runner.send(prompt, for: entry)
        #expect(accepted)
        let deadline = ContinuousClock.now + .seconds(600)
        while runner.isRunning, ContinuousClock.now < deadline {
            if let pending = runner.pendingApproval { pending.respond(false) }        // never hang on a tool question
            try await Task.sleep(for: .milliseconds(50))
        }
        let elapsed = (ContinuousClock.now - t0).components
        let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
        watcher.cancel()
        sampler.note(ModelSmoke.physMiB())
        var text = ""
        for case .message(let m) in runner.transcript.items.reversed() where m.role == .assistant { text = m.text; break }
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false).count
        let after = ModelSmoke.physMiB()
        let m = ModelSmoke.mlxMiB()
        if report {
            ModelSmoke.info(Self.row, String(format: "turn.%02d", n),
                            String(format: "tokens %d, %.1f s, %.2f tok/s; phys after %.0f MiB, peak %.0f MiB; mlx active %d>%d cache %d>%d peak %d MiB, cacheLimit %d MiB; error: %@",
                                    tokens, seconds, seconds > 0 ? Double(tokens) / seconds : 0, after, sampler.peak,
                                    before.active, m.active, before.cache, m.cache, m.peak, MLX.Memory.cacheLimit / 1_048_576, runner.errorMessage ?? "none"))
        }
        return Turn(n: n, tokens: tokens, seconds: seconds, peakMiB: sampler.peak, afterMiB: after, text: text)
    }
}
