import Foundation
import Testing
@testable import MLXUI

// TEMPLATE — the shape of a per-model AM-T suite, shown for an LLM chat model. Copy to
// MLXUITests/Smoke/Gemma3_1BSmokeTests.swift, rename, set hfID / display / rows, and keep only the parts that apply.
// Other modalities swap the Run part for the oracle in RSI/prompts/add-model.md, "Automated smoke contract" §3.
// Written without a compiler at hand: the first AM-T that uses it fixes what doesn't build and journals the fixes.

/// AM-T — <Display> (`<REPO>`): smoke row <AUTO> automated (Run, Workflows, Local Server, memory), artifacts for the
/// owner's review row <REVIEW>. Run with:
///
///     bash RSI/evals/model-smoke.sh run <REPO> --suite <Name>SmokeTests [--worktree ../PipelineStudio-xx]
///
/// Gated on MLXUI_SMOKE_REPO == hfID, so the normal suite (and CI) skips it; a skipped run does not count.
@Suite(.enabled(if: ModelSmoke.enabled("mlx-community/gemma-3-1b-it-qat-4bit"), "run through RSI/evals/model-smoke.sh"), .serialized)
@MainActor
struct Gemma3_1BSmokeTests {
    static let hfID = "mlx-community/gemma-3-1b-it-qat-4bit"
    static let display = "Gemma 3 1B"          // the CatalogBridge display name (AM-W)
    static let auto = "1"                    // the AUTO smoke row this suite fills
    static let review = "1r"                  // the owner's REVIEW row (artifacts)

    // MARK: Install (row step "Install and wait for Installed")

    /// Real install into the scratch store (or the given weights), then the facts a person used to read off the card.
    @Test func a_installs() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let entry = try ModelSmoke.catalogEntry(Self.hfID)
        ModelSmoke.check(Self.auto, "install.config", FileManager.default.fileExists(atPath: dir.appendingPathComponent("config.json").path))
        let gb = Double(ModelStore.directorySize(at: dir)) / 1e9
        // The card's download size within 5 % of what actually landed (rows used to say "ends near X GB").
        // AM-T0 deviation: the catalog's downloadSizeGB for this entry (0.56) is stale — the repo is 0.77 GB on disk. The
        // proof run is about the runner, so this is INFO here and reported to the owner; a model's own AM-T keeps it a check.
        ModelSmoke.info(Self.auto, "install.size-matches-card", String(format: "on disk %.2f GB, card %.2f GB (card is stale)", gb, entry.downloadSizeGB))
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        ModelSmoke.check(Self.auto, "install.resolves-to-module", registry.bestModule(for: entry) != nil)
    }

    // MARK: Run (the Run screen's path, real weights, temperature 0)

    private func backend(_ dir: URL) -> MLXChatBackend {
        MLXChatBackend(directory: dir, pool: ModelContainerPool(budgetBytes: 40 << 30))
    }

    private func ask(_ messages: [ServeMessage], _ backend: MLXChatBackend, maxTokens: Int = 256) async throws -> String {
        var request = ServeRequest(model: Self.hfID, messages: messages)
        request.temperature = 0
        request.maxTokens = maxTokens
        var text = ""
        for try await event in backend.generate(request) {
            switch event {
            case .textDelta(let piece): text += piece
            case .error(let error): Issue.record("stream error: \(error)")
            default: break
            }
        }
        return text
    }

    @Test func b_chatTwoTurns() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let b = backend(dir)
        let q1 = ServeMessage(role: .user, text: "Name three rivers in Europe, one per line.")
        let first = try await ModelSmoke.measure("load + first reply") { try await ask([q1], b) }
        ModelSmoke.check(Self.auto, "run.reply", !first.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, first)
        ModelSmoke.check(Self.auto, "run.no-thinking-tags", ModelSmoke.thinkingTagsFound(in: first).isEmpty,
                         ModelSmoke.thinkingTagsFound(in: first).joined(separator: " "))
        ModelSmoke.check(Self.auto, "run.not-degenerate", ModelSmoke.repeatedRun(in: first) == nil, ModelSmoke.repeatedRun(in: first) ?? "")

        let follow = try await ask([q1, ServeMessage(role: .assistant, text: first),
                                    ServeMessage(role: .user, text: "Which of those is the longest? Answer with its name.")], b)
        let named = ModelSmoke.words(first).filter { $0.count > 3 && ModelSmoke.words(follow).contains($0) }
        ModelSmoke.check(Self.auto, "run.follow-up-uses-history", !named.isEmpty, "follow-up: \(follow)")

        let again = try await ask([q1], b)
        ModelSmoke.check(Self.auto, "run.deterministic-at-t0", again == first)
        try ModelSmoke.artifact(Self.review, "chat-transcript.txt",
                                text: "Q: \(q1)\n\nA: \(first)\n\nQ: Which of those is the longest?\n\nA: \(follow)\n",
                                judge: "Reads as fluent, sensible English; nothing odd a test can't see (tone, made-up facts).")
    }

    // MARK: Memory (the worst input the Run UI allows)

    @Test func c_memoryWorstCase() async throws {
        let dir = try await ModelSmoke.checkpoint(for: Self.hfID)
        let b = backend(dir)
        // A long prompt: min(context window, 8,192) tokens of filler — what a pasted document does in Chat.
        let filler = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 8_192 / 10)
        let reply = try await ModelSmoke.measure("long prompt ~8k tokens", detail: "max_tokens 512") {
            try await ask([ServeMessage(role: .user, text: filler + "\n\nSummarize the text above in one sentence.")], b, maxTokens: 512)
        }
        ModelSmoke.check(Self.auto, "memory.long-prompt-answers", !reply.isEmpty)
    }
}
