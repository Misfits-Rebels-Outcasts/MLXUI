import Foundation
import Testing
@testable import MLXUI

/// MM3 — the real load of "MiMo V2.6 9B": the OptiQ per-layer 4/8-bit weights (134 layers at 8-bit,
/// 116 at 4-bit, `config.json` `quantization`) must load with no shape or `QuantizedLinear`
/// mismatch, and the output must be coherent. **Owner-run, gated on env** (a 7.1 GB checkpoint;
/// one real-model test at a time on the owner's Mac): pass the installed folder as
/// `TEST_RUNNER_MLXUI_TEST_MIMO_DIR=<folder>` and run the host unsandboxed so it can read it. The
/// exact command is in the MM3 journal. A skipped run does not count.
///
/// If the load throws, **stop and ask**: it would be a library gap in mlx-swift-lm 3.31.4, and no
/// package change may be made without owner approval. The first suspect for garbled (not failing)
/// output is `Qwen35.swift` `sanitize`'s norm shift (no `mtp.` weights ship in this checkpoint).
struct MiMoRealLoadTests {
    static let dir = ProcessInfo.processInfo.environment["MLXUI_TEST_MIMO_DIR"]

    private struct Answer {
        var text = ""
        var reasoning = ""
        var usage: ServeUsage?
    }

    private static func ask(_ prompt: String, thinking: Bool?, maxTokens: Int, backend: MLXChatBackend) async throws -> Answer {
        PromptCache.overrideForTesting = false          // MM-S owns the cache run; isolate the load
        defer { PromptCache.overrideForTesting = nil }
        var request = ServeRequest(model: "m", messages: [ServeMessage(role: .user, text: prompt)])
        request.temperature = 0
        request.maxTokens = maxTokens
        request.reasoning = thinking
        var answer = Answer()
        for try await event in backend.generate(request) {
            switch event {
            case .textDelta(let piece): answer.text += piece
            case .reasoningDelta(let piece): answer.reasoning += piece
            case .usage(let usage): answer.usage = usage
            case .error(let error): Issue.record("stream error: \(error)")
            default: break
            }
        }
        return answer
    }

    /// Coherent = real prose, not mojibake or a degenerate run (the signature of a wrong per-layer
    /// bit width or a bad norm shift is usually token soup, not an exception).
    private static func looksCoherent(_ text: String) -> Bool {
        let scalars = Array(text.unicodeScalars)
        guard scalars.count >= 1 else { return false }
        let plain = scalars.filter { $0.isASCII || $0.properties.isAlphabetic }.count
        return !text.contains("\u{FFFD}") && Double(plain) / Double(scalars.count) > 0.9
    }

    @Test(.enabled(if: dir != nil)) func theOptiQCheckpointLoadsAndAnswers() async throws {
        let url = URL(fileURLWithPath: try #require(Self.dir))
        let pool = ModelContainerPool(budgetBytes: 40 << 30)
        let backend = MLXChatBackend(directory: url, pool: pool)

        // Thinking off (the default; the template pre-closes an empty think block).
        let fact = try await Self.ask("What is the capital of France? Answer in one word.",
                                      thinking: nil, maxTokens: 48, backend: backend)
        print("MM3 factual: \(fact.text.debugDescription)")
        #expect(Self.looksCoherent(fact.text))
        #expect(fact.text.contains("Paris"))

        let sum = try await Self.ask("What is 17 times 23? Reply with just the number.",
                                     thinking: nil, maxTokens: 48, backend: backend)
        print("MM3 arithmetic: \(sum.text.debugDescription)")
        #expect(Self.looksCoherent(sum.text))
        #expect(sum.text.contains("391"))

        // Thinking on: the model opens `<think>`, so the reasoning must arrive separately.
        let thought = try await Self.ask("What is 17 times 23? Think it through, then answer.",
                                         thinking: true, maxTokens: 1024, backend: backend)
        print("MM3 thinking: reasoning \(thought.reasoning.count) chars, answer \(thought.text.debugDescription)")
        #expect(!thought.reasoning.isEmpty)
        #expect(Self.looksCoherent(thought.reasoning))
    }
}
