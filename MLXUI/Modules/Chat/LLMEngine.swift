import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// The single place MLX LLM text generation happens. `LLMStage` (pipeline) calls it;
/// the chat sheet (`ModelRunner`) uses the same `MLXLMCommon.generate` call and is to
/// be unified onto this engine in the OP4/OP5 ChatSDK cycle. Returns the final text
/// (non-streaming) — stages thread a single `Media` value, not a token stream.
enum LLMEngine {
    /// Where `InstallManager` places a model's files.
    nonisolated static func modelDirectory(for modelID: String) -> URL {
        ModelStore.shared.directory(forModelID: modelID)
    }

    /// Load the model at `modelDir` and generate a reply to `prompt`.
    nonisolated static func generate(
        prompt: String,
        modelDir: URL,
        maxTokens: Int,
        temperature: Float = 0.7
    ) async throws -> String {
        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            throw StageError.modelNotInstalled(id: modelDir.lastPathComponent)
        }
        do {
            let loader = HFTokenizerLoader()
            let container = try await LLMModelFactory.shared.loadContainer(from: modelDir, using: loader)
            return try await container.perform { context in
                // DA-11: ask a hybrid-reasoning template (Qwen3's) to skip its visible
                // `<think>…</think>` preamble — none of the flow-row LLM primitives
                // (Generate / Summarize / the deciders / Extract Structured / Text to Table,
                // and the Summarize agent tool) want the reasoning trace, only the answer, and
                // a 32-token tag gate has no room for a think block at all (F004). Port of
                // `catflow-mlx` `engines/llm.py::_render_chat_prompt` (`enable_thinking=False`
                // + fallback). Unlike the Python's `apply_chat_template` kwarg — which raises
                // `TypeError` on a template that rejects an unknown keyword — swift-transformers
                // merges this into the Jinja render *context* as a plain variable, so an
                // unrecognised key is inert and there is no TypeError case; the `catch` is
                // belt-and-braces for a template that throws for some other reason.
                let input: LMInput
                do {
                    input = try await context.processor.prepare(
                        input: UserInput(prompt: prompt, additionalContext: Self.noThinkingContext))
                } catch {
                    input = try await context.processor.prepare(input: UserInput(prompt: prompt))
                }
                // TCP-1 (`RSI/backlog.md`, journal `2026-230`): consume **raw token ids**, not
                // `.chunk` text. The decoded-text path runs every chunk through mlx-swift-lm's
                // `ToolCallProcessor`, which drops the text before a `<` whenever the `<…` tail
                // partially matches `<tool_call>` — so a reply containing `<div><span>` loses
                // the `>` between them. `LLMStage` (flow rows) never emits tool calls; skip
                // that scanner and detokenize the whole id stream once.
                let stream = try MLXLMCommon.generateTokens(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxTokens, temperature: temperature),
                    context: context)
                var tokenIds: [Int] = []
                for await generation in stream {
                    if case .token(let id) = generation { tokenIds.append(id) }
                }
                return Self.stripThinkBlock(context.tokenizer.decode(tokenIds: tokenIds))
            }
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "LLM", underlying: error)
        }
    }

    /// The chat-template render context DA-11 passes on every flow-row LLM call: `enable_thinking
    /// = false`, the Jinja-context equivalent of the reference's `apply_chat_template(…,
    /// enable_thinking=False)`. A Qwen3 template reads it and pre-closes the think block
    /// (`<think>\n\n</think>`), so the model answers directly; a template that never references
    /// the key ignores it.
    nonisolated static let noThinkingContext: [String: any Sendable] = ["enable_thinking": false]

    /// Port of `engines/llm.py::_strip_think_block` — remove one leading, **closed**
    /// `<think>…</think>` block. The reference's own caveat holds: an *unclosed* block (thinking
    /// that ran past `maxTokens` because the template ignored the flag) is left as-is rather
    /// than guessed at. So this is a safety net for a template that emits a *short* trace, not
    /// a substitute for `enable_thinking=false` taking effect — with `maxTokens: 32` on the
    /// yes/no gate a leaked block almost never closes, and the caller (`DeciderFrame.extractTag`)
    /// then correctly reports F004 rather than reading a tag out of reasoning text.
    nonisolated static func stripThinkBlock(_ text: String) -> String {
        guard let regex = try? NSRegularExpression(
            pattern: #"\A\s*<think>.*?</think>\s*"#, options: [.dotMatchesLineSeparators]) else {
            return text
        }
        let full = NSRange(text.startIndex..., in: text)
        return regex.stringByReplacingMatches(in: text, range: full, withTemplate: "")
    }
}
