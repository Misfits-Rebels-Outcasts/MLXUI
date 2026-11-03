import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// MC2-4 — the TranslateGemma 4B engine. One request in, one translation out.
///
/// The pieces that matter, each a trap the issue calls out:
///  * **The request is structured.** `UserInput(prompt: .messages(…))` with the template's one-item content list
///    (Trap 2); both language codes are validated against the table before rendering, because the template's lookup is
///    silent on a miss.
///  * **It stops at the end of the turn.** The installed `config.json`/`generation_config.json` declare no EOS ids, and
///    a folder load does not inherit mlx-swift-lm's registered TranslateGemma preset, so the default stop set is just
///    `<eos>` (1) and the model would run past `<end_of_turn>` (106). The engine puts 106 into the container's stop set
///    before generating and, belt and braces, never appends it to the output ids (Trap 1).
///  * **It counts the rendered prompt** with the model's own tokenizer and refuses over 2,048 tokens before any
///    generation (decision 8).
///  * **It decodes greedily** — temperature 0 is mlx-swift-lm's `ArgMaxSampler`, so the same request gives the same
///    translation (decision 7).
///  * **It leases the shared LLM pool** rather than loading weights itself, so a Translate row and a chat row share one
///    ledger and one resident container (Trap 8).
///
/// The implementation deliberately does **not** use `LLMRegistry.translategemma_4b_it_4bit`'s preset: the app loads only
/// files the installer downloaded (the A2 pattern), and `LLMModelFactory.loadContainer(from:)` — the shared pool's
/// loader — cannot carry a preset's `extraEOSTokens`. Routing around it with a second pool would fork the LLM weight
/// ledger; the post-load `configuration` update below is the smaller change. Implementer's call, pending owner
/// confirmation.
nonisolated enum TranslateGemmaEngine {

    /// The checkpoint's own turn markers. `<eos>` is the tokenizer's EOS (1) and always in the stop set;
    /// `<end_of_turn>` (106) must be added — the files name neither (Trap 1).
    static let eosTokenId = 1
    static let endOfTurnTokenId = 106
    static let stopTokenIds: Set<Int> = [eosTokenId, endOfTurnTokenId]

    /// One translation request: the text, and the two language codes from the pinned table.
    struct Request: Sendable, Equatable, Hashable {
        let text: String
        let sourceCode: String
        let targetCode: String

        init(text: String, sourceCode: String, targetCode: String) {
            self.text = text
            self.sourceCode = sourceCode
            self.targetCode = targetCode
        }
    }

    /// What a run produced. `cancelled` is a Stop, not a failure: the caller shows no translation.
    struct Outcome: Sendable, Equatable {
        let text: String
        let promptTokens: Int
        let generatedTokens: Int
        let cancelled: Bool

        static let cancelled = Outcome(text: "", promptTokens: 0, generatedTokens: 0, cancelled: true)
    }

    /// Where `InstallManager` places the model's files.
    nonisolated static func modelDirectory(for modelID: String) -> URL {
        ModelStore.shared.directory(forModelID: modelID)
    }

    /// Everything that can be refused before a single weight is loaded: an empty text, a code outside the table, or a
    /// missing install. The over-limit check needs the tokenizer, so it happens inside `translate` after `prepare`.
    static func validate(_ request: Request) throws {
        let trimmed = request.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw TranslateGemmaError.emptyText }
        _ = try TranslateGemmaPrompt.validatedCodes(sourceCode: request.sourceCode,
                                                    targetCode: request.targetCode)
    }

    /// Decision 8's limit, checked on the **rendered** prompt's token count (the template's own instruction text
    /// counts too).
    static func requireWithinLimit(promptTokens: Int, limit: Int = TranslateGemmaError.promptTokenLimit) throws {
        guard promptTokens <= limit else { throw TranslateGemmaError.promptTooLong(tokens: promptTokens) }
    }

    /// Translate one request. `onToken` sees each generated token id as it arrives (the gated Stop test uses the first
    /// one as its event; the Run screen passes nil). Cancellation of the calling task stops generation within a token.
    static func translate(
        _ request: Request,
        modelDirectory: URL,
        maxOutputTokens: Int = TranslateGemmaLanguages.maxOutputTokens,
        footprintBytes: Int64? = nil,
        holder: MemoryBudget.HolderKind = .flow,
        onToken: (@Sendable (Int) -> Void)? = nil,
        pool: ModelContainerPool = .shared
    ) async throws -> Outcome {
        try validate(request)
        guard FileManager.default.fileExists(atPath: modelDirectory.path) else {
            throw StageError.modelNotInstalled(id: modelDirectory.lastPathComponent)
        }
        do {
            let lease = try await pool.lease(directory: modelDirectory, footprintBytes: footprintBytes, holder: holder)
            defer { lease.release() }

            // Trap 1: the install names no EOS ids, so put `<end_of_turn>` into this container's stop set. The stop set is
            // read from `context.configuration` at generation time, and this container is only ever used by this engine.
            await lease.container.update { context in
                context.configuration.extraEOSTokens.insert("<end_of_turn>")
                context.configuration.eosTokenIds.insert(endOfTurnTokenId)
            }

            return try await lease.container.perform { context in
                let messages = try TranslateGemmaPrompt.messages(sourceCode: request.sourceCode,
                                                                 targetCode: request.targetCode,
                                                                 text: request.text)
                let input = try await context.processor.prepare(input: UserInput(prompt: .messages(messages)))
                let promptTokens = input.text.tokens.size
                // Trap 2's silent failure: a missing template renders nothing, and the processor falls back to joining
                // string contents, which are arrays here — a one-token prompt. Refuse rather than translate nothing.
                guard promptTokens > 1 else { throw TranslateGemmaError.unreadableTemplate }
                try requireWithinLimit(promptTokens: promptTokens)

                let stream = try MLXLMCommon.generateTokens(
                    input: input,
                    parameters: GenerateParameters(maxTokens: maxOutputTokens, temperature: 0),   // decision 7
                    context: context)
                var tokenIds: [Int] = []
                for await event in stream {
                    guard case .token(let id) = event else { continue }
                    onToken?(id)
                    guard !Self.stopTokenIds.contains(id) else { continue }   // Trap 1: never decode a turn marker
                    tokenIds.append(id)
                }
                let cancelled = Task.isCancelled
                let text = context.tokenizer.decode(tokenIds: tokenIds, skipSpecialTokens: true)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if cancelled { return Outcome.cancelled }
                guard !text.isEmpty else { throw TranslateGemmaError.noTranslation }
                return Outcome(text: text, promptTokens: promptTokens, generatedTokens: tokenIds.count,
                               cancelled: false)
            }
        } catch let error as TranslateGemmaError {
            throw error
        } catch is CancellationError {
            return .cancelled
        } catch let error as StageError {
            throw error
        } catch let error as MemoryBudget.WaitError {
            throw error          // SC-2a-b: the sentence names the model and who holds it
        } catch {
            // A cancelled run is never reported as an engine failure (MP5-3g's rule).
            if Task.isCancelled { return .cancelled }
            throw StageError.engineFailure(stage: "Translation", underlying: error)
        }
    }
}
