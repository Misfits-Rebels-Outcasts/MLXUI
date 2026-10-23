import Foundation
import MLX
import MLXLMCommon

/// The Clef Flash runner: prompt → backbone hidden states → joint schema head → one typed answer per
/// question. Ports `clef_mlx.py`'s `ClefMLX` plumbing and `systemone` (`:434–539`).
///
/// An actor, like `CLMEngine`: it owns the loaded backbone and head, both single-instance, and the
/// reference answers **one record at a time** — no batching, no cache (readme F9). The head's
/// `lexical_lookup` seam (`:370`) is filled here from the retained output-embedding rows
/// (``ClefLexicalRows``), which is the one piece of the reference's `_lexical` that does not live in
/// `ClefHead`.
///
/// Errors are the reference's own: `ContextTooLong`'s two sentences and `systemone`'s validation
/// messages all come from ``ClefPrompt``/``ClefRequest`` unchanged, so a caller sees
/// "request needs N tokens; maximum is M" rather than a new string.
/// `@unchecked Sendable` box so the head can be captured by `ModelContainer.perform`'s `@Sendable`
/// closure — the same route `CLMHeadPair` takes for its two `CLMHead`s. A loaded `Module` is immutable
/// in practice, and every call arrives through ``ClefEngine`` (an actor), so the only sharing is
/// between the engine and one `perform` body at a time.
nonisolated struct ClefHeadBox: @unchecked Sendable {
    let head: ClefHead

    func callAsFunction(
        hidden: MLXArray, record: ClefEncodedRecord, lexicalLookup: (MLXArray) -> MLXArray
    ) -> [MLXArray] {
        head(hidden: hidden, record: record, lexicalLookup: lexicalLookup)
    }
}

actor ClefEngine {
    private let backbone: ClefBackbone
    private let head: ClefHead
    private let tokenizer: ClefMLXTokenizer
    /// `usage.input_tokens` (`:517`): the token count of every prompt this engine has encoded.
    private(set) var tokensSpent = 0

    private init(backbone: ClefBackbone, head: ClefHead, tokenizer: ClefMLXTokenizer) {
        self.backbone = backbone
        self.head = head
        self.tokenizer = tokenizer
    }

    /// Loads both halves: the joint head from `joint_head_config.json` + `joint_head.safetensors`, and
    /// the backbone through gate B1's identity-head swap. The tokenizer goes through the app's own
    /// loader — which is why CF-5a-FIX-1's `Qwen3_5Tokenizer` fallback is load-bearing here.
    static func load(modelDirectory: URL) async throws -> ClefEngine {
        let head = try ClefHead.load(
            configURL: modelDirectory.appendingPathComponent("joint_head_config.json"),
            weightsURL: modelDirectory.appendingPathComponent("joint_head.safetensors"))
        let backbone = try await ClefBackbone.load(
            modelDirectory: modelDirectory, hiddenSize: head.config.hiddenSize)
        let upstream = try await HFTokenizerLoader().load(from: modelDirectory)
        return ClefEngine(
            backbone: backbone, head: head, tokenizer: ClefMLXTokenizer(upstream: upstream))
    }

    /// `systemone` for one record: one `LayaAnswer` per question, in the request's order.
    ///
    /// `maxLength` is gate **F2**'s 8,192-token budget (F1's 16,384 was measured at a 15 GB peak);
    /// `stateTruncated` is the app-side flag
    /// (CF-A0 #3) and reaches ``ClefAnswer`` from the encoded record.
    func answer(
        state: ClefJSON,
        questions: [(id: String, type: ClefQuestionType, instructions: ClefJSON?, criteria: ClefJSON?)],
        maxLength: Int = ClefPrompt.defaultMaxLength
    ) async throws -> [(id: String, answer: LayaAnswer)] {
        let request = ClefRequest(
            state: state,
            questions: questions.map {
                ClefQuestion(
                    id: $0.id, type: $0.type, instructions: $0.instructions, criteria: $0.criteria)
            })
        let record = try ClefPrompt.encodeRecord(
            tokenizer: tokenizer, record: request, maxLength: maxLength)
        tokensSpent += record.inputIDs.count

        let head = ClefHeadBox(head: self.head)
        let probabilities = try await backbone.withHidden(ids: record.inputIDs) { hidden, rows in
            // The reference's `predict` softmaxes in float32 (`:489`, `:509`); the head itself is
            // bfloat16, so the cast is the same one `logits()` makes.
            let logits = head(hidden: hidden, record: record, lexicalLookup: { rows($0) })
            return try logits.map { try MLXErrorGuard.read(softmax($0.asType(.float32)), as: Float.self).map(Double.init) }      // WW2-3-8
        }

        var results: [(id: String, answer: LayaAnswer)] = []
        results.reserveCapacity(request.questions.count)
        for (index, question) in request.questions.enumerated() {
            guard index < probabilities.count else {
                throw ClefError.invalidQuestion("\(question.id): the head produced no logits")
            }
            // `systemone_answer` keys probabilities by option id (`:523`), in the prompt's encode
            // order; the reported order is the request's own (`ClefAnswer` handles that).
            var byOptionID: [String: Double] = [:]
            for (option, probability) in zip(record.questions[index].optionIDs, probabilities[index]) {
                byOptionID[option] = probability
            }
            results.append((
                question.id,
                try ClefAnswer.answer(
                    question: question, probabilities: byOptionID,
                    stateTruncated: record.stateTruncated)))
        }
        return results
    }
}
