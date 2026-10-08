import Foundation

/// `chat.completion.chunk` framing for `ServePipeline.stream` (S3-2a). The first frame of a stream is
/// the `role: "assistant"` chunk, written once before whichever event arrives first; the last is
/// `data: [DONE]`. Frames are exactly the ones the route wrote before the pipeline was extracted.
nonisolated struct OpenAIChatStreamEncoder: ServeStreamEncoder {
    let id: String
    let created: Int
    let model: String
    let includeUsage: Bool
    private var wroteRole = false

    init(id: String, created: Int, model: String, includeUsage: Bool) {
        self.id = id
        self.created = created
        self.model = model
        self.includeUsage = includeUsage
    }

    var keepAlive: [UInt8] { OpenAIChatDialect.keepAlive }

    mutating func frames(for event: ServeEvent) -> [[UInt8]] {
        var out = ensureRole()
        switch event {
        case .textDelta(let piece):
            out.append(OpenAIChatDialect.sse(OpenAIChatDialect.contentChunk(
                id: id, created: created, model: model, text: piece, includeUsage: includeUsage)))
        case .toolCallDelta(let delta):
            out.append(OpenAIChatDialect.sse(OpenAIChatDialect.toolCallChunk(
                id: id, created: created, model: model, call: delta, includeUsage: includeUsage)))
        case .reasoningDelta(let piece):
            out.append(OpenAIChatDialect.sse(OpenAIChatDialect.reasoningChunk(
                id: id, created: created, model: model, text: piece, includeUsage: includeUsage)))
        case .usage, .finish, .error:
            break
        }
        return out
    }

    mutating func completion(finish: ServeFinishReason, usage: ServeUsage?) -> [[UInt8]] {
        var out = ensureRole()
        out.append(OpenAIChatDialect.sse(OpenAIChatDialect.finishChunk(
            id: id, created: created, model: model, finish: finish, includeUsage: includeUsage)))
        if includeUsage, let usage {
            out.append(OpenAIChatDialect.sse(OpenAIChatDialect.usageChunk(
                id: id, created: created, model: model, usage: usage)))
        }
        out.append(OpenAIChatDialect.done)
        return out
    }

    func failure(_ error: ServeError) -> [[UInt8]] {
        [OpenAIChatDialect.sse(OpenAIChatDialect.streamErrorChunk(error)), OpenAIChatDialect.done]
    }

    private mutating func ensureRole() -> [[UInt8]] {
        guard !wroteRole else { return [] }
        wroteRole = true
        return [OpenAIChatDialect.sse(OpenAIChatDialect.roleChunk(
            id: id, created: created, model: model, includeUsage: includeUsage))]
    }
}
