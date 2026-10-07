import Foundation
import Testing
@testable import MLXUI

// S2-6 (R23) — the context window a checkpoint declares, and the 400 that says a prompt doesn't fit.
//
// The per-family shapes below are the real ones (fetched 2026-10-07, journalled in
// `RSI/journal/2026-410-s2-6-precount.md`): the six plain decoders carry
// `max_position_embeddings` at the top level, the six multimodal wrappers (`qwen3_5` ×4,
// `mistral3` ×2) carry it only under `text_config`.

struct ModelContextWindowTests {

    private func config(_ json: String) -> Data { Data(json.utf8) }

    /// The catalog's shapes, one per family spelling.
    @Test func theCatalogShapesAreRead() {
        // Top level: qwen3 40,960; llama 131,072; gemma2 8,192; qwen2 32,768; gemma3_text 32,768;
        // gpt_oss 131,072.
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"qwen3","max_position_embeddings":40960}"#)) == 40960)
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"llama","max_position_embeddings":131072}"#)) == 131072)
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"gemma2","max_position_embeddings":8192}"#)) == 8192)
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"qwen2","max_position_embeddings":32768}"#)) == 32768)
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"gemma3_text","max_position_embeddings":32768}"#)) == 32768)
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"gpt_oss","max_position_embeddings":131072}"#)) == 131072)

        // Nested: the qwen3_5 and mistral3 wrappers carry it only in `text_config`.
        #expect(ModelContextWindow.limit(configData: config(
            #"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text","max_position_embeddings":262144}}"#)) == 262144)
        #expect(ModelContextWindow.limit(configData: config(
            #"{"model_type":"mistral3","text_config":{"model_type":"ministral3","max_position_embeddings":262144}}"#)) == 262144)
    }

    /// Nothing declares it → nothing to pre-count.
    @Test func aConfigWithoutAWindowIsNil() {
        #expect(ModelContextWindow.limit(configData: nil) == nil)
        #expect(ModelContextWindow.limit(configData: config("not json")) == nil)
        #expect(ModelContextWindow.limit(configData: config(#"{"model_type":"qwen3"}"#)) == nil)
        #expect(ModelContextWindow.limit(configData: config(#"{"text_config":{"hidden_size":8}}"#)) == nil)
        #expect(ModelContextWindow.limit(configData: config(#"{"max_position_embeddings":0}"#)) == nil)
        #expect(ModelContextWindow.limit(configData: config(#"{"max_position_embeddings":-1}"#)) == nil)
        #expect(ModelContextWindow.limit(configData: config(#"{"max_position_embeddings":"40960"}"#)) == nil)
    }

    /// Top level wins when a config somehow carries both (none of the 12 does).
    @Test func theTopLevelWinsWhenBothExist() {
        #expect(ModelContextWindow.limit(configData: config(
            #"{"max_position_embeddings":4096,"text_config":{"max_position_embeddings":8192}}"#)) == 4096)
    }

    /// The Connect sheet's line and the server's pre-count read the same parse (S2-5 / S2-6).
    @Test func theConnectSheetReadsTheSameNumber() {
        let json = config(#"{"model_type":"qwen3","max_position_embeddings":40960}"#)
        #expect(ConnectSnippets.contextLength(configJSON: json) == ModelContextWindow.limit(configData: json))
        #expect(ConnectSnippets.contextLength(configJSON: json) == 40960)
    }
}

struct ContextLengthMessageTests {

    /// Both numbers, in the shape the documented OpenAI error uses.
    @Test func theMessageCarriesTheLimitAndTheCount() {
        let message = ChatCompletionsRoute.contextLengthMessage(count: 41208, limit: 40960)
        #expect(message.contains("40960"))
        #expect(message.contains("41208"))
        #expect(message.contains("maximum context length"))
    }

    /// The wire error is a 400 `invalid_request_error` with the documented code, and the body
    /// carries exactly the three documented keys.
    @Test func theErrorBodyHasTheDocumentedShape() throws {
        let error = ServeError(status: 400, message: ChatCompletionsRoute.contextLengthMessage(count: 5, limit: 4),
                               code: "context_length_exceeded")
        let object = try #require(try JSONSerialization.jsonObject(with: error.body()) as? [String: Any])
        let inner = try #require(object["error"] as? [String: Any])
        #expect(inner["code"] as? String == "context_length_exceeded")
        #expect(inner["type"] as? String == "invalid_request_error")
        #expect((inner["message"] as? String)?.contains("5") == true)
    }
}
