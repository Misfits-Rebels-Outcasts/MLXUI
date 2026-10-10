import Testing
import Foundation
@testable import MLXUI

// S3-5a — memory admission before prefill (R47).

private let model = "mlx-community/Qwen3-8B-4bit"

/// Qwen3-8B: 36 layers × 8 KV heads × 128 × 2 (K,V) × 2 bytes = 147,456 B/token.
private let qwen3Config = Data(#"{"model_type":"qwen3","num_hidden_layers":36,"num_attention_heads":32,"num_key_value_heads":8,"head_dim":128,"hidden_size":4096}"#.utf8)
/// No `head_dim`: hidden / heads = 128. 32 × 8 × 128 × 4 = 131,072.
private let llamaConfig = Data(#"{"model_type":"llama","num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096}"#.utf8)
/// A hybrid (Qwen3.5-9B-shaped): 24 linear + 8 full-attention layers; only the 8 grow with the prompt.
private func hybridConfig(linearTypes: Bool = true) -> Data {
    let types = (0..<32).map { ($0 + 1) % 4 == 0 ? "full_attention" : "linear_attention" }
        .map { "\"\($0)\"" }.joined(separator: ",")
    return Data(#"""
    {"model_type":"qwen3_5","text_config":{"num_hidden_layers":32,"num_attention_heads":16,"num_key_value_heads":4,"head_dim":256,
     "hidden_size":4096,\#(linearTypes ? "\"layer_types\":[\(types)]," : "")"linear_num_value_heads":32,"linear_num_key_heads":16,
     "linear_key_head_dim":128,"linear_value_head_dim":128,"linear_conv_kernel_dim":4}}
    """#.utf8)
}

struct ServeMemoryEstimatorTests {
    @Test func theKVCostPerTokenFromConfig() throws {
        let qwen = try #require(ServeMemoryAdmission.kvCost(configData: qwen3Config))
        #expect(qwen.perToken == 147_456 && qwen.fixed == 0)                      // 144 KiB, F10
        let llama = try #require(ServeMemoryAdmission.kvCost(configData: llamaConfig))
        #expect(llama.perToken == 131_072 && llama.fixed == 0)                    // head_dim from hidden / heads
        #expect(ServeMemoryAdmission.kvCost(configData: Data(#"{"model_type":"x"}"#.utf8)) == nil)
        #expect(ServeMemoryAdmission.kvCost(configData: nil) == nil)
    }

    /// Only the full-attention layers grow; the linear ones are a fixed state.
    @Test func aHybridCountsOnlyItsFullAttentionLayers() throws {
        let hybrid = try #require(ServeMemoryAdmission.kvCost(configData: hybridConfig()))
        #expect(hybrid.perToken == 8 * 4 * 256 * 2 * 2)                           // 8 layers, 32 KiB/token
        #expect(hybrid.fixed > 0)
        // Without `layer_types` every layer is an attention layer (the safe default), no fixed state.
        let plain = try #require(ServeMemoryAdmission.kvCost(configData: hybridConfig(linearTypes: false)))
        #expect(plain.perToken == 32 * 4 * 256 * 2 * 2 && plain.fixed == 0)
    }

    /// The same numbers `AgentReadiness.kvBytes` (the S2-4 badge) computes for the same config.
    @Test func agreesWithTheBadgesEstimateForTheSameConfig() throws {
        for config in [qwen3Config, llamaConfig, hybridConfig(), hybridConfig(linearTypes: false)] {
            let kv = try #require(ServeMemoryAdmission.kvCost(configData: config))
            #expect(AgentReadiness.kvBytes(configData: config, contextTokens: 32_768) == kv.perToken * 32_768 + kv.fixed)
        }
    }

    @Test func theProfileScalesKVByTheOverheadFactorAndNeedsWeights() throws {
        let profile = try #require(ServeMemoryAdmission.profile(configData: qwen3Config, weightsBytes: 4_600_000_000,
                                                                 roomBytes: 11_000_000_000))
        #expect(profile.bytesPerToken == 147_456 * ServeMemoryAdmission.overheadFactor)
        #expect(profile.needBytes(promptTokens: 1_000, replyTokens: 24) == 4_600_000_000 + Int64(1_024 * profile.bytesPerToken))
        #expect(ServeMemoryAdmission.profile(configData: qwen3Config, weightsBytes: 0, roomBytes: 1) == nil)   // no weights: exempt
        #expect(ServeMemoryAdmission.profile(configData: nil, weightsBytes: 1, roomBytes: 1) == nil)            // no config: exempt
    }

    @Test func theEstimatedTokenCountIsBytesOverThreeRoundedUp() {
        #expect(ServeMemoryAdmission.estimatedTokens(utf8Bytes: 0) == 0)
        #expect(ServeMemoryAdmission.estimatedTokens(utf8Bytes: 3) == 1)
        #expect(ServeMemoryAdmission.estimatedTokens(utf8Bytes: 4) == 2)
        #expect(ServeMemoryAdmission.estimatedTokens(utf8Bytes: 300_000) == 100_000)
    }

    /// The boundary: weights 1,000, 10 B/token, room for exactly (n + 1,024) tokens.
    @Test func refuseVersusAdmitAtTheBoundary() {
        func profile(room: Int64) -> ServeMemoryProfile {
            ServeMemoryProfile(weightsBytes: 1_000, fixedBytes: 0, bytesPerToken: 10, roomBytes: room)
        }
        let n = 5_000
        let exact = profile(room: 1_000 + Int64(n + ServeMemoryAdmission.minimumReplyTokens) * 10)
        #expect(ServeMemoryAdmission.decide(profile: exact, promptTokens: n) == .admit(maxReplyTokens: 1_024))
        #expect(ServeMemoryAdmission.decide(profile: exact, promptTokens: n + 1)
                == .refuse(needBytes: exact.needBytes(promptTokens: n + 1, replyTokens: 1_024), roomBytes: exact.roomBytes))
        // More room: the reply ceiling is what the room holds beyond the prompt.
        let roomy = profile(room: 1_000 + 20_000 * 10)
        #expect(ServeMemoryAdmission.decide(profile: roomy, promptTokens: n) == .admit(maxReplyTokens: 15_000))
        // Fixed (hybrid) state comes off the top.
        let hybrid = ServeMemoryProfile(weightsBytes: 1_000, fixedBytes: 500, bytesPerToken: 10, roomBytes: 1_500 + 20_000 * 10)
        #expect(ServeMemoryAdmission.decide(profile: hybrid, promptTokens: n) == .admit(maxReplyTokens: 15_000))
    }

    /// R50: factor 1.75. On this Mac (room 11.6 GiB, Qwen3-8B's 4.6 GB of weights) the boundary sits near 29.4k
    /// prompt tokens: 29k is admitted (it ran in ~136 s with no swap growth), 30k is refused (2.0 stopped at ~25.6k).
    @Test func theCalibratedFactorPutsTheBoundaryNear29kTokens() throws {
        #expect(ServeMemoryAdmission.overheadFactor == 1.75)
        let thisMac = try #require(ServeMemoryAdmission.profile(configData: qwen3Config, weightsBytes: 4_600_000_000,
                                                                 roomBytes: Int64(11.6 * 1_073_741_824)))
        #expect(thisMac.bytesPerToken == 147_456 * 1.75)
        #expect(ServeMemoryAdmission.decide(profile: thisMac, promptTokens: 25_000) != .refuse(needBytes: 0, roomBytes: 0))
        guard case .admit = ServeMemoryAdmission.decide(profile: thisMac, promptTokens: 29_000) else {
            Issue.record("29k tokens must be admitted"); return
        }
        guard case .refuse = ServeMemoryAdmission.decide(profile: thisMac, promptTokens: 30_000) else {
            Issue.record("30k tokens must be refused"); return
        }
        let edge = thisMac.tokenCapacity - ServeMemoryAdmission.minimumReplyTokens
        #expect((29_300...29_500).contains(edge))
    }

    @Test func theRefusalSentenceNamesBothFigures() {
        let error = ServeMemoryAdmission.refusal(needBytes: 17_716_740_096, roomBytes: 11_596_411_699)
        #expect(error.status == 400 && error.code == "context_too_large_for_memory")
        #expect(error.message == "This prompt needs about 16.5 GB of memory, and this Mac can give the model about 10.8 GB. "
                + "Use a shorter prompt (for example, fewer tools) or a smaller model.")
    }

    @Test func roomIsTheSharedBudgetsCapacityAndAgentContextUsesTheSameEstimator() throws {
        #expect(ServeMemoryAdmission.room(budget: MemoryBudget(capacityBytes: 123_456)) == 123_456)
        let profile = try #require(ServeMemoryAdmission.profile(configData: qwen3Config, weightsBytes: 4_600_000_000,
                                                                 roomBytes: 36 * 1_073_741_824 * 6 / 10))
        #expect(ServeMemoryAdmission.fitsAgentContext(profile))                        // a 36 GiB Mac: 21.6 GiB room
        var small = profile
        small.roomBytes = Int64(10.8 * 1_073_741_824)                                  // an 18 GiB Mac
        #expect(!ServeMemoryAdmission.fitsAgentContext(small))
    }

    @Test func weightsAreTheSafetensorsOnly() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mem-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try Data(count: 3_000).write(to: dir.appendingPathComponent("model-00001.safetensors"))
        try Data(count: 2_000).write(to: dir.appendingPathComponent("model-00002.safetensors"))
        try Data(count: 9_999).write(to: dir.appendingPathComponent("tokenizer.json"))
        #expect(MLXChatBackend.weightsBytes(in: dir) == 5_000)
        #expect(MLXChatBackend.weightsBytes(in: dir.appendingPathComponent("missing")) == 0)
    }
}

// MARK: - Through the routes

private func environment(_ backend: ServeBackend, log: RequestLog = RequestLog()) -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [model] },
        installed: { InstalledModelIndex(entries: [
            .init(id: "mlx-community--Qwen3-8B-4bit", hfModelId: model, kind: .llm, ramGB: 3, downloadSizeGB: 2)]) },
        appleFoundationReadiness: { nil }, created: 1_700_000_000,
        backendFor: { $0 == model ? backend : nil }, queue: ServeQueue(), requestLog: log)
}

private func send(_ json: String, path: String, env: ServeEnvironment) async -> ServeResponse? {
    let body = Data(json.utf8)
    let head = ServeRequestHead(method: "POST", path: path, host: "127.0.0.1:1212", origin: nil, transferEncoding: nil,
                                contentLength: String(body.count), userAgent: nil)
    if case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env,
                                                         body: { body }) { return response }
    return nil
}

private func object(_ data: Data) -> [String: Any]? { (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] }

/// An 18 GiB Mac serving Qwen3-8B-4bit: 4.6 GB of weights, the 10.8 GiB budget as the room.
private func eighteenGiBProfile() -> ServeMemoryProfile {
    ServeMemoryAdmission.profile(configData: qwen3Config, weightsBytes: 4_600_000_000,
                                 roomBytes: Int64(10.8 * 1_073_741_824))!
}

private let chatBody = #"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#
private func chatBody(max: Int) -> String { #"{"model":"\#(model)","max_tokens":\#(max),"messages":[{"role":"user","content":"hi"}]}"# }
private func messagesBody(max: Int, extra: String = "") -> String {
    #"{"model":"\#(model)","max_tokens":\#(max),"messages":[{"role":"user","content":"hi"}]\#(extra)}"#
}
private let responsesBody = #"{"model":"\#(model)","input":"hi"}"#

struct ServeMemoryAdmissionRouteTests {
    private func backend(count: Int?, profile: ServeMemoryProfile? = eighteenGiBProfile()) -> MessagesBackend {
        let b = MessagesBackend(count: count, contextLimit: 40_960)
        b.memoryProfile = profile
        return b
    }

    /// INV-1's ~39.5k-token prompt: refused before the queue, in each dialect's own shape, and nothing generated.
    @Test func aPromptTheMachineCannotHoldIsRefusedInEachDialectsShape() async throws {
        let b = backend(count: 39_492)
        let chat = try #require(await send(chatBody, path: "/v1/chat/completions", env: environment(b)))
        #expect(chat.status == 400)
        let chatError = try #require(object(chat.body)?["error"] as? [String: Any])
        #expect(chatError["code"] as? String == "context_too_large_for_memory" && chatError["type"] as? String == "invalid_request_error")
        let message = try #require(chatError["message"] as? String)
        #expect(message.hasPrefix("This prompt needs about 14.0 GB of memory") && message.contains("this Mac can give the model about 10.8 GB"))
        #expect(message.hasSuffix("Use a shorter prompt (for example, fewer tools) or a smaller model."))

        let anthropic = try #require(await send(messagesBody(max: 1_000), path: "/v1/messages", env: environment(b)))
        #expect(anthropic.status == 400 && object(anthropic.body)?["type"] as? String == "error")
        let inner = try #require(object(anthropic.body)?["error"] as? [String: Any])
        #expect(inner["type"] as? String == "invalid_request_error" && inner["message"] as? String == message)

        let responses = try #require(await send(responsesBody, path: "/v1/responses", env: environment(b)))
        #expect(responses.status == 400)
        #expect((object(responses.body)?["error"] as? [String: Any])?["code"] as? String == "context_too_large_for_memory")
        #expect(b.generated.isEmpty)
    }

    /// The lean Claude Code case: ~17k prompt, `max_tokens: 32000` → admitted, with the reply clamped to what memory allows.
    @Test func theLeanClaudeCodeCaseIsAdmittedWithAClampedReply() async throws {
        let profile = eighteenGiBProfile()
        let b = backend(count: 17_154, profile: profile)
        let response = try #require(await send(messagesBody(max: 32_000), path: "/v1/messages", env: environment(b)))
        #expect(response.status == 200)
        let limit = try #require(b.generated.first?.effectiveMaxTokens)
        #expect(limit == profile.tokenCapacity - 17_154)                 // floor((room − weights) / bytesPerToken) − prompt
        #expect(limit < 23_806)                                          // below the window-only clamp (40,960 − 17,154)
        #expect(limit >= ServeMemoryAdmission.minimumReplyTokens && limit < 32_000)
    }

    @Test func theMemoryClampOnlyEverLowers() async throws {
        let b = backend(count: 17_154)
        _ = await send(messagesBody(max: 100), path: "/v1/messages", env: environment(b))
        #expect(b.generated.first?.effectiveMaxTokens == 100)             // a small explicit limit is untouched
        let c = backend(count: 17_154)
        _ = await send(chatBody, path: "/v1/chat/completions", env: environment(c))
        #expect(c.generated.first?.effectiveMaxTokens == ServeRequest.defaultMaxTokens)   // the default 4096 fits
        // Even a tiny prompt cannot ask for more reply than the memory holds beyond the weights.
        let profile = eighteenGiBProfile()
        let d = backend(count: 30, profile: profile)
        _ = await send(messagesBody(max: 32_000), path: "/v1/messages", env: environment(d))
        #expect(d.generated.first?.effectiveMaxTokens == profile.tokenCapacity - 30)
        let e = backend(count: 30, profile: profile)
        _ = await send(messagesBody(max: 8_000), path: "/v1/messages", env: environment(e))
        #expect(e.generated.first?.effectiveMaxTokens == 8_000)           // fits: unchanged
    }

    /// No count (a render that failed): the size is estimated from the request's UTF-8 bytes, for admission only.
    @Test func withoutACountTheSizeIsEstimatedFromBytes() async throws {
        let big = String(repeating: "x", count: 120_000)                  // ≈ 40k tokens by bytes ÷ 3
        let refused = try #require(await send(
            #"{"model":"\#(model)","messages":[{"role":"user","content":"\#(big)"}]}"#,
            path: "/v1/chat/completions", env: environment(backend(count: nil))))
        #expect(refused.status == 400)
        #expect((object(refused.body)?["error"] as? [String: Any])?["code"] as? String == "context_too_large_for_memory")

        let small = backend(count: nil)
        let admitted = try #require(await send(chatBody, path: "/v1/chat/completions", env: environment(small)))
        #expect(admitted.status == 200 && small.generated.count == 1)
    }

    /// `count_tokens` is not a generation: never refused for memory, and never answered with an estimate.
    @Test func countTokensIsNeverAdmissionControlled() async throws {
        let counted = try #require(await send(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#,
                                              path: "/v1/messages/count_tokens", env: environment(backend(count: 39_492))))
        #expect(counted.status == 200 && object(counted.body)?["input_tokens"] as? Int == 39_492)
        let uncounted = try #require(await send(#"{"model":"\#(model)","messages":[{"role":"user","content":"hi"}]}"#,
                                                path: "/v1/messages/count_tokens", env: environment(backend(count: nil))))
        #expect(uncounted.status == 501)                                   // "can't count", not an estimate
    }

    /// Exempt: a backend with no profile (Apple Foundation Models, doubles) is never refused for memory.
    @Test func aBackendWithoutAProfileIsExempt() async throws {
        let exempt = backend(count: 39_492, profile: nil)
        let response = try #require(await send(chatBody, path: "/v1/chat/completions", env: environment(exempt)))
        #expect(response.status == 200 && exempt.generated.count == 1)
        #expect(exempt.generated.first?.effectiveMaxTokens == 1_468)       // only the window clamp (40,960 − 39,492)
        let afm = AFMBackend(streamer: FakeChatStreamer([.snapshot("x")]), readiness: { .ready })
        #expect(afm.memoryProfile == nil)
    }

    /// The S2-6 window check still comes first.
    @Test func theWindowRefusalStillComesFirst() async throws {
        let response = try #require(await send(chatBody, path: "/v1/chat/completions", env: environment(backend(count: 41_000))))
        #expect((object(response.body)?["error"] as? [String: Any])?["code"] as? String == "context_length_exceeded")
    }

    @Test func theRefusalIsLoggedAsAMetadataOnlyLine() async throws {
        let log = RequestLog()
        _ = await send(chatBody, path: "/v1/chat/completions", env: environment(backend(count: 39_492), log: log))
        let entry = try #require(log.entries.last)
        #expect(entry.status == 400 && entry.model == model)
    }
}
