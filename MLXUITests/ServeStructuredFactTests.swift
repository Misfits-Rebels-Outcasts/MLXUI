import Testing
import Foundation
@testable import MLXUI

// WW4-1-5 (ruling BI) — structured output as a plain fact: `"structured_output"` on every `/v1/models` entry and one line on the
// Connect sheet ("Structured output (JSON schema): yes / no"), both derived from the allow-list the backend's 400 uses.
// New tests only: the existing /v1/models and Connect snapshot tests are untouched and still pass as written.

private let gemma = "mlx-community/gemma-4-e4b-it-4bit"
private let qwen = "mlx-community/Qwen3-8B-4bit"
private let llama = "mlx-community/Llama-3.2-3B-Instruct-4bit"

private func config(_ modelType: String) -> Data {
    Data(#"{"model_type":"\#(modelType)","max_position_embeddings":131072}"#.utf8)
}

/// What each served id's `config.json` says — the live environment reads these from disk; the test hands them over.
private let configs: [String: Data] = [gemma: config("gemma4"), qwen: config("qwen3"), llama: config("llama")]

private func environment() -> ServeEnvironment {
    func entry(_ hf: String) -> InstalledModelIndex.Entry {
        .init(id: hf.replacingOccurrences(of: "/", with: "--"), hfModelId: hf, kind: .llm, ramGB: 3, downloadSizeGB: 2)
    }
    var env = ServeEnvironment(
        servedModelIDs: { [gemma, qwen, llama, ServedModels.appleFoundationID] },
        installed: { InstalledModelIndex(entries: [entry(gemma), entry(qwen), entry(llama)]) },
        appleFoundationReadiness: { .ready },
        created: 1_700_000_000,
        backendFor: { _ in nil })
    env.modelStructuredOutput = { id in
        id == ServedModels.appleFoundationID ? false : ServeStructuredOutputFamilies.allows(configData: configs[id])
    }
    return env
}

private func models(env: ServeEnvironment) throws -> [String: [String: Any]] {
    let response = ServeAPI.respond(to: ServeRequestHead(method: "GET", path: "/v1/models", host: "127.0.0.1:1212", origin: nil),
                                    guard: RequestGuard(port: 1212), environment: env)
    #expect(response.status == 200)
    let object = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
    let data = try #require(object["data"] as? [[String: Any]])
    return Dictionary(uniqueKeysWithValues: data.compactMap { entry in (entry["id"] as? String).map { ($0, entry) } })
}

struct ServeStructuredFactModelsTests {
    @Test func modelsReportsStructuredOutputTrueForTheAllowListedFamiliesOnly() throws {
        let listed = try models(env: environment())
        #expect(listed.count == 4)
        #expect(listed[gemma]?["structured_output"] as? Bool == true)
        #expect(listed[qwen]?["structured_output"] as? Bool == true)
        #expect(listed[llama]?["structured_output"] as? Bool == false)
        #expect(listed[ServedModels.appleFoundationID]?["structured_output"] as? Bool == false, "Apple Foundation Models can't be constrained")
    }

    @Test func everyEntryCarriesTheFieldAsABooleanNextToContextLength() throws {
        var env = environment()
        env.modelContextLength = { _ in 40_960 }
        for (id, entry) in try models(env: env) {
            #expect(entry["structured_output"] is Bool, "\(id) has no boolean structured_output")
            #expect(entry["context_length"] as? Int == 40_960, "\(id)")
        }
    }

    @Test func aDoubleThatNeverSetItAnswersFalseRatherThanOmittingTheField() throws {
        var env = environment()
        env.modelStructuredOutput = { _ in false }
        for (id, entry) in try models(env: env) { #expect(entry["structured_output"] as? Bool == false, "\(id)") }
        let body = ModelsRoute.body(ids: ["x"], created: 1, contextLength: { _ in nil })
        let list = try #require(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        #expect((list["data"] as? [[String: Any]])?.first?["structured_output"] as? Bool == false)
    }

    @Test func theLiveAnswerForAppleFoundationAndAnUnknownIdIsFalse() {
        #expect(!ServeEnvironment.liveModelStructuredOutput(ServedModels.appleFoundationID))
        #expect(!ServeEnvironment.liveModelStructuredOutput("mlx-community/not-installed-anywhere"))
    }
}

struct ServeStructuredFactFactsTests {
    @Test func theConnectFactsFollowTheAllowList() {
        for type in ["gemma4", "qwen3"] {
            #expect(ServeModelFacts.facts(modelID: "m", configData: config(type)).structuredOutput, "\(type)")
        }
        for type in ["qwen3_5", "qwen3_moe", "qwen2", "llama", "gemma2", "gemma3", "gemma3_text", "mistral", ""] {
            #expect(!ServeModelFacts.facts(modelID: "m", configData: config(type)).structuredOutput, "\(type)")
        }
        #expect(!ServeModelFacts.facts(modelID: "m", configData: nil).structuredOutput)
    }

    @Test func everyFamilyOnTheAllowListIsReportedTrue() {
        #expect(!ServeStructuredOutputFamilies.verified.isEmpty)
        for type in ServeStructuredOutputFamilies.verified {
            #expect(ServeModelFacts.facts(modelID: "m", configData: config(type)).structuredOutput, "\(type)")
            #expect(ServeStructuredOutputFamilies.allows(configData: config(type)), "\(type)")
        }
    }

    /// The point of the single source: the fact the client is shown and the 400 the backend answers cannot disagree. Each family
    /// gets a directory holding only its `config.json`; the allow-list check is the first thing `prepareStructure` does, before
    /// any tokenizer or weights, so an allowed family fails later with a different code (the tokenizer isn't there).
    @Test func theFactAndTheBackendsRefusalNeverDisagree() async throws {
        let types = ["gemma4", "qwen3", "qwen3_5", "qwen3_moe", "qwen2", "llama", "gemma2", "gemma3", "gemma3_text", "mistral", "phi3", "glm4"]
            + Array(ServeStructuredOutputFamilies.verified)
        for type in types {
            let dir = FileManager.default.temporaryDirectory.appendingPathComponent("ww415-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: dir) }
            try config(type).write(to: dir.appendingPathComponent("config.json"))

            let fact = ServeModelFacts.facts(modelID: "m", configData: ModelContainerPool.configData(directory: dir)).structuredOutput
            var request = ServeRequest(model: "m", messages: [ServeMessage(role: .user, text: "hi")])
            request.structure = .jsonObject
            var refusedAsUnsupported = false
            do { _ = try await MLXChatBackend(directory: dir).prepareStructure(request) } catch let error as ServeError {
                refusedAsUnsupported = error.code == ServeStructureCheck.unsupportedModelCode
            } catch {}
            #expect(fact == !refusedAsUnsupported, "\(type): the fact says \(fact) but the backend \(refusedAsUnsupported ? "refuses" : "accepts")")
        }
    }
}

struct ServeStructuredFactConnectTests {
    @Test func theLineIsPlainWordsAndNothingElse() {
        #expect(ConnectSnippets.structuredOutputLine(true) == "Structured output (JSON schema): yes")
        #expect(ConnectSnippets.structuredOutputLine(false) == "Structured output (JSON schema): no")
    }

    /// R51/R56: a fact, not a verdict — no rating, badge or recommendation wording.
    @Test func theLineCarriesNoVerdictWording() {
        let banned = ["ready", "rating", "recommended", "best", "supported", "experimental", "beta", "badge", "works with", "not "]
        for supported in [true, false] {
            let line = ConnectSnippets.structuredOutputLine(supported).lowercased()
            for word in banned { #expect(!line.contains(word), "\(word) in: \(line)") }
        }
    }

    /// The fact is shown on the sheet and by `/v1/models`; it is not part of any client's snippet, so every snippet is
    /// byte-identical whatever the flag says (the existing snapshot tests keep passing unchanged).
    /// WW5-1 (ruling BN, 2026-10-24): the one exception is the Swift (Foundation Models) block, which writes
    /// `supportsGuidedGeneration` from the fact; its own tests are `SwiftFoundationModelsSnippetTests`.
    @Test func noClientSnippetChangesWithTheFact() {
        for harness in ConnectSnippets.Harness.allCases where harness != .swiftFoundationModels {
            for tools in [true, false] {
                var on = ConnectSnippets.ModelFacts(modelID: "mlx-community/Qwen3-8B-4bit", supportsTools: tools,
                                                    contextLength: 40_960, thinkingAvailable: tools)
                var off = on
                on.structuredOutput = true
                off.structuredOutput = false
                #expect(ConnectSnippets.text(for: harness, port: 1212, model: on) == ConnectSnippets.text(for: harness, port: 1212, model: off),
                        "\(harness.rawValue) changed with structuredOutput")
            }
        }
    }

    @Test func aPlainFactsValueSaysNo() {
        #expect(ConnectSnippets.ModelFacts.plain("m").structuredOutput == false)
    }
}

/// WW5-1 (ruling BN, copy row C10): the Swift (Foundation Models) block.
struct SwiftFoundationModelsSnippetTests {
    private func block(id: String, structured: Bool) -> String {
        var facts = ConnectSnippets.ModelFacts(modelID: id, supportsTools: true, contextLength: 131_072, thinkingAvailable: false)
        facts.structuredOutput = structured
        return ConnectSnippets.text(for: .swiftFoundationModels, port: 1212, model: facts)
    }

    @Test func aStructuredOutputModelWritesTrueExplicitly() {
        #expect(block(id: "mlx-community/gemma-4-e4b-it-4bit", structured: true) == """
        // Swift package (pre-release; pin this exact tag):
        //   .package(url: "https://github.com/apple/foundation-models-utilities", exact: "1.1.0-beta1")
        // Needs macOS 27 on the client Mac.
        import FoundationModels
        import FoundationModelsUtilities

        let model = ChatCompletionsLanguageModel(
            name: "mlx-community/gemma-4-e4b-it-4bit",
            url: URL(string: "http://127.0.0.1:1212/v1")!,
            supportsGuidedGeneration: true)
        let session = LanguageModelSession(model: model)
        """)
    }

    @Test func aModelWithoutItWritesFalseExplicitly() {
        let text = block(id: "mlx-community/Llama-3.1-8B-Instruct-4bit", structured: false)
        #expect(text.contains("supportsGuidedGeneration: false)"))
        #expect(text.contains(#"name: "mlx-community/Llama-3.1-8B-Instruct-4bit","#))
        #expect(text.contains("Needs macOS 27 on the client Mac."))
    }

    @Test func thePlainSheetTextNeverOmitsTheArgument() {
        #expect(ConnectSnippets.text(for: .swiftFoundationModels, port: 1212, modelID: "m").contains("supportsGuidedGeneration: false"))
    }

    @Test func theBlockFollowsThePort() {
        let text = ConnectSnippets.text(for: .swiftFoundationModels, port: 4321, modelID: "m")
        #expect(text.contains("http://127.0.0.1:4321/v1"))
    }

    /// WW5-1 (owner 2026-10-24): the flag is pinned from the checkpoint's `model_type` through the real allow-list
    /// (`allows(configData:)`, the same reader the backend's 400 uses): true for gemma4 and qwen3, false otherwise.
    @Test func theFlagFollowsTheModelTypeThroughTheAllowList() {
        for (modelType, expected) in [("gemma4", true), ("qwen3", true), ("qwen3_5", false), ("qwen2", false),
                                      ("llama", false), ("gemma3", false), ("gemma2", false), ("mistral3", false),
                                      ("gpt_oss", false)] {
            let config = Data(#"{"model_type": "\#(modelType)"}"#.utf8)
            var facts = ConnectSnippets.ModelFacts.plain("mlx-community/x")
            facts.structuredOutput = ServeStructuredOutputFamilies.allows(configData: config)
            let text = ConnectSnippets.text(for: .swiftFoundationModels, port: 1212, model: facts)
            #expect(text.contains("supportsGuidedGeneration: \(expected))"), "\(modelType)")
            #expect(!text.contains("supportsGuidedGeneration: \(!expected))"), "\(modelType)")
        }
    }

    /// The exemption in `noClientSnippetChangesWithTheFact` is real and only this block: it differs, the others do not.
    @Test func onlyTheSwiftBlockDiffersWithTheFact() {
        var on = ConnectSnippets.ModelFacts.plain("m"); on.structuredOutput = true
        var off = on; off.structuredOutput = false
        let differing = ConnectSnippets.Harness.allCases.filter {
            ConnectSnippets.text(for: $0, port: 1212, model: on) != ConnectSnippets.text(for: $0, port: 1212, model: off)
        }
        #expect(differing == [.swiftFoundationModels])
    }

    /// Xcode 27's own wording (owner, 2026-10-24): "Add a Chat Provider" ▸ "Locally Hosted".
    @Test func theXcodeBlockUsesXcode27sWording() {
        let text = ConnectSnippets.text(for: .xcode, port: 1212, modelID: "m")
        #expect(text.contains("Add a Chat Provider"))
    }
}
