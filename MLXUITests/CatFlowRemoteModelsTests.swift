import Testing
import Foundation
@testable import MLXUI

/// Phase RM — remote models (Claude, GPT, DeepSeek, and a keyless LAN endpoint), entirely
/// mock-driven at the network boundary (`ProviderAvailability.executorOverride`) — no test
/// here ever issues a real HTTP request, "mock by default" applied to the network exactly
/// as Phase AFM applied it to the OS. Every Keychain value used is a throwaway,
/// UUID-suffixed string, never printed — the same constraint that has run through KEY.
struct CatFlowRemoteModelsTests {

    private func resetOverrides() {
        ProviderAvailability.executorOverride = nil
    }

    private final class MockProviderExecutor: ProviderExecuting, @unchecked Sendable {
        var textToReturn = "a reply"
        var errorToThrow: ProviderRequestError?
        private(set) var lastPrompt: String?
        private(set) var lastMaxTokens: Int?

        func generate(instructions: String, prompt: String, maxTokens: Int, temperature: Double) async throws -> String {
            lastPrompt = prompt
            lastMaxTokens = maxTokens
            if let errorToThrow { throw errorToThrow }
            return textToReturn
        }
    }

    private func realExecutor(catalog: [ModelEntry] = []) -> RealExecutor {
        RealExecutor(
            workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory),
            flowID: "test",
            blobDirectory: FileManager.default.temporaryDirectory,
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "unused", kind: .llm) },
            installedModelIDs: [],
            catalog: catalog)
    }

    private func issues(for text: String) throws -> [FlowIssue] {
        let flow = try CatParser.parseForValidation(text)
        return FlowValidator.checkFlow(flow)
    }

    // MARK: - RM-1: the four ported manifests decode

    @Test func claudeManifestDecodesProviderFields() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "claude-sonnet-4.json"))
        #expect(manifest.id == "anthropic/claude-sonnet-4")
        #expect(manifest.display == "claude-sonnet @ anthropic")
        #expect(manifest.kind == "provider")
        #expect(manifest.engine == "anthropic-api")
        #expect(manifest.credentials == "anthropic")
        #expect(manifest.egress == nil)   // hosted APIs declare no egress — internet is the default
        #expect(manifest.baseURL == nil)   // fixed per engine, not manifest-declared
    }

    @Test func macstudioManifestIsTheKeylessLANEndpoint() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "macstudio-qwen3-32b.json"))
        #expect(manifest.kind == "provider")
        #expect(manifest.engine == "openai-compatible")
        #expect(manifest.egress == "lan")
        #expect(manifest.credentials == nil)
        #expect(manifest.baseURL == "http://mac-studio.local:8080/v1")
    }

    // MARK: - RM-2: TaskModels.providerModels / lanProviderDisplayNames

    @Test func providerModelsOffersAllFourPortedManifestsUnderGenerate() {
        let names = Set(TaskModels.providerDisplayNames)
        for expected in ["claude-sonnet @ anthropic", "gpt-5.6-luna @ openai",
                         "deepseek-v4-flash @ deepseek", "qwen3-32b @ http://mac-studio.local:8080"] {
            #expect(names.contains(expected), "expected \(expected) in the provider registry")
        }
    }

    @Test func lanProviderDisplayNamesContainsOnlyTheKeylessEndpoint() {
        let lan = TaskModels.lanProviderDisplayNames
        #expect(lan.contains("qwen3-32b @ http://mac-studio.local:8080"))
        #expect(!lan.contains("claude-sonnet @ anthropic"))
        #expect(!lan.contains("gpt-5.6-luna @ openai"))
        #expect(!lan.contains("deepseek-v4-flash @ deepseek"))
    }

    @Test func providerModelRefResourceNoteNamesTheProviderAndLeavingTheMac() {
        // Uncapitalized — the row's own text is lowercase, and the LAN case's "provider"
        // is a literal URL (`http://mac-studio.local:8080`), which `.capitalized` would
        // mangle. RM-3's refusal message capitalizes for a different reason (a named
        // hosted provider read as English prose); this is the row's own display text.
        let ref = try? #require(TaskModels.providerModelRef(forDisplay: "claude-sonnet @ anthropic"))
        #expect(ref?.resourceNote == "Runs on anthropic — leaves this Mac")
    }

    // MARK: - RM-2: ProviderCredential.readiness(for:) — the LAN case never asks for a key

    @Test func lanManifestReadinessIsAlwaysReadyRegardlessOfKeychain() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "macstudio-qwen3-32b.json"))
        #expect(ProviderCredential.readiness(for: manifest) == .ready)
    }

    @Test func keyedManifestReadinessTracksTheKeychain() throws {
        let manifest = try #require(CuratedManifest.load(manifestFile: "claude-sonnet-4.json"))
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        defer {
            if let original { KeychainHelper.save(original, account: account) }
            else { KeychainHelper.delete(account: account) }
        }
        KeychainHelper.delete(account: account)
        guard case .needsSetup = ProviderCredential.readiness(for: manifest) else {
            Issue.record("expected .needsSetup with no key in the Keychain")
            return
        }
        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        #expect(ProviderCredential.readiness(for: manifest) == .ready)
    }

    // MARK: - RM-3: CatalogBridge's provider-specific refusal

    @Test func resolveReturnsAProviderSlotForAPortedManifest() {
        guard case .runnable(.provider(let ref), let equivalence, _) = CatalogBridge.resolve("claude-sonnet @ anthropic", catalog: []) else {
            Issue.record("expected a .provider slot")
            return
        }
        #expect(ref.displayName == "claude-sonnet @ anthropic")
        #expect(equivalence == .same)
    }

    @Test func resolveRefusesAnUnportedProviderRowWithASpecificMessageAndButton() {
        guard case .notRunnable(let display, let reason, let action) = CatalogBridge.resolve("llama-3.3-70b @ groq", catalog: []) else {
            Issue.record("expected .notRunnable for an unported provider")
            return
        }
        #expect(display == "llama-3.3-70b @ groq")
        #expect(reason == "llama-3.3-70b @ groq runs on Groq's servers. Add your Groq key in Settings to use it.")
        #expect(action == .openSettings(.providers))
    }

    @Test func resolveKeepsTheGenericRefusalForAGenuinelyUnknownName() {
        guard case .notRunnable(_, let reason, let action) = CatalogBridge.resolve("Some Made Up Model", catalog: []) else {
            Issue.record("expected .notRunnable")
            return
        }
        #expect(reason.contains("isn't in the runnable-model table"))
        #expect(action == nil)
    }

    // MARK: - RM-4b: E120 — the check RM-4b builds

    @Test func remoteProviderRowWithoutOffdeviceRaisesE120() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Answer       (1)  claude-sonnet @ anthropic
        3. Save Text    out.md

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let found = try issues(for: text)
        #expect(found.contains { $0.code == "E120" })
    }

    @Test func remoteProviderRowWithOffdeviceRaisesNoE120() throws {
        let text = """
        mlxflow 0.8; offdevice
        1. Read Text    memo.txt
        2. Answer       (1)  claude-sonnet @ anthropic
        3. Save Text    out.md

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E120" })
    }

    @Test func appleFoundationSystemRowNeverRaisesE120() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Summarize    (1)  apple-foundation @ system
        3. Save Text    out.md

        models:
          apple-foundation @ system = apple/foundation-models
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E120" })
    }

    /// The instruction this cycle gave, and the reference does not: a keyless LAN endpoint
    /// never raises E120, even with no `offdevice` flag. Flagged as a deliberate divergence
    /// from `catflow-mlx`'s own `_check_offdevice_flag` at `FlowValidator.checkOffdeviceFlag`'s
    /// definition and in the journal.
    @Test func keylessLANEndpointNeverRaisesE120() throws {
        let text = """
        mlxflow 0.8
        1. Read Text    memo.txt
        2. Answer       (1)  qwen3-32b @ http://mac-studio.local:8080
        3. Save Text    out.md

        models:
          qwen3-32b @ http://mac-studio.local:8080 = macstudio/qwen3-32b
        """
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E120" })
    }

    @Test func e120MessageMatchesTheEstablishedTemplateAndNamesTheRow() throws {
        let text = """
        mlxflow 0.8
        1. Answer       claude-sonnet @ anthropic; "cite sources"

        models:
          claude-sonnet @ anthropic = anthropic/claude-sonnet-4
        """
        let found = try issues(for: text)
        let e120 = try #require(found.first { $0.code == "E120" })
        #expect(e120.row == "1")
        #expect(e120.message.contains("claude-sonnet @ anthropic"))
        #expect(e120.message.contains("leaves this building"))
        #expect(e120.message.contains("; offdevice"))
    }

    /// `44-FrontierEscalate.cat` already declares `offdevice` — RM-4b's own check must not
    /// newly break the shipping example flow it exists to protect.
    @Test func frontierEscalateGalleryFlowRaisesNoE120() throws {
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/44-FrontierEscalate.cat")
        let text = try String(contentsOf: url, encoding: .utf8)
        let found = try issues(for: text)
        #expect(!found.contains { $0.code == "E120" })
    }

    // MARK: - ProviderWireFormat — request/response mapping, network-free

    @Test func anthropicRequestShapeMatchesTheAPI() {
        let (url, headers, body) = ProviderWireFormat.anthropicRequest(
            prompt: "hello", apiModelName: "claude-sonnet-4", maxTokens: 100,
            temperature: 0, apiKey: "test-key")
        #expect(url.absoluteString == "https://api.anthropic.com/v1/messages")
        #expect(headers["x-api-key"] == "test-key")
        #expect(headers["anthropic-version"] == "2023-06-01")
        #expect(body["model"] as? String == "claude-sonnet-4")
        #expect(body["max_tokens"] as? Int == 100)
        let messages = body["messages"] as? [[String: String]]
        #expect(messages?.first?["content"] == "hello")
    }

    @Test func anthropicParseJoinsTextBlocks() {
        let json: [String: Any] = ["content": [["type": "text", "text": "hello "],
                                               ["type": "text", "text": "world"]]]
        #expect(ProviderWireFormat.anthropicParse(json) == "hello world")
    }

    @Test func deepseekRequestUsesTheNonVersionedPath() {
        let (url, headers, body) = ProviderWireFormat.deepseekRequest(
            prompt: "hi", apiModelName: "deepseek-v4-flash", maxTokens: 50,
            temperature: 0, apiKey: "key")
        #expect(url.absoluteString == "https://api.deepseek.com/chat/completions")
        #expect(headers["Authorization"] == "Bearer key")
        #expect(body["model"] as? String == "deepseek-v4-flash")
    }

    @Test func openAIRequestOmitsTemperatureAndUsesMaxCompletionTokens() {
        let (url, headers, body) = ProviderWireFormat.openAIRequest(
            prompt: "hi", apiModelName: "gpt-5.6-luna", maxTokens: 50, apiKey: "key")
        #expect(url.absoluteString == "https://api.openai.com/v1/chat/completions")
        #expect(headers["Authorization"] == "Bearer key")
        #expect(body["max_completion_tokens"] as? Int == 50)
        #expect(body["max_tokens"] == nil)
        #expect(body["temperature"] == nil)
    }

    @Test func openAICompatibleRequestOmitsAuthorizationWithNoKey() {
        let (url, headers, body) = ProviderWireFormat.openAICompatibleRequest(
            prompt: "hi", apiModelName: "qwen3-32b", maxTokens: 50, temperature: 0,
            baseURL: "http://mac-studio.local:8080/v1", apiKey: nil)
        #expect(url.absoluteString == "http://mac-studio.local:8080/v1/chat/completions")
        #expect(headers["Authorization"] == nil)
        #expect(body["model"] as? String == "qwen3-32b")
    }

    @Test func openAICompatibleRequestTrimsATrailingSlashFromBaseURL() {
        let (url, _, _) = ProviderWireFormat.openAICompatibleRequest(
            prompt: "hi", apiModelName: "m", maxTokens: 1, temperature: 0,
            baseURL: "https://api.groq.com/openai/v1/", apiKey: "k")
        #expect(url.absoluteString == "https://api.groq.com/openai/v1/chat/completions")
    }

    @Test func openAICompatibleParseReadsChoicesMessageContent() {
        let json: [String: Any] = ["choices": [["message": ["content": "the answer"]]]]
        #expect(ProviderWireFormat.openAICompatibleParse(json) == "the answer")
    }

    @Test func apiModelNameStripsTheOrgPrefix() {
        #expect(ProviderWireFormat.apiModelName("anthropic/claude-sonnet-4") == "claude-sonnet-4")
        #expect(ProviderWireFormat.apiModelName("macstudio/qwen3-32b") == "qwen3-32b")
        #expect(ProviderWireFormat.apiModelName("no-slash-here") == "no-slash-here")
    }

    // MARK: - RM-2: RealExecutor dispatches a provider row (mocked at the network boundary)

    @Test func answerRowDispatchesToTheMockProviderExecutor() async throws {
        resetOverrides()
        defer { resetOverrides() }
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        defer {
            if let original { KeychainHelper.save(original, account: account) }
            else { KeychainHelper.delete(account: account) }
        }
        let mock = MockProviderExecutor()
        mock.textToReturn = "a cited answer"
        ProviderAvailability.executorOverride = mock

        let executor = realExecutor()
        let row = Row(task: "Answer", model: "claude-sonnet @ anthropic", settings: "\"cite sources\"")
        let input = Asset(items: [Item(kind: .text, value: "some context", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input])
        #expect(output.items.first?.value == "a cited answer")
    }

    @Test func answerRowRefusesPlainlyWhenNoKeyIsSet() async throws {
        resetOverrides()
        defer { resetOverrides() }
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        KeychainHelper.delete(account: account)
        defer { if let original { KeychainHelper.save(original, account: account) } }

        let executor = realExecutor()
        let row = Row(task: "Answer", model: "claude-sonnet @ anthropic", settings: nil)
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        await #expect(throws: (any Error).self) {
            _ = try await executor.execute(path: "1", row: row, inputs: [input])
        }
    }

    @Test func providerErrorSurfacesAsR904NeverTheRawBody() async throws {
        resetOverrides()
        defer { resetOverrides() }
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        defer {
            if let original { KeychainHelper.save(original, account: account) }
            else { KeychainHelper.delete(account: account) }
        }
        let mock = MockProviderExecutor()
        mock.errorToThrow = ProviderRequestError(status: "401 Unauthorized")
        ProviderAvailability.executorOverride = mock

        let executor = realExecutor()
        let row = Row(task: "Answer", model: "claude-sonnet @ anthropic", settings: nil)
        let input = Asset(items: [Item(kind: .text, value: "text", path: nil, sourceText: nil)])
        do {
            _ = try await executor.execute(path: "1", row: row, inputs: [input])
            Issue.record("expected a thrown error")
        } catch {
            let message = String(describing: error)
            #expect(message.contains("claude-sonnet @ anthropic"))
            #expect(message.contains("401"))
            #expect(message.contains("Your key is configured in Settings"))
        }
    }

    // MARK: - RM-2: deciders — F004 strict-parse, F010 disclosed (never suppressed)

    @Test func decideRowFiresATagViaStrictParseAndDisclosesF010() async throws {
        resetOverrides()
        defer { resetOverrides() }
        let account = KeychainHelper.providerAccount("anthropic")
        let original = KeychainHelper.get(account: account)
        KeychainHelper.save("test-key-\(UUID().uuidString)", account: account)
        defer {
            if let original { KeychainHelper.save(original, account: account) }
            else { KeychainHelper.delete(account: account) }
        }
        let mock = MockProviderExecutor()
        mock.textToReturn = "ship"
        ProviderAvailability.executorOverride = mock

        let executor = realExecutor()
        let row = Row(task: "Gate", model: "claude-sonnet @ anthropic",
                     settings: "\"Ready to ship?\"", tags: ["ship", "hold"])
        let input = Asset(items: [Item(kind: .text, value: "looks solid", path: nil, sourceText: nil)])
        let output = try await executor.execute(path: "1", row: row, inputs: [input])
        #expect(executor.lastTag == "ship")
        #expect(output.items.first?.value == "looks solid")
        let flag = try #require(executor.lastProviderDeciderFlag)
        #expect(flag.code == "F010")
        #expect(flag.message.contains("claude-sonnet @ anthropic"))
        #expect(flag.message.contains("parsed"))
    }

    @Test func nonProviderRowNeverDisclosesF010() async throws {
        resetOverrides()
        defer { resetOverrides() }
        let executor = realExecutor()
        // No model resolves for this display, so this exercises only the flag-reset path —
        // any thrown error is fine; the assertion is about the box, not the outcome.
        let row = Row(task: "Read Text", model: nil, settings: "memo.txt")
        _ = try? await executor.execute(path: "1", row: row, inputs: [])
        #expect(executor.lastProviderDeciderFlag == nil)
    }
}
