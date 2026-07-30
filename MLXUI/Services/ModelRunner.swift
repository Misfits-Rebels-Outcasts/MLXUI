import Foundation
import AppKit
import MLX
import MLXLLM
import MLXLMCommon

// Tokenization is handled by the real `HFTokenizerLoader` (swift-transformers) — see
// `Core/HFTokenizerLoader.swift`. The former hand-rolled `SimpleTokenizer` stub was removed
// (journal/2026-37): it did whitespace-split "tokenization" with no chat template, which
// produced garbage tokens for every model and broke VLM vision placeholders entirely.

// MARK: - Chat Message

struct ChatMessage: Identifiable, Sendable {
    enum Role: Sendable { case user, assistant }
    let id = UUID()
    let role: Role
    var text: String
}

// MARK: - Pending approval

/// A tool call awaiting the user's decision in the approval sheet (AG1). `respond` resumes the
/// `AgentSession` approval continuation.
struct PendingApproval: Identifiable {
    let id = UUID()
    let toolName: String
    let toolDescription: String
    let arguments: [String: JSONValue]
    let respond: (Bool) -> Void
}

// MARK: - Model Runner

@Observable
final class ModelRunner {
    /// Renderable transcript (text bubbles + tool-call cards), assembled by the pure reducer.
    private(set) var transcript = TranscriptBuilder()
    var isRunning = false
    var errorMessage: String?

    /// A tool call currently blocked on user approval (drives the approval sheet).
    var pendingApproval: PendingApproval?

    /// Tools available to the agent this build, and which are currently enabled (toggles).
    let availableTools: [any AgentTool] = ModelRunner.defaultTools()
    var enabledToolNames: Set<String> = Set(ModelRunner.defaultTools().map(\.name))

    private var currentTask: Task<Void, Never>?

    // Cache the loaded model so repeated sends in one session don't reload weights.
    private var loadedContainer: ModelContainer?
    private var loadedModelID: String?

    /// Tools that ship in this build. `fetch_url` (AG2) is the first real tool and ships in all
    /// builds; DEBUG additionally carries the `echo` demo tool so the approval/toggle UI has a
    /// gated subject to render. More real tools land in AG3–AG5.
    static func defaultTools() -> [any AgentTool] {
        #if DEBUG
        [FetchURLTool(), EchoDemoTool()]
        #else
        [FetchURLTool()]
        #endif
    }

    /// Called when the chat sheet opens. Resets history when switching models and
    /// frees the previously loaded model's weights.
    func prepare(for model: ModelEntry) {
        if loadedModelID != model.id {
            loadedContainer = nil
            loadedModelID = nil
            transcript.reset()
        }
        errorMessage = nil
    }

    /// Generate a reply to `prompt`, running the agent tool loop via `AgentSession` and
    /// streaming assistant text + tool-call cards into the transcript as they arrive.
    func send(_ prompt: String, for model: ModelEntry) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isRunning else { return }

        let modelDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("AI Browser/models/\(model.id)")

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            errorMessage = "Model files not found. Please reinstall."
            return
        }

        errorMessage = nil
        transcript.addUserMessage(trimmed)
        isRunning = true

        // Snapshot the enabled tools for this turn (Sendable to hand to the nonisolated session).
        let registry = ToolRegistry(availableTools.filter { enabledToolNames.contains($0.name) })

        currentTask = Task { [weak self] in
            guard let self else { return }
            do {
                let container = try await self.containerForModel(model, dir: modelDir)

                let session = AgentSession(
                    model: container,
                    registry: registry,
                    parameters: GenerateParameters(maxTokens: 512, temperature: 0.7),
                    approve: { [weak self] _, tool in
                        await self?.requestApproval(for: tool) ?? false
                    },
                    onEvent: { [weak self] event in
                        Task { @MainActor in self?.transcript.apply(event) }
                    }
                )

                for try await chunk in session.streamResponse(to: trimmed) {
                    self.transcript.appendAssistantChunk(chunk)
                }
            } catch is CancellationError {
                // stopped by the user — leave the partial transcript in place
            } catch {
                if !Task.isCancelled {
                    self.errorMessage = "Failed: \(error.localizedDescription)"
                }
            }
            self.isRunning = false
            self.currentTask = nil
            self.pendingApproval?.respond(false)  // release any in-flight approval
            self.pendingApproval = nil
        }
    }

    func stop() {
        currentTask?.cancel()
        currentTask = nil
        isRunning = false
        pendingApproval?.respond(false)
        pendingApproval = nil
    }

    // MARK: - Approval

    /// Present the approval sheet and await the user's decision. Called by `AgentSession` for
    /// tools whose `requiresApproval` is true.
    private func requestApproval(for tool: any AgentTool) async -> Bool {
        await withCheckedContinuation { continuation in
            self.pendingApproval = PendingApproval(
                toolName: tool.name,
                toolDescription: tool.toolDescription,
                arguments: [:],
                respond: { continuation.resume(returning: $0) }
            )
        }
    }

    /// Called by the approval sheet's Allow/Deny buttons.
    func resolveApproval(_ approved: Bool) {
        let pending = pendingApproval
        pendingApproval = nil
        pending?.respond(approved)
    }

    // MARK: - Non-LLM entry point

    /// Entry point for non-LLM model types — shows the "unsupported" alert.
    func run(_ model: ModelEntry) {
        switch model.modelType {
        case .llm:
            break // LLMs are driven through the chat sheet via prepare(for:)/send(_:for:)
        case .asr:
            showUnsupported("Speech-to-text", model)
        case .tts:
            showUnsupported("Text-to-speech", model)
        case .embedding:
            showUnsupported("Embeddings", model)
        case .vision, .ocr:
            showUnsupported("Vision/OCR", model)
        case .video:
            showUnsupported("Video", model)
        }
    }

    private func containerForModel(_ model: ModelEntry, dir: URL) async throws -> ModelContainer {
        if let cached = loadedContainer, loadedModelID == model.id {
            return cached
        }
        let loader = HFTokenizerLoader()
        let container = try await LLMModelFactory.shared.loadContainer(from: dir, using: loader)

        // Wire the tool-call parser to match the model family. Qwen3.5 emits the XML
        // `<tool_call><function=…>` form (`ToolCallFormat.xmlFunction`); without this the
        // generate path parses tool calls as `.json` and never fires them (AG0 finding).
        if let format = AgentSession.toolCallFormat(forModelType: Self.modelTypeString(model, dir: dir)) {
            await container.update { $0.configuration.toolCallFormat = format }
        }

        loadedContainer = container
        loadedModelID = model.id
        return container
    }

    /// The checkpoint's `model_type` (from `config.json`), falling back to the catalog
    /// architecture string. Used to pick the tool-call format.
    nonisolated private static func modelTypeString(_ model: ModelEntry, dir: URL) -> String {
        if let data = try? Data(contentsOf: dir.appendingPathComponent("config.json")),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let modelType = obj["model_type"] as? String {
            return modelType
        }
        return model.architecture ?? ""
    }

    private func showUnsupported(_ type: String, _ model: ModelEntry) {
        let alert = NSAlert()
        alert.messageText = "\(type) Not Yet Supported"
        alert.informativeText = """
        \(model.displayName) is a \(type.lowercased()) model.

        Running \(type.lowercased()) models will be available in a future update.
        Currently only LLM (chat/text) models can be run locally via MLX.
        """
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }
}
