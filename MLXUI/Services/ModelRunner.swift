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

struct ChatMessage: Identifiable, Codable, Sendable, Equatable {
    enum Role: String, Codable, Sendable { case user, assistant }
    let id: UUID
    let role: Role
    var text: String

    init(id: UUID = UUID(), role: Role, text: String) {
        self.id = id
        self.role = role
        self.text = text
    }
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
    /// This is the working copy of the **active conversation's** items.
    private(set) var transcript = TranscriptBuilder()
    var isRunning = false
    var errorMessage: String?

    /// Persisted conversations for the current model, newest first, and the one on screen.
    private(set) var conversations: [Conversation] = []
    private(set) var activeConversationID: UUID?
    /// The model whose conversations are loaded (set in `prepare`, independent of weight
    /// loading, which happens lazily on the first send).
    private var preparedModelID: String?

    /// A tool call currently blocked on user approval (drives the approval sheet).
    var pendingApproval: PendingApproval?

    /// Tools available to the agent this build, and which are currently enabled. The enabled
    /// set is the build default with the user's persisted toggles applied (AG6b-2) — mutate it
    /// through `setToolEnabled(_:for:)` so the choice sticks across launches.
    let availableTools: [any AgentTool] = ModelRunner.defaultTools()
    private(set) var enabledToolNames: Set<String> = AgentToolSettings.load()
        .enabledToolNames(defaults: ModelRunner.defaultEnabledToolNames())

    /// Persisted agent-tool configuration (AG6b): per-tool approval policy (`.ask` / `.always`
    /// / `.never`), per-tool enabled overrides, and the per-reply tool-call limit.
    private(set) var agentToolSettings: AgentToolSettings = .load()

    /// Session audit trail of tool activity (AG6b-2). In-memory only — tool results can
    /// contain user file contents, so it is never written to disk.
    private(set) var auditLog = AgentAuditLog()

    /// Folders the user has granted the file tools access to (AG5) — mirrors the persisted
    /// `FolderGrants` store so the wrench menu updates live.
    private(set) var folderGrantPaths: [String] = FolderGrants.grantedPaths()

    private var currentTask: Task<Void, Never>?

    // Cache the loaded model so repeated sends in one session don't reload weights.
    private var loadedContainer: ModelContainer?
    private var loadedModelID: String?

    /// Tools that ship in this build. `fetch_url` (AG2), the AG3 compute tools, and the AG4 in-app
    /// MLX tools (`embed_text`, `summarize`, `semantic_search`, `transcribe_audio`, `ocr_image`)
    /// ship in all builds; DEBUG additionally carries the `echo` demo tool.
    static func defaultTools() -> [any AgentTool] {
        var tools: [any AgentTool] = [
            FetchURLTool(),
            RunJavaScriptTool(),
            CalculatorTool(),
            DateTimeTool(),
            ReadClipboardTool(),
            WriteClipboardTool(),
            EmbedTextTool(),
            SummarizeTool(),
            SemanticSearchTool(),
            TranscribeAudioTool(),
            OCRImageTool(),
        ]
        #if DEBUG
        tools.append(EchoDemoTool())
        #endif
        // Shell, filesystem and subprocess tools ship only in the notarized
        // direct-download edition — the sandbox blocks them anyway, and the
        // implementations aren't compiled into the App Store target at all
        // (`Apps/Direct/` is synchronized into MLXUI-Direct only).
        // See Design/dual-distribution.md.
        // The sandboxed edition gets the same file-tool names scoped to
        // user-granted folders instead (AG5 — security-scoped bookmarks).
        #if DIRECT_BUILD
        tools.append(contentsOf: DirectTools.all())
        #else
        tools.append(contentsOf: ScopedFileTools.all())
        #endif
        return tools
    }

    /// Which tools start enabled. Everything except the ones a user should have to
    /// opt into: `run_shell` and `write_file` are visible in the tool list but
    /// unticked, so a model can never propose a shell command on first launch.
    /// Per-call approval still gates them once enabled.
    static func defaultEnabledToolNames() -> Set<String> {
        var names = Set(defaultTools().map(\.name))
        #if DIRECT_BUILD
        names.subtract(DirectTools.disabledByDefault)
        #else
        names.subtract(ScopedFileTools.disabledByDefault)
        #endif
        return names
    }

    /// Called when the chat sheet opens. Frees the previous model's weights when switching
    /// models and loads the new model's persisted conversations (most recent selected, or a
    /// fresh empty one). Reopening the same model keeps everything in place.
    func prepare(for model: ModelEntry) {
        if loadedModelID != model.id {
            loadedContainer = nil
            loadedModelID = nil
        }
        if preparedModelID != model.id {
            preparedModelID = model.id
            conversations = ConversationStore.conversations(forModelID: model.id)
            if let mostRecent = conversations.first {
                activeConversationID = mostRecent.id
                transcript.load(mostRecent.items)
            } else {
                startNewConversation(modelID: model.id)
            }
        }
        errorMessage = nil
    }

    // MARK: - Conversations

    /// Start a new conversation (an existing empty one is reused instead of stacking blanks).
    func newConversation() {
        guard let modelID = preparedModelID else { return }
        if isRunning { stop() }
        saveActiveConversation()
        if let existing = conversations.first(where: { $0.items.isEmpty }) {
            activeConversationID = existing.id
            transcript.load(existing.items)
            return
        }
        startNewConversation(modelID: modelID)
    }

    /// Show another conversation; an in-flight generation is stopped and the partial
    /// transcript saved first.
    func selectConversation(_ id: UUID) {
        guard id != activeConversationID,
              let conversation = conversations.first(where: { $0.id == id })
        else { return }
        if isRunning { stop() }
        saveActiveConversation()
        activeConversationID = id
        transcript.load(conversation.items)
    }

    /// Delete a conversation from the list and from disk. Deleting the one on screen moves
    /// to the most recent remaining conversation, or a fresh empty one.
    func deleteConversation(_ id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        if id == activeConversationID, isRunning { stop() }
        ConversationStore.delete(id: id)
        conversations.remove(at: index)
        guard id == activeConversationID else { return }
        if let next = conversations.first {
            activeConversationID = next.id
            transcript.load(next.items)
        } else if let modelID = preparedModelID {
            startNewConversation(modelID: modelID)
        } else {
            activeConversationID = nil
            transcript.reset()
        }
    }

    private func startNewConversation(modelID: String) {
        let conversation = Conversation(modelID: modelID)
        conversations.insert(conversation, at: 0)
        activeConversationID = conversation.id
        transcript.reset()
    }

    /// Sync the working transcript into the active conversation, bump it to the top, and
    /// persist. No-op when nothing changed; empty conversations stay memory-only so blank
    /// chats never litter the disk.
    private func saveActiveConversation() {
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID }),
              conversations[index].items != transcript.items
        else { return }
        conversations[index].items = transcript.items
        conversations[index].updatedAt = Date()
        if !transcript.items.isEmpty {
            ConversationStore.save(conversations[index])
        }
        if index != 0 {
            let conversation = conversations.remove(at: index)
            conversations.insert(conversation, at: 0)
        }
    }

    /// Generate a reply to `prompt`, running the agent tool loop via `AgentSession` and
    /// streaming assistant text + tool-call cards into the transcript as they arrive.
    func send(_ prompt: String, for model: ModelEntry) {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !isRunning else { return }

        let modelDir = ModelStore.shared.directory(forModelID: model.id)

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            errorMessage = "Model files not found. Please reinstall."
            return
        }

        errorMessage = nil

        // Prior turns become the model's rehydrated history; the new prompt goes on top.
        if activeConversationID == nil {
            preparedModelID = model.id
            startNewConversation(modelID: model.id)
        }
        let history = Conversation.chatHistory(from: transcript.items)
        if let index = conversations.firstIndex(where: { $0.id == activeConversationID }),
           conversations[index].items.isEmpty, conversations[index].title == Conversation.untitled {
            conversations[index].title = Conversation.title(forFirstPrompt: trimmed)
        }
        let turnConversationID = activeConversationID

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
                    history: history,
                    registry: registry,
                    // 512 was too tight once reasoning models entered the picture:
                    // Qwen3's <think> block alone can run 250–500 tokens, so the
                    // generation was being truncated *before* the tool call was
                    // emitted. The symptom is indistinguishable from "the model
                    // chose not to call a tool" — reasoning text, then nothing.
                    parameters: GenerateParameters(maxTokens: 2048, temperature: 0.7),
                    maxToolCalls: self.agentToolSettings.toolCallLimit,
                    approve: { [weak self] _, tool in
                        switch await self?.approvalDecision(for: tool) {
                        case .autoApprove:     return true
                        case .prompt:          return await self?.requestApproval(for: tool) ?? false
                        case .autoDeny, .none: return false
                        }
                    },
                    onEvent: { [weak self] event in
                        Task { @MainActor [weak self] in
                            guard let self else { return }
                            // Late events stay out of a switched-to conversation; the
                            // audit log records the session regardless.
                            if self.activeConversationID == turnConversationID {
                                self.transcript.apply(event)
                            }
                            self.auditLog.record(event)
                        }
                    }
                )

                for try await chunk in session.streamResponse(to: trimmed) {
                    // A late chunk must not leak into a conversation the user switched to
                    // (switching stops the task, but a chunk can already be in flight).
                    guard self.activeConversationID == turnConversationID else { break }
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
            // Persist the finished (or interrupted) turn. When the user already switched
            // conversations, the switch saved the partial transcript itself.
            if self.activeConversationID == turnConversationID {
                self.saveActiveConversation()
            }
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

    /// The standing approval policy for a tool (drives the wrench-menu picker).
    func toolPolicy(for toolName: String) -> ToolApprovalPolicy {
        agentToolSettings.policy(for: toolName)
    }

    /// Set and persist a tool's approval policy (AG6b).
    func setToolPolicy(_ policy: ToolApprovalPolicy, for toolName: String) {
        agentToolSettings = agentToolSettings.setting(policy, for: toolName)
        agentToolSettings.save()
    }

    /// The gate decision for an approval-required tool, from its persisted policy.
    private func approvalDecision(for tool: any AgentTool) -> ApprovalDecision {
        agentToolSettings.decision(for: tool.name)
    }

    // MARK: - Tool enable / budget / audit (AG6b-2)

    /// Toggle a tool for the agent and persist the choice across launches.
    func setToolEnabled(_ enabled: Bool, for toolName: String) {
        agentToolSettings = agentToolSettings.settingEnabled(
            enabled, for: toolName,
            defaultEnabled: Self.defaultEnabledToolNames().contains(toolName))
        agentToolSettings.save()
        if enabled { enabledToolNames.insert(toolName) }
        else { enabledToolNames.remove(toolName) }
    }

    /// Per-reply tool-call budget (the `ToolBudget` limit handed to `AgentSession`).
    var toolCallLimit: Int { agentToolSettings.toolCallLimit }

    /// Set and persist the per-reply tool-call budget; applies from the next send.
    func setToolCallLimit(_ limit: Int) {
        agentToolSettings = agentToolSettings.settingToolCallLimit(limit)
        agentToolSettings.save()
    }

    func clearAuditLog() {
        auditLog.clear()
    }

    // MARK: - Folder grants (AG5)

    /// Persist a security-scoped grant for a folder the user picked in the open panel.
    func grantFolderAccess(to url: URL) {
        do {
            try FolderGrants.addGrant(for: url)
            folderGrantPaths = FolderGrants.grantedPaths()
        } catch {
            errorMessage = "Could not save folder access: \(error.localizedDescription)"
        }
    }

    /// Remove a granted folder (takes effect on the next tool call).
    func revokeFolderAccess(path: String) {
        FolderGrants.removeGrant(path: path)
        folderGrantPaths = FolderGrants.grantedPaths()
    }

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
