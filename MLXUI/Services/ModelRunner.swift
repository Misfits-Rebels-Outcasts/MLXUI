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

struct ChatMessage: Identifiable, Sendable, Equatable {
    enum Role: String, Codable, Sendable { case user, assistant }
    let id: UUID
    let role: Role
    var text: String
    /// CW-3: files this turn carried. The bubble shows `text` plus chips; the model sees
    /// `ChatAttachment.modelFacing(text:attachments:)`.
    var attachments: [ChatAttachment]

    init(id: UUID = UUID(), role: Role, text: String, attachments: [ChatAttachment] = []) {
        self.id = id
        self.role = role
        self.text = text
        self.attachments = attachments
    }
}

/// Hand-written Codable: `attachments` must be `decodeIfPresent`, or every conversation saved
/// before CW-3 would fail to decode and `ConversationStore` would silently skip it — the user's
/// chat history would vanish (the backlog's trap).
extension ChatMessage: Codable {
    private enum CodingKeys: String, CodingKey { case id, role, text, attachments }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        role = try c.decode(Role.self, forKey: .role)
        text = try c.decode(String.self, forKey: .text)
        attachments = try c.decodeIfPresent([ChatAttachment].self, forKey: .attachments) ?? []
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(role, forKey: .role)
        try c.encode(text, forKey: .text)
        try c.encode(attachments, forKey: .attachments)
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

    /// Where this runner persists conversations. `prepare(for:in:)` sets it, so a test's temp
    /// directory and the app's real one both flow through the same save/delete path (CW-2).
    private var conversationDirectory = ConversationStore.defaultDirectory()

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

    /// Bumped by every `send` and by `stop()`. A turn captures it before its attachment import and
    /// re-checks it after, so a `stop()` during the import invalidates the turn even though there
    /// is no `currentTask` to cancel yet (CW-3-FIX-1 B1).
    private var turnGeneration = 0

    #if DEBUG
    /// Test seam only (CW-3-FIX-1 B1). Replaces the OCR step of the attachment import, which is the
    /// one part of `send` that can take seconds with no model loaded. `nil` in the app.
    @ObservationIgnored var attachmentOCRForTesting: (@Sendable (URL) async -> String?)?

    /// Test seam only. Runs inside `send`, right after the import and the transcript append, before
    /// the weights are touched — so a test can observe the runner mid-turn (busy, generation
    /// claimed, message appended) and then fail the turn on purpose rather than loading a model.
    /// `nil` in the app, which is the only thing that makes the `try` below a no-op.
    @ObservationIgnored var afterImportForTesting: (@Sendable () async throws -> Void)?

    /// Test seam only. Where `send` looks for a model's files; the app always uses `ModelStore`.
    @ObservationIgnored var modelDirectoryForTesting: (@Sendable (String) -> URL)?

    /// Test seam only. Lets `send`'s turn end on purpose (a throw) instead of loading weights —
    /// that is what makes a B1 test able to run the *whole* turn, past the append and the save,
    /// with no model. `nil` in the app.
    @ObservationIgnored var containerForTesting: (@Sendable () async throws -> ModelContainer)?

    /// Test seam only (MLX-ERR-3). Supplies the streaming session **instead of** loading weights, so
    /// a test can drive a whole turn against a stream that reports an MLX error part-way through —
    /// and assert the turn stopped there. It has to sit before the load: no test in this suite can
    /// build a `ModelContainer`, so a seam that replaced the session only *after* `containerForModel`
    /// would be unreachable. `nil` in the app.
    @ObservationIgnored var streamingSessionForTesting: (@Sendable () -> any ChatStreamingSession)?
    #endif

    /// S1-1: loaded weights live in the shared `ModelContainerPool` (chat, flow rows and the
    /// Local Server share one owner), so repeated sends — and a flow row using the same model —
    /// don't reload.
    @ObservationIgnored private let pool: ModelContainerPool

    /// SC-5a (gate D): asks before a turn's model load pushes out a model other apps are being served; the chat sheet
    /// presents its question.
    let loadConfirmer: ServedLoadConfirmer

    init(pool: ModelContainerPool = .shared, loadConfirmer: ServedLoadConfirmer? = nil) {
        self.pool = pool
        self.loadConfirmer = loadConfirmer ?? ServedLoadConfirmer()
    }

    /// SC-5a: `true` to go ahead with the turn. Asks only when the model isn't already resident (a resident one adds
    /// nothing to the ledger) and a served model would be stopped for it.
    private func confirmLoad(_ model: ModelEntry, dir: URL) async -> Bool {
        guard await pool.residentBytes(directory: dir) == nil else { return true }
        let bytes = ModelWeightCharge.bytes(directory: dir, catalogFallback: Int64(model.ramGB * 1_073_741_824))
        return await loadConfirmer.confirm(modelName: model.displayName, loadBytes: bytes, requester: .chat)
    }

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
    /// opt into: `run_shell`, `write_file` and `edit_file` are visible in the tool
    /// list but unticked, so a model can never propose a shell command or a write on
    /// first launch. Per-call approval still gates them once enabled.
    static func defaultEnabledToolNames() -> Set<String> {
        var names = Set(defaultTools().map(\.name))
        #if DIRECT_BUILD
        names.subtract(DirectTools.disabledByDefault)
        #else
        names.subtract(ScopedFileTools.disabledByDefault)
        #endif
        return names
    }

    /// Called when the chat sheet opens. Loads the new model's persisted conversations (most
    /// recent selected, or a fresh empty one). Reopening the same model keeps everything in
    /// place. Switching models no longer frees the old weights here — the `ModelContainerPool`'s
    /// LRU budget decides (S1-1).
    ///
    /// `directory` is injected like `ConversationStore`'s own entry points so a test can pin the
    /// store it reads; the app always takes the default.
    func prepare(
        for model: ModelEntry, in directory: URL = ConversationStore.defaultDirectory()
    ) {
        conversationDirectory = directory
        if preparedModelID != model.id {
            preparedModelID = model.id
            conversations = ConversationStore.conversations(
                forModelID: model.id, in: directory)
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

    // CX-2-FIX-1 (N3): test-only seams must not ship in the Release binary.
    #if DEBUG
    /// Append a user prompt to the working transcript without a model — the test seam for
    /// `exportSnapshot`. `transcript` is `private(set)` so nothing outside the runner can
    /// replace the working copy wholesale; these two hand out the same mutation `send` performs.
    /// The app never calls them.
    func appendUserMessageForTesting(_ text: String) {
        transcript.addUserMessage(text)
    }

    /// Append a streamed assistant chunk without a model — see `appendUserMessageForTesting`.
    func appendAssistantChunkForTesting(_ chunk: String) {
        transcript.appendAssistantChunk(chunk)
    }
    #endif

    /// A conversation to export (CX-2). For the conversation on screen the **live** transcript
    /// wins, because it can be ahead of the saved copy mid-turn — `saveActiveConversation` only
    /// runs when a turn ends, so exporting the stored copy can miss the last turn. Any other
    /// conversation has no working copy, so its stored items are the truth.
    ///
    /// `nil` when there is no such conversation, or when it has no items (D5: nothing to export).
    func exportSnapshot(for id: UUID) -> Conversation? {
        guard let stored = conversations.first(where: { $0.id == id }) else { return nil }
        guard id == activeConversationID else {
            return stored.items.isEmpty ? nil : stored
        }
        var live = stored
        live.items = transcript.items
        return live.items.isEmpty ? nil : live
    }

    /// Delete a conversation from the list and from disk. Deleting the one on screen moves
    /// to the most recent remaining conversation, or a fresh empty one.
    func deleteConversation(_ id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        if id == activeConversationID, isRunning { stop() }
        ConversationStore.delete(id: id, in: conversationDirectory)
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

    // MARK: - Attached folders (CW-2, D1/D2/D5)

    /// The active conversation's attached folders, in attach order. What the composer's chips and
    /// the "+" menu render.
    var activeAttachedFolders: [String] {
        conversations.first { $0.id == activeConversationID }?.attachedFolders ?? []
    }

    /// The active conversation's working folder, or its first attached folder (C8).
    var activeWorkingFolder: String? {
        conversations.first { $0.id == activeConversationID }?.effectiveWorkingFolder
    }

    /// Attach a folder to the active chat.
    ///
    /// App Store: **grants it first** when no grant covers the path, so "+" → Add folder is granting
    /// and attaching in one step (D1). Direct: records the path and bumps the C9 MRU. In both, the
    /// first folder attached becomes the working folder (C8).
    func attachFolder(_ url: URL) {
        let path = url.path
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID })
        else { return }
        #if !DIRECT_BUILD
        if FolderGrants.grant(covering: path, in: FolderGrants.load()) == nil {
            grantFolderAccess(to: url)
            // B5: if the grant did not take (the panel's bookmark could not be created, say),
            // attaching would put a chip on a folder every tool then filters out. Refuse instead;
            // `grantFolderAccess` has already set `errorMessage`.
            guard FolderGrants.grant(covering: path, in: FolderGrants.load()) != nil else { return }
        }
        #endif
        recordAttachment(path, at: index)
    }

    /// The shared tail of both attach routes: add the path unless the same folder is already
    /// attached under another spelling, default the working folder, persist.
    private func recordAttachment(_ path: String, at index: Int) {
        if !FolderGrants.contains(conversations[index].attachedFolders, path) {
            conversations[index].attachedFolders.append(path)
        }
        if conversations[index].workingFolder == nil {
            conversations[index].workingFolder = conversations[index].attachedFolders.first
        }
        persistConversation(at: index)
        #if DIRECT_BUILD
        Self.recordRecentFolder(path)
        #endif
    }

    /// Attach a path that is already granted (the "+" menu's one-click list, or a test).
    func attachFolder(path: String) {
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID })
        else { return }
        recordAttachment(path, at: index)
    }

    /// Detach a folder. Removing the working folder promotes the next attached one, or clears it
    /// (C8).
    func detachFolder(path: String) {
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID })
        else { return }
        conversations[index].attachedFolders.removeAll { FolderGrants.samePath($0, path) }
        if let working = conversations[index].workingFolder, FolderGrants.samePath(working, path) {
            conversations[index].workingFolder = conversations[index].attachedFolders.first
        }
        persistConversation(at: index)
    }

    /// Choose which attached folder relative paths resolve against. A path that is not attached is
    /// ignored.
    func setWorkingFolder(path: String) {
        guard let index = conversations.firstIndex(where: { $0.id == activeConversationID }),
              let attached = conversations[index].attachedFolders.first(where: { FolderGrants.samePath($0, path) })
        else { return }
        conversations[index].workingFolder = attached
        persistConversation(at: index)
    }

    /// Save an attach/detach/working-folder change straight to disk.
    ///
    /// Deliberately **not** `saveActiveConversation()`: that returns early unless `items` changed,
    /// so a folder attached to a chat with history would never be written (CW-2 trap). A blank
    /// conversation stays memory-only, exactly as before — it is persisted by the first send.
    private func persistConversation(at index: Int) {
        conversations[index].updatedAt = Date()
        guard !conversations[index].items.isEmpty else { return }
        ConversationStore.save(conversations[index], in: conversationDirectory)
    }

    #if DIRECT_BUILD
    private static let recentFoldersKey = "chatRecentFolders"

    /// Direct has no grants, so "+" lists a most-recently-used set instead (C9, ≤ 8 paths).
    static func recentFolders(from defaults: UserDefaults = .standard) -> [String] {
        defaults.stringArray(forKey: recentFoldersKey) ?? []
    }

    static func recordRecentFolder(_ path: String, in defaults: UserDefaults = .standard) {
        var recent = recentFolders(from: defaults).filter { !FolderGrants.samePath($0, path) }
        recent.insert(path, at: 0)
        defaults.set(Array(recent.prefix(8)), forKey: recentFoldersKey)
    }
    #endif

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
            ConversationStore.save(conversations[index], in: conversationDirectory)
        }
        if index != 0 {
            let conversation = conversations.remove(at: index)
            conversations.insert(conversation, at: 0)
        }
    }

    /// K1's copier: `AttachmentImporter.copy` inside the covering grant's security scope when
    /// there is one, and the plain call when there is not.
    static func scopedCopier() -> (URL, URL, Int) throws -> URL {
        { source, directory, ordinal in
            #if DIRECT_BUILD
            return try AttachmentImporter.copy(source, into: directory, ordinal: ordinal)
            #else
            let grants = FolderGrants.load()
            guard FolderGrants.grant(covering: source.path, in: grants) != nil else {
                return try AttachmentImporter.copy(source, into: directory, ordinal: ordinal)
            }
            guard let copied = FolderGrants.withScopedRead(
                to: source,
                grants: grants,
                container: ModelStore.shared.baseDirectory,
                body: { scoped in
                    try AttachmentImporter.copy(scoped, into: directory, ordinal: ordinal)
                }
            ) else {
                throw CocoaError(.fileReadNoPermission)
            }
            return copied
            #endif
        }
    }

    /// The OCR step for one attachment copy: the test override when there is one, else the real
    /// `OCRImageTool` entry point.
    private static func ocrForAttachment(
        override: (@Sendable (URL) async -> String?)?
    ) -> @Sendable (URL) async -> String? {
        if let override { return override }
        return { stored in await OCRImageTool.recognizeText(in: stored) }
    }

    /// Where a model's files are: the test seam when set, else `ModelStore`.
    private func modelDirectory(for model: ModelEntry) -> URL {
        #if DEBUG
        modelDirectoryForTesting?(model.id) ?? ModelStore.shared.directory(forModelID: model.id)
        #else
        ModelStore.shared.directory(forModelID: model.id)
        #endif
    }

    /// CW-4: the environment block the next send would carry, rebuilt on demand for the inspector.
    /// Shares `environmentBlock(…)` with `send`, so the meter measures the text the model actually
    /// gets rather than a second guess at it.
    func currentEnvironmentBlock(for model: ModelEntry) -> String? {
        let conversation = conversations.first { $0.id == activeConversationID }
        #if DIRECT_BUILD
        let edition = FileScope.Edition.direct
        let grants: [FolderGrant] = []
        let homePath: String? = FileManager.default.homeDirectoryForCurrentUser.path
        #else
        let edition = FileScope.Edition.appStore
        let grants = FolderGrants.load()
        let homePath: String? = nil
        #endif
        return environmentBlock(
            conversation: conversation, edition: edition, grants: grants,
            homePath: homePath, attachmentsNote: nil, modelDirectory: modelDirectory(for: model))
    }

    /// One place builds the C3 block: `send` and the inspector both call this.
    private func environmentBlock(
        conversation: Conversation?,
        edition: FileScope.Edition,
        grants: [FolderGrant],
        homePath: String?,
        attachmentsNote: String?,
        modelDirectory: URL
    ) -> String? {
        // CW-FIX: a template that can't render tools gets no folder or tool text. `.unknown` stays
        // conservative (the block is kept). Reads `model_type` the way the Local Server does.
        let modelType = ModelContainerPool.modelTypeString(
            configData: ModelContainerPool.configData(directory: modelDirectory)) ?? ""
        if ToolCallFormatTable.toolSupport(forModelType: modelType) == .templateIgnoresTools { return nil }
        let scope = FileScope.forConversation(
            attached: conversation?.attachedFolders ?? [],
            workingFolder: conversation?.effectiveWorkingFolder,
            grants: grants,
            edition: edition)
        return ChatEnvironment.instructions(
            scope: scope,
            edition: edition,
            fileToolsEnabled: enabledToolNames,
            writeEnabled: enabledToolNames.contains("write_file")
                || enabledToolNames.contains("edit_file"),
            homePath: homePath,
            attachmentsNote: attachmentsNote)
    }

    /// Import the files this message carries (CW-3): copy into the conversation's attachments
    /// folder, extract text, apply the C11 budget. Returns those that made it — a refusal is
    /// reported and skipped, never a reason to lose the message. `async` because the image branch
    /// awaits OCR.
    private func prepareAttachments(
        _ urls: [URL], for model: ModelEntry, in modelDirectory: URL, of conversationID: UUID
    ) async -> [ChatAttachment] {
        guard !urls.isEmpty else { return [] }
        var chosen = urls
        if chosen.count > ChatAttachment.maxFilesPerMessage {
            errorMessage = "Only the first \(ChatAttachment.maxFilesPerMessage) files were"
                + " attached; the rest were left out."
            chosen = Array(chosen.prefix(ChatAttachment.maxFilesPerMessage))
        }
        let window = ContextWindow.tokens(for: model, modelDirectory: modelDirectory)
        let directory = ConversationStore.attachmentsDirectory(for: conversationID)
        var result: [ChatAttachment] = []
        var inlineBytes = 0
        // K1: a file inside a granted folder is only readable inside that grant's security scope.
        // A panel-picked file and a dropped file carry the extension the panel/drop handed out, but
        // "Attach to message" from the sidebar tree hands over a plain path — so after a relaunch
        // the copy would fail with "Operation not permitted". Same class as CW-A F1. Only the
        // **copy** needs the scope: extraction and OCR read the container copy.
        let copier = Self.scopedCopier()
        // R1 (CW-3-FIX-2): the seam only exists in DEBUG, so the reference must be inside an `#if`
        // too — otherwise the Release configuration does not compile at all.
        #if DEBUG
        let ocrOverride = attachmentOCRForTesting
        #else
        let ocrOverride: (@Sendable (URL) async -> String?)? = nil
        #endif
        for (index, url) in chosen.enumerated() {
            do {
                let attachment = try await AttachmentImporter.importAttachment(
                    from: url, into: directory, ordinal: index, window: window,
                    inlineBytesSoFar: inlineBytes,
                    ocr: Self.ocrForAttachment(override: ocrOverride),
                    copier: copier)
                inlineBytes += attachment.inlineText?.utf8.count ?? 0
                result.append(attachment)
            } catch let failure as AttachmentImporter.Failure {
                errorMessage = failure.message
            } catch {
                errorMessage = "Could not attach \(url.lastPathComponent): \(error.localizedDescription)"
            }
        }
        return result
    }

    /// Generate a reply to `prompt`, running the agent tool loop via `AgentSession` and
    /// streaming assistant text + tool-call cards into the transcript as they arrive.
    ///
    /// Returns `false` only when the user **declined** the load question (SC-5a-b): the turn never started, so the caller
    /// can give the composer back what it cleared. Every other exit — sent, ignored, failed — is `true`.
    @discardableResult
    func send(_ prompt: String, attachments: [URL] = [], for model: ModelEntry) async -> Bool {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        // M1: a file on its own is a turn — "what is in this file?" needs no typed text. The text
        // check that used to live here now runs *after* the import, against what actually arrived.
        guard !trimmed.isEmpty || !attachments.isEmpty, !isRunning else { return true }

        #if DEBUG
        let modelDir = modelDirectoryForTesting?(model.id)
            ?? ModelStore.shared.directory(forModelID: model.id)
        #else
        let modelDir = ModelStore.shared.directory(forModelID: model.id)
        #endif

        guard FileManager.default.fileExists(atPath: modelDir.path) else {
            errorMessage = "Model files not found. Please reinstall."
            return true
        }

        // SC-7 (gate F): in Dedicated server mode no model loads here. false = nothing sent, so the composer keeps what was typed.
        if let refusal = RunAdmission.canLoadModel(surface: .chat) {
            errorMessage = refusal.sentence
            return false
        }

        errorMessage = nil

        // Prior turns become the model's rehydrated history; the new prompt goes on top.
        if activeConversationID == nil {
            preparedModelID = model.id
            startNewConversation(modelID: model.id)
        }
        let history = Conversation.chatHistory(from: transcript.items)
        if let index = conversations.firstIndex(where: { $0.id == activeConversationID }),
           conversations[index].items.isEmpty, conversations[index].title == Conversation.untitled,
           !trimmed.isEmpty {
            conversations[index].title = Conversation.title(forFirstPrompt: trimmed)
        }

        // B1: the turn is claimed **before** the import. `isRunning` is what the composer and the
        // Send button read, so a second send during a slow OCR import is a no-op rather than a
        // second turn against one model; the existing spinner covers the import.
        isRunning = true
        turnGeneration += 1
        let generation = turnGeneration
        let turnConversationID = activeConversationID

        // CW-3: copy + extract the attachments before the append, so the bubble and the persisted
        // turn both carry the final result. A file that cannot be attached reports and drops out;
        // the message still sends.
        var readyAttachments: [ChatAttachment] = []
        if !attachments.isEmpty, let conversationID = turnConversationID {
            readyAttachments = await prepareAttachments(
                attachments, for: model, in: modelDir, of: conversationID)
        }

        // B1: the import can take seconds, so re-check what it was started for. A `stop()` bumps
        // the generation and a conversation switch moves `activeConversationID`.
        #if DEBUG
        // B1's test seam sits **before** the append: a test can move the conversation or stop the
        // runner while the turn is parked, which is exactly the window the fix is about. Without
        // the seam this is a no-op and the re-check below runs immediately after the import.
        if let hook = afterImportForTesting { try? await hook() }
        #endif
        guard generation == turnGeneration,
              activeConversationID == turnConversationID,
              isRunning
        else { return true }
        // M1: nothing to send. `RunChatView` let an attachment-only message through and cleared its
        // chips, so if every file then failed to import there is no text and no attachment — an
        // empty turn would be appended and a generation spent on nothing. The import has already
        // reported why (`errorMessage`), so keep that, release the turn and send nothing.
        guard !trimmed.isEmpty || !readyAttachments.isEmpty else {
            isRunning = false
            return true
        }

        // SC-5a (gate D): ask before the turn if loading this model would stop keeping a served one ready. Cancel: the
        // turn doesn't start — nothing appended, nothing loaded, no error banner.
        guard await confirmLoad(model, dir: modelDir) else {
            if generation == turnGeneration { isRunning = false }
            return false
        }
        guard generation == turnGeneration, activeConversationID == turnConversationID, isRunning else { return true }

        transcript.addUserMessage(trimmed, attachments: readyAttachments)
        // CW-C-FIX-1: the model gets the same thing the transcript shows. `send` used to pass the
        // bare `trimmed` text as the prompt, so an attachment's contents reached the **bubble** and
        // `chatHistory` but not the turn that needed them — the model answered "you'll need to
        // attach it first" while looking at a message that already had it. `chatHistory` covers
        // earlier turns; this covers the turn being sent.
        let modelPrompt = ChatAttachment.prompt(text: trimmed, attachments: readyAttachments)

        // Snapshot the enabled tools for this turn (Sendable to hand to the nonisolated session).
        let registry = ToolRegistry(availableTools.filter { enabledToolNames.contains($0.name) })

        // CW-2: the scope is the conversation's attached folders (G1), rebuilt on *every* turn so a
        // revoke or a detach takes effect on the next message.
        let conversation = conversations.first { $0.id == activeConversationID }
        #if DIRECT_BUILD
        let edition = FileScope.Edition.direct
        let grants: [FolderGrant] = []
        let homePath: String? = FileManager.default.homeDirectoryForCurrentUser.path
        #else
        let edition = FileScope.Edition.appStore
        let grants = FolderGrants.load()
        let homePath: String? = nil
        #endif
        let fileScope = FileScope.forConversation(
            attached: conversation?.attachedFolders ?? [],
            workingFolder: conversation?.effectiveWorkingFolder,
            grants: grants,
            edition: edition)
        let attachmentsNote = ChatAttachment.environmentNote(for: readyAttachments)
        let instructions = environmentBlock(
            conversation: conversation, edition: edition, grants: grants,
            homePath: homePath, attachmentsNote: attachmentsNote,
            modelDirectory: modelDir)

        currentTask = Task { [weak self] in
            guard let self else { return }
            let turnLease = TurnLease()
            defer { turnLease.release() }      // SC-2a: the model is in use for the turn — return it on every exit
            do {
                // MLX-ERR-3: the session (and with it the weights load) is built through a factory
                // so the test seam can replace it, and the whole turn — load, generation and the
                // in-app MLX tools that run through the session's tool dispatch on this same task —
                // sits inside one guard.
                let session = try await self.streamingSession(
                    for: model, dir: modelDir, instructions: instructions, history: history,
                    registry: registry, fileScope: fileScope,
                    turnConversationID: turnConversationID, turn: turnLease)

                // R-ME3-1 (owner, 2026-10-12): **stop at the error.** `makeStream` must be a
                // closure: the producer's `Task {}` only inherits mlx-swift's `@TaskLocal` handler
                // if it is created inside the guard (journal 2026-469).
                try await MLXErrorGuard.guardedStream({ session.streamResponse(to: modelPrompt) }) { chunk in
                    // A late chunk must not leak into a conversation the user switched to
                    // (switching stops the task, but a chunk can already be in flight).
                    guard self.activeConversationID == turnConversationID else { return false }
                    self.transcript.appendAssistantChunk(chunk)
                    return true
                }
            } catch is CancellationError {
                // stopped by the user — leave the partial transcript in place
            } catch {
                if !Task.isCancelled {
                    self.errorMessage = "Failed: \(MLXErrorGuard.sentence(for: error))"
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
        return true
    }

    func stop() {
        // B1: bump before anything else, so an in-flight attachment import (which has no
        // `currentTask` yet) is invalidated too.
        turnGeneration += 1
        loadConfirmer.respond(false)        // SC-5a: a question still open is Cancel
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
    /// tools whose `requiresApproval` is true. The call's arguments ride along so the sheet can
    /// show what is being approved (EF-3, D10); nothing else about the gate changes.
    private func requestApproval(for tool: any AgentTool, call: ToolCall) async -> Bool {
        await withCheckedContinuation { continuation in
            self.pendingApproval = PendingApproval(
                toolName: tool.name,
                toolDescription: tool.toolDescription,
                arguments: call.function.arguments,
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
        case .image:
            showUnsupported("Image generation", model)
        case .music:
            showUnsupported("Music generation", model)
        case .segmentation:
            showUnsupported("Image segmentation", model)
        case .upscale:
            showUnsupported("Image upscaling", model)
        case .rerank:
            // Dead once RerankSDK is registered (MoC-4-3, RSI/DelegateMoCBacklog.md):
            // AppState.runModel(_:) checks the registry before ever calling here, and
            // RerankSDK claims every .rerank model — so this case only fires if this
            // legacy entry point is ever called directly, bypassing the registry check.
            // RerankUI.makeRunView is what a user actually sees (an explanation that
            // Rerank is a headless flow task, not a chat-style surface).
            showUnsupported("Rerank", model)
        case .decision:
            // Dead once LayaSDK is registered (LY-4, RSI/DelegateLayaBacklog.md): the
            // registry check ahead of this entry point will claim every .decision model,
            // the same way MoC-4-3 made the .rerank case above unreachable in practice.
            showUnsupported("Decision", model)
        case .diarization:
            // MP4-1: no SortformerSDK is registered yet, so AppState.runModel(_:) lands here and the user sees
            // the "unsupported" alert. Becomes unreachable once MP4-3 registers the module (same as .rerank above).
            showUnsupported("Speaker diarization", model)
        case .enhance:
            // Unreachable in practice since MP5-3a registered DeepFilterNetModule (same as .diarization above).
            showUnsupported("Audio cleanup", model)
        case .voice:
            // Unreachable in practice since MP6-3a registered LFMAudioModule (same as .enhance above).
            showUnsupported("Voice conversation", model)
        case .interpolate:
            // Unreachable in practice since RIFE-3d registered RIFEModule (same as .voice above).
            showUnsupported("Frame interpolation", model)
        }
    }

    /// The container for `model`, from the shared pool (which also wires the tool-call format).
    func containerForModel(_ model: ModelEntry, dir: URL) async throws -> ModelContainer {
        #if DEBUG
        if let injected = containerForTesting { return try await injected() }
        #endif
        return try await pool.container(for: model, directory: dir)
    }

    /// SC-2a: the container for one chat turn, **leased** — in use until the turn ends, so another model's load
    /// (a flow row, a Run UI, a server request) can't evict it mid-reply. The lease goes into `turn`, which `send`
    /// returns on every exit. With the test seam, nothing is leased.
    private func leasedContainerForModel(_ model: ModelEntry, dir: URL, turn: TurnLease) async throws -> ModelContainer {
        #if DEBUG
        if let injected = containerForTesting { return try await injected() }
        #endif
        let lease = try await pool.lease(for: model, directory: dir, holder: .chat)
        turn.hold = lease.hold
        return lease.container
    }

    /// The memory lease a chat turn holds (SC-2a); returned by `release()` or when the turn's task lets go of it.
    private final class TurnLease: @unchecked Sendable {
        var hold: MemoryBudget.Hold?
        func release() { hold?.release(); hold = nil }
    }

    /// Build the streaming session for one turn — the single place `send` gets one (MLX-ERR-3).
    ///
    /// Extracted from `send` so the test seam can replace the **whole** session, weights load
    /// included: `streamingSessionForTesting` is checked first, so a test never needs a
    /// `ModelContainer`, which nothing in this suite can build. The `parameters`/`approve`/`onEvent`
    /// wiring is unchanged from when it lived inline in `send`.
    private func streamingSession(
        for model: ModelEntry,
        dir: URL,
        instructions: String?,
        history: [Chat.Message],
        registry: ToolRegistry,
        fileScope: FileScope?,
        turnConversationID: UUID?,
        turn: TurnLease
    ) async throws -> any ChatStreamingSession {
        #if DEBUG
        if let injected = streamingSessionForTesting { return injected() }
        #endif
        let container = try await leasedContainerForModel(model, dir: dir, turn: turn)
        return AgentSession(
            model: container,
            instructions: instructions,
            history: history,
            registry: registry,
            fileScope: fileScope,
            // 512 was too tight once reasoning models entered the picture:
            // Qwen3's <think> block alone can run 250–500 tokens, so the
            // generation was being truncated *before* the tool call was
            // emitted. The symptom is indistinguishable from "the model
            // chose not to call a tool" — reasoning text, then nothing.
            parameters: GenerateParameters(maxTokens: 2048, temperature: 0.7),
            maxToolCalls: agentToolSettings.toolCallLimit,
            approve: { [weak self] call, tool in
                switch await self?.approvalDecision(for: tool) {
                case .autoApprove:     return true
                case .prompt:
                    return await self?.requestApproval(for: tool, call: call) ?? false
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
