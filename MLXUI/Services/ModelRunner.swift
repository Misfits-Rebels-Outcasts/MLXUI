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

struct ChatMessage: Identifiable {
    enum Role { case user, assistant }
    let id = UUID()
    let role: Role
    var text: String
}

// MARK: - Model Runner

@Observable
final class ModelRunner {
    /// Conversation shown in the Run chat sheet.
    var transcript: [ChatMessage] = []
    var isRunning = false
    var errorMessage: String?

    private var currentTask: Task<Void, Never>?

    // Cache the loaded model so repeated sends in one session don't reload weights.
    private var loadedContainer: ModelContainer?
    private var loadedModelID: String?

    /// Called when the chat sheet opens. Resets history when switching models and
    /// frees the previously loaded model's weights.
    func prepare(for model: ModelEntry) {
        if loadedModelID != model.id {
            loadedContainer = nil
            loadedModelID = nil
            transcript.removeAll()
        }
        errorMessage = nil
    }

    /// Generate a reply to `prompt` for the given (LLM) model, streaming tokens into
    /// the transcript as they arrive.
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
        transcript.append(ChatMessage(role: .user, text: trimmed))
        let assistant = ChatMessage(role: .assistant, text: "")
        transcript.append(assistant)
        let assistantID = assistant.id
        isRunning = true

        currentTask = Task {
            do {
                let container = try await containerForModel(model, dir: modelDir)
                let result = try await container.perform { context in
                    let input = try await context.processor.prepare(input: UserInput(prompt: trimmed))
                    return try MLXLMCommon.generate(
                        input: input,
                        parameters: GenerateParameters(maxTokens: 512, temperature: 0.7),
                        context: context
                    ) { tokens in
                        // `tokens` is cumulative — decode the whole reply and replace
                        // the assistant message text (don't append, or it duplicates).
                        let text = context.tokenizer.decode(tokenIds: tokens)
                        Task { @MainActor in
                            self.updateAssistant(id: assistantID, text: text)
                        }
                        return tokens.count >= 512 ? .stop : .more
                    }
                }
                await MainActor.run {
                    if !result.output.isEmpty {
                        self.updateAssistant(id: assistantID, text: result.output)
                    }
                }
            } catch {
                if !Task.isCancelled {
                    await MainActor.run {
                        self.errorMessage = "Failed: \(error.localizedDescription)"
                    }
                }
            }
            await MainActor.run {
                self.isRunning = false
                self.currentTask = nil
            }
        }
    }

    func stop() {
        currentTask?.cancel()
        currentTask = nil
        isRunning = false
    }

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
        loadedContainer = container
        loadedModelID = model.id
        return container
    }

    @MainActor
    private func updateAssistant(id: UUID, text: String) {
        guard let idx = transcript.firstIndex(where: { $0.id == id }) else { return }
        transcript[idx].text = text
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
