import Foundation
import CoreGraphics
import CoreImage
import MLX
import MLXLMCommon
import MLXNN
import MLXVLM

/// MC4-4/MC4-5 — the Qwen3-VL Embedding 2B multimodal embedder.
///
/// Every input, queries and documents alike, is the model card's conversation: a `system` message
/// `"Represent the user's input."`, then a `user` message whose content is an image item first and
/// a text item after it, rendered by the model's own chat template with
/// `add_generation_prompt=True` (decision 3). The vector is the **last token's** final-norm hidden
/// state, all 2048 dimensions, L2-normalised. No per-task instruction, no Matryoshka truncation.
///
/// Image preprocessing goes through mlx-swift-lm's **public** `Qwen3VLProcessor.preprocess` (its
/// `Qwen2VLImageProcessorFast` resize/patchify is the model's own settings: min 4,096 / max
/// 1,310,720 pixels ⇒ at most 1,280 vision tokens). Prompt rendering is the processor's own recipe
/// too — `Qwen3VLMessageGenerator` + the chat template + the pad expansion — but tokenised here
/// rather than through `Qwen3VLProcessor.prepare`, because the reference recipe's final vector is
/// the token **after** the tokenizer's post-processor appends `<|endoftext|>` (the official
/// `scripts/qwen3_vl_embedding.py` tokenises with `add_special_tokens=True`); `prepare` returns the
/// template tokens alone. The forward pass is the trimmed copy in `Qwen3VLForward.swift`; DeepStack
/// and interleaved M-RoPE stay exactly as upstream has them (issue Trap 3).
///
/// Batching is implementer's call, pending owner confirmation: **one input at a time**, each
/// pooling its own last real token (the issue's "or run one input at a time"). It keeps the memory
/// of an Images-tab run at one forward pass and makes the batching equality test a statement about
/// the recipe, not about padding.
nonisolated final class Qwen3VLEmbedder {

    /// The recipe's system prompt (decision 3), the model card's default instruction.
    static let systemPrompt = "Represent the user's input."
    /// Decision 4: this version accepts 8,192 tokens after rendering (the model supports 32K).
    static let maxInputTokens = 8_192
    /// The model's embedding width — all of it, no truncation (decision 3).
    static let dimensions = 2_048

    private let model: Qwen3VLEmbedModel
    private let processor: Qwen3VLProcessor
    private let tokenizer: any MLXLMCommon.Tokenizer
    /// The tokenizer's post-processor special tokens (the reference's trailing `<|endoftext|>`,
    /// id 151643) — appended exactly when `add_special_tokens=True` would append them.
    private let appendedTokenIDs: [Int]
    private let visionStartTokenID: Int
    private let visionEndTokenID: Int
    private let imageTokenID: Int
    private let mergeSize: Int

    private init(
        model: Qwen3VLEmbedModel, processor: Qwen3VLProcessor,
        tokenizer: any MLXLMCommon.Tokenizer, configuration: Qwen3VLConfiguration,
        appendedTokenIDs: [Int]
    ) {
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.appendedTokenIDs = appendedTokenIDs
        self.visionStartTokenID = configuration.visionStartTokenId
        self.visionEndTokenID = configuration.visionEndTokenId
        self.imageTokenID = configuration.imageTokenIndex
        self.mergeSize = configuration.visionConfiguration.spatialMergeSize
    }

    /// Load the **installed** model directory (the A2 pattern): tokenizer and processor config from
    /// the repo files, then the checkpoint through the trimmed forward pass.
    static func load(from directory: URL) async throws -> Qwen3VLEmbedder {
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw StageError.modelNotInstalled(id: directory.lastPathComponent)
        }
        do {
            let tokenizer = try await HFTokenizerLoader().load(from: directory)
            let processorData = try Data(
                contentsOf: directory.appendingPathComponent("preprocessor_config.json"))
            let processorConfig = try JSONDecoder().decode(
                Qwen3VLProcessorConfiguration.self, from: processorData)
            let processor = Qwen3VLProcessor(processorConfig, tokenizer: tokenizer)
            let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
            let configuration = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: configData)
            let model = try Qwen3VLEmbedModel.load(from: directory)
            return Qwen3VLEmbedder(
                model: model, processor: processor, tokenizer: tokenizer,
                configuration: configuration,
                appendedTokenIDs: Self.appendedSpecialTokenIDs(in: directory))
        } catch let error as StageError {
            throw error
        } catch let error as Qwen3VLEmbedError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "Qwen3-VL Embedding", underlying: error)
        }
    }

    /// Embed each text into an L2-normalised vector, one input at a time.
    static func embedTexts(_ texts: [String], modelDirectory: URL) async throws -> [[Float]] {
        let embedder = try await load(from: modelDirectory)
        return try await embedder.embedTexts(texts)
    }

    /// Embed each text into an L2-normalised vector, one input at a time.
    func embedTexts(_ texts: [String]) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        for text in texts {
            vectors.append(try embed(.text(text)))
        }
        return vectors
    }

    /// Embed one image, optionally with a caption (the model reads both in the same vector).
    func embedImage(_ image: CGImage, text: String? = nil) async throws -> [Float] {
        try embed(.image(image, text: text))
    }

    // MARK: - The recipe

    private enum Input {
        case text(String)
        case image(CGImage, text: String?)
    }

    /// The tokens/grids one input hands the forward pass, plus the raw token ids for the seams.
    struct PreparedInput {
        let tokenIDs: [Int]
        let pixels: MLXArray?
        let frames: [THW]?
        let mergeSize: Int

        var imagePadCount: Int {
            guard let frames else { return 0 }
            return frames.map { $0.product / (mergeSize * mergeSize) }.reduce(0, +)
        }
    }

    /// Render one input with the reference recipe: the model's chat template, the image placeholder
    /// expanded to the merged patch count, then the tokenizer's post-processor special tokens.
    func prepare(text: String?, image: CGImage?) throws -> PreparedInput {
        let messages = Qwen3VLMessageGenerator().generate(messages: [
            .system(Self.systemPrompt),
            .user(
                text ?? "",
                images: image.map { [.ciImage(CIImage(cgImage: $0))] } ?? []),
        ])
        var tokenIDs = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: nil)

        var pixels: MLXArray?
        var frames: [THW]?
        if let image {
            let (processedPixels, frame) = try processor.preprocess(
                images: [CIImage(cgImage: image)], processing: nil)
            pixels = processedPixels
            frames = [frame]
            tokenIDs = try Self.expandImagePlaceholders(
                tokenIDs, frames: [frame], mergeSize: mergeSize,
                visionStart: visionStartTokenID, imageToken: imageTokenID,
                visionEnd: visionEndTokenID)
        }
        tokenIDs.append(contentsOf: appendedTokenIDs)
        return PreparedInput(
            tokenIDs: tokenIDs, pixels: pixels, frames: frames, mergeSize: mergeSize)
    }

    private func embed(_ input: Input) throws -> [Float] {
        let prepared: PreparedInput
        switch input {
        case .text(let text):
            prepared = try prepare(text: text, image: nil)
        case .image(let image, let text):
            prepared = try prepare(text: text, image: image)
        }

        guard prepared.tokenIDs.count <= Self.maxInputTokens else {
            throw Qwen3VLEmbedError.inputTooLong(tokens: prepared.tokenIDs.count)
        }

        let tokens = MLXArray(prepared.tokenIDs.map { Int32($0) })
            .reshaped(1, prepared.tokenIDs.count)
        let hidden = try model.hiddenStates(
            inputIds: tokens, imagePixels: prepared.pixels, imageFrames: prepared.frames)

        // The last real token (one input, no padding), after the final norm, L2-normalised in
        // float32 so cosine similarity is a dot product.
        let lastIndex = hidden.dim(1) - 1
        let last = hidden[0, lastIndex].asType(.float32)
        let norm = MLX.sqrt((last * last).sum())
        let vector = last / MLX.maximum(norm, MLXArray(Float(1e-12)))
        // WW2-3-8: evaluate and read through the guard — a failed array must throw, not segfault.
        let values = try MLXErrorGuard.read(vector, as: Float.self)
        guard values.count == Self.dimensions else {
            throw Qwen3VLEmbedError.unexpectedDimensions(
                expected: Self.dimensions, actual: values.count)
        }
        return values
    }

    /// Replace the single `<|vision_start|><|image_pad|><|vision_end|>` placeholder with one
    /// `<|image_pad|>` per merged patch — the expansion `Qwen3VLProcessor.prepare` performs.
    static func expandImagePlaceholders(
        _ tokens: [Int], frames: [THW], mergeSize: Int,
        visionStart: Int, imageToken: Int, visionEnd: Int
    ) throws -> [Int] {
        let placeholder = [visionStart, imageToken, visionEnd]
        var result: [Int] = []
        result.reserveCapacity(tokens.count + 1_280)
        var index = 0
        var frameIndex = 0
        while index < tokens.count {
            if frameIndex < frames.count,
               index + placeholder.count <= tokens.count,
               Array(tokens[index..<(index + placeholder.count)]) == placeholder {
                let paddingCount = frames[frameIndex].product / (mergeSize * mergeSize)
                result.append(visionStart)
                result.append(contentsOf: repeatElement(imageToken, count: paddingCount))
                result.append(visionEnd)
                index += placeholder.count
                frameIndex += 1
            } else {
                result.append(tokens[index])
                index += 1
            }
        }
        guard frameIndex == frames.count else {
            throw Qwen3VLEmbedError.featureTokenMismatch(
                expected: frames.count, actual: frameIndex)
        }
        return result
    }

    /// The ids the tokenizer's post-processor appends to every sequence when
    /// `add_special_tokens=True` (for Qwen2/Qwen3 tokenizers, the trailing `<|endoftext|>`); read
    /// straight out of the installed `tokenizer.json` so the recipe matches the reference exactly.
    static func appendedSpecialTokenIDs(in directory: URL) -> [Int] {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let postProcessor = object["post_processor"] as? [String: Any] else { return [] }

        func ids(inTemplate template: Any?, tokenMap: [String: Any]) -> [Int] {
            guard let entries = template as? [[String: Any]] else { return [] }
            var out: [Int] = []
            for entry in entries {
                guard let special = entry["SpecialToken"] as? [String: Any],
                      let name = special["id"] as? String,
                      let info = tokenMap[name] as? [String: Any],
                      let values = info["ids"] as? [Int] else { continue }
                out.append(contentsOf: values)
            }
            return out
        }

        if let processors = postProcessor["processors"] as? [[String: Any]] {
            var out: [Int] = []
            for processor in processors {
                let tokenMap = processor["special_tokens"] as? [String: Any] ?? [:]
                out.append(contentsOf: ids(inTemplate: processor["single"], tokenMap: tokenMap))
            }
            return out
        }
        let tokenMap = postProcessor["special_tokens"] as? [String: Any] ?? [:]
        return ids(inTemplate: postProcessor["single"], tokenMap: tokenMap)
    }
}

// MARK: - Errors

/// One plain sentence for each way this embedder can refuse an input — the Run screens render
/// `LocalizedError.errorDescription`, never a raw error dump.
nonisolated enum Qwen3VLEmbedError: LocalizedError {
    case inputTooLong(tokens: Int)
    case featureTokenMismatch(expected: Int, actual: Int)
    case unexpectedDimensions(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .inputTooLong(let tokens):
            return "This model accepts up to 8,192 tokens after rendering; this input rendered to \(tokens). Shorten it and try again."
        case .featureTokenMismatch(let expected, let actual):
            return "The image did not match the prompt's image placeholders (\(expected) placeholders, \(actual) features). Try a different image."
        case .unexpectedDimensions(let expected, let actual):
            return "The model returned \(actual) dimensions instead of \(expected). Re-install the model and try again."
        }
    }
}

// MARK: - Loading the trimmed model

extension Qwen3VLEmbedModel {

    /// The top-level `quantization` block of the 4-bit repo's `config.json`.
    private struct Quantization: Codable {
        let groupSize: Int
        let bits: Int
        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    private struct RootConfig: Codable {
        let quantization: Quantization?
    }

    /// Load the installed checkpoint through the trimmed forward pass. Each layer is quantized only
    /// when `<path>.scales` exists — the same rule mlx-swift-lm's loader uses — so the unquantized
    /// vision tower stays float while the language model loads its 4-bit weights.
    static func load(from directory: URL) throws -> Qwen3VLEmbedModel {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: configData)
        let quantization = try? JSONDecoder().decode(RootConfig.self, from: configData).quantization

        let model = Qwen3VLEmbedModel(config)

        var weights: [String: MLXArray] = [:]
        for file in try Self.weightFiles(in: directory) {
            try MLX.loadArrays(url: file).forEach { weights[$0.key] = $0.value }
        }
        let sanitized = model.sanitize(weights: weights)

        if let quantization {
            quantize(model: model) { path, _ in
                sanitized["\(path).scales"] != nil
                    ? (quantization.groupSize, quantization.bits, QuantizationMode.affine)
                    : nil
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        eval(model)
        return model
    }

    /// `model.safetensors`, or the shards named by `model.safetensors.index.json`.
    private static func weightFiles(in directory: URL) throws -> [URL] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let data = try Data(contentsOf: indexURL)
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            let map = object?["weight_map"] as? [String: String] ?? [:]
            let names = Set(map.values).sorted()
            let urls = names.map { directory.appendingPathComponent($0) }
            if !urls.isEmpty, urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) {
                return urls
            }
        }
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
        return files
            .filter { $0.pathExtension == "safetensors" && $0.lastPathComponent.hasPrefix("model") }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}

// MARK: - Test seam

extension Qwen3VLEmbedder {

    /// What the recipe hands the model for one input, without loading the weights — so MC4-4 can
    /// compare the rendered token ids with the reference fixture *before* comparing vectors, and
    /// MC4-5 can assert the image-pad count.
    static func preparedInput(
        text: String?, image: CGImage?, modelDirectory: URL
    ) async throws -> PreparedInput {
        let tokenizer = try await HFTokenizerLoader().load(from: modelDirectory)
        let processorData = try Data(
            contentsOf: modelDirectory.appendingPathComponent("preprocessor_config.json"))
        let processorConfig = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self, from: processorData)
        let processor = Qwen3VLProcessor(processorConfig, tokenizer: tokenizer)
        let configData = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(Qwen3VLConfiguration.self, from: configData)
        let embedder = Qwen3VLEmbedder(
            model: Qwen3VLEmbedModel(configuration), processor: processor, tokenizer: tokenizer,
            configuration: configuration,
            appendedTokenIDs: appendedSpecialTokenIDs(in: modelDirectory))
        return try embedder.prepare(text: text, image: image)
    }
}

// MARK: - The Run screen's session

/// One loaded embedder carried across the Run screen's task boundary. The Images tab embeds up to
/// 200 images, so the 1.8 GB checkpoint is loaded once and reused; loading it per image would take
/// minutes and re-read the weights 200 times.
///
/// `@unchecked Sendable`: MLX arrays are not `Sendable`, but every graph this type builds is
/// evaluated and read on the task that built it (`MLXErrorGuard.read`, WW2-3-8) and the weights
/// were materialised at load (`eval(model)`), so no unevaluated MLX graph crosses a thread here.
nonisolated final class Qwen3VLEmbeddingSession: @unchecked Sendable {
    private let embedder: Qwen3VLEmbedder

    private init(embedder: Qwen3VLEmbedder) {
        self.embedder = embedder
    }

    static func load(from directory: URL) async throws -> Qwen3VLEmbeddingSession {
        Qwen3VLEmbeddingSession(embedder: try await Qwen3VLEmbedder.load(from: directory))
    }

    /// One query/document text → one L2-normalised 2048-d vector.
    func embedText(_ text: String) async throws -> [Float] {
        guard let vector = try await embedder.embedTexts([text]).first else {
            throw Qwen3VLEmbedError.unexpectedDimensions(
                expected: Qwen3VLEmbedder.dimensions, actual: 0)
        }
        return vector
    }

    /// One image (optionally with a caption) → one L2-normalised 2048-d vector.
    func embedImage(_ image: CGImage, text: String? = nil) async throws -> [Float] {
        try await embedder.embedImage(image, text: text)
    }
}

/// Loads (once) the session both Images-tab closures share. The actor is the "load once" seam: a
/// concurrent first `load()` waits for the one in progress and gets the same session.
actor Qwen3VLEmbeddingSessionLoader {
    private let directory: URL
    private var session: Qwen3VLEmbeddingSession?
    private var loadTask: Task<Qwen3VLEmbeddingSession, Error>?

    init(directory: URL) {
        self.directory = directory
    }

    func load() async throws -> Qwen3VLEmbeddingSession {
        if let session { return session }
        if let loadTask { return try await loadTask.value }
        let directory = self.directory
        let task = Task<Qwen3VLEmbeddingSession, Error> {
            try await Qwen3VLEmbeddingSession.load(from: directory)
        }
        loadTask = task
        do {
            let loaded = try await task.value
            session = loaded
            loadTask = nil
            return loaded
        } catch {
            loadTask = nil
            throw error
        }
    }
}
