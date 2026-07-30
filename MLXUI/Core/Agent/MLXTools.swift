//
//  MLXTools.swift
//  MLXUI — agentic chat tools, slice AG4a.
//
//  In-app MLX tools: let the chat model orchestrate the local models the app already runs.
//  AG4a ships the two text-in / text-out tools (no media input, no on-disk index):
//    • embed_text  — semantic embedding via a local embedding model (dims, or cosine of two texts)
//    • summarize   — condense text via a local chat model
//
//  Later AG4 sub-slices add the media + index tools: semantic_search (AG4b, on-disk index) and
//  transcribe_audio / ocr_image (AG4c, input via http URL — no file entitlement needed).
//
//  Each tool auto-picks the smallest installed model of the modality it needs, via
//  `InstalledModelIndex` (reads the bundled catalog for modality + the on-disk `.installed`
//  markers). As in AG2/AG3 the risky/pure logic lives in `static` helpers exercised by
//  synchronous unit tests; real inference (engine loads weights) is not unit-tested.
//

import Foundation
import MLXLMCommon

/// Resolves an installed model of a given modality. Modality comes from the bundled catalog
/// (`browser.json` → `ModelEntry.runnerKind`); "installed" is the on-disk `.installed` marker
/// (the same signal the install registry reconciles against). `nonisolated` + `Sendable` so a
/// tool can build it off the MainActor.
nonisolated struct InstalledModelIndex: Sendable {
    struct Entry: Sendable, Equatable {
        let id: String
        let kind: RunnerKind
        let ramGB: Double
    }

    /// Installed models only.
    let entries: [Entry]

    /// The smallest installed model (by RAM) of the given kind — a fast, deterministic default.
    func best(kind: RunnerKind) -> String? {
        entries.filter { $0.kind == kind }.min { $0.ramGB < $1.ramGB }?.id
    }

    /// Pure builder — keeps only installed ids. Testable without touching disk.
    static func make(catalog: [Entry], installedIDs: Set<String>) -> InstalledModelIndex {
        InstalledModelIndex(entries: catalog.filter { installedIDs.contains($0.id) })
    }

    /// The live index from the bundled catalog + on-disk install markers.
    static func loadInstalled() -> InstalledModelIndex {
        make(catalog: catalogEntries(), installedIDs: installedIDsOnDisk())
    }

    /// All catalog models flattened to `(id, kind, ramGB)`. Decoded through a local
    /// `nonisolated` DTO rather than the MainActor-isolated `BrowserData`/`ModelEntry`, keeping
    /// the resolver off the MainActor (the same boundary the nonisolated engines observe).
    private static func catalogEntries() -> [Entry] {
        guard let url = Bundle.main.url(forResource: "browser", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let catalog = try? JSONDecoder().decode(CatalogDTO.self, from: data)
        else { return [] }
        return catalog.domains.flatMap(\.all).map {
            Entry(id: $0.id, kind: Self.kind(modelType: $0.modelType, family: $0.family), ramGB: $0.ramGB)
        }
    }

    /// Minimal `nonisolated` mirror of the bundled catalog — only the fields the resolver needs.
    private struct CatalogDTO: Decodable {
        struct Node: Decodable {
            let children: [Node]?
            let models: [Model]?
            var all: [Model] { models ?? children?.flatMap(\.all) ?? [] }
        }
        struct Model: Decodable {
            let id: String
            let family: String
            let modelType: String
            let ramGB: Double
        }
        let domains: [Node]
    }

    /// Modality mapping — mirrors `ModelEntry.runnerKind` (the single source of truth), including
    /// its catalog-mislabel overrides, kept in sync here so the resolver stays nonisolated.
    private static func kind(modelType: String, family: String) -> RunnerKind {
        switch family {
        case "Llama-OuteTTS":             return .tts
        case "LFM2.5-Audio", "sam-audio": return .unsupported
        default:                          break
        }
        switch modelType {
        case "llm":       return .llm
        case "asr":       return .asr
        case "tts":       return .tts
        case "vision":    return .vision
        case "embedding": return .embedding
        case "ocr":       return .ocr
        default:          return .unsupported  // video / unknown
        }
    }

    private static var modelsBase: URL {
        ModelStore.shared.modelsDirectory
    }

    /// Ids whose `.installed` marker exists on disk (the atomic "install succeeded" signal).
    private static func installedIDsOnDisk() -> Set<String> {
        guard let dirs = try? FileManager.default.contentsOfDirectory(
            at: modelsBase, includingPropertiesForKeys: nil) else { return [] }
        var ids: Set<String> = []
        for dir in dirs where FileManager.default.fileExists(
            atPath: dir.appendingPathComponent(".installed").path) {
            ids.insert(dir.lastPathComponent)
        }
        return ids
    }
}

/// Computes a semantic embedding for text using a locally-installed embedding model. With one
/// text it reports the vector's dimensionality; with two it reports their cosine similarity.
nonisolated struct EmbedTextTool: AgentTool {
    let name = "embed_text"
    let toolDescription = """
        Compute a semantic embedding for text using a locally-installed embedding model. With \
        one text, reports the embedding's dimensionality; with a second text (via 'compare_to'), \
        reports the cosine similarity of the two (−1 to 1, higher = more similar in meaning).
        """
    let parameters: [ToolParameter] = [
        .required("text", type: .string, description: "The text to embed."),
        .optional("compare_to", type: .string,
                  description: "Optional second text; if given, returns the cosine similarity of the two texts."),
    ]

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let text = (arguments.string("text") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "Error: missing required 'text' argument." }
        guard let modelID = InstalledModelIndex.loadInstalled().best(kind: .embedding) else {
            return "Error: no embedding model is installed. Install one from the Browse tab first."
        }
        let dir = EmbeddingEngine.installedModelDirectory(id: modelID)

        let other = arguments.string("compare_to")?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let other, !other.isEmpty {
            let vectors = try await EmbeddingEngine.embed([text, other], modelDirectory: dir)
            guard vectors.count == 2 else { return "Error: embedding did not return both vectors." }
            return Self.formatSimilarity(Self.cosine(vectors[0], vectors[1]),
                                         model: modelID, dim: vectors[0].count)
        }
        let vectors = try await EmbeddingEngine.embed([text], modelDirectory: dir)
        guard let v = vectors.first else { return "Error: embedding failed to produce a vector." }
        return Self.formatSingle(dim: v.count, model: modelID)
    }

    /// Cosine similarity of two equal-length vectors; 0 for mismatched/empty/zero vectors.
    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in a.indices {
            dot += a[i] * b[i]
            na += a[i] * a[i]
            nb += b[i] * b[i]
        }
        let denom = na.squareRoot() * nb.squareRoot()
        return denom == 0 ? 0 : dot / denom
    }

    static func formatSimilarity(_ score: Float, model: String, dim: Int) -> String {
        let s = String(format: "%.3f", score)
        return "Cosine similarity: \(s) (−1 to 1, higher = more similar). Model: \(model), \(dim)-dim."
    }

    static func formatSingle(dim: Int, model: String) -> String {
        "Computed a \(dim)-dimensional embedding using \(model). "
            + "(Pass a 'compare_to' text to measure similarity between two texts.)"
    }
}

/// Summarizes a block of text using a locally-installed chat model. Optionally caps the summary
/// length with 'max_words'. Loads its own copy of a chat model (engines load per call — see the
/// deferred "cache the engine instance" backlog item), so it may pick a different, smaller model
/// than the one driving the chat.
nonisolated struct SummarizeTool: AgentTool {
    let name = "summarize"
    let toolDescription = """
        Summarize a block of text using a locally-installed chat model. Optionally cap the \
        length with 'max_words'. Useful for condensing long content (e.g. a fetched web page) \
        into a short overview.
        """
    let parameters: [ToolParameter] = [
        .required("text", type: .string, description: "The text to summarize."),
        .optional("max_words", type: .int,
                  description: "Optional maximum length of the summary, in words."),
    ]

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let text = (arguments.string("text") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return "Error: missing required 'text' argument." }
        guard let modelID = InstalledModelIndex.loadInstalled().best(kind: .llm) else {
            return "Error: no chat model is installed. Install one from the Browse tab first."
        }
        let maxWords = arguments.int("max_words")
        return try await LLMEngine.generate(
            prompt: Self.prompt(for: text, maxWords: maxWords),
            modelDir: LLMEngine.modelDirectory(for: modelID),
            maxTokens: Self.maxTokens(forMaxWords: maxWords),
            temperature: 0.3)
    }

    /// The instruction handed to the chat model. Pure, so it is unit-testable.
    static func prompt(for text: String, maxWords: Int?) -> String {
        let limit = maxWords.map { " in at most \($0) words" } ?? ""
        return """
            Summarize the following text\(limit). Respond with only the summary, no preamble.

            \(text)
            """
    }

    /// Token budget for the summary — roughly 2 tokens/word, clamped to a sane range.
    static func maxTokens(forMaxWords maxWords: Int?) -> Int {
        guard let maxWords, maxWords > 0 else { return 256 }
        return min(1024, max(32, maxWords * 2))
    }
}
