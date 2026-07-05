import Foundation
import NaturalLanguage

/// Sentence-aware text splitting via `NaturalLanguage`. Provided as a pure utility:
/// a true fan-out "Chunk" node (one input → many segments) needs the branch engine
/// deferred to later cycles (open-pipeline-nodes §8), so for now this feeds callers
/// that want to summarize a long transcript piece by piece.
nonisolated enum TextChunker {
    /// Split `text` into sentences using NL's sentence tokenizer; blank pieces dropped.
    static func sentences(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        var result: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let piece = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !piece.isEmpty { result.append(piece) }
            return true
        }
        return result
    }

    /// Group sentences into chunks no longer than `maxCharacters`. A single sentence
    /// longer than the limit becomes its own (over-limit) chunk rather than being split.
    static func chunks(_ text: String, maxCharacters: Int) -> [String] {
        guard maxCharacters > 0 else { return [text] }
        var chunks: [String] = []
        var current = ""
        for sentence in sentences(text) {
            if current.isEmpty {
                current = sentence
            } else if current.count + 1 + sentence.count <= maxCharacters {
                current += " " + sentence
            } else {
                chunks.append(current)
                current = sentence
            }
        }
        if !current.isEmpty { chunks.append(current) }
        return chunks
    }
}
