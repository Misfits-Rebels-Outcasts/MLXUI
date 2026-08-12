import Foundation

/// SDXL CLIP text tokenizer, ported verbatim from `FluxCLIPTokenizer` (which is a port of
/// HuggingFace's `CLIPTokenizer` from `ml-explore/mlx-examples/flux`). SDXL uses the **same**
/// BPE as FLUX: `tokenizer/vocab.json` + `tokenizer/merges.txt`, 77-token window, vocab 49408.
/// Both of SDXL's tokenizers (`tokenizer/` for CLIP-L and `tokenizer_2/` for OpenCLIP-G) use
/// this identical BPE — the engine instantiates one `SDCLIPTokenizer` per directory.
///
/// `bpeRanks` is keyed by an ordered **pair** of subwords (the reference builds it from
/// `tuple(m.split())` over `merges.txt` lines), NOT a single joined string — the rank lookup
/// and the merge comparisons both use pairs, so keying by `"a b"` would silently never fire.
nonisolated enum SDCLIPTokenizer {
    static let maxLength = 77

    private static let bos = "<|startoftext|>"
    private static let eos = "<|endoftext|>"

    /// A BPE bigram — `("a", "b")`. Hashable so it can key the rank map.
    struct Bigram: Hashable {
        let a: String
        let b: String
        init(_ a: String, _ b: String) { self.a = a; self.b = b }
    }

    /// The token regex from the reference (letters, numbers, punctuation, and the special
    /// start/end markers), case-insensitive.
    private static let pattern = try! NSRegularExpression(
        pattern: #"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"#,
        options: [.caseInsensitive]
    )

    /// Tokenize a prompt to CLIP ids: lowercase → regex split → BPE → bos/eos → truncate to 77.
    /// Mirrors `CLIPTokenizer.tokenize`.
    static func tokenize(_ text: String, bpeRanks: [Bigram: Int], vocabulary: [String: Int]) -> [Int] {
        let clean = text.lowercased()
            .replacingOccurrences(of: "\\s+", with: " ", options: .regularExpression)
        let range = NSRange(clean.startIndex..., in: clean)
        let matches = pattern.matches(in: clean, range: range)
        let tokens = matches.map { String(clean[Range($0.range, in: clean)!]) }

        var out = tokens.flatMap { bpe($0, bpeRanks: bpeRanks) }
            .compactMap { vocabulary[$0] }

        if let bosID = vocabulary[bos] { out.insert(bosID, at: 0) }
        if let eosID = vocabulary[eos] { out.append(eosID) }

        if out.count > maxLength {
            out = Array(out.prefix(maxLength))
            if let eosID = vocabulary[eos] { out[out.count - 1] = eosID }
        }
        return out
    }

    /// BPE merge (`tokenizers.py` `bpe`): repeatedly merge the lowest-ranked bigram.
    static func bpe(_ text: String, bpeRanks: [Bigram: Int]) -> [String] {
        var unigrams = text.dropLast().map { String($0) } + ["\(text.last!)</w>"]
        var uniqueBigrams = Set(zip(unigrams, unigrams.dropFirst()).map { Bigram($0.0, $0.1) })

        while !uniqueBigrams.isEmpty {
            let best = uniqueBigrams.min { a, b in
                (bpeRanks[a] ?? Int.max) < (bpeRanks[b] ?? Int.max)
            }!
            guard bpeRanks[best] != nil else { break }

            var newUnigrams: [String] = []
            var skip = false
            for (a, b) in zip(unigrams, unigrams.dropFirst()) {
                if skip { skip = false; continue }
                if Bigram(a, b) == best {
                    newUnigrams.append(a + b)
                    skip = true
                } else {
                    newUnigrams.append(a)
                }
            }
            if !skip, let last = unigrams.last { newUnigrams.append(last) }

            unigrams = newUnigrams
            uniqueBigrams = Set(zip(unigrams, unigrams.dropFirst()).map { Bigram($0.0, $0.1) })
        }
        return unigrams
    }
}
