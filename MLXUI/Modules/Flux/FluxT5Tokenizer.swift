import Foundation

/// FLUX T5 tokenizer, ported from `ml-explore/mlx-examples/flux/flux/tokenizers.py` +
/// `google/sentencepiece` `unigram_model.cc` (`EncodeOptimized`) and `normalizer.cc`.
/// Loads the installed model's raw `spiece.model` (a `ModelProto`) directly — swift-transformers
/// has no SentencePiece reader (its `T5Tokenizer` needs a `tokenizer.json`, which FLUX.1 repos
/// don't ship). AM4c.
///
/// Behavior mirrors `sentencepiece.SentencePieceProcessor`:
/// - normalization: leading-space trim, add dummy prefix `▁`, collapse runs of spaces to a
///   single `▁`, escape spaces as `▁` (per the model's `NormalizerSpec` flags). The model's
///   precompiled charsmap is approximated with NFKC compatibility mapping — identity for ASCII,
///   which covers the common FLUX prompt case; exact non-ASCII parity is out of scope.
/// - encoding: the byte-level Viterbi `EncodeOptimized` over a prefix trie of all pieces
///   (skipping UNUSED), with a single-char UNK fallback and rectified-flow of token scores.
nonisolated enum FluxT5Tokenizer {
    static let maxLength = 512

    // MARK: - Trie

    private final class TrieNode {
        var children: [UInt8: TrieNode] = [:]
        var id: Int?   // piece id when this node is a terminal
    }

    private static func buildTrie(_ model: FluxSentencePiece.Model) -> TrieNode {
        let root = TrieNode()
        for (id, piece) in model.pieces.enumerated() {
            var node = root
            for byte in piece.text.utf8 {
                node = node.children[byte] ?? {
                    let child = TrieNode()
                    node.children[byte] = child
                    return child
                }()
            }
            node.id = id
        }
        return root
    }

    // MARK: - Normalization

    /// The number of bytes in one UTF-8 code point starting at the given byte.
    private static func oneCharLen(_ byte: UInt8) -> Int {
        if byte < 0x80 { return 1 }
        if byte & 0xE0 == 0xC0 { return 2 }
        if byte & 0xF0 == 0xE0 { return 3 }
        if byte & 0xF8 == 0xF0 { return 4 }
        return 1
    }

    /// Approximate the model's charsmap (nmt_nfkc) with Foundation's NFKC compatibility mapping.
    private static func charsmap(_ text: String) -> String {
        text.precomposedStringWithCompatibilityMapping
    }

    /// The `Normalizer::Normalize` behavior for the model's flags (identity charsmap semantics).
    static func normalize(_ input: String, model: FluxSentencePiece.Model) -> String {
        var text = charsmap(input)
        if model.removeExtraWhitespaces {
            while let first = text.first, first == " " {
                text.removeFirst()
            }
        }
        if text.isEmpty { return "" }

        var output = ""
        let spaceSymbol = model.escapeWhitespaces ? "▁" : " "
        func addWS() { output.append(contentsOf: spaceSymbol) }

        if model.addDummyPrefix { addWS() }

        var isPrevSpace = model.removeExtraWhitespaces
        for scalar in text.unicodeScalars {
            var sp = String(scalar)
            if isPrevSpace, sp.hasPrefix(" ") { sp = "" }
            if !sp.isEmpty {
                for c in sp {
                    if c == " " { addWS() } else { output.append(c) }
                }
                isPrevSpace = sp.hasSuffix(" ")
            }
        }
        if model.removeExtraWhitespaces {
            while output.hasSuffix(spaceSymbol) {
                output.removeLast(spaceSymbol.count)
            }
        }
        return output
    }

    // MARK: - EncodeOptimized

    /// The byte-level Viterbi from `unigram_model.cc` `EncodeOptimized`: returns piece ids for
    /// `text` (already normalized). This is the unigram best-path decode over the piece trie.
    static func encode(_ text: String, model: FluxSentencePiece.Model) -> [Int] {
        let normalized = normalize(text, model: model)
        let bytes = Array(normalized.utf8)
        let size = bytes.count
        if size == 0 { return [] }

        let trie = buildTrie(model)
        let minScore = model.pieces
            .filter { $0.type == 1 }
            .map(\.score)
            .min() ?? 0
        let unkScore = minScore - 10.0

        var nodeID = [Int](repeating: -1, count: size + 1)
        var bestScore = [Float](repeating: 0, count: size + 1)
        var startsAtOf = [Int](repeating: -1, count: size + 1)

        let kScoreResetThreshold: Float = 100_000
        var maxFrontier = 0
        var startsAt = 0
        while startsAt < size {
            var scoreTillHere = bestScore[startsAt]
            if scoreTillHere < -kScoreResetThreshold || scoreTillHere > kScoreResetThreshold {
                let offset = scoreTillHere
                for i in startsAt...maxFrontier {
                    if i == startsAt || startsAtOf[i] != -1 {
                        bestScore[i] -= offset
                    }
                }
                scoreTillHere = 0
            }

            let mblen = oneCharLen(bytes[startsAt])
            var node = trie
            var keyPos = startsAt
            var hasSingleNode = false

            while keyPos < size, let child = node.children[bytes[keyPos]] {
                node = child
                keyPos += 1
                guard let id = node.id else { continue }
                if model.pieces[id].type == 5 { continue }  // UNUSED
                maxFrontier = max(maxFrontier, keyPos)
                let length = keyPos - startsAt
                let score: Float
                if model.pieces[id].type == 4 {
                    score = 0.1 * Float(length - 1)
                } else {
                    score = model.pieces[id].score
                }
                let candidate = score + scoreTillHere
                if startsAtOf[keyPos] == -1 || candidate > bestScore[keyPos] {
                    bestScore[keyPos] = candidate
                    startsAtOf[keyPos] = startsAt
                    nodeID[keyPos] = id
                }
                if !hasSingleNode && length == mblen {
                    hasSingleNode = true
                }
            }

            if !hasSingleNode {
                maxFrontier = max(maxFrontier, startsAt + mblen)
                let target = startsAt + mblen
                let candidate = unkScore + scoreTillHere
                if startsAtOf[target] == -1 || candidate > bestScore[target] {
                    bestScore[target] = candidate
                    startsAtOf[target] = startsAt
                    nodeID[target] = model.unkID
                }
            }
            startsAt += mblen
        }

        var result: [Int] = []
        result.reserveCapacity(size / 4 + 1)
        var endsAt = size
        while endsAt > 0 {
            let s = startsAtOf[endsAt]
            guard s >= 0 else { break }
            result.append(nodeID[endsAt])
            endsAt = s
        }
        result.reverse()
        return result
    }

    // MARK: - T5 wrapper (tokenizers.py T5Tokenizer)

    /// `T5Tokenizer.tokenize(text, prepend_bos:, append_eos:, pad:)` — the ids a single prompt
    /// produces, matching the Python reference (bos=-1 → none prepended; eos=1 appended; padded
    /// to `maxLength` with the pad id 0).
    static func tokenize(
        _ text: String,
        model: FluxSentencePiece.Model,
        prependBOS: Bool = true,
        appendEOS: Bool = true,
        pad: Bool = true
    ) -> [Int] {
        var tokens = encode(text, model: model)
        if prependBOS, model.bosID >= 0 { tokens.insert(model.bosID, at: 0) }
        if appendEOS, model.eosID >= 0 { tokens.append(model.eosID) }
        if pad, tokens.count < maxLength, model.padID >= 0 {
            tokens += Array(repeating: model.padID, count: maxLength - tokens.count)
        }
        return tokens
    }
}
