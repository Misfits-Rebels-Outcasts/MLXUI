import Foundation
import MLXLMCommon

/// Errors from tokenization. Every sentence is one plain line, names no input text and no span text
/// (decision 10); token ids and counts are metadata and may appear.
nonisolated enum PrivacyFilterTokenizerError: Error, LocalizedError, Equatable {
    case missingTokenizerFile
    case loadFailed
    case missingTokenPiece(tokenID: Int)
    case unmappableTokenPiece(tokenID: Int)
    case roundTripMismatch(tokenCount: Int, byteCount: Int)

    var errorDescription: String? {
        switch self {
        case .missingTokenizerFile:
            return "Privacy Filter: the model folder is missing tokenizer.json. Install the model again."
        case .loadFailed:
            return "Privacy Filter: the model's tokenizer could not be loaded. Install the model again."
        case .missingTokenPiece(let tokenID):
            return "Privacy Filter: the model's tokenizer has no text for token \(tokenID). Install the model again."
        case .unmappableTokenPiece(let tokenID):
            return "Privacy Filter: token \(tokenID) does not map back to bytes. Install the model again."
        case .roundTripMismatch(let tokenCount, let byteCount):
            return "Privacy Filter: the tokenizer did not reproduce the input text exactly (\(tokenCount) tokens, \(byteCount) bytes), so span offsets would be wrong."
        }
    }
}

/// The checkpoint's tokenizer plus character-accurate UTF-8 byte offsets (Trap 1).
///
/// `MLXUI/Core/HFTokenizerLoader` gives token ids only — no offsets. Offsets are rebuilt here from
/// the byte-level BPE vocabulary: each token id → its vocabulary string → bytes through the
/// GPT-2/o200k byte-level alphabet, accumulated into per-token UTF-8 byte ranges. The concatenation
/// is then compared with the input's exact UTF-8 bytes; if they differ (a normalizer, an added
/// prefix space, or a special token that is not part of the input) the call throws one plain
/// sentence instead of guessing.
///
/// The checkpoint's `tokenizer.json` is exactly tiktoken `o200k_base`
/// (`normalizer: null`; the ByteLevel pre-tokenizer is tiktoken's o200k pattern), so ids match the
/// reference encoder. The round-trip assertion, not that fact, is what every span relies on.
nonisolated final class PrivacyFilterTokenizer: Sendable {
    /// One encode: token ids and each token's exact UTF-8 byte range in the input.
    struct Tokenization: Sendable {
        let tokenIDs: [Int]
        /// Per token, start byte offset into the input UTF-8 (inclusive).
        let utf8ByteStarts: [Int]
        /// Per token, end byte offset into the input UTF-8 (exclusive).
        let utf8ByteEnds: [Int]

        var tokenCount: Int { tokenIDs.count }
    }

    private let tokenizer: any MLXLMCommon.Tokenizer

    init(modelDirectory: URL) async throws {
        let tokenizerFile = modelDirectory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerFile.path) else {
            throw PrivacyFilterTokenizerError.missingTokenizerFile
        }
        do {
            tokenizer = try await HFTokenizerLoader().load(from: modelDirectory)
        } catch {
            throw PrivacyFilterTokenizerError.loadFailed
        }
    }

    /// Encode `text` with no special tokens and return ids plus per-token UTF-8 byte offsets.
    func tokenize(_ text: String) throws -> Tokenization {
        let tokenIDs = tokenizer.encode(text: text, addSpecialTokens: false)
        var pieces: [[UInt8]] = []
        pieces.reserveCapacity(tokenIDs.count)
        for tokenID in tokenIDs {
            guard let piece = tokenizer.convertIdToToken(tokenID) else {
                throw PrivacyFilterTokenizerError.missingTokenPiece(tokenID: tokenID)
            }
            pieces.append(try Self.bytes(forTokenPiece: piece, tokenID: tokenID))
        }
        return try Self.tokenization(
            tokenIDs: tokenIDs, tokenPieces: pieces, inputUTF8: Array(text.utf8))
    }

    // MARK: Offset construction (testable without a checkpoint)

    /// Map one vocabulary string back to bytes through the byte-level BPE alphabet.
    ///
    /// The o200k/GPT-2 byte-level alphabet is the identity for the printable byte ranges and maps
    /// every other byte to a Unicode scalar starting at U+0100; see
    /// `tiktoken`'s/`openai`'s `bytes_to_unicode`. A vocabulary string containing a scalar outside
    /// that alphabet cannot come from the checkpoint, so it is refused rather than guessed.
    static func bytes(forTokenPiece piece: String, tokenID: Int = 0) throws -> [UInt8] {
        var bytes: [UInt8] = []
        bytes.reserveCapacity(piece.utf8.count)
        for scalar in piece.unicodeScalars {
            guard let byte = byteLevelInverse[scalar] else {
                throw PrivacyFilterTokenizerError.unmappableTokenPiece(tokenID: tokenID)
            }
            bytes.append(byte)
        }
        return bytes
    }

    /// Accumulate per-token offsets and assert the pieces are exactly the input's UTF-8 bytes.
    static func tokenization(
        tokenIDs: [Int],
        tokenPieces: [[UInt8]],
        inputUTF8: [UInt8]
    ) throws -> Tokenization {
        var starts: [Int] = []
        var ends: [Int] = []
        starts.reserveCapacity(tokenIDs.count)
        ends.reserveCapacity(tokenIDs.count)
        var joined: [UInt8] = []
        joined.reserveCapacity(inputUTF8.count)
        var cursor = 0
        for piece in tokenPieces {
            starts.append(cursor)
            cursor += piece.count
            ends.append(cursor)
            joined.append(contentsOf: piece)
        }
        guard joined.count == inputUTF8.count, joined == inputUTF8 else {
            throw PrivacyFilterTokenizerError.roundTripMismatch(
                tokenCount: tokenIDs.count, byteCount: inputUTF8.count)
        }
        return Tokenization(tokenIDs: tokenIDs, utf8ByteStarts: starts, utf8ByteEnds: ends)
    }

    /// Byte → scalar direction (`bytes_to_unicode`): printable ranges map to themselves, the rest to
    /// U+0100 in byte order.
    static let byteLevelAlphabet: [UInt8: Unicode.Scalar] = {
        var printable: [Int] = []
        printable.append(contentsOf: 33 ... 126)
        printable.append(contentsOf: 161 ... 172)
        printable.append(contentsOf: 174 ... 255)
        var bytes = printable
        var scalars = printable
        var next = 256
        for byte in 0 ..< 256 where !printable.contains(byte) {
            bytes.append(byte)
            scalars.append(next)
            next += 1
        }
        var table: [UInt8: Unicode.Scalar] = [:]
        for (byte, scalarValue) in zip(bytes, scalars) {
            if let scalar = Unicode.Scalar(UInt32(scalarValue)) {
                table[UInt8(byte)] = scalar
            }
        }
        return table
    }()

    /// Scalar → byte direction, the inverse of ``byteLevelAlphabet``.
    static let byteLevelInverse: [Unicode.Scalar: UInt8] = {
        var inverse: [Unicode.Scalar: UInt8] = [:]
        for (byte, scalar) in byteLevelAlphabet {
            inverse[scalar] = byte
        }
        return inverse
    }()
}
