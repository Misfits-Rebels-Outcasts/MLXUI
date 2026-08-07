import Foundation

/// A minimal reader for SentencePiece's `spiece.model` (a serialized `ModelProto`). Only the
/// fields FLUX needs are decoded: the unigram pieces (piece string, float score, type), the
/// special-token ids from `trainer_spec`, and the normalizer flags. See
/// `google/sentencepiece` `src/sentencepiece_model.proto`. Ported for AM4c because
/// swift-transformers' `T5Tokenizer` is a `UnigramTokenizer` that needs a `tokenizer.json` —
/// the FLUX.1 repo ships only the raw `spiece.model`.
nonisolated enum FluxSentencePiece {
    /// One unigram piece. `id` is its index in the `pieces` list.
    struct Piece {
        let text: String
        let score: Float
        let type: Int  // 1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 5 UNUSED, 6 BYTE
    }

    struct Model {
        let pieces: [Piece]
        let unkID: Int
        let bosID: Int
        let eosID: Int
        let padID: Int
        let addDummyPrefix: Bool
        let removeExtraWhitespaces: Bool
        let escapeWhitespaces: Bool
    }

    // MARK: - Wire format

    private struct Reader {
        let data: [UInt8]
        var index = 0

        mutating func varint() -> UInt64 {
            var value: UInt64 = 0
            var shift: UInt64 = 0
            while index < data.count {
                let byte = data[index]; index += 1
                value |= UInt64(byte & 0x7f) << shift
                if byte & 0x80 == 0 { break }
                shift += 7
            }
            return value
        }

        mutating func bytes(_ n: Int) -> [UInt8] {
            let slice = Array(data[index ..< min(index + n, data.count)])
            index += slice.count
            return slice
        }
    }

    // MARK: - Parse

    /// Parse a `spiece.model` file into the model structure. Throws if the file isn't a
    /// readable SentencePiece `ModelProto`.
    static func parse(from url: URL) throws -> Model {
        let data = try Data(contentsOf: url)
        return try parse(from: [UInt8](data))
    }

    static func parse(from data: [UInt8]) throws -> Model {
        var pieces: [Piece] = []
        var unkID = 0, bosID = 1, eosID = 2, padID = -1   // proto defaults
        var addDummyPrefix = true, removeExtra = true, escapeWS = true

        var reader = Reader(data: data)
        while reader.index < data.count {
            let tag = reader.varint()
            let field = Int(tag >> 3)
            let wireType = Int(tag & 7)
            switch field {
            case 1:  // repeated SentencePiece pieces
                guard wireType == 2 else { throw TokenizerError.malformedModel }
                let length = Int(reader.varint())
                let pieceBytes = reader.bytes(length)
                let piece = try parsePiece(pieceBytes)
                pieces.append(piece)
            case 2:  // trainer_spec
                guard wireType == 2 else { throw TokenizerError.malformedModel }
                let length = Int(reader.varint())
                let spec = reader.bytes(length)
                let ids = try parseTrainerSpec(spec)
                unkID = ids.unk; bosID = ids.bos; eosID = ids.eos; padID = ids.pad
            case 3:  // normalizer_spec
                guard wireType == 2 else { throw TokenizerError.malformedModel }
                let length = Int(reader.varint())
                let spec = reader.bytes(length)
                let flags = try parseNormalizerSpec(spec)
                addDummyPrefix = flags.addDummyPrefix
                removeExtra = flags.removeExtraWhitespaces
                escapeWS = flags.escapeWhitespaces
            default:
                // Skip unknown top-level fields by wire type.
                switch wireType {
                case 0: _ = reader.varint()
                case 1: _ = reader.bytes(8)
                case 2:
                    let length = Int(reader.varint()); _ = reader.bytes(length)
                case 5: _ = reader.bytes(4)
                default: throw TokenizerError.malformedModel
                }
            }
        }

        guard !pieces.isEmpty else { throw TokenizerError.malformedModel }
        return Model(
            pieces: pieces, unkID: unkID, bosID: bosID, eosID: eosID, padID: padID,
            addDummyPrefix: addDummyPrefix,
            removeExtraWhitespaces: removeExtra,
            escapeWhitespaces: escapeWS
        )
    }

    private static func parsePiece(_ bytes: [UInt8]) throws -> Piece {
        var text = ""
        var score: Float = 0
        var type = 1  // NORMAL
        var reader = Reader(data: bytes)
        while reader.index < bytes.count {
            let tag = reader.varint()
            let field = Int(tag >> 3)
            let wireType = Int(tag & 7)
            switch field {
            case 1:
                guard wireType == 2 else { throw TokenizerError.malformedModel }
                let length = Int(reader.varint())
                text = String(decoding: reader.bytes(length), as: UTF8.self)
            case 2:
                guard wireType == 5 else { throw TokenizerError.malformedModel }
                let raw = reader.bytes(4)
                score = raw.withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
            case 3:
                guard wireType == 0 else { throw TokenizerError.malformedModel }
                type = Int(reader.varint())
            default:
                switch wireType {
                case 0: _ = reader.varint()
                case 1: _ = reader.bytes(8)
                case 2:
                    let length = Int(reader.varint()); _ = reader.bytes(length)
                case 5: _ = reader.bytes(4)
                default: throw TokenizerError.malformedModel
                }
            }
        }
        return Piece(text: text, score: score, type: type)
    }

    private static func parseTrainerSpec(_ bytes: [UInt8]) throws -> (unk: Int, bos: Int, eos: Int, pad: Int) {
        var unk = 0, bos = 1, eos = 2, pad = -1
        var reader = Reader(data: bytes)
        while reader.index < bytes.count {
            let tag = reader.varint()
            let field = Int(tag >> 3)
            let wireType = Int(tag & 7)
            // Special-token ids are int32 varints (proto defaults: unk 0, bos 1, eos 2, pad -1).
            if wireType == 0 {
                let raw = reader.varint()
                let value = Int(truncatingIfNeeded: UInt64(bitPattern: Int64(bitPattern: raw)))
                switch field {
                case 40: unk = value
                case 41: bos = value
                case 42: eos = value
                case 43: pad = value
                default: break
                }
            } else if wireType == 2 {
                let length = Int(reader.varint()); _ = reader.bytes(length)
            } else if wireType == 5 {
                _ = reader.bytes(4)
            } else if wireType == 1 {
                _ = reader.bytes(8)
            } else {
                throw TokenizerError.malformedModel
            }
        }
        return (unk, bos, eos, pad)
    }

    private static func parseNormalizerSpec(_ bytes: [UInt8]) throws
        -> (addDummyPrefix: Bool, removeExtraWhitespaces: Bool, escapeWhitespaces: Bool) {
        var addDummy = true, removeExtra = true, escape = true
        var reader = Reader(data: bytes)
        while reader.index < bytes.count {
            let tag = reader.varint()
            let field = Int(tag >> 3)
            let wireType = Int(tag & 7)
            if wireType == 0 {
                let value = reader.varint() != 0
                switch field {
                case 3: addDummy = value
                case 4: removeExtra = value
                case 5: escape = value
                default: break
                }
            } else if wireType == 2 {
                let length = Int(reader.varint()); _ = reader.bytes(length)
            } else if wireType == 5 {
                _ = reader.bytes(4)
            } else if wireType == 1 {
                _ = reader.bytes(8)
            } else {
                throw TokenizerError.malformedModel
            }
        }
        return (addDummy, removeExtra, escape)
    }
}

/// TokenizerError for the FLUX tokenizer port.
nonisolated enum TokenizerError: Error {
    case malformedModel
    case invalidState(String)
}
