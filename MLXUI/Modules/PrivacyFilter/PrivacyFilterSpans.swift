import Foundation

/// One detected span in the original input: exact UTF-8 byte offsets (the source of truth for
/// redaction and exported JSON) plus a `Character`-count range for the UI.
///
/// `text` is the covered text; it is never logged (decision 10). The byte offsets always fall on
/// Unicode-scalar boundaries, so slicing the original UTF-8 with them is exact.
nonisolated struct PrivacyFilterSpan: Sendable, Equatable {
    let category: PrivacyFilterCategory
    /// Byte offset into the original input UTF-8, inclusive.
    let utf8Start: Int
    /// Byte offset into the original input UTF-8, exclusive.
    let utf8End: Int
    /// The covered text (`text.utf8[utf8Start ..< utf8End]`).
    let text: String
    /// The span's range in Swift `Character`s, snapped **outward** to grapheme boundaries so a
    /// highlighted span never splits an emoji or an accented letter. For the UI only.
    var characterRange: Range<Int>

    init(
        category: PrivacyFilterCategory,
        utf8Start: Int,
        utf8End: Int,
        text: String,
        characterRange: Range<Int>
    ) {
        self.category = category
        self.utf8Start = utf8Start
        self.utf8End = utf8End
        self.text = text
        self.characterRange = characterRange
    }
}

/// One label span over token indices, before it is mapped to character/byte offsets
/// (`opf/_core/spans.py::labels_to_spans` output).
nonisolated struct PrivacyFilterTokenSpan: Sendable, Equatable {
    /// Span-label index into ``PrivacyFilterLabelInfo/spanClassNames``.
    let labelIndex: Int
    /// First token index of the span (inclusive).
    let tokenStart: Int
    /// Token index after the span's last token (exclusive).
    let tokenEnd: Int
}

/// One label span over **Unicode-scalar** indices after the token→character mapping
/// (`opf/_core/spans.py::token_spans_to_char_spans` output).
nonisolated struct PrivacyFilterCharacterSpan: Sendable, Equatable {
    let labelIndex: Int
    /// Scalar index of the span start, inclusive.
    let start: Int
    /// Scalar index after the span's last scalar, exclusive.
    let end: Int
}

/// The character-accurate span math of `opf/_core/spans.py`, ported verbatim.
///
/// The reference works in Python `str` indices (Unicode scalars) and its per-token offsets come
/// from tiktoken's byte pieces. This port keeps the same scalar-index math — including
/// `token_spans_to_char_spans`' interior snapping — and converts to UTF-8 byte offsets only where
/// the original input bytes are needed, so the result is exact for every byte outside a span and
/// for non-ASCII input.
///
/// Note the direction of the two bisects: a token whose start falls inside a scalar begins at that
/// scalar (outward), while a token whose end splits a scalar stops before it (inward) — that is
/// `token_spans_to_char_spans`, and it is why a span can cover a byte range that starts before its
/// first label token. The whitespace trim then works on scalars, exactly like `str.isspace()`.
nonisolated enum PrivacyFilterSpans {

    // MARK: Offset tables

    /// Per-scalar UTF-8 byte ranges of `text`: `starts[i] ..< ends[i]` is scalar `i`.
    static func scalarByteRanges(_ text: String) -> (starts: [Int], ends: [Int]) {
        var starts: [Int] = []
        var ends: [Int] = []
        starts.reserveCapacity(text.unicodeScalars.count)
        ends.reserveCapacity(text.unicodeScalars.count)
        var cursor = 0
        for scalar in text.unicodeScalars {
            starts.append(cursor)
            cursor += Self.utf8Width(scalar)
            ends.append(cursor)
        }
        return (starts, ends)
    }

    /// UTF-8 byte offsets at which each Swift `Character` (grapheme cluster) starts, plus the total
    /// byte count as the final entry. Strictly increasing, so `bisectLeft` works on it.
    static func graphemeByteStarts(_ text: String) -> [Int] {
        var starts: [Int] = []
        starts.reserveCapacity(text.count + 1)
        var cursor = 0
        for character in text {
            starts.append(cursor)
            cursor += character.utf8.count
        }
        starts.append(cursor)
        return starts
    }

    /// The `Character`-count range covering `utf8Start ..< utf8End`, snapped **outward** so a span
    /// never starts inside a grapheme cluster.
    static func graphemeRange(
        utf8Start: Int,
        utf8End: Int,
        graphemeStarts: [Int]
    ) -> Range<Int> {
        guard graphemeStarts.count >= 1, utf8Start >= 0, utf8End >= utf8Start else { return 0 ..< 0 }
        let lastStart = graphemeStarts.count - 1
        // Start: the last Character whose first byte is at or before `utf8Start` — i.e. the Character
        // containing the span's first byte, snapped outward to its beginning.
        var startIndex = bisectRight(graphemeStarts, utf8Start) - 1
        if startIndex < 0 { startIndex = 0 }
        if startIndex > lastStart { startIndex = lastStart }
        // End (exclusive): the first Character whose bytes start at or after `utf8End`; if the last
        // covered byte falls inside a Character, that Character is included.
        var endIndex = bisectLeft(graphemeStarts, utf8End)
        if endIndex > lastStart { endIndex = lastStart }
        if endIndex < startIndex { endIndex = startIndex }
        return startIndex ..< endIndex
    }

    // MARK: opf/_core/spans.py

    /// `labels_to_spans`: token label ids → token-index spans.
    static func labelsToTokenSpans(
        labelsByTokenIndex: [Int: Int],
        labelInfo: PrivacyFilterLabelInfo
    ) -> [PrivacyFilterTokenSpan] {
        var spans: [PrivacyFilterTokenSpan] = []
        var currentLabel: Int?
        var startIndex: Int?
        var previousIndex: Int?
        let backgroundSpanLabel = labelInfo.backgroundSpanLabel

        func flush(upTo endExclusive: Int) {
            if let currentLabel, let startIndex {
                spans.append(PrivacyFilterTokenSpan(
                    labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: endExclusive))
            }
            currentLabel = nil
            startIndex = nil
        }

        for tokenIndex in labelsByTokenIndex.keys.sorted() {
            let labelID = labelsByTokenIndex[tokenIndex] ?? labelInfo.backgroundTokenLabel
            let spanLabel = labelInfo.tokenToSpanLabel[labelID]
            let boundaryTag = labelInfo.tokenBoundaryTags[labelID]

            if let previousIndex, tokenIndex != previousIndex + 1 {
                flush(upTo: previousIndex + 1)
            }

            guard let spanLabel else {
                previousIndex = tokenIndex
                continue
            }

            if spanLabel == backgroundSpanLabel {
                if let currentLabel, let startIndex {
                    spans.append(PrivacyFilterTokenSpan(
                        labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: tokenIndex))
                }
                currentLabel = nil
                startIndex = nil
                previousIndex = tokenIndex
                continue
            }

            switch boundaryTag {
            case "S":
                if let currentLabel, let startIndex, let previousIndex {
                    spans.append(PrivacyFilterTokenSpan(
                        labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: previousIndex + 1))
                }
                spans.append(PrivacyFilterTokenSpan(
                    labelIndex: spanLabel, tokenStart: tokenIndex, tokenEnd: tokenIndex + 1))
                currentLabel = nil
                startIndex = nil
            case "B":
                if let currentLabel, let startIndex, let previousIndex {
                    spans.append(PrivacyFilterTokenSpan(
                        labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: previousIndex + 1))
                }
                currentLabel = spanLabel
                startIndex = tokenIndex
            case "I":
                if currentLabel == nil || currentLabel != spanLabel {
                    if let currentLabel, let startIndex, let previousIndex {
                        spans.append(PrivacyFilterTokenSpan(
                            labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: previousIndex + 1))
                    }
                    currentLabel = spanLabel
                    startIndex = tokenIndex
                }
            case "E":
                if let openLabel = currentLabel, let openStart = startIndex, openLabel == spanLabel {
                    spans.append(PrivacyFilterTokenSpan(
                        labelIndex: openLabel, tokenStart: openStart, tokenEnd: tokenIndex + 1))
                    currentLabel = nil
                } else {
                    if let currentLabel, let startIndex, let previousIndex {
                        spans.append(PrivacyFilterTokenSpan(
                            labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: previousIndex + 1))
                    }
                    spans.append(PrivacyFilterTokenSpan(
                        labelIndex: spanLabel, tokenStart: tokenIndex, tokenEnd: tokenIndex + 1))
                    currentLabel = nil
                }
                startIndex = nil
            default:
                if let currentLabel, let startIndex, let previousIndex {
                    spans.append(PrivacyFilterTokenSpan(
                        labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: previousIndex + 1))
                }
                currentLabel = nil
                startIndex = nil
            }

            previousIndex = tokenIndex
        }

        if let currentLabel, let startIndex, let previousIndex {
            spans.append(PrivacyFilterTokenSpan(
                labelIndex: currentLabel, tokenStart: startIndex, tokenEnd: previousIndex + 1))
        }
        return spans
    }

    /// `token_spans_to_char_spans`: token-index spans → scalar-index spans, using each token's
    /// UTF-8 byte range and the text's scalar byte table. Interior token boundaries snap inward,
    /// exactly like the reference's two bisects.
    static func tokenSpansToCharacterSpans(
        _ spans: [PrivacyFilterTokenSpan],
        tokenByteStarts: [Int],
        tokenByteEnds: [Int],
        scalarStarts: [Int],
        scalarEnds: [Int]
    ) -> [PrivacyFilterCharacterSpan] {
        guard !spans.isEmpty else { return [] }
        var converted: [PrivacyFilterCharacterSpan] = []
        converted.reserveCapacity(spans.count)
        for span in spans {
            guard span.tokenStart >= 0,
                  span.tokenStart < span.tokenEnd,
                  span.tokenEnd <= tokenByteStarts.count,
                  span.tokenEnd <= tokenByteEnds.count else { continue }
            let characterStart = bisectRight(scalarEnds, tokenByteStarts[span.tokenStart])
            var characterEnd = bisectLeft(scalarStarts, tokenByteEnds[span.tokenEnd - 1])
            if characterEnd < characterStart { characterEnd = characterStart }
            if characterEnd <= characterStart { continue }
            converted.append(PrivacyFilterCharacterSpan(
                labelIndex: span.labelIndex, start: characterStart, end: characterEnd))
        }
        return converted
    }

    /// `trim_char_spans_whitespace`: trim leading/trailing Python-`isspace()` scalars.
    static func trimCharacterSpansWhitespace(
        _ spans: [PrivacyFilterCharacterSpan],
        text: String
    ) -> [PrivacyFilterCharacterSpan] {
        let isSpace = text.unicodeScalars.map(Self.isPythonWhitespace)
        var trimmed: [PrivacyFilterCharacterSpan] = []
        trimmed.reserveCapacity(spans.count)
        for span in spans {
            var start = span.start
            var end = span.end
            guard start >= 0, start < end, end <= isSpace.count else { continue }
            while start < end, isSpace[start] { start += 1 }
            while end > start, isSpace[end - 1] { end -= 1 }
            if end > start {
                trimmed.append(PrivacyFilterCharacterSpan(
                    labelIndex: span.labelIndex, start: start, end: end))
            }
        }
        return trimmed
    }

    /// Python `str.isspace()` for one scalar: Unicode's White_Space characters **plus** the four
    /// information separators U+001C…U+001F (which are `isspace()` in CPython but not White_Space).
    static func isPythonWhitespace(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x09 ... 0x0D, 0x1C ... 0x1F, 0x20, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000:
            return true
        case 0x2000 ... 0x200A:
            return true
        default:
            return false
        }
    }

    /// UTF-8 encoded length of one Unicode scalar (1…4 bytes; scalars exclude surrogates).
    private static func utf8Width(_ scalar: Unicode.Scalar) -> Int {
        switch scalar.value {
        case 0 ..< 0x80: return 1
        case 0x80 ..< 0x800: return 2
        case 0x800 ..< 0x10000: return 3
        default: return 4
        }
    }

    // MARK: bisect (Python's `bisect` on strictly-increasing lists)

    /// `bisect.bisect_left`: first index whose value is `>= value`.
    static func bisectLeft(_ values: [Int], _ value: Int) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] < value { low = mid + 1 } else { high = mid }
        }
        return low
    }

    /// `bisect.bisect_right`: first index whose value is `> value`.
    static func bisectRight(_ values: [Int], _ value: Int) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] <= value { low = mid + 1 } else { high = mid }
        }
        return low
    }
}
