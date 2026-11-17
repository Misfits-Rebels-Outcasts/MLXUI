import Foundation

/// Errors from the redaction assembly. Every sentence is one plain line and never quotes input text.
nonisolated enum PrivacyFilterRedactionError: Error, LocalizedError, Equatable {
    case scoreRowCountMismatch(tokenCount: Int, scoreRows: Int)

    var errorDescription: String? {
        switch self {
        case .scoreRowCountMismatch(let tokenCount, let scoreRows):
            return "Privacy Filter: the model returned \(scoreRows) score rows for \(tokenCount) tokens. Try again; if it repeats, re-install the model."
        }
    }
}

/// The result of one redaction: the detected spans (all categories, trimmed, non-overlapping, sorted
/// by start) and the redacted text for the requested category subset.
nonisolated struct PrivacyFilterRedaction: Sendable {
    /// Selected spans, sorted by start (decision 4). These are **all** detected categories — the
    /// category subset only controls which of them are replaced in ``redactedText``, so the Run
    /// screen can re-render for a checkbox change without re-running the model.
    let spans: [PrivacyFilterSpan]
    let redactedText: String
    /// Number of model tokens the input produced.
    let tokenCount: Int
}

/// Detected spans → redacted text, ported from `opf/_core/runtime.py::predict_text` with
/// `trim_whitespace=true`, `discard_overlapping_predicted_spans=false`, `output_mode="typed"`.
///
/// The span math runs in `opf`'s Unicode-scalar indices (``PrivacyFilterSpans``) and is converted to
/// UTF-8 byte offsets once; the redacted text is rebuilt from the **original input bytes**, so every
/// byte outside a replaced span is unchanged (decision 4).
nonisolated enum PrivacyFilterRedactor {

    /// The detected, trimmed, non-overlapping spans for one text and its score matrix.
    ///
    /// `logProbs` is `[tokenCount][33]` fp32 log-probabilities — `PrivacyFilterModel.logProbs`'s
    /// output. Decoding follows `predict_text`: ``PrivacyFilterDecoder/decode(logProbs:)``, falling
    /// back to per-token argmax if the decoder's output length is not the token count (the
    /// reference's guard).
    static func detectedSpans(
        _ text: String,
        logProbs: [[Float]],
        tokenization: PrivacyFilterTokenizer.Tokenization,
        decoder: PrivacyFilterDecoder
    ) throws -> [PrivacyFilterSpan] {
        let labelInfo = decoder.labelInfo
        let tokenCount = tokenization.tokenIDs.count
        guard logProbs.count == tokenCount else {
            throw PrivacyFilterRedactionError.scoreRowCountMismatch(
                tokenCount: tokenCount, scoreRows: logProbs.count)
        }
        guard tokenCount > 0 else { return [] }

        var labels = decoder.decode(logProbs: logProbs)
        if labels.count != tokenCount {
            labels = PrivacyFilterDecoder.argmaxPath(logProbs, numClasses: labelInfo.classNames.count)
        }

        var labelsByTokenIndex: [Int: Int] = [:]
        labelsByTokenIndex.reserveCapacity(tokenCount)
        for (index, label) in labels.enumerated() {
            labelsByTokenIndex[index] = label
        }
        let tokenSpans = PrivacyFilterSpans.labelsToTokenSpans(
            labelsByTokenIndex: labelsByTokenIndex, labelInfo: labelInfo)
        guard !tokenSpans.isEmpty else { return [] }

        let scalarRanges = PrivacyFilterSpans.scalarByteRanges(text)
        let characterSpans = PrivacyFilterSpans.tokenSpansToCharacterSpans(
            tokenSpans,
            tokenByteStarts: tokenization.utf8ByteStarts,
            tokenByteEnds: tokenization.utf8ByteEnds,
            scalarStarts: scalarRanges.starts,
            scalarEnds: scalarRanges.ends)
        let trimmedSpans = PrivacyFilterSpans.trimCharacterSpansWhitespace(characterSpans, text: text)
        guard !trimmedSpans.isEmpty else { return [] }

        let inputBytes = Array(text.utf8)
        let graphemeStarts = PrivacyFilterSpans.graphemeByteStarts(text)
        var detected: [PrivacyFilterSpan] = []
        detected.reserveCapacity(trimmedSpans.count)
        for characterSpan in trimmedSpans {
            guard let category = labelInfo.category(forSpanLabel: characterSpan.labelIndex),
                  characterSpan.start >= 0,
                  characterSpan.start < characterSpan.end,
                  characterSpan.end <= scalarRanges.starts.count else { continue }
            let utf8Start = scalarRanges.starts[characterSpan.start]
            let utf8End = scalarRanges.ends[characterSpan.end - 1]
            guard utf8Start >= 0, utf8Start < utf8End, utf8End <= inputBytes.count else { continue }
            let covered = String(decoding: inputBytes[utf8Start ..< utf8End], as: UTF8.self)
            let characterRange = PrivacyFilterSpans.graphemeRange(
                utf8Start: utf8Start, utf8End: utf8End, graphemeStarts: graphemeStarts)
            detected.append(PrivacyFilterSpan(
                category: category,
                utf8Start: utf8Start,
                utf8End: utf8End,
                text: covered,
                characterRange: characterRange))
        }
        return selectNonOverlappingSpans(detected)
    }

    /// Full pipeline for one text: decode labels, reconstruct spans, trim, select, redact.
    ///
    /// `categories == nil` means all eight (decision 4). A subset only changes which selected spans
    /// are replaced; ``PrivacyFilterRedaction/spans`` still carries every detected category.
    static func redact(
        _ text: String,
        logProbs: [[Float]],
        tokenization: PrivacyFilterTokenizer.Tokenization,
        decoder: PrivacyFilterDecoder,
        categories: Set<PrivacyFilterCategory>?
    ) throws -> PrivacyFilterRedaction {
        let spans = try detectedSpans(
            text, logProbs: logProbs, tokenization: tokenization, decoder: decoder)
        return PrivacyFilterRedaction(
            spans: spans,
            redactedText: redactedText(text, spans: spans, categories: categories),
            tokenCount: tokenization.tokenIDs.count)
    }

    /// Rebuild the redacted text from a cached span list (the Run screen's checkbox re-render, with
    /// no model call). `categories == nil` replaces every span.
    static func redactedText(
        _ text: String,
        spans: [PrivacyFilterSpan],
        categories: Set<PrivacyFilterCategory>?
    ) -> String {
        let inputBytes = Array(text.utf8)
        var output: [UInt8] = []
        output.reserveCapacity(inputBytes.count + 16)
        var cursor = 0
        for span in spans {
            guard span.utf8Start >= cursor,
                  span.utf8Start < span.utf8End,
                  span.utf8End <= inputBytes.count else { continue }
            output.append(contentsOf: inputBytes[cursor ..< span.utf8Start])
            let replace = categories.map { $0.contains(span.category) } ?? true
            if replace {
                output.append(contentsOf: span.category.placeholder.utf8)
            } else {
                output.append(contentsOf: inputBytes[span.utf8Start ..< span.utf8End])
            }
            cursor = span.utf8End
        }
        if cursor < inputBytes.count {
            output.append(contentsOf: inputBytes[cursor...])
        }
        return String(decoding: output, as: UTF8.self)
    }

    /// `runtime._select_non_overlapping_spans`: sort by `(start, -(length), label)` and keep a
    /// left-to-right non-overlapping subset.
    static func selectNonOverlappingSpans(_ spans: [PrivacyFilterSpan]) -> [PrivacyFilterSpan] {
        let ordered = spans.sorted { a, b in
            if a.utf8Start != b.utf8Start { return a.utf8Start < b.utf8Start }
            let aLength = a.utf8End - a.utf8Start
            let bLength = b.utf8End - b.utf8Start
            if aLength != bLength { return aLength > bLength }
            return a.category.modelLabel < b.category.modelLabel
        }
        var kept: [PrivacyFilterSpan] = []
        kept.reserveCapacity(ordered.count)
        var cursor = 0
        for span in ordered {
            if span.utf8Start < cursor { continue }
            if span.utf8End <= span.utf8Start { continue }
            kept.append(span)
            cursor = span.utf8End
        }
        return kept
    }
}
