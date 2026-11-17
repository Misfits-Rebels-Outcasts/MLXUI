import CoreFoundation
import Foundation

/// Errors from reading `viterbi_calibration.json`. One plain sentence each; they never quote input.
nonisolated enum PrivacyFilterCalibrationError: Error, LocalizedError, Equatable {
    case missingFile
    case unreadable
    case malformed(String)

    var errorDescription: String? {
        switch self {
        case .missingFile:
            return "Privacy Filter: the model folder is missing viterbi_calibration.json. Install the model again."
        case .unreadable:
            return "Privacy Filter: viterbi_calibration.json could not be read. Install the model again."
        case .malformed(let detail):
            return "Privacy Filter: viterbi_calibration.json is not the expected calibration file (\(detail)). Install the model again."
        }
    }
}

/// The six transition biases of `opf`'s constrained Viterbi decoder
/// (`opf/_core/decoding.py::VITERBI_BIAS_KEYS`), at the checkpoint's default operating point.
///
/// The shipped `viterbi_calibration.json` is all zeros, which is exactly ``zero``; the file is still
/// parsed rather than assumed so a future artifact cannot be silently ignored.
nonisolated struct PrivacyFilterCalibration: Sendable, Equatable {
    let transitionBiasBackgroundStay: Float
    let transitionBiasBackgroundToStart: Float
    let transitionBiasInsideToContinue: Float
    let transitionBiasInsideToEnd: Float
    let transitionBiasEndToBackground: Float
    let transitionBiasEndToStart: Float

    /// The default operating point (all-zero biases; the shipped file's contents).
    static let zero = PrivacyFilterCalibration(
        transitionBiasBackgroundStay: 0,
        transitionBiasBackgroundToStart: 0,
        transitionBiasInsideToContinue: 0,
        transitionBiasInsideToEnd: 0,
        transitionBiasEndToBackground: 0,
        transitionBiasEndToStart: 0)

    static let filename = "viterbi_calibration.json"

    private static let biasKeys = [
        "transition_bias_background_stay",
        "transition_bias_background_to_start",
        "transition_bias_inside_to_continue",
        "transition_bias_inside_to_end",
        "transition_bias_end_to_background",
        "transition_bias_end_to_start",
    ]

    /// Read the default operating point from `viterbi_calibration.json` in `directory`.
    ///
    /// Validates the artifact's exact key set the way `opf` does
    /// (`resolve_viterbi_biases_from_calibration_path`): a file with extra, missing or non-numeric
    /// fields is refused rather than half-applied.
    static func load(from directory: URL) throws -> PrivacyFilterCalibration {
        let url = directory.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PrivacyFilterCalibrationError.missingFile
        }
        let data: Data
        do {
            data = try Data(contentsOf: url)
        } catch {
            throw PrivacyFilterCalibrationError.unreadable
        }
        let root: Any
        do {
            root = try JSONSerialization.jsonObject(with: data)
        } catch {
            throw PrivacyFilterCalibrationError.malformed("it is not valid JSON")
        }
        guard let object = root as? [String: Any] else {
            throw PrivacyFilterCalibrationError.malformed("the top level is not an object")
        }
        try requireExactKeys(object, expected: ["operating_points"], field: "the artifact")
        guard let operatingPoints = object["operating_points"] as? [String: Any] else {
            throw PrivacyFilterCalibrationError.malformed("operating_points is not an object")
        }
        try requireExactKeys(operatingPoints, expected: ["default"], field: "operating_points")
        guard let defaultPoint = operatingPoints["default"] as? [String: Any] else {
            throw PrivacyFilterCalibrationError.malformed("operating_points.default is not an object")
        }
        try requireExactKeys(defaultPoint, expected: ["biases"], field: "operating_points.default")
        guard let biases = defaultPoint["biases"] as? [String: Any] else {
            throw PrivacyFilterCalibrationError.malformed("operating_points.default.biases is not an object")
        }
        try requireExactKeys(biases, expected: Set(biasKeys), field: "operating_points.default.biases")
        return PrivacyFilterCalibration(
            transitionBiasBackgroundStay: try float(biases, "transition_bias_background_stay"),
            transitionBiasBackgroundToStart: try float(biases, "transition_bias_background_to_start"),
            transitionBiasInsideToContinue: try float(biases, "transition_bias_inside_to_continue"),
            transitionBiasInsideToEnd: try float(biases, "transition_bias_inside_to_end"),
            transitionBiasEndToBackground: try float(biases, "transition_bias_end_to_background"),
            transitionBiasEndToStart: try float(biases, "transition_bias_end_to_start"))
    }

    private static func float(_ biases: [String: Any], _ key: String) throws -> Float {
        guard let number = biases[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else {
            throw PrivacyFilterCalibrationError.malformed("\(key) is not numeric")
        }
        return number.floatValue
    }

    private static func requireExactKeys(
        _ object: [String: Any], expected: Set<String>, field: String
    ) throws {
        let actual = Set(object.keys)
        guard actual == expected else {
            let missing = expected.subtracting(actual).sorted().joined(separator: ", ")
            let extra = actual.subtracting(expected).sorted().joined(separator: ", ")
            throw PrivacyFilterCalibrationError.malformed(
                "\(field) has missing keys [\(missing)] and extra keys [\(extra)]")
        }
    }
}

/// `opf`'s constrained BIOES Viterbi decoder, ported verbatim from
/// `opf/_core/decoding.py::ViterbiCRFDecoder`.
///
/// The structural transition/start/end tables use `-1e9` for impossible edges and the six
/// calibration biases for the allowed ones, exactly as the reference builds them; decoding is fp32
/// and follows torch's tie-break (the **first** / lowest index wins a tie, for both the transition
/// `max(dim=0)` and the final `argmax`). The fallback when no score is finite is per-token argmax.
nonisolated struct PrivacyFilterDecoder: Sendable {
    /// Large negative score used to mask impossible CRF transitions (`decoding._NEG_INF`).
    static let negativeInfinity: Float = -1e9

    let labelInfo: PrivacyFilterLabelInfo
    let calibration: PrivacyFilterCalibration

    private let startScores: [Float]
    private let endScores: [Float]
    private let transitionScores: [[Float]]
    private let numClasses: Int

    init(labelInfo: PrivacyFilterLabelInfo, calibration: PrivacyFilterCalibration) {
        self.labelInfo = labelInfo
        self.calibration = calibration
        let numClasses = labelInfo.classNames.count
        self.numClasses = numClasses

        let backgroundToken = labelInfo.backgroundTokenLabel
        let backgroundSpan = labelInfo.backgroundSpanLabel

        var startScores = [Float](repeating: Self.negativeInfinity, count: numClasses)
        var endScores = [Float](repeating: Self.negativeInfinity, count: numClasses)
        var transitionScores = [[Float]](
            repeating: [Float](repeating: Self.negativeInfinity, count: numClasses),
            count: numClasses)

        for idx in 0 ..< numClasses {
            let tag = labelInfo.tokenBoundaryTags[idx]
            let spanLabel = labelInfo.tokenToSpanLabel[idx]
            if tag == "B" || tag == "S" || idx == backgroundToken {
                startScores[idx] = 0
            }
            if tag == "E" || tag == "S" || idx == backgroundToken {
                endScores[idx] = 0
            }
            for nextIdx in 0 ..< numClasses {
                let nextTag = labelInfo.tokenBoundaryTags[nextIdx]
                let nextSpanLabel = labelInfo.tokenToSpanLabel[nextIdx]
                guard Self.isValidTransition(
                    prevTag: tag,
                    prevSpan: spanLabel,
                    nextTag: nextTag,
                    nextSpan: nextSpanLabel,
                    backgroundTokenizerLabel: backgroundToken,
                    backgroundSpanLabel: backgroundSpan,
                    nextIdx: nextIdx) else { continue }
                transitionScores[idx][nextIdx] = Self.transitionBias(
                    prevTag: tag,
                    prevSpan: spanLabel,
                    nextTag: nextTag,
                    nextSpan: nextSpanLabel,
                    backgroundTokenizerLabel: backgroundToken,
                    backgroundSpanLabel: backgroundSpan,
                    prevIdx: idx,
                    nextIdx: nextIdx,
                    calibration: calibration)
            }
        }

        self.startScores = startScores
        self.endScores = endScores
        self.transitionScores = transitionScores
    }

    /// `ViterbiCRFDecoder._transition_bias` — the bias of one structurally allowed edge.
    private static func transitionBias(
        prevTag: String?,
        prevSpan: Int?,
        nextTag: String?,
        nextSpan: Int?,
        backgroundTokenizerLabel: Int,
        backgroundSpanLabel: Int,
        prevIdx: Int,
        nextIdx: Int,
        calibration: PrivacyFilterCalibration
    ) -> Float {
        let prevIsBackground = prevSpan == backgroundSpanLabel || prevIdx == backgroundTokenizerLabel
        let nextIsBackground = nextSpan == backgroundSpanLabel || nextIdx == backgroundTokenizerLabel

        if prevIsBackground {
            if nextIsBackground { return calibration.transitionBiasBackgroundStay }
            if nextTag == "B" || nextTag == "S" { return calibration.transitionBiasBackgroundToStart }
            return 0
        }
        if prevTag == "B" || prevTag == "I" {
            if nextTag == "I", prevSpan == nextSpan { return calibration.transitionBiasInsideToContinue }
            if nextTag == "E", prevSpan == nextSpan { return calibration.transitionBiasInsideToEnd }
            return 0
        }
        if prevTag == "E" || prevTag == "S" {
            if nextIsBackground { return calibration.transitionBiasEndToBackground }
            if nextTag == "B" || nextTag == "S" { return calibration.transitionBiasEndToStart }
            return 0
        }
        return 0
    }

    /// `ViterbiCRFDecoder._is_valid_transition` — whether one CRF edge is structurally allowed.
    private static func isValidTransition(
        prevTag: String?,
        prevSpan: Int?,
        nextTag: String?,
        nextSpan: Int?,
        backgroundTokenizerLabel: Int,
        backgroundSpanLabel: Int,
        nextIdx: Int
    ) -> Bool {
        let nextIsBackground = nextSpan == backgroundSpanLabel || nextIdx == backgroundTokenizerLabel
        if (nextSpan == nil || nextTag == nil) && !nextIsBackground {
            return false
        }
        if prevSpan == nil || prevTag == nil {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        let prevIsBackground = prevSpan == backgroundSpanLabel
        if prevIsBackground {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        if prevTag == "E" || prevTag == "S" {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        if prevTag == "B" || prevTag == "I" {
            let sameSpan = prevSpan == nextSpan
            return sameSpan && (nextTag == "I" || nextTag == "E")
        }
        return false
    }

    /// Decode one `[seq_len, 33]` fp32 log-probability matrix into label ids.
    func decode(logProbs: [[Float]]) -> [Int] {
        let seqLength = logProbs.count
        guard seqLength > 0 else { return [] }
        guard logProbs.allSatisfy({ $0.count == numClasses }) else {
            // Structurally impossible input for this decoder; mirror opf's argmax fallback rather
            // than indexing out of bounds.
            return Self.argmaxPath(logProbs, numClasses: numClasses)
        }

        var scores = [Float](repeating: 0, count: numClasses)
        for label in 0 ..< numClasses {
            scores[label] = logProbs[0][label] + startScores[label]
        }

        var backpointers = [[Int]](repeating: [Int](repeating: 0, count: numClasses), count: max(0, seqLength - 1))
        for step in 1 ..< seqLength {
            var bestPrev = [Int](repeating: 0, count: numClasses)
            var bestScores = [Float](repeating: 0, count: numClasses)
            for next in 0 ..< numClasses {
                var bestScore = -Float.infinity
                var bestIndex = 0
                for prev in 0 ..< numClasses {
                    let candidate = scores[prev] + transitionScores[prev][next]
                    if Self.isBetter(candidate, than: bestScore) {
                        bestScore = candidate
                        bestIndex = prev
                    }
                }
                bestScores[next] = bestScore
                bestPrev[next] = bestIndex
            }
            for label in 0 ..< numClasses {
                scores[label] = bestScores[label] + logProbs[step][label]
            }
            backpointers[step - 1] = bestPrev
        }

        guard scores.contains(where: { $0.isFinite }) else {
            return Self.argmaxPath(logProbs, numClasses: numClasses)
        }

        var finalScores = [Float](repeating: 0, count: numClasses)
        for label in 0 ..< numClasses {
            finalScores[label] = scores[label] + endScores[label]
        }
        var last = Self.argmaxIndex(finalScores)
        var path = [Int](repeating: 0, count: seqLength)
        path[seqLength - 1] = last
        if seqLength >= 2 {
            for step in stride(from: seqLength - 2, through: 0, by: -1) {
                last = backpointers[step][last]
                path[step] = last
            }
        }
        return path
    }

    /// Decode one `[seq_len, 33]` matrix as per-token argmax (opf's no-finite-score fallback and
    /// the argmax decode mode).
    static func argmaxPath(_ logProbs: [[Float]], numClasses: Int) -> [Int] {
        logProbs.map { row in
            var bestIndex = 0
            var bestValue = row.first ?? -Float.infinity
            for label in row.indices.dropFirst() where isBetter(row[label], than: bestValue) {
                bestValue = row[label]
                bestIndex = label
            }
            return min(bestIndex, max(0, numClasses - 1))
        }
    }

    /// torch's `max` / `argmax` tie-break: the **first** (lowest-index) maximum wins; NaN wins over
    /// any non-NaN, and the first NaN wins among NaNs.
    @inline(__always)
    private static func isBetter(_ candidate: Float, than best: Float) -> Bool {
        if candidate.isNaN { return !best.isNaN }
        if best.isNaN { return false }
        return candidate > best
    }

    private static func argmaxIndex(_ values: [Float]) -> Int {
        var bestIndex = 0
        var bestValue = values.first ?? -Float.infinity
        for index in values.indices.dropFirst() where isBetter(values[index], than: bestValue) {
            bestValue = values[index]
            bestIndex = index
        }
        return bestIndex
    }
}
