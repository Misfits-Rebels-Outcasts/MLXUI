import Foundation

/// Clef Flash's answer mapping — a from-reference port of `clef_mlx.py::systemone_answer`
/// (`:520–539`) onto the app's shared ``LayaAnswer`` shape (the same one Laya's and CLM's engines
/// produce), so a Run screen and the deciders keep one answer type.
///
/// Three things differ from CLM 8B and are the easy mistakes here (readme T8, port map §4):
///
/// 1. **`confidence` is the maximum probability** (`:529`, `:536`) — not CLM's
///    top-minus-mean-of-the-rest.
/// 2. **A `choice` is reported in the criteria' insertion order** (`:524`), while the *prompt*
///    encodes the same options id-sorted (`:83`). Python's `max` returns the first maximum, so a
///    tie resolves to the option inserted earliest — not to the alphabetically first id.
/// 3. **Every reported number is `round(x, 4)`** — Python's round-half-even on the exact double,
///    which is neither `%.4f` nor Swift's `rounded()`.
///
/// The reference's `noul` body carries only `P(true)` and no `confidence`; `LayaAnswer.confidence`
/// is non-optional, so it is filled with the larger of the two probabilities here (flagged, not
/// invented — see ``answer(question:probabilities:stateTruncated:)``).
/// `scoreLevels` holds display text, as in the CLM port; the reference's `legend` keeps raw values.
nonisolated enum ClefAnswer {
    /// `systemone_answer` (`:520–539`).
    ///
    /// - Parameters:
    ///   - question: the request's own question — its criteria order is what `choice` reports.
    ///   - probabilities: `option_id → P(option)`, keyed the way the prompt encoded them
    ///     (`choice`: ids; `score`: `"0"`, `"1"`, …; `noul`: `"true"`/`"false"`).
    ///   - stateTruncated: the app-side flag from ``ClefEncodedRecord`` (CF-A0 #3).
    static func answer(
        question: ClefQuestion,
        probabilities: [String: Double],
        stateTruncated: Bool
    ) throws -> LayaAnswer {
        let options = try reportedOptions(question)
        let raw = options.map { probabilities[$0.id] ?? 0 }

        var result = LayaAnswer(
            type: question.type.layaType,
            confidence: round4(raw.max() ?? 0),
            probabilities: raw.map(round4),
            optionLabels: options.map(\.id),
            stateTruncated: stateTruncated)

        switch question.type {
        case .noul:
            // `{"type": "noul", "noul": round(probabilities["true"], 4)}` — the false side is not
            // reported. `confidence` is this port's fill (the reference has none for noul).
            result.noulProbability = round4(raw.first ?? 0)
            result.confidence = round4(raw.max() ?? 0)

        case .choice:
            // `max(options, key=probabilities.__getitem__)` — first maximum, insertion order.
            var best = 0
            for index in 1 ..< raw.count where raw[index] > raw[best] { best = index }
            result.choiceLabel = options[best].id
            result.confidence = round4(raw[best])

        case .score:
            // `sum(i * probabilities[lv] for i, lv in enumerate(levels))`, levels `"0"`, `"1"`, …
            result.scoreValue = round4(raw.enumerated().reduce(0.0) { $0 + Double($1.offset) * $1.element })
            result.confidence = round4(raw.max() ?? 0)
            result.scoreLevels = options.map { text($0.description) }
        }
        return result
    }

    /// The order the **answer** reports, which is the request's own order (`:524`, `:532`) — not
    /// the prompt's sorted encode order for `choice`.
    static func reportedOptions(_ question: ClefQuestion) throws -> [(id: String, description: ClefJSON?)] {
        switch question.type {
        case .noul:
            return [(id: "true", description: nil), (id: "false", description: nil)]

        case .choice:
            guard case .object(let fields)? = question.criteria, !fields.isEmpty else {
                throw ClefError.invalidQuestion("\(question.id): criteria must not be empty")
            }
            return fields.map { (id: $0.key, description: $0.value == .null ? nil : $0.value) }

        case .score:
            guard case .array(let items)? = question.criteria, !items.isEmpty else {
                throw ClefError.invalidQuestion("\(question.id): criteria must not be empty")
            }
            return items.enumerated().map {
                (id: String($0.offset), description: $0.element == .null ? nil : $0.element)
            }
        }
    }

    /// Python's `round(x, 4)`. `Decimal(Double)` is the double's **exact** value and
    /// `NSDecimalRound(.bankers)` is the same half-even tie rule, so this does not double-round the
    /// way `(x * 1e4).rounded() / 1e4` can. Past ~1e15 four decimals sit below the ulp, and Python
    /// returns the value unchanged too.
    static func round4(_ value: Double) -> Double {
        guard value.isFinite, abs(value) < 1e15 else { return value }
        // `%.30f` is the exact double to well past the 4th decimal for |x| < 1e15, so the
        // half-even decision is made on the value CPython sees. (Both `Decimal(value)` and
        // `NSDecimalNumber.doubleValue` lose an ulp on this path — `0.88` came back
        // `0.8800000000000001` — hence the string round trip.) Checked against `repr(round(x, 4))`
        // for 822 values: the table in `ClefAnswerTests` plus 400 random values in [0, 1) and 400
        // more at −10⁶·[0, 1) compared bit-for-bit with CPython.
        guard var source = Decimal(string: String(format: "%.30f", value)) else { return value }
        var rounded = Decimal()
        NSDecimalRound(&rounded, &source, 4, .bankers)
        return Double(rounded.description) ?? value
    }

    /// Display text for a criterion value (`legend`/`scoreLevels`). Strings render as themselves —
    /// `DataNormalizer`-free, exactly the level's own words.
    static func text(_ value: ClefJSON?) -> String {
        guard let value else { return "" }
        switch value {
        case .string(let s): return s
        case .null: return ""
        case .bool(let b): return b ? "true" : "false"
        case .int(let i): return String(i)
        case .bigInt(let text): return text
        case .double(let d): return ClefJSON.pythonFloat(d)
        case .array, .object: return ClefJSON.render(value)
        }
    }
}

extension ClefQuestionType {
    /// The shared decider answer type (`Modules/Laya/LayaPrompt.swift:69`).
    nonisolated var layaType: LayaQuestionType {
        switch self {
        case .noul: return .noul
        case .choice: return .choice
        case .score: return .score
        }
    }
}
