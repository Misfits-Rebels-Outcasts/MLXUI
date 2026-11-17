import Foundation

/// The eight personal-data categories the checkpoint can label. `rawValue` is the UI name from the
/// issue (decision 7's `only=` vocabulary); `modelLabel` is the checkpoint's span-class name and
/// `placeholder` is `opf`'s redaction placeholder for it (`runtime._label_placeholder`).
nonisolated enum PrivacyFilterCategory: String, CaseIterable, Sendable {
    case person
    case email
    case phone
    case url
    case address
    case date
    case account
    case secret

    /// The span-class name in the checkpoint's label space (`config.json`'s class names after the
    /// BIOES prefix is removed).
    var modelLabel: String {
        switch self {
        case .person: return "private_person"
        case .email: return "private_email"
        case .phone: return "private_phone"
        case .url: return "private_url"
        case .address: return "private_address"
        case .date: return "private_date"
        case .account: return "account_number"
        case .secret: return "secret"
        }
    }

    /// `opf`'s placeholder text: the class name upper-cased inside angle brackets
    /// (`runtime._label_placeholder`: non-alphanumerics become `_`, trimmed).
    var placeholder: String {
        "<" + Self.placeholderLabel(for: modelLabel) + ">"
    }

    static func placeholderLabel(for modelLabel: String) -> String {
        var normalized = ""
        var lastWasUnderscore = false
        for scalar in modelLabel.uppercased().unicodeScalars {
            let isAlphanumeric = CharacterSet.alphanumerics
                .contains(scalar)
            if isAlphanumeric {
                normalized.unicodeScalars.append(scalar)
                lastWasUnderscore = false
            } else if !lastWasUnderscore {
                normalized.append("_")
                lastWasUnderscore = true
            }
        }
        while normalized.hasPrefix("_") { normalized.removeFirst() }
        while normalized.hasSuffix("_") { normalized.removeLast() }
        return normalized.isEmpty ? "REDACTED" : normalized
    }

    init?(modelLabel: String) {
        guard let match = Self.allCases.first(where: { $0.modelLabel == modelLabel }) else { return nil }
        self = match
    }
}

/// Errors from building the label space. One plain sentence each; they never quote input text.
nonisolated enum PrivacyFilterLabelError: Error, LocalizedError, Equatable {
    case missingBackground
    case malformedClassName(String)
    case duplicateClassName(String)
    case missingBoundary(baseLabel: String, boundary: String)

    var errorDescription: String? {
        switch self {
        case .missingBackground:
            return "Privacy Filter: the model's label list has no background label. Install the model again."
        case .malformedClassName(let name):
            return "Privacy Filter: the model's label list entry \"\(name)\" is not a BIOES label. Install the model again."
        case .duplicateClassName(let name):
            return "Privacy Filter: the model's label list repeats \"\(name)\". Install the model again."
        case .missingBoundary(let baseLabel, let boundary):
            return "Privacy Filter: the model's label list is missing the \(boundary)-\(baseLabel) label. Install the model again."
        }
    }
}

/// The token-label → span-label tables `opf` builds in
/// `opf/_core/sequence_labeling.py::build_label_info`, ported verbatim.
///
/// The checkpoint's 33-class label order (`config.json`'s `id2label`) is not the span-class order:
/// `opf` derives the span space by first appearance, so building the tables rather than hard-coding
/// them is what keeps the Viterbi decoder and the span reconstruction identical to the reference.
nonisolated struct PrivacyFilterLabelInfo: Sendable {
    /// The frozen class-name order of `openai/privacy-filter`'s `config.json` `id2label`
    /// (ids 0…32). Used by ``standard``; a checkpoint whose config disagrees fails the label build
    /// instead of silently decoding wrong.
    static let canonicalLabelNames: [String] = [
        "O",
        "B-account_number", "I-account_number", "E-account_number", "S-account_number",
        "B-private_address", "I-private_address", "E-private_address", "S-private_address",
        "B-private_date", "I-private_date", "E-private_date", "S-private_date",
        "B-private_email", "I-private_email", "E-private_email", "S-private_email",
        "B-private_person", "I-private_person", "E-private_person", "S-private_person",
        "B-private_phone", "I-private_phone", "E-private_phone", "S-private_phone",
        "B-private_url", "I-private_url", "E-private_url", "S-private_url",
        "B-secret", "I-secret", "E-secret", "S-secret",
    ]

    /// The label space for the shipped checkpoint, built from ``canonicalLabelNames``.
    ///
    /// The constant cannot be malformed — it is the frozen `id2label` list above — so a failure here
    /// is a programming error, not something a runtime input can cause.
    static let standard: PrivacyFilterLabelInfo = {
        do {
            return try PrivacyFilterLabelInfo(classNames: canonicalLabelNames)
        } catch {
            preconditionFailure("PrivacyFilterLabelInfo.standard is malformed: \(error)")
        }
    }()

    /// The 33 BIOES names in id order, exactly as given (``build_label_info``'s `class_names`).
    let classNames: [String]
    /// Span-class names in `opf`'s derived order: index 0 is `"O"`, then each base label in first
    /// appearance order.
    let spanClassNames: [String]
    /// `baseLabel → boundary tag ("B"/"I"/"E"/"S") → token-label id`.
    let boundaryLabelLookup: [String: [String: Int]]
    /// Token-label id → span-label index (``build_label_info``'s `token_to_span_label`).
    let tokenToSpanLabel: [Int: Int]
    /// Token-label id → boundary tag; background (and only background) is absent.
    let tokenBoundaryTags: [Int: String]
    /// Token-label id of `"O"` (``build_label_info``'s `background_token_label`).
    let backgroundTokenLabel: Int
    /// Span-label index of `"O"` (``build_label_info``'s `background_span_label`; always 0).
    let backgroundSpanLabel: Int
    /// Span-label index → category; `nil` for `"O"`.
    let spanLabelCategories: [PrivacyFilterCategory?]

    init(classNames: [String]) throws {
        var spanClassNames: [String] = [Self.backgroundClassName]
        var spanLabelLookup: [String: Int] = [Self.backgroundClassName: 0]
        var boundaryLabelLookup: [String: [String: Int]] = [:]
        var tokenToSpanLabel: [Int: Int] = [:]
        var tokenBoundaryTags: [Int: String] = [:]
        var backgroundTokenLabel: Int?

        var seen = Set<String>()
        for (idx, name) in classNames.enumerated() {
            guard seen.insert(name).inserted else {
                throw PrivacyFilterLabelError.duplicateClassName(name)
            }
            if name == Self.backgroundClassName {
                backgroundTokenLabel = idx
                tokenToSpanLabel[idx] = spanLabelLookup[Self.backgroundClassName]
                continue
            }
            let pieces = name.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false)
            guard pieces.count == 2, Self.boundaryTags.contains(String(pieces[0])), !pieces[1].isEmpty else {
                throw PrivacyFilterLabelError.malformedClassName(name)
            }
            let boundary = String(pieces[0])
            let baseLabel = String(pieces[1])
            let spanIndex: Int
            if let existing = spanLabelLookup[baseLabel] {
                spanIndex = existing
            } else {
                spanIndex = spanClassNames.count
                spanClassNames.append(baseLabel)
                spanLabelLookup[baseLabel] = spanIndex
            }
            tokenToSpanLabel[idx] = spanIndex
            tokenBoundaryTags[idx] = boundary
            boundaryLabelLookup[baseLabel, default: [:]][boundary] = idx
        }

        guard let backgroundTokenLabel else {
            throw PrivacyFilterLabelError.missingBackground
        }
        for (baseLabel, mapping) in boundaryLabelLookup {
            for boundary in Self.boundaryTags where mapping[boundary] == nil {
                throw PrivacyFilterLabelError.missingBoundary(baseLabel: baseLabel, boundary: boundary)
            }
        }

        self.classNames = classNames
        self.spanClassNames = spanClassNames
        self.boundaryLabelLookup = boundaryLabelLookup
        self.tokenToSpanLabel = tokenToSpanLabel
        self.tokenBoundaryTags = tokenBoundaryTags
        self.backgroundTokenLabel = backgroundTokenLabel
        self.backgroundSpanLabel = spanLabelLookup[Self.backgroundClassName] ?? 0
        self.spanLabelCategories = spanClassNames.map { name in
            name == Self.backgroundClassName ? nil : PrivacyFilterCategory(modelLabel: name)
        }
    }

    private static let backgroundClassName = "O"
    private static let boundaryTags = ["B", "I", "E", "S"]

    /// The category for one *span*-label index, or `nil` for background.
    func category(forSpanLabel spanLabel: Int) -> PrivacyFilterCategory? {
        guard spanLabel >= 0, spanLabel < spanLabelCategories.count else { return nil }
        return spanLabelCategories[spanLabel]
    }

    /// The category for one *token*-label id, or `nil` for background.
    func category(forTokenLabel tokenLabel: Int) -> PrivacyFilterCategory? {
        guard let spanLabel = tokenToSpanLabel[tokenLabel] else { return nil }
        return category(forSpanLabel: spanLabel)
    }

    /// The token-label id for `category`'s `boundary` tag, from ``boundaryLabelLookup``.
    func tokenLabelID(category: PrivacyFilterCategory, boundary: String) -> Int? {
        boundaryLabelLookup[category.modelLabel]?[boundary]
    }
}
