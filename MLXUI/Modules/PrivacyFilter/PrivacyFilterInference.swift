import Foundation

/// Errors surfaced by the inference orchestration. Every sentence is one plain line, never quotes
/// input text, and says what to do (decision 10: metadata such as counts is fine).
nonisolated enum PrivacyFilterInferenceError: Error, LocalizedError, Equatable {
    case emptyText
    case tooManyTokens(Int)
    case invalidWindow
    case unexpectedLabelSpace

    var errorDescription: String? {
        switch self {
        case .emptyText:
            return "Privacy Filter: there is no text to redact."
        case .tooManyTokens(let count):
            return "Privacy Filter: this text is \(count) tokens; the model takes at most \(PrivacyFilterInference.maxTokens). Shorten it and try again."
        case .invalidWindow:
            return "Privacy Filter: the window settings are invalid."
        case .unexpectedLabelSpace:
            return "Privacy Filter: the checkpoint's labels are not the expected eight categories. Install the model again."
        }
    }
}

/// PF-6 — the decode-side pipeline the engine and the parity suites call: tokenize → model
/// log-probabilities → constrained Viterbi labels → spans → redacted text.
///
/// Two paths produce the same rows: the full-mask path (`model.logProbs` over the whole input; the
/// network's banded attention is memory-linear) and ``windowedLogProbs`` for inputs long enough that
/// a single call would hold too much state. The windowed path feeds each core window `halo` tokens
/// of context on both sides with **absolute** RoPE positions, so the core rows are the full-mask
/// rows — the receptive field is `blocks × band` = 8 × 128 = 1024 tokens each side (Trap 3).
nonisolated struct PrivacyFilterInference {

    /// Decision 5's documented context: an input over this many tokens is refused **before** any
    /// model work.
    static let maxTokens = 128_000

    /// The core-token window size used for long inputs.
    static let defaultWindowSize = 2_048

    /// Halo on each side: at least the receptive field (`numLayers × slidingWindow` = 1024).
    static let defaultHalo = 1_024

    /// Inputs at or below this many tokens run through one full-mask call; longer ones are windowed.
    private static let fullMaskTokenLimit = 2_048

    let config: PrivacyFilterConfig
    let model: PrivacyFilterModel
    let tokenizer: PrivacyFilterTokenizer
    let decoder: PrivacyFilterDecoder

    /// Load the checkpoint, tokenizer and default calibration from an installed model folder.
    ///
    /// The tokenizer load is the only suspension point; everything after it is synchronous, so no
    /// MLX array crosses an `await` (Trap 7).
    init(modelDirectory: URL) async throws {
        let tokenizer = try await PrivacyFilterTokenizer(modelDirectory: modelDirectory)
        let config = try PrivacyFilterConfig.load(from: modelDirectory)
        let labelInfo: PrivacyFilterLabelInfo
        do {
            labelInfo = try PrivacyFilterLabelInfo(classNames: config.labelNames)
        } catch {
            throw PrivacyFilterInferenceError.unexpectedLabelSpace
        }
        guard labelInfo.classNames.count == config.numLabels,
              Set(labelInfo.spanLabelCategories.compactMap { $0 }) == Set(PrivacyFilterCategory.allCases) else {
            throw PrivacyFilterInferenceError.unexpectedLabelSpace
        }
        let calibration = try PrivacyFilterCalibration.load(from: modelDirectory)
        let model = try PrivacyFilterModel.load(from: modelDirectory)
        self.tokenizer = tokenizer
        self.config = config
        self.model = model
        self.decoder = PrivacyFilterDecoder(labelInfo: labelInfo, calibration: calibration)
    }

    /// Full pipeline: tokenize → log-probabilities → decode → spans/redaction.
    ///
    /// Input checks happen first — empty/whitespace-only and over ``maxTokens`` are refused before any
    /// model work (decision 5). `progress` receives `0…1`; `isCancelled` is consulted before the model
    /// call and between windows, and throws `CancellationError` (Stop).
    func redact(
        _ text: String,
        categories: Set<PrivacyFilterCategory>?,
        progress: (@Sendable (Double) -> Void)?,
        isCancelled: @Sendable () -> Bool
    ) throws -> PrivacyFilterRedaction {
        try Self.validateInput(text)
        let tokenization = try tokenizer.tokenize(text)
        try Self.validateTokenCount(tokenization.tokenIDs.count)
        try Self.throwIfCancelled(isCancelled)

        let logProbs: [[Float]]
        if tokenization.tokenIDs.count <= Self.fullMaskTokenLimit {
            progress?(0)
            logProbs = try model.logProbs(tokenIDs: tokenization.tokenIDs)
            progress?(1)
        } else {
            logProbs = try windowedLogProbs(
                tokenization.tokenIDs,
                windowSize: Self.defaultWindowSize,
                halo: Self.defaultHalo,
                isCancelled: isCancelled,
                progress: progress)
        }
        try Self.throwIfCancelled(isCancelled)
        return try PrivacyFilterRedactor.redact(
            text,
            logProbs: logProbs,
            tokenization: tokenization,
            decoder: decoder,
            categories: categories)
    }

    /// The linear-memory path: windows of `windowSize` core tokens with `halo` tokens of context on
    /// each side, **absolute** RoPE positions, only the core rows kept.
    ///
    /// A core token's banded attention never reaches past `numLayers × slidingWindow` tokens, so with
    /// `halo` at least that large every core row is identical to the full-mask row (Trap 3; the
    /// windowed-vs-full test pins it). Cancellation is checked before every window.
    func windowedLogProbs(
        _ tokenIDs: [Int],
        windowSize: Int,
        halo: Int,
        isCancelled: @Sendable () -> Bool
    ) throws -> [[Float]] {
        try windowedLogProbs(
            tokenIDs, windowSize: windowSize, halo: halo, isCancelled: isCancelled, progress: nil)
    }

    private func windowedLogProbs(
        _ tokenIDs: [Int],
        windowSize: Int,
        halo: Int,
        isCancelled: @Sendable () -> Bool,
        progress: (@Sendable (Double) -> Void)?
    ) throws -> [[Float]] {
        guard windowSize > 0, halo >= 0 else {
            throw PrivacyFilterInferenceError.invalidWindow
        }
        guard !tokenIDs.isEmpty else { return [] }

        let coreCount = (tokenIDs.count + windowSize - 1) / windowSize
        var rows: [[Float]] = []
        rows.reserveCapacity(tokenIDs.count)
        for windowIndex in 0 ..< coreCount {
            try Self.throwIfCancelled(isCancelled)
            let coreStart = windowIndex * windowSize
            let coreEnd = min(coreStart + windowSize, tokenIDs.count)
            let sliceStart = max(0, coreStart - halo)
            let sliceEnd = min(tokenIDs.count, coreEnd + halo)
            let slice = Array(tokenIDs[sliceStart ..< sliceEnd])
            let sliceRows = try model.logProbs(tokenIDs: slice, positionOffset: sliceStart)
            guard sliceRows.count == slice.count else {
                throw PrivacyFilterInferenceError.invalidWindow
            }
            let firstCoreRow = coreStart - sliceStart
            let lastCoreRow = coreEnd - sliceStart
            rows.append(contentsOf: sliceRows[firstCoreRow ..< lastCoreRow])
            progress?(Double(windowIndex + 1) / Double(coreCount))
        }
        return rows
    }

    // MARK: - Input validation (pure; testable without weights)

    /// Decision 5: text only, at least one non-whitespace character.
    static func validateInput(_ text: String) throws {
        let hasNonWhitespace = text.unicodeScalars.contains { !PrivacyFilterSpans.isPythonWhitespace($0) }
        guard hasNonWhitespace else { throw PrivacyFilterInferenceError.emptyText }
    }

    /// Decision 5: refuse over ``maxTokens`` **before** any model work.
    static func validateTokenCount(_ count: Int) throws {
        guard count <= maxTokens else {
            throw PrivacyFilterInferenceError.tooManyTokens(count)
        }
    }

    private static func throwIfCancelled(_ isCancelled: @Sendable () -> Bool) throws {
        if isCancelled() { throw CancellationError() }
    }
}
