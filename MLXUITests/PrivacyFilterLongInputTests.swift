import Foundation
import Testing
@testable import MLXUI

/// PF-6, Trap 3 — the long-input path.
///
/// `windowedLogProbs` runs core-token windows with `halo` tokens of context on each side and
/// **absolute** RoPE positions; because the receptive field is `numLayers × slidingWindow`
/// (8 × 128 = 1024), a halo of at least 1024 makes every core row identical to the full-mask row.
/// This suite pins that equality on the ~6000-token corpus document, pins identical **spans**, and
/// pins the 128,000-token refusal (decision 5) without a forward pass.
///
/// Opt-in: `MLXUI_TEST_PRIVACY=1` and `MLXUI_TEST_PRIVACY_DIR=/private/tmp/pf-model`
/// (`TEST_RUNNER_` prefixes reach the test host through xcodebuild).
struct PrivacyFilterLongInputTests {

    /// Thread-safe progress/cancel probes (the closures are `@Sendable`).
    final class Probe: @unchecked Sendable {
        private let lock = NSLock()
        private var progressCalls = 0
        private var lastProgress: Double = -1
        private var cancelCalls = 0
        private var cancelAfter: Int?

        func recordProgress(_ value: Double) {
            lock.lock(); progressCalls += 1; lastProgress = value; lock.unlock()
        }

        var progressCount: Int {
            lock.lock(); defer { lock.unlock() }
            return progressCalls
        }

        var lastProgressValue: Double {
            lock.lock(); defer { lock.unlock() }
            return lastProgress
        }

        /// Stop returning `false` after `count` checks (a deterministic mid-run Stop).
        func cancelAfter(_ count: Int) {
            lock.lock(); cancelAfter = count; cancelCalls = 0; lock.unlock()
        }

        func isCancelled() -> Bool {
            lock.lock(); defer { lock.unlock() }
            cancelCalls += 1
            if let cancelAfter { return cancelCalls > cancelAfter }
            return false
        }
    }

    static func maxAbsDifference(_ a: [[Float]], _ b: [[Float]]) -> Float {
        guard a.count == b.count else { return .infinity }
        var worst: Float = 0
        for (rowA, rowB) in zip(a, b) {
            guard rowA.count == rowB.count else { return .infinity }
            for (valueA, valueB) in zip(rowA, rowB) {
                worst = max(worst, abs(valueA - valueB))
            }
        }
        return worst
    }

    /// Per-row max |Δ| over the columns.
    static func rowMaxDiffs(_ a: [[Float]], _ b: [[Float]]) -> [Float] {
        guard a.count == b.count else { return [] }
        var worst: [Float] = []
        worst.reserveCapacity(a.count)
        for (rowA, rowB) in zip(a, b) where rowA.count == rowB.count {
            var rowWorst: Float = 0
            for (valueA, valueB) in zip(rowA, rowB) {
                rowWorst = max(rowWorst, abs(valueA - valueB))
            }
            worst.append(rowWorst)
        }
        return worst
    }

    /// Pearson correlation over the flattened matrices (the issue's `r >= 0.999` shape).
    static func pearson(_ a: [[Float]], _ b: [[Float]]) -> Float {
        let flatA = a.flatMap { $0 }
        let flatB = b.flatMap { $0 }
        guard flatA.count == flatB.count, !flatA.isEmpty else { return 0 }
        let n = Float(flatA.count)
        var sumA: Float = 0
        var sumB: Float = 0
        for (x, y) in zip(flatA, flatB) { sumA += x; sumB += y }
        let meanA = sumA / n
        let meanB = sumB / n
        var covariance: Float = 0
        var varianceA: Float = 0
        var varianceB: Float = 0
        for (x, y) in zip(flatA, flatB) {
            let dx = x - meanA
            let dy = y - meanB
            covariance += dx * dy
            varianceA += dx * dx
            varianceB += dy * dy
        }
        let denominator = (varianceA * varianceB).squareRoot()
        return denominator > 0 ? covariance / denominator : 1
    }

    static func fixtureLongDocument() throws -> PrivacyFilterDecodeTestSupport.Golden.Case {
        let golden = try PrivacyFilterDecodeTestSupport.loadGolden()
        return try #require(golden.long_document, "the corpus has no long_document")
    }

    static func makeInference() async throws -> PrivacyFilterInference {
        let directory = try #require(PrivacyFilterDecodeTestSupport.weightsDirectory)
        return try await PrivacyFilterInference(modelDirectory: directory)
    }

    // MARK: - Windowed == full mask (PF-6 acceptance)

    @Suite(.serialized, .enabled(
        if: PrivacyFilterDecodeTestSupport.canRunRealWeights && PrivacyFilterDecodeTestSupport.goldenExists,
        "needs MLXUI_TEST_PRIVACY=1, MLXUI_TEST_PRIVACY_DIR and the checked-in fixtures"))
    struct WindowedEqualityTests {

        /// The issue's long-input acceptance: the windowed path must reproduce the full-mask path
        /// "at the logit tolerance from PF-5, with identical spans".
        ///
        /// Criterion (implementer's call, cross-checked with the network owner and the Lead's
        /// ruling): identical spans (next test), Pearson `r >= 0.999` over every log-prob element,
        /// and **the windowed path adds no more row-level divergence than a mere call-length change
        /// does**. The measured row histograms are printed. The outliers are MoE router near-tie
        /// flips: the router is discrete, so a last-bit hidden-state difference (shape-dependent
        /// GEMM reduction order) can change a token's top-4 experts and move that row by several
        /// units (the issue's Trap 4 says exactly this and requires exact span equality, which we
        /// have). Two controls distinguish that from a windowing bug:
        /// - the same graph twice must be bit-identical (MLX is deterministic);
        /// - the same model over a **shorter call length** (prefix) is compared with the full call's
        ///   prefix: on the 6,019-token document this control alone already diverges, so a bare
        ///   "all rows within 0.05" bound is not achievable at this length and is not what the
        ///   windowing code adds.
        ///
        /// Measured on this host (2026-11-18, bf16, 18 GB MacBook Pro; after network's fused
        /// fp32 GEMM+bias `linear` fix, which cuts the reference parity gap but leaves this
        /// call-shape comparison unchanged):
        /// - windowed-vs-full: Pearson 0.9995544, max|Δ| 6.96875, 5,551/6,019 rows (92%) within 0.05;
        /// - call-length control: Pearson 0.99693173, max|Δ| 17.584, 3,505/4,096 rows (85%) within
        ///   0.05 (the worst control rows are the 1,024 at the call's tail, which lack the future
        ///   context the full call gives their 8-layer receptive field);
        /// - spans byte-identical: 425/425 on the long document (also equal to opf's).
        @Test func windowedLogProbsEqualTheFullMaskPath() async throws {
            let inference = try await PrivacyFilterLongInputTests.makeInference()
            let longDocument = try PrivacyFilterLongInputTests.fixtureLongDocument()
            let tokenIDs = longDocument.token_ids
            let windowSize = PrivacyFilterInference.defaultWindowSize
            let halo = PrivacyFilterInference.defaultHalo
            #expect(tokenIDs.count > windowSize, "the long document should span several windows")
            #expect(tokenIDs.count <= PrivacyFilterInference.maxTokens)

            let full = try inference.model.logProbs(tokenIDs: tokenIDs)
            let fullAgain = try inference.model.logProbs(tokenIDs: tokenIDs)
            let windowed = try inference.windowedLogProbs(
                tokenIDs, windowSize: windowSize, halo: halo, isCancelled: { false })
            let windowedAgain = try inference.windowedLogProbs(
                tokenIDs, windowSize: windowSize, halo: halo, isCancelled: { false })
            #expect(full.count == tokenIDs.count)
            #expect(windowed.count == tokenIDs.count)

            // Control A — the same graph twice. Without this, a windowed-vs-full delta has no meaning.
            #expect(full == fullAgain, "the full-mask forward is not deterministic between calls")
            #expect(windowed == windowedAgain)

            let worst = PrivacyFilterLongInputTests.maxAbsDifference(full, windowed)
            let correlation = PrivacyFilterLongInputTests.pearson(full, windowed)
            let rowWorst = PrivacyFilterLongInputTests.rowMaxDiffs(full, windowed)
            let rowCount = rowWorst.count
            let rowsTight = rowWorst.filter { $0 <= 0.05 }.count
            let tightFraction = rowCount > 0 ? Double(rowsTight) / Double(rowCount) : 1
            let rowsOverPointOne = rowWorst.filter { $0 > 0.1 }.count
            let rowsOverOne = rowWorst.filter { $0 > 1.0 }.count
            let outlierRows = rowWorst.enumerated()
                .filter { $0.element > 0.05 }
                .sorted { $0.element > $1.element }
                .prefix(8)
                .map { "\($0.offset):\($0.element)" }
                .joined(separator: ",")
            print("PF6 windowed-vs-full: tokens=\(tokenIDs.count) window=\(windowSize) halo=\(halo) pearson=\(correlation) maxAbsDelta=\(worst) rowsWithin0.05=\(rowsTight)/\(rowCount) (\(Int(tightFraction * 100))%) rowsOver0.1=\(rowsOverPointOne) rowsOver1.0=\(rowsOverOne) outlierRows=[\(outlierRows)]")
            PrivacyFilterDecodeTestSupport.record("PF6 windowed-vs-full: tokens=\(tokenIDs.count) window=\(windowSize) halo=\(halo) pearson=\(correlation) maxAbsDelta=\(worst) rowsWithin0.05=\(rowsTight)/\(rowCount) (\(Int(tightFraction * 100))%) rowsOver0.1=\(rowsOverPointOne) rowsOver1.0=\(rowsOverOne) outlierRows=[\(outlierRows)]")
            #expect(correlation >= 0.999, "windowed vs full-mask Pearson r = \(correlation)")

            // Control B — the same model over the first two windows' worth of tokens, so the CALL
            // LENGTH differs but the tokens and RoPE positions are identical. A shape-dependent GERM
            // reduction order shows up here without any windowing code involved.
            let prefixCount = min(tokenIDs.count, windowSize * 2)
            let prefixCall = try inference.model.logProbs(tokenIDs: Array(tokenIDs.prefix(prefixCount)))
            let fullPrefix = Array(full.prefix(prefixCount))
            let prefixWorst = PrivacyFilterLongInputTests.maxAbsDifference(fullPrefix, prefixCall)
            let prefixCorrelation = PrivacyFilterLongInputTests.pearson(fullPrefix, prefixCall)
            let prefixRowWorst = PrivacyFilterLongInputTests.rowMaxDiffs(fullPrefix, prefixCall)
            let prefixRowsTight = prefixRowWorst.filter { $0 <= 0.05 }.count
            let prefixTightFraction = prefixRowWorst.isEmpty
                ? 1 : Double(prefixRowsTight) / Double(prefixRowWorst.count)
            let prefixOutlierRows = prefixRowWorst.enumerated()
                .filter { $0.element > 0.05 }
                .sorted { $0.element > $1.element }
                .prefix(8)
                .map { "\($0.offset):\($0.element)" }
                .joined(separator: ",")
            print("PF6 length-control: prefix=\(prefixCount) pearson=\(prefixCorrelation) maxAbsDelta=\(prefixWorst) rowsWithin0.05=\(prefixRowsTight)/\(prefixRowWorst.count) (\(Int(prefixTightFraction * 100))%) outlierRows=[\(prefixOutlierRows)]")
            PrivacyFilterDecodeTestSupport.record("PF6 length-control: prefix=\(prefixCount) pearson=\(prefixCorrelation) maxAbsDelta=\(prefixWorst) rowsWithin0.05=\(prefixRowsTight)/\(prefixRowWorst.count) (\(Int(prefixTightFraction * 100))%) outlierRows=[\(prefixOutlierRows)]")
            #expect(prefixCorrelation >= 0.99,
                    "the call-length control unexpectedly disagrees (r=\(prefixCorrelation))")

            // The windowed code must add **no more** divergence than a mere call-length change does:
            // same model, same positions, same band, only the call's shape differs in the control.
            // That is the Trap-4-safe form of "at the logit tolerance from PF-5" for a 128-expert
            // MoE, where a shape-dependent GEMM reduction order flips top-4 router choices on a
            // fraction of rows (measured: see the two printed lines; spans are byte-identical in
            // `windowedPathGivesIdenticalSpansAndRedactedText`).
            #expect(tightFraction >= prefixTightFraction - 0.05,
                    "the windowed path diverged more than the call-shape control: windowed \(Int(tightFraction * 100))% of rows within 0.05 vs control \(Int(prefixTightFraction * 100))% (windowed r=\(correlation), control r=\(prefixCorrelation), maxAbsDelta windowed=\(worst) control=\(prefixWorst))")
        }

        @Test func windowedPathGivesIdenticalSpansAndRedactedText() async throws {
            let inference = try await PrivacyFilterLongInputTests.makeInference()
            let longDocument = try PrivacyFilterLongInputTests.fixtureLongDocument()
            let tokenIDs = longDocument.token_ids

            let full = try inference.model.logProbs(tokenIDs: tokenIDs)
            let windowed = try inference.windowedLogProbs(
                tokenIDs,
                windowSize: PrivacyFilterInference.defaultWindowSize,
                halo: PrivacyFilterInference.defaultHalo,
                isCancelled: { false })
            let tokenization = PrivacyFilterDecodeTestSupport.tokenization(for: longDocument)
            let fullSpans = try PrivacyFilterRedactor.detectedSpans(
                longDocument.text, logProbs: full, tokenization: tokenization, decoder: inference.decoder)
            let windowedSpans = try PrivacyFilterRedactor.detectedSpans(
                longDocument.text, logProbs: windowed, tokenization: tokenization, decoder: inference.decoder)
            #expect(fullSpans == windowedSpans, "windowed spans differ from full-mask spans")

            // The fixture's opf spans are the full-mask reference.
            #expect(windowedSpans.count == longDocument.spans.count,
                    "\(windowedSpans.count) spans vs \(longDocument.spans.count)")
            if windowedSpans.count == longDocument.spans.count {
                for (span, expected) in zip(windowedSpans, longDocument.spans) {
                    #expect(span.category.modelLabel == expected.label)
                    #expect(span.utf8Start == expected.start_byte)
                    #expect(span.utf8End == expected.end_byte)
                }
            }
            print("PF6 long document: \(tokenIDs.count) tokens, \(windowedSpans.count) spans identical windowed vs full")
            PrivacyFilterDecodeTestSupport.record("PF6 long document: \(tokenIDs.count) tokens, \(windowedSpans.count) spans identical windowed vs full")
        }

        @Test func redactUsesTheWindowedPathAndMatchesOpfOnTheLongDocument() async throws {
            let inference = try await PrivacyFilterLongInputTests.makeInference()
            let longDocument = try PrivacyFilterLongInputTests.fixtureLongDocument()
            let probe = PrivacyFilterLongInputTests.Probe()
            let redaction = try inference.redact(
                longDocument.text,
                categories: nil,
                progress: { probe.recordProgress($0) },
                isCancelled: { false })
            #expect(redaction.tokenCount == longDocument.token_ids.count)
            #expect(redaction.spans.count == longDocument.spans.count)
            #expect(redaction.redactedText == longDocument.redacted_text,
                    "windowed redaction differs from opf's on the long document")
            // A multi-window input reports progress per window.
            #expect(probe.progressCount >= 2)
            #expect(probe.lastProgressValue == 1.0)
        }

        @Test func stopCancelsBetweenWindows() async throws {
            let inference = try await PrivacyFilterLongInputTests.makeInference()
            let longDocument = try PrivacyFilterLongInputTests.fixtureLongDocument()
            let probe = PrivacyFilterLongInputTests.Probe()
            probe.cancelAfter(1) // let one window run, then Stop
            do {
                _ = try inference.windowedLogProbs(
                    longDocument.token_ids,
                    windowSize: PrivacyFilterInference.defaultWindowSize,
                    halo: PrivacyFilterInference.defaultHalo,
                    isCancelled: { probe.isCancelled() })
                Issue.record("Stop was ignored")
            } catch is CancellationError {
                // expected: Stop is a CancellationError, not a plain sentence.
            }
        }
    }

    // MARK: - The 128,000-token refusal (decision 5)

    @Suite(.serialized, .enabled(
        if: PrivacyFilterDecodeTestSupport.canRunRealWeights && PrivacyFilterDecodeTestSupport.goldenExists,
        "needs MLXUI_TEST_PRIVACY=1, MLXUI_TEST_PRIVACY_DIR and the checked-in fixtures"))
    struct InputLimitTests {
        @Test func overTheLimitIsRefusedBeforeAnyForwardPass() async throws {
            let inference = try await PrivacyFilterLongInputTests.makeInference()
            let longDocument = try PrivacyFilterLongInputTests.fixtureLongDocument()
            // Build a text that is definitely over the documented limit: repeat the corpus document
            // until tokenising it reports more than maxTokens.
            var text = longDocument.text
            var tokenCount = longDocument.token_ids.count
            let tokensPerRepeat = max(1, longDocument.token_ids.count + 1)
            var repeats = PrivacyFilterInference.maxTokens / tokensPerRepeat + 2
            var attempts = 0
            while tokenCount <= PrivacyFilterInference.maxTokens, attempts < 8 {
                text = Array(repeating: longDocument.text, count: repeats).joined(separator: "\n")
                tokenCount = try inference.tokenizer.tokenize(text).tokenCount
                repeats *= 2
                attempts += 1
            }
            #expect(tokenCount > PrivacyFilterInference.maxTokens)

            let probe = PrivacyFilterLongInputTests.Probe()
            do {
                _ = try inference.redact(
                    text,
                    categories: nil,
                    progress: { probe.recordProgress($0) },
                    isCancelled: { false })
                Issue.record("over-limit input was accepted")
            } catch let error as PrivacyFilterInferenceError {
                guard case .tooManyTokens(let refusedCount) = error else {
                    Issue.record("wrong refusal: \(error)")
                    return
                }
                #expect(refusedCount == tokenCount)
                #expect(error.errorDescription?.contains("128000") == true)
                // Refused before the model ran: the progress callback never fired.
                #expect(probe.progressCount == 0)
            }
            print("PF6 input limit: refused \(tokenCount) tokens before any forward pass")
            PrivacyFilterDecodeTestSupport.record("PF6 input limit: refused \(tokenCount) tokens before any forward pass")
        }

        @Test func emptyAndWhitespaceOnlyInputIsRefusedWithoutAForwardPass() async throws {
            let inference = try await PrivacyFilterLongInputTests.makeInference()
            let probe = PrivacyFilterLongInputTests.Probe()
            for text in ["", " \t\r\n \u{00a0}"] {
                do {
                    _ = try inference.redact(
                        text, categories: nil,
                        progress: { probe.recordProgress($0) },
                        isCancelled: { false })
                    Issue.record("empty/whitespace input was accepted")
                } catch let error as PrivacyFilterInferenceError {
                    #expect(error == .emptyText)
                }
            }
            #expect(probe.progressCount == 0)
        }
    }
}
