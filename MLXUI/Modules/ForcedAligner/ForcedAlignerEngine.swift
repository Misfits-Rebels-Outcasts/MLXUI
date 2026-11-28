import Foundation
import MLX
import MLXAudioSTT

/// One aligned unit: the library's own text unit (a word in space-separated languages, a character in Chinese) with the
/// times it is spoken, in seconds. FA-3 for GitHub issue #13.
///
/// Decision 4's contract, applied by ``ForcedAlignerEngine/units(from:duration:)``: one item per aligned unit in
/// transcript order, times rounded to the millisecond and clipped to `[0, audio duration]`, `end >= start`, and starts
/// never decrease.
nonisolated struct AlignedUnit: Sendable, Equatable, Hashable {
    let text: String
    let start: Double
    let end: Double

    var duration: Double { end - start }
}

/// MLX-native forced alignment via mlx-audio-swift's `Qwen3ForcedAlignerModel` (the pinned `mlx-audio-swift` @ `5c4c12b`,
/// `Sources/MLXAudioSTT/Models/Qwen3ASR/Qwen3ForcedAligner.swift`). Give it a 16 kHz mono buffer and the transcript, and
/// it returns when each word starts and ends. Loads the **installed** directory (the A2 pattern, `fromModelDirectory`,
/// never `fromPretrained`, which would download behind the app's back).
///
/// **Not a port.** The library ships the runner in `MLXAudioSTT`, which both app targets link; this engine is the
/// installed-folder loader, the 16 kHz/limits contract, the language table and decision 4's clean-up.
///
/// **Install side effect (readme T5, trap 1).** On first load the library calls
/// `Qwen3ASRModel.generateTokenizerJSONIfMissing`, which writes `tokenizer.json` into the model folder, built from
/// `vocab.json` + `merges.txt` + `tokenizer_config.json` — the same side effect Qwen3-ASR has. `ForcedAlignerInstallTests`
/// shows it is harmless to `InstallManager`.
///
/// **Thread-bound streams (KICKOFF rule 17; journals 2026-557/558).** `Qwen3ForcedAligner.swift` @ `5c4c12b` has no
/// array-index-after-another-index gather on its path: `callAsFunction` splices embeddings with plain `Range` slices, the
/// timestamp read is `inputIdsFlat[i].item(…)` / `outputIdsFlat[i].item(…)` over a single subscript, and `generate`
/// returns `[ForcedAlignItem]` — Swift values only. Everything is built, run and read inside one synchronous `generate`
/// call, so no lazy array crosses an actor hop. `MLXErrorGuard.evaluate` evaluates the input buffer up front, so a failed
/// evaluation throws before the library touches it (rule 17b); nothing is read back through `MLXErrorGuard.read` because
/// nothing MLX-typed is handed back. `MLXAudioThread` is not used (``usesAudioThread`` is `false`), and the main-actor
/// probe in `ForcedAlignerRealLoadTests` is the evidence.
///
/// **One call at a time per model (`ForcedAlignerGate`).** The load, the forward pass and the buffer-cache release of
/// one call finish before the next call for the same model folder starts. `model.generate` is synchronous (no
/// suspension inside the forward pass); the load before it is async, and the gate's turn is held across it. The input
/// refusals come before the gate, so they never wait.
///
/// **Memory.** The library calls `Memory.clearCache()` after reading the result; the engine returns the buffer cache
/// again once the model local is released, as `Qwen3ASREngine` and `SortformerEngine` do.
///
/// **The booking (FA-R20).** Each call books the `.align` row's whole figure (files x 1.10 plus the per-call working set,
/// ``charge(directory:catalogFallback:)``) on the shared `MemoryBudget` before the load, inside the gate, and gives it
/// back when the call ends. `ForcedAlignerSDK` is `StageBooksOwnWeights`, so `EngineCache` books 0 for the stage.
///
/// **Library findings worth knowing (read from the pinned checkout; reported on issue #13).**
/// - The 300 s limit and the input refusals are app-side: the library has no length limit and no chunking, and its
///   timestamp head allows up to about 400 s (5,000 classes x 80 ms) whatever the clip's length.
/// - The aligner's `preprocessAudio` calls `MLXAudioCore.computeMelSpectrogram` **without** `melScale: .slaney` and
///   `hannPeriodic: true`, which the ASR path (`Qwen3ASR.swift:1056-1064`) passes and which is what the checkpoint's
///   WhisperFeatureExtractor applies. Upstream's fix (#247 / bf14ae0c) missed the aligner. The module cannot patch it
///   (decision 2: report, never patch the pinned package). It is why the parity test gives Chinese its own tolerance.
///   Owner ruling 2 (Boo, 2026-11-27): "Keep ±80 ms as the contract for English and Japanese. Chinese gets its own
///   pinned tolerance of ±160 ms, with a comment that names the mel-frontend difference."
/// - The splitter drops punctuation and whitespace and splits Japanese kana runs/Korean words with the space path
///   (Python's reference tokenises them); `ForcedAlignerEngine.libraryTranscript` normalises whitespace so a pasted
///   newline cannot merge two English words.
/// - The mel difference, the Japanese/Korean splitter and the newline split are reported upstream:
///   https://github.com/Neywa-Labs/mlx-audio-swift/issues/422
/// - `peakMemoryUsage` is the process-wide `Memory.peakMemory`, not a per-run figure; FA-7's footprint test resets it.
enum ForcedAlignerEngine {
    /// The model's input rate. The stage resamples; never pass the decoded rate through (trap 5).
    nonisolated static let sampleRate = 16_000
    /// The model card's limit: about five minutes of speech per call (owner decision 5).
    nonisolated static let maxSeconds: Double = 300
    /// The timestamp head's class width: 80 ms per class (`timestamp_segment_time` in `config.json`).
    nonisolated static let timestampStepSeconds: Double = 0.08
    /// The 11 supported languages, as the codes `Transcribe` already uses (owner decision 6).
    nonisolated static let languageCodes = ["zh", "en", "yue", "fr", "de", "it", "ja", "ko", "pt", "ru", "es"]
    nonisolated static let defaultLanguageCode = "en"

    /// Rule 17's one-line switch (see the type comment). `false`: a call runs on the concurrent pool, where the gate
    /// puts it.
    nonisolated static let usesAudioThread = false

    /// Code → the library's language name (`config.json`'s `support_languages`). The name selects the splitter: the
    /// library splits Chinese per character and every other language by spaces (see its `encodeTimestamp`).
    nonisolated static let languageNames: [String: String] = [
        "zh": "Chinese", "en": "English", "yue": "Cantonese", "fr": "French", "de": "German",
        "it": "Italian", "ja": "Japanese", "ko": "Korean", "pt": "Portuguese", "ru": "Russian", "es": "Spanish",
    ]

    /// A picker row: the code a `.cat` row writes and the name a person reads.
    nonisolated struct Language: Sendable, Equatable, Identifiable {
        let code: String
        let name: String
        var id: String { code }
    }

    nonisolated static var languages: [Language] {
        languageCodes.compactMap { code in
            languageNames[code].map { Language(code: code, name: $0) }
        }
    }

    /// The library's name for a code, or nil when the code isn't one of the 11 (refused by name, never guessed).
    nonisolated static func libraryLanguageName(for code: String) -> String? {
        languageNames[code.lowercased()]
    }

    /// A number of seconds as a person reads it: "5 minutes", "12 seconds", "49 seconds".
    nonisolated static func spoken(_ seconds: Double) -> String {
        let total = max(Int(seconds.rounded()), 0)
        let (m, s) = (total / 60, total % 60)
        func unit(_ n: Int, _ word: String) -> String { "\(n) \(word)\(n == 1 ? "" : "s")" }
        if m == 0 { return unit(s, "second") }
        return s == 0 ? unit(m, "minute") : "\(unit(m, "minute")) \(unit(s, "second"))"
    }

    /// Errors with one plain sentence, shown as written (a `LocalizedError`, so `MLXErrorGuard.sentence(for:)` keeps it).
    enum EngineError: Error, LocalizedError, Equatable {
        case emptyAudio
        case emptyTranscript
        case noWords
        case overLimit(seconds: Double)
        case unsupportedLanguage(code: String)
        /// No room on the memory ledger within the wait: who was in the way, if the ledger knew.
        case memoryBusy(MemoryBudget.Blocker?)

        var errorDescription: String? {
            switch self {
            case .emptyAudio:
                return "The audio file has no samples; choose a recording with sound in it."
            case .emptyTranscript:
                return "The transcript is empty. Type or open the text that was spoken — the aligner times the words it is given, it does not transcribe."
            case .noWords:
                return "The transcript has no words to time. Type the words that were spoken — the aligner times words, not punctuation."
            case .overLimit(let seconds):
                return "This recording is \(ForcedAlignerEngine.spoken(seconds)) long. Forced alignment handles up to "
                    + "\(ForcedAlignerEngine.spoken(ForcedAlignerEngine.maxSeconds)) at a time; trim the recording or split it into parts."
            case .unsupportedLanguage(let code):
                return "The language code '\(code)' isn't one this model supports. Use one of "
                    + ForcedAlignerEngine.languageCodes.joined(separator: ", ") + "."
            case .memoryBusy(let blocker):
                let who = blocker.map { blocker in
                    "\(blocker.model) is in use" + (blocker.holder.map { " by \($0.phrase)" } ?? "")
                } ?? "another model is still loading or in use"
                return "There isn't room in memory to align yet because \(who), so try again when it finishes."
            }
        }
    }

    /// Throws ``EngineError/overLimit(seconds:)`` for a recording over ``maxSeconds`` (checked before any model is loaded).
    nonisolated static func requireWithinLimit(seconds: Double) throws {
        if seconds > maxSeconds { throw EngineError.overLimit(seconds: seconds) }
    }

    /// The transcript as the library should receive it: no case, punctuation or number normalisation (trap 6), and only
    /// its whitespace **layout** normalised — surrounding whitespace trimmed and internal runs collapsed to one space.
    ///
    /// The collapse goes beyond trap 6's "apart from trimming surrounding whitespace". Owner ruling 3 (Boo, 2026-11-27):
    /// "Keep it only if a test shows the library's units are identical with and without it, for the English, Chinese and
    /// Japanese clips. Otherwise remove it and pass the transcript as given, trimmed only (Trap 6)." They are identical
    /// (`ForcedAlignerEngineTests.theWhitespaceCollapseLeavesTheClipUnitsUnchanged`), so it stays.
    ///
    /// What it is for: the pinned library splits on the ASCII space alone (`ForceAlignProcessor.tokenizeSpaceLang`),
    /// while the Python reference splits on any whitespace (`text.split()`, `qwen3_forced_aligner.py:141`, mlx-audio
    /// 0.4.4). Without the collapse a pasted `Hello\nworld` would reach the library as one word and merge two units.
    /// Collapsing keeps every word and character exactly as given. The newline split is reported upstream:
    /// https://github.com/Neywa-Labs/mlx-audio-swift/issues/422
    nonisolated static func libraryTranscript(_ transcript: String) -> String {
        transcript
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)
            .joined(separator: " ")
    }

    // MARK: - Decision 4's clean-up

    /// The library's items as ``AlignedUnit``s under decision 4: times rounded to the millisecond, clipped to
    /// `0...duration`, `end >= start`, starts never decreasing, transcript order kept.
    ///
    /// The pinned library already runs an LIS pass (`ForceAlignProcessor.fixTimestamp`) that makes its raw timestamps
    /// non-decreasing, so `end >= start` and ordered starts hold there too; what it does **not** do is clip to the
    /// recording (a confident class can be past the clip's end), so this pass is what guarantees the whole contract.
    nonisolated static func units(from items: [ForcedAlignItem], duration: Double) -> [AlignedUnit] {
        let upper = max(0, duration)
        var out: [AlignedUnit] = []
        out.reserveCapacity(items.count)
        var previousStart = 0.0
        for item in items {
            let start = min(max(0, milliseconds(item.startTime)), upper)
            let end = min(max(0, milliseconds(item.endTime)), upper)
            // Decision 4: starts never decrease, and end >= start (the clip above can invert a pair that started late).
            let orderedStart = max(start, previousStart)
            let orderedEnd = min(max(end, orderedStart), upper)
            out.append(AlignedUnit(text: item.text, start: orderedStart, end: orderedEnd))
            previousStart = orderedStart
        }
        return out
    }

    /// A time in seconds rounded to the millisecond.
    nonisolated static func milliseconds(_ seconds: Double) -> Double {
        guard seconds.isFinite else { return 0 }
        return (seconds * 1000).rounded() / 1000
    }

    // MARK: - The memory booking

    /// What one call books: the model's resident charge (files x 1.10, or `catalogFallback` — the catalog's `ramGB` in
    /// bytes — with no files to measure) plus the `.align` row's per-call working set.
    nonisolated static func charge(directory: URL, catalogFallback: Int64? = nil) -> Int64 {
        ModelLoadProfile.charge(kind: .align, directory: directory, catalogFallback: catalogFallback)
            + ModelLoadProfile.workingSetBytes(for: .align)
    }

    /// The budget's view of the aligner: nothing stays resident between calls, so there is never anything to evict.
    private nonisolated final class RunBooking: MemoryBudgetParticipant {
        func evictLeastRecentlyUsed() async -> Bool { false }
    }

    nonisolated private static let bookingOwner = RunBooking()

    /// Books `bytes` for one call, **held** (so another load waits for it rather than going over budget), under the
    /// holder the call runs under. The budget may evict other caches' idle engines to make room. Waits at most
    /// `maxWait`, then ``EngineError/memoryBusy(_:)``; a call cancelled while it waits is a `CancellationError`
    /// (ruling 20). The caller ends it with `budget.release(reservation)`.
    ///
    /// Owner ruling 7 (Boo, 2026-11-27): "Set it from your own FA-7 run: the highest measured peak rounded up to 0.1 GB.
    /// Keep the `.align` booking at or above that peak." The booking has to be on the ledger to keep that, so it is
    /// reserved here for the length of the call. This design (a transient reservation by the engine, `EngineCache`
    /// booking 0) is the implementer's call, pending owner confirmation.
    nonisolated private static func book(_ bytes: Int64, for modelDirectory: URL, in budget: MemoryBudget,
                                         maxWait: Duration) async throws
        -> (reservation: MemoryBudget.Reservation, hold: MemoryBudget.Hold) {
        let holder = MemoryBudget.RunWindowScope.holder ?? .flow
        let label = modelDirectory.path
        let reservation: MemoryBudget.Reservation
        do {
            reservation = try await budget.reserveBounded(bytes: bytes, for: bookingOwner, label: label,
                                                          maxWait: maxWait, pinned: .access(for: holder))
        } catch let error as MemoryBudget.WaitError {
            try Task.checkCancellation()
            throw EngineError.memoryBusy(error.blockers.first)
        }
        let hold = budget.hold(reservation, label: label, holder: holder)   // held before it is committed
        budget.commit(reservation)
        return (reservation, hold)
    }

    // MARK: - The run

    /// The load and the forward pass of one call: what runs inside the gate. Tests pass their own.
    typealias Step = @Sendable (_ samples: [Float], _ transcript: String, _ language: String,
                                _ duration: Double, _ modelDirectory: URL) async throws -> [AlignedUnit]

    /// The app's gate: every caller (flow rows and Run sheets) takes its turn here.
    nonisolated static let sharedGate = ForcedAlignerGate()

    /// The gate's key for a model folder.
    nonisolated static func gateKey(_ modelDirectory: URL) -> String { modelDirectory.standardizedFileURL.path }

    /// Align 16 kHz mono float samples against `transcript`. Loads the model per call (matches the other engines), one
    /// call at a time per model folder, with ``charge(directory:catalogFallback:)`` booked on `budget` for that call.
    /// Throws ``EngineError`` for the input refusals and for no room in memory, `CancellationError` for a cancelled call
    /// (a waiting one never runs, ruling 20 at `ForcedAlignerGate`) and `StageError.engineFailure` for a load/run
    /// failure.
    nonisolated static func align(_ samples: [Float], transcript: String,
                                  language languageCode: String = defaultLanguageCode,
                                  modelDirectory: URL,
                                  catalogFallback: Int64? = nil,
                                  budget: MemoryBudget = .shared,
                                  maxWait: Duration = MemoryBudget.defaultMaxWait,
                                  gate: ForcedAlignerGate = sharedGate,
                                  step: @escaping Step = loadAndGenerate) async throws -> [AlignedUnit] {
        let text = libraryTranscript(transcript)
        guard !samples.isEmpty else { throw EngineError.emptyAudio }
        guard !text.isEmpty else { throw EngineError.emptyTranscript }
        guard let language = libraryLanguageName(for: languageCode) else {
            throw EngineError.unsupportedLanguage(code: languageCode)
        }
        let seconds = Double(samples.count) / Double(sampleRate)
        try requireWithinLimit(seconds: seconds)
        do {
            return try await gate.run(gateKey(modelDirectory)) {
                // Inside the gate, so one model never has two bookings; before the load.
                let booking = try await book(charge(directory: modelDirectory, catalogFallback: catalogFallback),
                                             for: modelDirectory, in: budget, maxWait: maxWait)
                // Every exit: the model (a local of `generate`) is gone with `step`; return MLX's buffer cache, as
                // Qwen3ASREngine / SortformerEngine do, then give the booking back.
                defer {
                    Memory.clearCache()
                    budget.release(booking.reservation)   // first: one step, holds included
                    booking.hold.release()                // nothing left to do; keeps the hold alive until here
                }
                let units = try await step(samples, text, language, seconds, modelDirectory)
                // The splitter drops punctuation, so a transcript of punctuation alone makes no units. Refuse with a
                // plain sentence rather than show an empty table.
                guard !units.isEmpty else { throw EngineError.noWords }
                return units
            }
        } catch let error as CancellationError {
            throw error
        } catch let error as EngineError {
            throw error
        } catch let error as StageError {
            throw error
        } catch {
            throw StageError.engineFailure(stage: "Forced aligner", underlying: error)
        }
    }

    /// The real ``Step``.
    nonisolated static func loadAndGenerate(_ samples: [Float], transcript: String, language: String,
                                            duration: Double, modelDirectory: URL) async throws -> [AlignedUnit] {
        try await perform {
            try await generate(samples, text: transcript, language: language,
                               duration: duration, modelDirectory: modelDirectory)
        }
    }

    nonisolated private static func perform(_ body: @escaping @Sendable () async throws -> [AlignedUnit]) async throws -> [AlignedUnit] {
        usesAudioThread ? try await MLXAudioThread.run(body) : try await body()
    }

    nonisolated private static func generate(_ samples: [Float], text: String, language: String,
                                             duration: Double, modelDirectory: URL) async throws -> [AlignedUnit] {
        // Stop: a cancelled run loads nothing more.
        try Task.checkCancellation()
        let model = try await Qwen3ForcedAlignerModel.fromModelDirectory(modelDirectory)
        try Task.checkCancellation()
        let audio = MLXArray(samples)
        // Rule 17b: evaluate the buffer up front, so a failed evaluation throws before the library touches it.
        try MLXErrorGuard.evaluate(audio)
        // The library's own preprocessAudio computes the 16 kHz mel; see `ForcedAlignerEngine.sampleRate`.
        let result = model.generate(audio: audio, text: text, language: language)
        return units(from: result.items, duration: duration)
    }

    /// The installed-model directory for a catalog id, mirroring `InstallManager`'s layout
    /// (`Application Support/AI Browser/models/{id}`).
    nonisolated static func installedModelDirectory(id: String) -> URL {
        ModelStore.shared.directory(forModelID: id)
    }
}
