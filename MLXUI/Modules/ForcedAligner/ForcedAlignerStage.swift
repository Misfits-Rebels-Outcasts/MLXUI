import Foundation

/// `audio → text` stage backed by `ForcedAlignerEngine`. Resamples the input to 16 kHz mono, refuses a recording past
/// `ForcedAlignerEngine.maxSeconds` before loading anything, aligns the given transcript and returns the result as the
/// `text,start,end` CSV (`AlignmentExport.csv`). There is no table `Media` case, and adding one would touch every switch
/// on `Media`; the CSV is the stage's output and the Run screen / the `Align` flow row read it back with
/// `AlignmentExport.parseCSV` — the same shape `SortformerStage` uses for `speaker,start,end`.
///
/// The engine is injectable so the stage's logic is unit-testable without a model (the seam `ASRStage` uses). The stage
/// holds no transcript, so one cached stage serves every transcript: it arrives as run input through
/// `AlignTranscript.current` (ruling 17), and a run without it is refused.
nonisolated struct ForcedAlignerStage: PipelineStage {
    let id: String
    let name: String
    var accepts: MediaKind { .audio }
    var produces: MediaKind { .text }

    private let language: String
    /// 16 kHz mono float samples + transcript + language → aligned units.
    private let align: @Sendable ([Float], String, String) async throws -> [AlignedUnit]

    init(id: String, name: String, language: String,
         align: @escaping @Sendable ([Float], String, String) async throws -> [AlignedUnit]) {
        self.id = id
        self.name = name
        self.language = language
        self.align = align
    }

    /// The real engine on the installed folder of `modelID`, timing in `language` (a code from
    /// `ForcedAlignerEngine.languageCodes`).
    init(modelID: String, language: String = ForcedAlignerEngine.defaultLanguageCode) {
        let directory = ForcedAlignerEngine.installedModelDirectory(id: modelID)
        self.init(id: "forced-aligner.\(modelID)", name: "Qwen3 Aligner", language: language) { samples, text, language in
            try await ForcedAlignerEngine.align(samples, transcript: text, language: language, modelDirectory: directory)
        }
    }

    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
        try require(input, .audio)
        guard case let .audio(buffer) = input else {
            throw StageError.kindMismatch(expected: .audio, got: input.kind)
        }
        guard let transcript = AlignTranscript.current else { throw AlignTranscript.Missing() }
        progress(0.1)
        let mono16k = try AudioResampler.resample(buffer, to: ForcedAlignerEngine.sampleRate)
        let duration = Double(mono16k.samples.count) / Double(ForcedAlignerEngine.sampleRate)
        try ForcedAlignerEngine.requireWithinLimit(seconds: duration)
        let units = try await align(mono16k.samples, transcript, language)
        progress(1.0)
        return .text(AlignmentExport.csv(units))
    }
}
