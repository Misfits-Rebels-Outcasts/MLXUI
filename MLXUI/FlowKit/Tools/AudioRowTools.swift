import Foundation

/// ER-10 — `Identify Speakers` (SPEC-Q238) and `Clean Audio` (SPEC-Q239), owner rulings R1 and R11.
/// The engines already exist (`SortformerStage`, `DeepFilterNetStage`, claimed through `ModelSDK`); this file is only the
/// flow-row seams: the no-settings parse shared with the validator's check 25 (E723 / E724), and the table shaping.
nonisolated struct AudioRowSettingsError: Error, Equatable {
    /// Completes "Row N's `<task>` ..." (E723 / E724).
    var detail: String
}

nonisolated enum AudioRowSettings {
    /// Python's `parse_no_settings`: both rows take none in v1.
    static func parseNone(_ raw: String?) throws {
        if let given = firstGiven(raw) {
            throw AudioRowSettingsError(detail: "doesn't take settings -- the model is named on the row and v1 has nothing else to set; remove \(given)")
        }
    }

    /// The first item the row was given, as the error names it: a bare value as `` `bare` ``, a key as `` `key=` ``.
    static func firstGiven(_ raw: String?) -> String? {
        let s = FlowSettings(raw)
        if let bare = s.firstBare() { return "`\(bare)`" }
        return s.orderedKeys.first.map { "`\($0)=`" }
    }
}

nonisolated enum IdentifySpeakersTable {  // SPEC-Q238
    static let columns = ["speaker", "start", "end"]

    /// The Sortformer stage's CSV (`speaker,start,end`, speaker 1-based, seconds) as the row's table rows. `speaker` is the
    /// stage's own label ("Speaker 1", numbered by first appearance: what the Run UI shows and what Python's mock uses);
    /// `start` and `end` are numbers in seconds; sorted by start, then end, then speaker.
    static func rows(fromCSV csv: String) throws -> [[Any?]] {
        try SpeakerSegments.parseCSV(csv)
            .sorted { ($0.start, $0.end, $0.speaker) < ($1.start, $1.end, $1.speaker) }
            .map { [$0.label, $0.start, $0.end] as [Any?] }
    }

    /// Python's mock table (`tools/audio_rows.py::MOCK_TURNS`): two speakers, four turns.
    static let mockRows: [[Any?]] = [
        ["Speaker 1", 0.0, 4.2], ["Speaker 2", 4.2, 9.8], ["Speaker 1", 9.8, 13.5], ["Speaker 2", 13.5, 18.0],
    ]
}
