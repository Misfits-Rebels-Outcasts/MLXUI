import Foundation

/// WW2-3-1 (owner-approved 2026-10-22, journal 2026-552 §3): this branch's own Swift outputs, recorded under
/// `Fixtures/{CLM,Clef}/swift-snapshot-mlx-swift-0.32.3/`. The Python goldens carry the mlx-core margins
/// (CLM/Clef tests); the snapshots are tight, so the next Swift-side change shows up even inside those margins.
///
/// Re-record deliberately, never to make a failing comparison pass: run the snapshot tests (output is printed, not written — see `write`) with
/// `TEST_RUNNER_WW2_RECORD_SWIFT_SNAPSHOT=1` (xcodebuild forwards it), review the diff, and say why in the journal.
enum SwiftSnapshot {
    static let directoryName = "swift-snapshot-mlx-swift-0.32.3"
    static let cosineTolerance = 1e-5
    static let probabilityTolerance = 1e-4

    static var isRecording: Bool {
        ProcessInfo.processInfo.environment["WW2_RECORD_SWIFT_SNAPSHOT"] == "1"
    }

    /// `family` is "CLM" or "Clef".
    static func url(_ family: String, _ file: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/\(family)/\(directoryName)/\(file)")
    }

    /// The sandboxed test host cannot write into the checkout, so a recording is **printed** between markers;
    /// `scripts`-free: extract it from the xcodebuild log (journal 2026-552 shows the one-liner) into `url`.
    static func write(_ object: Any, to url: URL) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        print("SWIFT-SNAPSHOT-BEGIN \(url.path)\n\(String(decoding: data, as: UTF8.self))\nSWIFT-SNAPSHOT-END")
    }

    static func read(_ url: URL) throws -> Any {
        try JSONSerialization.jsonObject(with: Data(contentsOf: url))
    }

    static func cosine(_ a: [Double], _ b: [Double]) -> Double {
        let dot = zip(a, b).reduce(0.0) { $0 + $1.0 * $1.1 }
        let normA = a.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        let normB = b.reduce(0.0) { $0 + $1 * $1 }.squareRoot()
        return dot / (normA * normB)
    }
}
