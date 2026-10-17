import Darwin
import Foundation
import MLX
import Testing

@testable import MLXUI

/// CF-5b's measurement row (the contract's "measure on the owner's Mac"): load time and
/// `Memory.peakMemory`, the number that matters on unified memory — RSS undercounts it (CL-4b's
/// lesson). Prints `CLEF-MEASURE …` for the journal; the assertions are only sanity, since the
/// numbers are machine-dependent and belong in prose, not in a threshold.
///
/// Needs the installed 6.2 GB checkpoint, so it skips on a fresh clone.
private let clefMeasurementModelDirectory: URL? = {
    let identifier = "mlx-community--clef-flash-4bit"
    var candidates = [ModelStore.shared.directory(forHFModelID: "mlx-community/clef-flash-4bit")]
    if let entry = getpwuid(getuid()), let path = entry.pointee.pw_dir {
        candidates.append(
            URL(fileURLWithPath: String(cString: path), isDirectory: true)
                .appendingPathComponent("Library/Application Support/AI Browser/models/\(identifier)"))
    }
    let needed = ["config.json", "tokenizer.json", "joint_head.safetensors"]
    for candidate in candidates
    where needed.allSatisfy({ FileManager.default.fileExists(atPath: candidate.appendingPathComponent($0).path) }) {
        return candidate
    }
    return nil
}()

@Suite(.serialized, .enabled(if: clefMeasurementModelDirectory != nil))
struct ClefMeasurementTests {
    private static func state(words: Int) -> ClefJSON {
        .string(String(repeating: "The nightly export failed and the dashboard is empty. ", count: max(1, words / 9)))
    }

    private static let question: (id: String, type: ClefQuestionType, instructions: ClefJSON?, criteria: ClefJSON?) =
        (id: "q", type: .noul, instructions: .string("Does this need attention today?"), criteria: nil)

    @Test func loadTimeAndPeakMemory() async throws {
        let directory = try #require(clefMeasurementModelDirectory)
        Memory.peakMemory = 0  // the setter resets the counter

        let started = Date()
        let engine = try await ClefEngine.load(modelDirectory: directory)
        let loadSeconds = Date().timeIntervalSince(started)
        let peakAfterLoad = Memory.peakMemory

        let oneK = try await engine.answer(state: Self.state(words: 1_000), questions: [Self.question])
        let peakAfterOneK = Memory.peakMemory

        let eightK = try await engine.answer(state: Self.state(words: 8_000), questions: [Self.question])
        let peakAfterEightK = Memory.peakMemory

        let megabytes = { (bytes: Int) in Double(bytes) / 1_048_576 }
        print(
            "CLEF-MEASURE loadSeconds=\(String(format: "%.1f", loadSeconds)) "
                + "peakAfterLoad=\(String(format: "%.0f", megabytes(peakAfterLoad)))MB "
                + "peakAfter1k=\(String(format: "%.0f", megabytes(peakAfterOneK)))MB "
                + "peakAfter8k=\(String(format: "%.0f", megabytes(peakAfterEightK)))MB "
                + "tokensSpent=\(await engine.tokensSpent)")

        #expect(loadSeconds > 0)
        #expect(peakAfterLoad > 1_000_000_000, "a 6.2 GB checkpoint must show up in peak memory")
        #expect(peakAfterEightK >= peakAfterLoad)
        #expect(oneK.count == 1 && eightK.count == 1)
    }
}
