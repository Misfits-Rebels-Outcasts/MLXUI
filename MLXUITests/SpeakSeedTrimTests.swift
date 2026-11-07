import Testing
import Foundation
@testable import MLXUI

/// BS-6 (`RSI/DelegateBasicShowcaseBacklog.md`, owner rulings R6 + R7): a `Speak` row's `seed=` and `trim_silence=` reach the
/// stage config (they were dropped before), a value that cannot be read is refused by name, and the engine's trim is exact.
struct SpeakSeedTrimTests {

    private func config(_ settings: String) throws -> StageConfig {
        let exec = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory), flowID: "bs6",
                                blobDirectory: FileManager.default.temporaryDirectory,
                                makeModelStage: { _, _ in throw StageError.unsupportedSetting(setting: "none") },
                                installedModelIDs: [], catalog: [])
        let desc = try #require(TaskCatalog.get("Speak"))
        return try exec.stageConfig(for: desc, row: Row(task: "Speak", model: "Kokoro 82M", settings: settings), path: "2")
    }

    @Test func flowNineRowCarriesSeedSeven() throws {
        let c = try config("af_heart; seed=7")
        #expect(c.seed == 7)
        #expect(c.voice == "af_heart")
        #expect(c.flowTask == "Speak")
        #expect(c.trimSilence == nil)
    }

    @Test func noSeedMeansUnsetAndAutoMeansUnset() throws {
        #expect(try config("af_heart").seed == nil)
        #expect(try config("af_heart; seed=auto").seed == nil)
    }

    @Test func aSeedThatIsNotAWholeNumberIsRefusedByName() {
        for bad in ["seed=banana", "seed=-1", "seed=1.5", "seed=99999999999999999999"] {
            do {
                _ = try config("af_heart; \(bad)")
                Issue.record("\(bad) should be refused")
            } catch let FlowError.invalidSettings(row, setting, _) {
                #expect(row == "2" && setting == "seed")
            } catch {
                Issue.record("\(bad): wrong error \(error)")
            }
        }
    }

    @Test func trimSilenceIsCarriedAndAnythingElseIsRefusedByName() throws {
        #expect(try config("af_heart; trim_silence=true").trimSilence == true)
        #expect(try config("af_heart; trim_silence=false").trimSilence == false)
        #expect(try config("af_heart; trim_silence=TRUE").trimSilence == true)
        do {
            _ = try config("af_heart; trim_silence=maybe")
            Issue.record("trim_silence=maybe should be refused")
        } catch let FlowError.invalidSettings(_, setting, _) {
            #expect(setting == "trim_silence")
        } catch {
            Issue.record("wrong error \(error)")
        }
    }

    @Test func seedAndTrimDoNotChangeWhichWeightsLoad() throws {
        let a = try config("af_heart; seed=7; trim_silence=true")
        let b = try config("af_heart; seed=8; trim_silence=false")
        #expect(a != b)
        #expect(a.weightsOnly == b.weightsOnly)
    }

    // MARK: the engine's trim (pure)

    @Test func trimmedSilenceDropsLeadAndTailButKeepsAPad() {
        let rate = 24_000
        let pad = Int(0.05 * Double(rate))
        let samples = [Float](repeating: 0, count: 10_000) + [Float](repeating: 0.5, count: 5_000) + [Float](repeating: 0, count: 8_000)
        let out = KokoroEngine.trimmedSilence(samples, sampleRate: rate)
        #expect(out.count == 5_000 + 2 * pad)
        #expect(out[pad] == 0.5 && out[pad - 1] == 0)
    }

    @Test func trimmedSilenceNeverCutsIntoSpeechAndHandlesEdges() {
        #expect(KokoroEngine.trimmedSilence([0, 0, 0]) .isEmpty)
        #expect(KokoroEngine.trimmedSilence([]).isEmpty)
        let loud = [Float](repeating: 0.2, count: 100)
        #expect(KokoroEngine.trimmedSilence(loud) == loud)
        // sub-threshold hiss on both ends is silence; the pad is clamped to the array
        let hiss = [Float](repeating: 0.0005, count: 50) + [0.3] + [Float](repeating: 0.0005, count: 50)
        #expect(KokoroEngine.trimmedSilence(hiss) == hiss)
    }
}
