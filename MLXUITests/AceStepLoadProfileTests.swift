import Foundation
import MLX
import Testing
@testable import MLXUI

/// MC5-T-mem: where the process footprint sits above MLX's own counters while ACE-Step loads. OPT-IN, real weights, run by the lead (never in the normal suite: it needs
/// `MLXUI_TEST_ACESTEP_DIR`, which is unset there, so it skips). It runs the engine twice on a 10 s instrumental song, once with `syncAtStageBoundaries = false` (the MC5-T-run-1
/// behaviour) and once with `true` (the fix), and prints one `LOADPROFILE` line per engine event and one per mode:
///
///     LOADPROFILE|<mode>|<event>|<phys_footprint MiB>|<peak since the previous line, MiB>|MLX active MiB|MLX cache MiB|MLX peak MiB
///
/// Command (unsandboxed host, app closed, nothing else holding the smoke lock; ~1 minute):
///
///     TEST_RUNNER_MLXUI_TEST_ACESTEP_DIR=/Volumes/Ovid/Users/chinpinboo/Documents/PipelineStudio/RSI/notes/modelcandidates/mc5/oracle/Ace-Step1.5 \
///     xcodebuild test -project MLXUI.xcodeproj -scheme MLXUI -destination 'platform=macOS,arch=arm64' -parallel-testing-enabled NO \
///         CODE_SIGN_ENTITLEMENTS=MLXUI/MLXUI-Direct.entitlements ENABLE_APP_SANDBOX=NO -only-testing:MLXUITests/AceStepLoadProfileTests 2>&1 | grep LOADPROFILE
@Suite(.enabled(if: AceStepLoadProfile.directory != nil, "needs TEST_RUNNER_MLXUI_TEST_ACESTEP_DIR=<Ace-Step1.5 folder>"), .serialized)
@MainActor
struct AceStepLoadProfileTests {

    @Test func footprintAtEveryEngineEventWithAndWithoutTheStageBoundarySync() async throws {
        let dir = try #require(AceStepLoadProfile.directory)
        for sync in [false, true] {
            let mode = sync ? "sync-at-boundaries" : "clearCache-only"
            try await Task.sleep(for: .seconds(3))
            Stream.gpu.synchronize(); Memory.clearCache()
            try await Task.sleep(for: .seconds(2))
            let log = AceStepLoadProfile.Log(mode: mode)
            var engine = AceStepEngine(directory: dir)
            engine.configuration.syncAtStageBoundaries = sync
            engine.observer = { log.note("\($0)") }
            log.start()
            _ = try await engine.generate(AceStepRequest(caption: "warm lo-fi hip hop, mellow piano, vinyl crackle, 85 bpm", duration: 10, seed: 7))
            log.note("generate returned")
            log.finish()
            try await Task.sleep(for: .seconds(2))
            print("LOADPROFILE|\(mode)|settled 2 s after return|\(Int(ModelSmoke.physMiB()))|-|\(ModelSmoke.mlxMiB().active)|\(ModelSmoke.mlxMiB().cache)|\(ModelSmoke.mlxMiB().peak)")
        }
    }
}

nonisolated enum AceStepLoadProfile {
    static let directory: URL? = {
        guard let path = ProcessInfo.processInfo.environment["MLXUI_TEST_ACESTEP_DIR"], !path.isEmpty,
              FileManager.default.fileExists(atPath: path + "/acestep-v15-turbo/model.safetensors") else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }()

    /// Samples the footprint every 20 ms; each `note` prints the peak seen since the previous one.
    final class Log: @unchecked Sendable {
        private let lock = NSLock()
        private let mode: String
        private var peak = 0.0
        private var running = false
        private var task: Task<Void, Never>?
        private var base = 0.0
        init(mode: String) { self.mode = mode }
        func start() {
            lock.withLock { running = true; base = ModelSmoke.physMiB(); peak = base }
            MLX.Memory.peakMemory = 0
            task = Task.detached { [self] in
                while lock.withLock({ running }) {
                    let m = ModelSmoke.physMiB()
                    lock.withLock { peak = max(peak, m) }
                    try? await Task.sleep(for: .milliseconds(20))
                }
            }
            print("LOADPROFILE|\(mode)|start|\(Int(base))|\(Int(base))|\(ModelSmoke.mlxMiB().active)|\(ModelSmoke.mlxMiB().cache)|\(ModelSmoke.mlxMiB().peak)")
        }
        func note(_ event: String) {
            let now = ModelSmoke.physMiB()
            let p = lock.withLock { () -> Double in let v = max(peak, now); peak = now; return v }
            let m = ModelSmoke.mlxMiB()
            print("LOADPROFILE|\(mode)|\(event.replacingOccurrences(of: "|", with: "/"))|\(Int(now))|\(Int(p))|\(m.active)|\(m.cache)|\(m.peak)")
        }
        func finish() { lock.withLock { running = false } }
    }
}
