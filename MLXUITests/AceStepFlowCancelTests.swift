import Foundation
import MLX
import Testing
@testable import MLXUI

/// MC5-T-fix (c): cancelling a one-row ACE-Step flow mid-run, through the executor path, on the tiny checkpoint, tears the engine down and returns MLX's memory to the baseline.
@Suite(.serialized)
@MainActor
struct AceStepFlowCancelTests {

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var n = 0
        func bump() { lock.withLock { n += 1 } }
        var value: Int { lock.withLock { n } }
    }

    private final class CancelBox: @unchecked Sendable {
        private let lock = NSLock()
        private var cancel: (@Sendable () -> Void)?
        private var fired = false
        func set(_ f: @escaping @Sendable () -> Void) { lock.withLock { cancel = f; if fired { f() } } }
        func fire() { let f = lock.withLock { () -> (@Sendable () -> Void)? in fired = true; return cancel }; f?() }
    }

    @Test func cancellingAOneRowAceStepFlowMidRunTearsTheEngineDownAndReleasesMLXMemory() async throws {
        let dir = try AceStepEngineTests.makeFolder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let repoRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")))
            .domains.flatMap { $0.allModels }
        let teardowns = Counter()
        let box = CancelBox()
        let completed = Counter()
        // The flow stage's generator is the real engine on the tiny checkpoint (the same engine code a real row runs); it cancels the flow at denoise step 1.
        let make: @Sendable (ModelEntry, StageConfig) async throws -> any PipelineStage = { model, config in
            AceStepStage(id: "acestep.tiny", name: "tiny", settings: AceStepRunSettings(seed: config.seed, steps: config.steps)) { request, onPhase in
                var engine = AceStepEngineTests.engine(dir, observer: { event in
                    if case .phase(let p) = event {
                        onPhase(p)
                        if case .denoise(let step, _) = p, step == 1 { box.fire() }
                    }
                })
                engine.teardown = { Stream.gpu.synchronize(); Memory.clearCache(); teardowns.bump() }
                let audio = try await engine.generate(request).audio
                completed.bump()
                return audio
            }
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("acestep-flowcancel-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executor = RealExecutor(workspace: FlowWorkspace(root: root), flowID: "mc5tfixc", blobDirectory: root, makeModelStage: make,
                                    installedModelIDs: ["ACE-Step--Ace-Step1.5"], catalog: catalog)
        let doc = try CatParser.parse("mlxflow 0.8\n1. Generate Sound   ACE-Step 1.5; steps=4; seed=7 \"calm piano\"\n\nmodels:\n  ACE-Step 1.5 = ACE-Step/Ace-Step1.5\n")

        // One whole tiny run first, so lazily-created process-wide MLX state (kernels, RNG) is not counted against the cancelled row.
        _ = try await AceStepEngineTests.engine(dir).generate(AceStepRequest(caption: "warm up", duration: 1, seed: 1))
        Stream.gpu.synchronize(); Memory.clearCache()
        let before = Memory.snapshot().activeMemory
        let task = Task { _ = try await FlowInterpreter.run(doc, executor: executor, definitions: doc.definitions, presets: doc.presets) }
        box.set { task.cancel() }
        _ = try? await task.value

        #expect(completed.value == 0, "the row was cancelled, not finished")
        #expect(teardowns.value == 1, "the engine tore down exactly once (and had started: it reached step 1)")
        var active = Memory.snapshot().activeMemory
        for _ in 0 ..< 100 where active != before {
            try await Task.sleep(for: .milliseconds(100))
            Memory.clearCache()
            active = Memory.snapshot().activeMemory
        }
        #expect(active == before, "MLX active memory \(active) after the cancelled flow row, \(before) before")
    }
}
