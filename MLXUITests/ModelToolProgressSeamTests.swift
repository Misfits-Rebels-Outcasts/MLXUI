import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// MT1-3b: the progress seam. Every helper is prefixed `MT13b`.
nonisolated final class MT13bBox<T: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var _items: [T] = []
    func add(_ item: T) { lock.withLock { _items.append(item) } }
    var items: [T] { lock.withLock { _items } }
}

/// A FlowExecutor that emits from inside `execute` (as a stage would) and records whether a sink was bound.
nonisolated final class MT13bEmitter: FlowExecutor, @unchecked Sendable {
    let emits: [(String, Double?)]
    let sinkWasBound = MT13bBox<Bool>()
    let sixArgCalls = MT13bBox<String>()
    init(emits: [(String, Double?)]) { self.emits = emits }

    func execute(path: String, row: Row, inputs: [Asset], transcript: [FlowInterpreter.TranscriptEntry]?,
                 context: [(label: String, content: String)]?, usedFlowContent: String?) async throws -> Asset {
        sixArgCalls.add(path)
        sinkWasBound.add(RowProgress.sink != nil)
        for (message, fraction) in emits { RowProgress.report(message, fraction: fraction) }
        return Asset(items: [Item(kind: .text, value: "out")])
    }
    var lastCacheHit: Bool { false }
    var lastTag: String? { nil }
    var lastTimeoutFlag: (code: String, message: String)? { nil }
    var lastProviderDeciderFlag: (code: String, message: String)? { nil }
    var lastStaged: (id: String, kind: String, summary: String)? { nil }
    var lastDeciderDetail: (tag: String, confidence: Double, probabilities: [(label: String, probability: Double)],
                            expectedLevel: Double?, stateTruncated: Bool)? { nil }
}

@Suite struct ModelToolProgressSeamTests {
    private let row = Row(task: "Transcribe")

    @Test func theProgressEntryDeliversEmittedProgressInOrder() async throws {
        let exec = MT13bEmitter(emits: [("Loading…", nil), ("Transcribing…", 0.25), ("Transcribing…", 0.75)])
        let got = MT13bBox<RowProgress>()
        let out = try await exec.execute(path: "1", row: row, inputs: [], transcript: nil, context: nil,
                                         usedFlowContent: nil, progress: { got.add($0) })
        #expect(out.items.first?.value == "out")
        #expect(got.items == [RowProgress(message: "Loading…"),
                              RowProgress(message: "Transcribing…", fraction: 0.25),
                              RowProgress(message: "Transcribing…", fraction: 0.75)])
        #expect(exec.sinkWasBound.items == [true])
    }

    @Test func noClosureIsANoOpAndTheAssetIsIdentical() async throws {
        let exec = MT13bEmitter(emits: [("Transcribing…", 0.5)])
        let plain = try await exec.execute(path: "1", row: row, inputs: [], transcript: nil, context: nil, usedFlowContent: nil)
        let viaNew = try await exec.execute(path: "1", row: row, inputs: [], transcript: nil, context: nil,
                                            usedFlowContent: nil, progress: nil)
        #expect(plain.items.map(\.value) == viaNew.items.map(\.value))
        #expect(plain.items.map(\.kind) == viaNew.items.map(\.kind))
        #expect(exec.sinkWasBound.items == [false, false])
    }

    @Test func theSixArgumentEntryBindsNothingAndReportingIsANoOp() async throws {
        let exec = MT13bEmitter(emits: [("x", 1)])
        let viaProtocol: any FlowExecutor = exec
        _ = try await viaProtocol.execute(path: "7", row: row, inputs: [])
        #expect(exec.sinkWasBound.items == [false])
        #expect(RowProgress.sink == nil)
        RowProgress.report("nobody listening")   // must not crash
    }

    @Test func theBindingEndsWithTheProgressEntry() async throws {
        let exec = MT13bEmitter(emits: [])
        _ = try await exec.execute(path: "1", row: row, inputs: [], transcript: nil, context: nil,
                                   usedFlowContent: nil, progress: { _ in })
        #expect(RowProgress.sink == nil)
    }

    @Test func fractionIsClamped() {
        #expect(RowProgress(message: "a", fraction: 2).fraction == 1)
        #expect(RowProgress(message: "a", fraction: -1).fraction == 0)
    }
}

@MainActor @Suite struct ModelToolProgressRunnerTests {
    private func runtime(emitting emits: [(String, Double?)], exec: MT13bEmitter) throws -> ModelToolRuntime {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let cat = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        let parakeet = try #require(cat.first { $0.hfModelId == "mlx-community/parakeet-tdt-0.6b-v3" })
        return ModelToolRuntime(catalog: cat, installedModelIDs: [parakeet.id], claimableModelIDs: claimable, totalRAMGB: 19,
                                makeExecutor: { _, _ in exec }, pathRoots: [FileManager.default.temporaryDirectory])
    }

    private func call(host: ModelToolHost?, exec: MT13bEmitter, audio: URL) async throws -> String {
        let rt = try runtime(emitting: [], exec: exec)
        let runner = ModelToolRunner(spec: try #require(ModelToolSpec.spec(named: "transcribe_audio")), runtime: rt)
        return await ModelToolHost.$current.withValue(host) {
            await runner.run(arguments: ["path": .string(audio.path)])
        }
    }

    @Test func withAHostProgressReachesHostProgress() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mt13b-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = dir.appendingPathComponent("a.wav"); try Data([0, 1]).write(to: audio)
        let seen = MT13bBox<ModelToolProgress>()
        let host = ModelToolHost(assets: ChatAssetStore(conversationID: UUID(), in: dir),
                                 progress: { seen.add($0) })
        let exec = MT13bEmitter(emits: [("Transcribing…", 0.5), ("Transcribing…", 1)])
        let out = try await call(host: host, exec: exec, audio: audio)
        #expect(out.contains("\"asset\""))
        let details = seen.items.compactMap(\.detail)
        #expect(details == ["Transcribing…", "Transcribing…"])
        #expect(seen.items.compactMap(\.fraction) == [0.5, 1])
        #expect(exec.sinkWasBound.items == [true])
    }

    @Test func withoutAHostNothingIsBoundAndTheResultIsPlainText() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mt13b-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = dir.appendingPathComponent("a.wav"); try Data([0, 1]).write(to: audio)
        let exec = MT13bEmitter(emits: [("Transcribing…", 0.5)])
        let out = try await call(host: nil, exec: exec, audio: audio)
        #expect(out == "out")
        #expect(exec.sinkWasBound.items == [false])
    }
}
