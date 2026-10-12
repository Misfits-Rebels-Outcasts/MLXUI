import Testing
import Foundation
@testable import MLXUI

/// SC-2c — every non-LLM load books its real size (`ModelLoadProfile`), not the catalog's `ramGB`; and the flow
/// editor shows a failed row's sentence.
struct RealSizesTests {

    private static let gib = Int64(1_073_741_824)

    private nonisolated struct Stub: PipelineStage {
        let id: String
        var name: String { id }
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
    }

    /// A model folder whose safetensors are `sizes` bytes each (sparse: GiB-sized files cost no disk).
    private func modelDir(_ sizes: [(String, Int64)]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("sc2c-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (path, size) in sizes {
            let url = dir.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: UInt64(size))
            try handle.close()
        }
        return dir
    }

    private func budget(_ gibs: Int64) -> MemoryBudget {
        MemoryBudget(capacityBytes: gibs * Self.gib,
                     cacheClearer: MLXCacheClearer(capActive: { false }, clear: {}, delay: .milliseconds(1)))
    }

    private func scaled(_ bytes: Int64, _ factor: Double) -> Int64 { Int64((Double(bytes) * factor).rounded()) }

    // MARK: - the table

    @Test func theTableIsTheMeasuredFactorsRoundedUp() {
        #expect(ModelLoadProfile.entry(for: .ocr) == .init(factor: 1.75, workingSetBytes: 0, measured: true))
        #expect(ModelLoadProfile.entry(for: .asr) == .init(factor: 2.05, workingSetBytes: 0, measured: true))
        #expect(ModelLoadProfile.entry(for: .decision) == .init(factor: 1.20, workingSetBytes: 0, measured: true))
        #expect(ModelLoadProfile.entry(for: .vision) == .init(factor: 1.10, workingSetBytes: 2 * Self.gib, measured: true))
        #expect(ModelLoadProfile.entry(for: .llm).factor == ModelWeightCharge.loadFactor)
        for unmeasured in [RunnerKind.tts, .image, .video, .upscale, .music, .segmentation, .embedding, .rerank] {
            #expect(!ModelLoadProfile.entry(for: unmeasured).measured, "\(unmeasured)")
        }
        // measured against: dots.ocr 5.64 GiB added / 3.28 GiB files = 1.72; Voxtral 5.94 / 2.92 = 2.03; Laya 0.93 / 0.78 = 1.18
        #expect(1.75 >= 5.64 / 3.28 && 2.05 >= 5.94 / 2.92 && 1.20 >= 0.93 / 0.78)
    }

    @Test func aChargeFollowsTheTableAndCountsSubfolders() throws {
        let dir = try modelDir([("model.safetensors", 3 * Self.gib), ("encoder/w.safetensors", Self.gib)])
        defer { try? FileManager.default.removeItem(at: dir) }
        let files = 4 * Self.gib
        #expect(ModelLoadProfile.charge(kind: .ocr, directory: dir, catalogFallback: Self.gib) == scaled(files, 1.75))
        #expect(ModelLoadProfile.charge(kind: .asr, directory: dir, catalogFallback: Self.gib) == scaled(files, 2.05))
        #expect(ModelLoadProfile.charge(kind: .decision, directory: dir, catalogFallback: nil) == scaled(files, 1.20))
        #expect(ModelLoadProfile.charge(kind: .vision, directory: dir, catalogFallback: nil) == scaled(files, 1.10))
        // the catalog figure no longer decides a measured kind, in either direction
        #expect(ModelLoadProfile.charge(kind: .ocr, directory: dir, catalogFallback: 20 * Self.gib) == scaled(files, 1.75))
        #expect(ModelLoadProfile.charge(kind: .ocr, directory: dir, catalogFallback: 0) == scaled(files, 1.75))
    }

    @Test func missingFilesFallBackToTheCatalogFigure() throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("sc2c-missing-\(UUID().uuidString)")
        #expect(ModelLoadProfile.charge(kind: .ocr, directory: missing, catalogFallback: 1_300_000_000) == 1_300_000_000)
        #expect(ModelLoadProfile.charge(kind: .tts, directory: missing, catalogFallback: 700) == 700)
        #expect(ModelLoadProfile.charge(kind: .asr, directory: missing, catalogFallback: nil) == 0)
        // a folder with no safetensors (an unreadable layout) falls back the same way
        let bare = try modelDir([])
        defer { try? FileManager.default.removeItem(at: bare) }
        try Data(count: 10).write(to: bare.appendingPathComponent("weights.bin"))
        #expect(ModelLoadProfile.charge(kind: .ocr, directory: bare, catalogFallback: 4_000) == 4_000)
    }

    @Test func aKindNobodyHasMeasuredNeverBooksBelowItsFilesOrItsOldFigure() throws {
        let dir = try modelDir([("model.safetensors", 2 * Self.gib)])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelLoadProfile.charge(kind: .tts, directory: dir, catalogFallback: 10 * Self.gib) == 10 * Self.gib)   // catalog floor
        #expect(ModelLoadProfile.charge(kind: .image, directory: dir, catalogFallback: Self.gib) == scaled(2 * Self.gib, 1.10))   // files floor
    }

    @Test func theLLMRuleIsUnchanged() throws {
        let dir = try modelDir([("model.safetensors", 5 * Self.gib)])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelLoadProfile.charge(kind: .llm, directory: dir, catalogFallback: 1) == ModelWeightCharge.bytes(directory: dir, catalogFallback: 1))
    }

    // MARK: - EngineCache books the real charge

    private func entry(_ id: String, kind: ModelType, ramGB: Double) -> ModelEntry {
        makeEntry(id: id, modelType: kind, ramGB: ramGB)
    }

    @Test func theCacheBooksTheRealChargeNotTheCatalogRamGB() async throws {
        let dir = try modelDir([("model.safetensors", 3 * Self.gib)])
        defer { try? FileManager.default.removeItem(at: dir) }
        let b = budget(20)
        let cache = EngineCache(budget: b, countsLLMWeights: false, modelDirectory: { _ in dir })
        // dots.ocr's situation: catalog says 1.3 GiB, the files say otherwise
        _ = try await cache.stage(for: entry("dots", kind: .ocr, ramGB: 1.3), config: .default) { m, _ in Stub(id: m.id) }
        #expect(b.usedBytes == scaled(3 * Self.gib, 1.75))
        #expect(b.usedBytes > Int64(1.3 * Double(Self.gib)) * 4)
    }

    @Test func theCacheFallsBackToRamGBWhenTheFilesAreMissing() async throws {
        let missing = FileManager.default.temporaryDirectory.appendingPathComponent("sc2c-none-\(UUID().uuidString)")
        let b = budget(20)
        let cache = EngineCache(budget: b, countsLLMWeights: false, modelDirectory: { _ in missing })
        _ = try await cache.stage(for: entry("x", kind: .asr, ramGB: 0.9), config: .default) { m, _ in Stub(id: m.id) }
        #expect(b.usedBytes == Int64(0.9 * Double(Self.gib)))
    }

    @Test func noDoubleBookingForStagesWhoseEngineBooksItsOwnWeights() async throws {
        let dir = try modelDir([("model.safetensors", 3 * Self.gib)])
        defer { try? FileManager.default.removeItem(at: dir) }
        let b = budget(20)
        let cache = EngineCache(budget: b, countsLLMWeights: false, modelDirectory: { _ in dir })
        let model = entry("vlm", kind: .vision, ramGB: 3.4)
        let stage = try await cache.leasedStage(for: model, config: .default, booksWeights: false) { m, _ in Stub(id: m.id) }
        #expect(b.usedBytes == 0)                                    // the pool books these weights, not the cache
        _ = stage
        // and an LLM stage still books 0 (the pool owns it)
        _ = try await cache.stage(for: entry("llm", kind: .llm, ramGB: 6), config: .default) { m, _ in Stub(id: m.id) }
        #expect(b.usedBytes == 0)
        #expect(VLMSDK() is StageBooksOwnWeights && OCRSDK() is StageBooksOwnWeights
                && LayaSDK() is StageBooksOwnWeights && CLMSDK() is StageBooksOwnWeights)
        #expect(!(DotsOCRSDK() is StageBooksOwnWeights) && !(VoxtralSDK() is StageBooksOwnWeights))
    }

    // MARK: - the OCR → VLM sequence on an 8 GiB cap, as booked

    /// Sparse stand-ins with the measured sizes: dots.ocr 3.28 GiB of safetensors, Gemma-3-4b 3.17 GiB.
    @Test func anOcrThenAVlmOnEightGiBEvictsTheIdleOcrInsteadOfOvershooting() async throws {
        let ocrDir = try modelDir([("model.safetensors", Int64(3.28 * Double(Self.gib)))])
        let vlmDir = try modelDir([("model.safetensors", Int64(3.17 * Double(Self.gib)))])
        defer { try? FileManager.default.removeItem(at: ocrDir); try? FileManager.default.removeItem(at: vlmDir) }
        let b = budget(8)
        let cache = EngineCache(budget: b, countsLLMWeights: false, modelDirectory: { _ in ocrDir })
        let window = LeasingStage(model: entry("dots", kind: .ocr, ramGB: 1.3), config: .default,
                                  probe: Stub(id: "dots"), cache: cache)
        _ = try await window.run(.text("x")) { _ in }                 // the OCR run ends: warm, idle
        #expect(b.usedBytes == scaled(Int64(3.28 * Double(Self.gib)), 1.75))
        #expect(Double(b.usedBytes) / Double(Self.gib) > 5.5)         // booked ≈ the 5.7 GB measured, not 1.3

        // the VLM's weights share the budget through the pool participant; an idle OCR entry makes room
        let pool = ResidentPool<Box>(budget: b)
        let weights = ModelWeightCharge.bytes(directory: vlmDir, catalogFallback: nil)
        let vlm = try await pool.lease(for: vlmDir.path, bytes: weights, maxWait: .seconds(2)) { Box() }
        #expect(cache.count == 0)                                      // evicted, not stacked on top
        #expect(b.usedBytes <= 8 * Self.gib)
        // …and the call's working set (2 GiB, reserved beside the resident weights) fits beside them
        #expect(b.usedBytes + ModelLoadProfile.workingSetBytes(for: .vision) <= 8 * Self.gib)
        vlm.release()
    }

    @Test func aRunningOcrMakesTheVlmWaitAndNameIt() async throws {
        let ocrDir = try modelDir([("model.safetensors", Int64(3.28 * Double(Self.gib)))])
        let vlmDir = try modelDir([("model.safetensors", Int64(3.17 * Double(Self.gib)))])
        defer { try? FileManager.default.removeItem(at: ocrDir); try? FileManager.default.removeItem(at: vlmDir) }
        let b = budget(8)
        let cache = EngineCache(budget: b, countsLLMWeights: false, modelDirectory: { _ in ocrDir })
        let held = try await cache.lease(for: entry("dots", kind: .ocr, ramGB: 1.3), config: .default,
                                         holder: .runWindow) { m, _ in Stub(id: m.id) }
        let pool = ResidentPool<Box>(budget: b)
        let weights = ModelWeightCharge.bytes(directory: vlmDir, catalogFallback: nil)
        do {
            _ = try await pool.lease(for: vlmDir.path, bytes: weights, maxWait: .milliseconds(150)) { Box() }
            Issue.record("expected the VLM to wait for the running OCR")
        } catch let error as MemoryBudget.WaitError {
            #expect(error.description == "Waiting for memory: dots is in use by a Run window. Try again when it finishes.")
        }
        #expect(b.usedBytes <= 8 * Self.gib)
        held.hold.release()
    }

    private final class Box: Sendable {}
    private struct Boom: Error {}

    // MARK: - the VLM call's working set

    @Test func aVLMCallReservesItsWorkingSetBesideTheWeightsAndReleasesIt() async throws {
        let vlmDir = try modelDir([("model.safetensors", 3 * Self.gib)])
        defer { try? FileManager.default.removeItem(at: vlmDir) }
        let b = budget(20)
        let pool = ModelContainerPool(budget: b, loader: { _, _ in throw Boom() })
        let weights = ModelWeightCharge.bytes(directory: vlmDir, catalogFallback: nil)
        let working = try await VLMEngine.reserveCallWorkingSet(pool: pool, modelDir: vlmDir, maxWait: .seconds(1))
        // nothing is resident here, so the whole need is reserved: weights + the 2 GiB working set
        #expect(b.usedBytes == weights + 2 * Self.gib)
        b.release(working)
        #expect(b.usedBytes == 0)
    }

    @Test func aVLMCallThatCannotFitEvenAloneIsRefusedNotRunOverBudget() async throws {
        let vlmDir = try modelDir([("model.safetensors", 3 * Self.gib)])
        defer { try? FileManager.default.removeItem(at: vlmDir) }
        let pool = ModelContainerPool(budget: budget(4), loader: { _, _ in throw Boom() })     // 3.3 + 2 GiB > 4 GiB
        await #expect(throws: StageError.self) {
            _ = try await VLMEngine.reserveCallWorkingSet(pool: pool, modelDir: vlmDir, maxWait: .milliseconds(200))
        }
    }

    // MARK: - an eviction frees MLX's buffers before the next load starts

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock(); private var n = 0
        func bump() { lock.lock(); n += 1; lock.unlock() }
        var value: Int { lock.lock(); defer { lock.unlock() }; return n }
    }

    @Test func aBudgetEvictionClearsTheBufferCacheBeforeTheLoadProceeds() async throws {
        let cleared = Counter()
        let clearer = MLXCacheClearer(capActive: { false }, clear: { cleared.bump() }, delay: .seconds(30))   // deferred path never fires
        let b = MemoryBudget(capacityBytes: 10 * Self.gib, cacheClearer: clearer)
        let pool = ResidentPool<Box>(budget: b)
        _ = try await pool.value(for: "idle", bytes: 6 * Self.gib) { Box() }
        #expect(cleared.value == 0)
        // 6 GiB more needs the idle model's room: evicted, and the buffers are cleared by the time `lease` returns
        let lease = try await pool.lease(for: "next", bytes: 6 * Self.gib, maxWait: .seconds(2)) { Box() }
        #expect(cleared.value >= 1)
        #expect(clearer.clearCount >= 1)
        lease.release()
    }

    @Test func aServerGenerationsScopedLimitDefersTheImmediateClear() async throws {
        let cleared = Counter()
        let capActive = Counter()   // 0 = no cap
        let clearer = MLXCacheClearer(capActive: { capActive.value == 0 ? false : true }, clear: { cleared.bump() },
                                      delay: .milliseconds(20), retryEvery: .milliseconds(20), giveUpAfter: .seconds(10))
        capActive.bump()
        clearer.clearNow()
        #expect(cleared.value == 0)                                // not while the server's cap is in force
    }

    // MARK: - FlowEditorView shows a failed row's sentence

    private struct WaitFailingExecutor: FlowExecutor {
        let inner: MockExecutor
        func execute(path: String, row: Row, inputs: [Asset], transcript: [FlowInterpreter.TranscriptEntry]?,
                     context: [(label: String, content: String)]?, usedFlowContent: String?) async throws -> Asset {
            if row.task == "Summarize" {
                throw MemoryBudget.WaitError(label: "m", blockers: [.init(model: "Qwen3-8B-4bit", holder: .serverRequest)])
            }
            return try await inner.execute(path: path, row: row, inputs: inputs, transcript: transcript,
                                           context: context, usedFlowContent: usedFlowContent)
        }
        var lastCacheHit: Bool { inner.lastCacheHit }
        var lastTag: String? { inner.lastTag }
        var lastTimeoutFlag: (code: String, message: String)? { inner.lastTimeoutFlag }
        var lastProviderDeciderFlag: (code: String, message: String)? { inner.lastProviderDeciderFlag }
        var lastStaged: (id: String, kind: String, summary: String)? { inner.lastStaged }
        var lastDeciderDetail: (tag: String, confidence: Double, probabilities: [(label: String, probability: Double)],
                                expectedLevel: Double?, stateTruncated: Bool)? { inner.lastDeciderDetail }
    }

    @Test func aFailedRowsWaitingSentenceReachesTheInspectorNote() async throws {
        let doc = try CatParser.parse("""
        mlxflow 0.8
        1. Template   "hello"
        2. Summarize   Qwen3 8B; "one line"
        """)
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("sc2c-flow-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let blob = base.appendingPathComponent("blobs")
        let ctx = FlowRunner.RunContext(flowID: "t", workspace: FlowWorkspace(root: base.appendingPathComponent("flows")),
                                        blobDirectory: blob, executor: WaitFailingExecutor(inner: MockExecutor(blobDirectory: blob)))
        let session = FlowRunSession()
        session.prepareInstall(FlowPreflight.Result())
        session.start(doc: doc, runner: FlowRunner(), context: ctx)
        let deadline = Date().addingTimeInterval(10)
        while session.isRunning && Date() < deadline { try? await Task.sleep(for: .milliseconds(50)) }
        let row = doc.rows[1]
        #expect(session.status(for: row.id) == .failed)
        let note = try #require(session.statusNote(for: row.id))
        #expect(!note.isSkip)
        #expect(note.text.contains("Waiting for memory: Qwen3-8B-4bit is in use by a server request. Try again when it finishes."))
        #expect(session.statusNote(for: doc.rows[0].id) == nil)
    }

    /// The editor and the flow page must both hand the inspector the selected row's note. A view body can't be
    /// rendered here, so this reads the two sources: each builds a `FlowInspectorPane` that is given `statusNote:`.
    @Test func bothFlowHostsPassTheStatusNoteToTheirInspector() throws {
        let views = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Views/Flows")
        for file in ["FlowListView.swift", "FlowEditorView.swift"] {
            let source = try String(contentsOf: views.appendingPathComponent(file), encoding: .utf8)
            let start = try #require(source.range(of: "FlowInspectorPane("), "\(file) builds no FlowInspectorPane")
            let call = source[start.upperBound...].prefix(900)
            #expect(call.contains("statusNote:"), "\(file) doesn't hand the inspector the row's status note")
            #expect(call.contains("session.statusNote(for:"), "\(file) doesn't read it from the run session")
        }
    }
}
