import Testing
import Foundation
@testable import MLXUI

/// CL-8 (`RSI/DelegateCLMBacklog.md` Phase CL-B) — `78-InvoiceDesk`'s own end-to-end proof,
/// through the exact composition a real Run uses (`CachingExecutor(RealExecutor)`,
/// `AppFlowExecutorFactory.cachingContext`'s own shape), with a fake `askCLM` so no real
/// checkpoint loads. Flow 77 shipped without this class of test and hit all three of
/// READ-UPSTREAM-1 (`Read Text` never checked its upstream `.file` item), CACHE-SIGNALS-1
/// (`CachingExecutor` never forwarded a decider's fired tag, so a real Run's block silently
/// ended right after the decider), and the 2026-09-30 CL-8 amendment's own bug (a missing
/// `GalleryLoader.bundledAssets` case, LY-9-FIX-1). This file exercises all three paths at
/// once for flow 78, cold and warm, before a human ever runs it.
struct CLMGalleryInvoiceDeskTests {
    private func clmEntry() -> ModelEntry {
        makeEntry(id: "RealityCat--CLM-v0.1-8B-MLX-8bit", family: "CLM", displayName: "CLM 8B",
                 modelType: .decision, source: .mlx, ramGB: 12.2, downloadSizeGB: 8.12,
                 hfRepo: "RealityCat", hfModelId: "RealityCat/CLM-v0.1-8B-MLX-8bit")
    }

    private func fakeAnswer(type: LayaQuestionType, label: String, tags: [String]) -> LayaAnswer {
        let index = tags.firstIndex(of: label) ?? 0
        var probs = [Double](repeating: 0.1 / Double(max(tags.count - 1, 1)), count: tags.count)
        probs[index] = 0.9
        var answer = LayaAnswer(type: type, confidence: 0.8, probabilities: probs,
                                optionLabels: tags, stateTruncated: false)
        answer.choiceLabel = label
        return answer
    }

    /// The five invoices' own distinguishing sender name, keyed to the (Gate, Score) outcome
    /// `RSI/DelegateCLMBacklog.md` CL-8 names as expected (not guaranteed, for a real model —
    /// scripted exactly here, since this test fakes the engine).
    private let scriptedOutcomes: [String: (gate: String, score: String?)] = [
        "Northwind Office Supply": ("approve", nil),
        "Brightline Logistics": ("hold", "major"),
        "Cedar & Pine Catering": ("hold", "minor"),
        "Apex Cloud Services": ("hold", "blocking"),
        "Harbor Print Co": ("approve", nil),
    ]

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var _count = 0
        func increment() { lock.withLock { _count += 1 } }
        var count: Int { lock.withLock { _count } }
    }

    @Test func flow78MaterialisesReadsAndRoutesEveryInvoiceColdAndWarm() async throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("clm-invoicedesk-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = FlowWorkspace(root: base.appendingPathComponent("flows"))
        let store = FlowCacheStore(root: base.appendingPathComponent("cache"))
        let flowID = "78-InvoiceDesk"

        // The exact materialization path Open / Duplicate & Edit / a gallery Run all take.
        try ws.prepare(flowID: flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        let flowDir = ws.directory(for: flowID)
        for n in 1 ... 5 {
            let invoice = flowDir.appendingPathComponent("invoices/invoice-0\(n).txt")
            #expect(FileManager.default.fileExists(atPath: invoice.path), "\(invoice.lastPathComponent) was never materialised")
        }

        let doc = try GalleryLoader.loadDocument(flowID: flowID)
        let clm = clmEntry()
        let outcomes = scriptedOutcomes
        let seenGateStates = Counter()
        let seenScoreStates = Counter()

        var real = RealExecutor(
            workspace: ws, flowID: flowID, blobDirectory: base.appendingPathComponent("blobs"),
            makeModelStage: { _, _ in throw StageError.unsupportedModel(id: "probe", kind: .llm) },
            installedModelIDs: [clm.id], catalog: [clm])
        real.askCLM = { _, state, type, _, _ in
            guard case .string(let text) = state,
                  let sender = outcomes.keys.first(where: { text.contains($0) })
            else {
                Issue.record("askCLM saw a state that doesn't match any known invoice: \(state)")
                return self.fakeAnswer(type: type, label: "approve", tags: ["approve", "hold"])
            }
            guard let outcome = outcomes[sender] else {
                Issue.record("no scripted outcome for \(sender)")
                return self.fakeAnswer(type: type, label: "approve", tags: ["approve", "hold"])
            }
            switch type {
            case .choice:
                seenGateStates.increment()
                return self.fakeAnswer(type: .choice, label: outcome.gate, tags: ["approve", "hold"])
            case .score:
                seenScoreStates.increment()
                guard let scoreTag = outcome.score else {
                    Issue.record("\(sender) was approved — Score should never be asked")
                    return self.fakeAnswer(type: .score, label: "minor", tags: ["minor", "major", "blocking"])
                }
                return self.fakeAnswer(type: .score, label: scoreTag, tags: ["minor", "major", "blocking"])
            case .noul:
                Issue.record("Gate/Score never ask a .noul question")
                return self.fakeAnswer(type: .noul, label: "false", tags: ["false", "true"])
            }
        }
        let caching = CachingExecutor(inner: real, store: store, cacheTier: "real",
                                      catalog: [clm], runSeed: 0, workspace: ws, flowID: flowID)

        func assertRun(_ events: [FlowInterpreter.PathEvent], label: String) throws -> String {
            let f003 = events.filter { $0.kind == .flagRaised && $0.code == "F003" }
            #expect(f003.isEmpty, "\(label): no invoice should fail: \(f003.map(\.message))")
            // Every branch this script actually routes to must complete: the merge (2.8),
            // APPROVE (2.3, invoices 01/05), HOLD-MAJOR (2.6, invoice 02),
            // HOLD-MINOR (2.5, invoice 03), HOLD-BLOCKING (2.7, invoice 04).
            for path in ["2.3", "2.5", "2.6", "2.7", "2.8"] {
                #expect(events.contains { $0.kind == .rowCompleted && $0.path == path },
                        "\(label): row \(path) never completed")
            }
            let saved = try String(contentsOf: flowDir.appendingPathComponent("invoice-desk.md"), encoding: .utf8)
            #expect(saved.contains("APPROVE"))
            #expect(saved.contains("HOLD-MAJOR"))
            #expect(saved.contains("HOLD-MINOR"))
            #expect(saved.contains("HOLD-BLOCKING"))
            // Sort direction=asc: APPROVE < HOLD-BLOCKING < HOLD-MAJOR < HOLD-MINOR.
            if let approveIndex = saved.range(of: "APPROVE")?.lowerBound,
               let blockingIndex = saved.range(of: "HOLD-BLOCKING")?.lowerBound,
               let majorIndex = saved.range(of: "HOLD-MAJOR")?.lowerBound,
               let minorIndex = saved.range(of: "HOLD-MINOR")?.lowerBound {
                #expect(approveIndex < blockingIndex)
                #expect(blockingIndex < majorIndex)
                #expect(majorIndex < minorIndex)
            } else {
                Issue.record("\(label): couldn't locate all four group labels in the saved file to check ordering")
            }
            return saved
        }

        let cold = try await FlowInterpreter.run(doc, executor: caching)
        let coldSaved = try assertRun(cold, label: "cold")
        #expect(seenGateStates.count == 5, "cold: every invoice must ask Gate for real")
        #expect(seenScoreStates.count == 3, "cold: only the 3 held invoices should reach Score")

        let warm = try await FlowInterpreter.run(doc, executor: caching)
        let warmSaved = try assertRun(warm, label: "warm")
        #expect(seenGateStates.count == 5, "warm: a cache hit must never call askCLM's Gate again")
        #expect(seenScoreStates.count == 3, "warm: a cache hit must never call askCLM's Score again")
        #expect(warm.contains { $0.kind == .cacheHit && $0.path == "2.2" }, "warm: Gate's own path should be a cache hit")
        #expect(coldSaved == warmSaved, "identical output on both runs")
    }
}
