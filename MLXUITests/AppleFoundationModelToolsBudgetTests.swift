import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT4-2 — the Apple Intelligence tool-token budget (1,500, enforced by `AppleFoundationRouter`) with the MODEL TOOLS in the
/// set. Owner-run, real Apple Intelligence (`TEST_RUNNER_MLXUI_TEST_REAL_AFM=1`); skips otherwise. The model tools run against a
/// scripted executor, so no model is loaded: the measurement is the real framework's token count and a real answered turn.
@MainActor
@Suite(.serialized)
struct AppleFoundationModelToolsBudgetTests {

    /// Wraps the real chatter, recording the tool names and instructions each session was built with.
    nonisolated final class RecordingChatter: AppleFoundationToolChatting, @unchecked Sendable {
        private let real: any AppleFoundationToolChatting
        private let lock = NSLock()
        private(set) var toolNames: [[String]] = []
        private(set) var instructions: [String?] = []
        init(_ real: any AppleFoundationToolChatting) { self.real = real }
        var contextSize: Int? { real.contextSize }
        func makeSession(instructions: String?, tools: [AppleFoundationToolSpec],
                         handler: @escaping AppleFoundationToolHandler) -> any AppleFoundationToolSession {
            lock.withLock { toolNames.append(tools.map(\.name)); self.instructions.append(instructions) }
            return real.makeSession(instructions: instructions, tools: tools, handler: handler)
        }
        func toolTokenCount(instructions: String?, tools: [AppleFoundationToolSpec]) async -> Int? {
            await real.toolTokenCount(instructions: instructions, tools: tools)
        }
        var last: (tools: [String], instructions: String?)? {
            lock.withLock { toolNames.last.map { ($0, instructions.last ?? nil) } }
        }
    }

    private func runtime() throws -> ModelToolRuntime {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        let cat = try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
        let ids = Set(["mlx-community/GLM-OCR-4bit", "mlx-community/parakeet-tdt-0.6b-v3", "mlx-community/Kokoro-82M-bf16"].compactMap { name in
            cat.first { $0.hfModelId == name }?.id })
        let registry = ModelRegistry()
        for module in installedModules { module.register(into: registry) }
        let claimable = Set(cat.filter { registry.bestModule(for: $0) != nil }.map(\.id))
        return ModelToolRuntime(catalog: cat, installedModelIDs: ids, claimableModelIDs: claimable, totalRAMGB: 19,
                                makeExecutor: { _, _ in MT13Executor.text("TOTAL 42.00, 3 items") })
    }

    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realModelToolTurnsFitTheBudgetAndAnswer() async throws {
        let edition: String = {
            #if DIRECT_BUILD
            return "Direct"
            #else
            return "AppStore"
            #endif
        }()
        let fm = FileManager.default
        let store = fm.temporaryDirectory.appendingPathComponent("mt42-afm-\(UUID().uuidString)")
        let files = fm.temporaryDirectory.appendingPathComponent("mt42-afm-f-\(UUID().uuidString)")
        for d in [store, files] { try fm.createDirectory(at: d, withIntermediateDirectories: true) }
        let png = files.appendingPathComponent("receipt.png"); try Data([0x89, 0x50, 0x4E, 0x47, 0]).write(to: png)
        let m4a = files.appendingPathComponent("meeting.m4a"); try Data([0, 1, 2, 3]).write(to: m4a)
        let csv = files.appendingPathComponent("rates.csv")
        try Data(String(repeating: "date,rate\n2026-01-01,4.25\n", count: 4_000).utf8).write(to: csv)
        let recorder = RecordingChatter(try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true)))
        AppleFoundationAvailability.toolChatterOverride = recorder
        defer { try? fm.removeItem(at: store); try? fm.removeItem(at: files); AppleFoundationAvailability.toolChatterOverride = nil }

        let runner = ModelRunner()
        runner.prepare(for: AppleIntelligenceChat.entry, in: store)
        let runtime = try runtime()
        runner.modelToolRuntimeProvider = { runtime }
        runner.modelToolsEnabled = true

        let turns: [(String, String, [URL])] = [
            ("plain", "What is the capital of France?", []),
            ("image", "What's the total on this receipt?", [png]),
            ("audio", "Summarize this meeting recording.", [m4a]),
            ("big text", "Which date has the highest rate in this file?", [csv]),
            ("image + all generators called for", "Read this receipt, translate it to Malay, say it aloud, draw a picture of it and make some music.", [png]),
            ("time", "What time is it?", []),
        ]
        for (label, prompt, attachments) in turns {
            runner.newConversation(for: AppleIntelligenceChat.modelID)
            await runner.send(prompt, attachments: attachments, for: AppleIntelligenceChat.entry)
            let deadline = ContinuousClock.now.advanced(by: .seconds(180))
            while runner.isRunning, ContinuousClock.now < deadline { try? await Task.sleep(for: .milliseconds(50)) }
            let r = runner.appleRouter()?.lastTrimReport
            let sent = recorder.last
            print("MT42 \(edition) | \(label) | tools:", sent?.tools as Any, "| instructions chars:", sent?.instructions?.count as Any,
                  "| before:", r?.tokensBefore as Any, "after:", r?.tokensAfter as Any, "dropped:", r?.dropped as Any,
                  "count secs:", r?.seconds as Any, "| reply:", (runner.transcript.lastAssistantText ?? "").prefix(60) as Any,
                  "| error:", runner.errorMessage as Any)
            if let after = r?.tokensAfter { #expect(after <= AppleFoundationToolBudget.tokenBudget, "\(label) is over the budget") }
            #expect(runner.errorMessage == nil, "\(label) did not answer")
            #expect(!(runner.transcript.lastAssistantText ?? "").isEmpty, "\(label) gave no reply")
        }
    }

    /// What each model tool costs in tokens (with the `asset` argument advertised, as in a chat), over an empty tool list.
    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realPerModelToolCosts() async throws {
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let base = try #require(await chatter.toolTokenCount(instructions: nil, tools: []))
        var costs: [(String, Int)] = []
        for tool in ModelTool.all(advertisingAssets: true) {
            let n = try #require(await chatter.toolTokenCount(instructions: nil, tools: [AppleFoundationRouter.spec(for: tool)]))
            costs.append((tool.name, n - base))
        }
        print("MT42 model-tool cost (tokens, over an empty tool list of \(base)):", costs.map { "\($0.0)=\($0.1)" }.joined(separator: ", "))
        let six = ["list_capabilities", "ocr_image", "describe_image", "upscale_image", "segment_image", "estimate_depth"]
        let specs = ModelTool.all(advertisingAssets: true).filter { six.contains($0.name) }.map(AppleFoundationRouter.spec(for:))
        let sixCount = try #require(await chatter.toolTokenCount(instructions: nil, tools: specs))
        print("MT42 the six image tools together (no instructions):", sixCount)
        #expect(sixCount > 0)
    }
}
