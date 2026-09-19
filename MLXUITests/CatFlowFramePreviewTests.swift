import Testing
import Foundation
@testable import MLXUI

/// FV-2 (`RSI/DelegateFrameViewBacklog.md`) — `FlowKit/FramePreview.swift`, the single render
/// function the Properties-tab "Prompt frame" preview and `RealExecutor`'s frame-backed rows
/// both call. Covers the three render branches (fact 7 — generator / decider-Judge /
/// decider-Think, with everything else in `TaskCatalog.deciderTasks` going through the plain
/// decider path), the "every `.frame` task resolves a bundled frame file" bundle-lookup gate,
/// the owner's own row (`RSI/DelegateFrameViewBacklog.md` §2's trace), and the structural
/// guarantee that a preview's stand-in copy can never reach `RealExecutor`'s execution path.
struct CatFlowFramePreviewTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func asset(_ text: String) -> Asset {
        Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }

    // MARK: - The three render branches (fact 7)

    @Test func generatorTaskRendersThroughFrameRenderer() throws {
        // Summarize is `refKind == .frame`, not in `TaskCatalog.deciderTasks` — the plain
        // generator branch, matching `FrameRenderer.render` exactly.
        let rendered = try FramePreview.render(
            task: "Summarize", refName: "frames/Summarize.frame.txt",
            settings: "\"3 words\"", assets: [asset("hello world")], tags: [])
        let frame = try FrameRenderer.loadFrame(named: "Summarize")
        let expected = try FrameRenderer.render(frameText: frame, settings: "\"3 words\"",
                                                 assets: [asset("hello world")])
        #expect(rendered == expected)
        #expect(rendered.contains("Instruction: \"3 words\""))
    }

    @Test func judgeTaskLabelsCandidatesByTag() throws {
        let rendered = try FramePreview.render(
            task: "Judge", refName: "frames/Judge.frame.txt", settings: "pick the clearest one",
            assets: [asset("first"), asset("second")], tags: ["a", "b"])
        #expect(rendered.contains("a: first"))
        #expect(rendered.contains("b: second"))
        #expect(rendered.contains("Criterion: pick the clearest one"))
    }

    @Test func thinkTaskSubstitutesToolsAndTranscript() throws {
        let edges = [ClauseEdge(tag: "search", target: .call(number: 3)),
                     ClauseEdge(tag: "final", target: .resume)]
        let tools = DeciderFrame.renderThinkTools(edges: edges)
        let rendered = try FramePreview.render(
            task: "Think", refName: "frames/Think.frame.txt", settings: nil,
            assets: [asset("solve this")], tags: ["search", "final"], tools: tools,
            transcript: "⟨what the agent has tried so far, once it runs⟩")
        #expect(rendered.contains("- search: calls row 3."))
        #expect(rendered.contains("- final: finish and give your answer."))
        #expect(rendered.contains("⟨what the agent has tried so far, once it runs⟩"))
        #expect(rendered.contains("Task: solve this"))
    }

    @Test func plainDeciderTaskRendersThroughDeciderFrame() throws {
        // Gate — a decider that isn't Judge or Think — goes through `DeciderFrame.renderFrame`.
        let rendered = try FramePreview.render(
            task: "Gate", refName: "frames/Gate.frame.txt", settings: "must be polite",
            assets: [asset("hi there")], tags: ["pass", "fail"])
        #expect(rendered.contains("Criterion: must be polite"))
        #expect(rendered.contains("Allowed tags: pass, fail"))
        #expect(rendered.contains("Input:\nhi there"))
    }

    @Test func nonFrameTaskRefuses() {
        #expect(throws: (any Error).self) {
            try FramePreview.render(task: "Decide", refName: "engines.llm.decide",
                                    settings: nil, assets: [], tags: [])
        }
    }

    // MARK: - Bundle-lookup gate (fact 8 — catches a next framed task's typo'd `refName`)

    @Test func everyFrameTaskResolvesAFrameFileFromTheBundle() throws {
        let frameTasks = TaskCatalog.allTasks().filter { $0.refKind == .frame }
        #expect(frameTasks.count == 16, "expected 16 framed tasks (§0's table); got \(frameTasks.count)")
        for desc in frameTasks {
            let name = FramePreview.frameFileName(refName: desc.refName)
            #expect((try? FrameRenderer.loadFrame(named: name)) != nil,
                    "\(desc.name)'s refName '\(desc.refName)' didn't resolve to a bundled frame file")
        }
    }

    // MARK: - The owner's own row (§2's trace)

    @Test func ownersRowRendersTheQuotedInstructionVerbatim() throws {
        // Basic Gallery → Summary From Audio, row 3: `3. Summarize   Qwen3 8B; "3 words"`.
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/BasicGallery/2-SummaryFromAudio.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        let row = try #require(doc.rows.first { $0.task == "Summarize" })
        #expect(row.settings == "\"3 words\"")

        let desc = try #require(TaskCatalog.get("Summarize"))
        let rendered = try FramePreview.render(
            task: "Summarize", refName: desc.refName, settings: row.settings,
            assets: [asset("⟨the text from row 2 — Transcribe⟩")], tags: [])
        // Q231 — the box above shows `3 words` unquoted; the frame the model actually reads
        // gets the raw settings string, quote marks included. Both are correct; see the
        // journal for this cycle and `catflow-mlx/SPEC_QUESTIONS.md` Q231.
        #expect(rendered.contains("Instruction: \"3 words\""))
    }

    // MARK: - Stand-in text cannot reach a run

    /// The "⟨…⟩" stand-in glyph is FV-2's own invention (§7) for the Properties-tab preview —
    /// it must never appear in the execution path itself. If a future edit "helpfully" adds a
    /// stand-in fallback for an unbound input inside `RealExecutor`/`FramePreview`/
    /// `FrameRenderer`/`DeciderFrame`, this fails immediately rather than silently sending a
    /// placeholder string to a real model.
    @Test func standInGlyphNeverAppearsInTheExecutionPath() throws {
        let executionFiles = [
            "MLXUI/FlowKit/RealExecutor.swift",
            "MLXUI/FlowKit/FramePreview.swift",
            "MLXUI/FlowKit/FrameRenderer.swift",
            "MLXUI/FlowKit/DeciderFrame.swift",
        ]
        for file in executionFiles {
            let text = try String(contentsOf: repoRoot.appendingPathComponent(file), encoding: .utf8)
            #expect(!text.contains("⟨"),
                    "\(file) must never hardcode a preview stand-in — the run path only ever sees real bound data")
        }
    }
}
