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

    /// FV-4-2: the refusal names the task and says there's nothing to preview — not
    /// `missingFrame`'s "reinstall to restore it", which is the wrong fix for a task that was
    /// never frame-backed to begin with.
    @Test func nonFrameTaskRefusesWithItsOwnErrorVoice() throws {
        do {
            _ = try FramePreview.render(task: "Decide", refName: "engines.llm.decide",
                                        settings: nil, assets: [], tags: [])
            Issue.record("expected FramePreview.render to throw for a non-frame task")
        } catch let error as FrameError {
            #expect(error == .noPublishedFrame(task: "Decide"))
            #expect(error.description == "Decide has no published frame — nothing to preview.")
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

    // MARK: - FV-4-1: bound-ref count, not signature-slot count

    private func allRows(_ rows: [Row]) -> [Row] {
        rows.flatMap { [$0] + allRows($0.children) }
    }

    /// A task's own signature slot count (`FlowEditorModel.inputSlotCount`'s rule, reproduced
    /// here without a live document: `.tupleOf` counts its kinds, everything else is 1).
    private func signatureSlots(_ accepts: Shape) -> Int {
        if case .tupleOf(let kinds) = accepts { return kinds.count }
        return 1
    }

    /// The regression itself: `Revise`/`Verify` both declare a single-slot signature, but
    /// every corpus row of either task binds **two** refs, and both frames read
    /// `{input[0]}`/`{input[1]}`. Before FV-4-1, `renderedFramePreview` sized its stand-in
    /// assets off the signature alone, so `FrameRenderer` threw `frameInputOutOfRange` for
    /// every such row — silently, because `framePreviewSection`'s `try?` swallowed it (the
    /// reason the 16-task bundle-lookup walk above stayed green despite the bug). This walks
    /// the same corpus with the row's *actual* bound-ref count, exactly as the fixed View
    /// code now does (`max(row.refs.count, inputSlotCount, 1)`), and fails loudly if any
    /// framed row can't render.
    @Test func everyFramedCorpusRowRendersWithItsBoundRefCount() throws {
        let dirs = ["MLXUI/Resources/BasicGallery", "MLXUI/Resources/Gallery", "MLXUI/Resources/Workspaces"]
        var checked = 0
        var failures: [String] = []
        for dir in dirs {
            let dirURL = repoRoot.appendingPathComponent(dir)
            let files = try FileManager.default.contentsOfDirectory(atPath: dirURL.path)
                .filter { $0.hasSuffix(".cat") }.sorted()
            for file in files {
                let text = try String(contentsOf: dirURL.appendingPathComponent(file), encoding: .utf8)
                let doc = try CatParser.parse(text)
                for row in allRows(doc.rows) {
                    guard let task = row.task, let desc = TaskCatalog.get(task),
                          desc.refKind == .frame else { continue }
                    checked += 1
                    let slots = max(row.refs.count, signatureSlots(desc.accepts), 1)
                    let assets = (1...slots).map { _ in asset("⟨stand-in⟩") }
                    do {
                        _ = try FramePreview.render(task: task, refName: desc.refName,
                                                    settings: row.settings, assets: assets,
                                                    tags: RealExecutor.declaredTags(row))
                    } catch {
                        failures.append("\(dir)/\(file) — \(task) (\(row.refs.count) refs): \(error)")
                    }
                }
            }
        }
        #expect(checked >= 100, "expected the framed-row corpus walk to cover ~111 rows (§0); got \(checked)")
        #expect(failures.isEmpty, "frame render failed for:\n\(failures.joined(separator: "\n"))")
    }

    /// The exit criterion, named directly: `11-ReflexionWriter.cat` row 6, `Revise (3,4)`.
    @Test func reviseRowWithTwoBoundRefsRendersTwoDistinctStandIns() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/11-ReflexionWriter.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        let row = try #require(doc.rows.first { $0.task == "Revise" })
        #expect(row.refs.count == 2)

        let desc = try #require(TaskCatalog.get("Revise"))
        let rendered = try FramePreview.render(
            task: "Revise", refName: desc.refName, settings: row.settings,
            assets: [asset("⟨the text from row 3⟩"), asset("⟨the text from row 4⟩")], tags: [])
        #expect(rendered.contains("Draft:      ⟨the text from row 3⟩"))
        #expect(rendered.contains("Weaknesses: ⟨the text from row 4⟩"))
    }

    /// The reviewer's other named repro: `46-IndexSelfTest.cat` row 10, `Verify (9,3)`.
    @Test func verifyRowWithTwoBoundRefsRendersTwoDistinctStandIns() throws {
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/46-IndexSelfTest.cat")
        let doc = try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
        // Row 10 is nested inside a block, not top-level — `allRows` walks children.
        let row = try #require(allRows(doc.rows).first { $0.task == "Verify" })
        #expect(row.refs.count == 2)

        let desc = try #require(TaskCatalog.get("Verify"))
        let rendered = try FramePreview.render(
            task: "Verify", refName: desc.refName, settings: row.settings,
            assets: [asset("⟨the text from row 9⟩"), asset("⟨the text from row 3⟩")], tags: [])
        #expect(rendered.contains("Text (claims): ⟨the text from row 9⟩"))
        #expect(rendered.contains("Source:        ⟨the text from row 3⟩"))
    }
}
