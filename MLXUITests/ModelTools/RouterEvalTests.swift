import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT4-6 — the CI half of the router eval: no model, no network, deterministic. The eval data is well formed, the policy offers
/// the expected tool for every case it should, the example arguments validate, the harness scores a scripted router correctly
/// (also through the real `AppleFoundationRouter` and `MLXChatRouter` classes with fixture sessions), and the findings the eval
/// inherited are pinned as measurements. Real-router runs are in `RouterEvalRealTests` (env-gated, owner-run).
@MainActor
@Suite(.serialized)
struct RouterEvalTests {

    private func suite() throws -> RouterEvalSuite { try RouterEvalSuite.load() }

    // MARK: the data

    @Test func thirtyCasesCoverEveryV1ModelToolAndTheAmbiguousGateCues() throws {
        let cases = try suite().cases
        #expect(cases.count == 30)
        #expect(Set(cases.map(\.id)).count == 30)
        let expectedTools = Set(cases.compactMap(\.expectedTool))
        for spec in ModelToolSpec.all { #expect(expectedTools.contains(spec.name), "\(spec.name) has no case") }
        for name in ToolExposurePolicy.ambiguousTools { #expect(expectedTools.contains(name), "\(name) has no case") }
        #expect(cases.filter { $0.expectedTool == nil }.count >= 4)                       // plain questions, "capital of France"
        #expect(cases.contains { $0.prompt == "What is the capital of France?" })
        #expect(cases.filter { $0.attached?.contains("image") == true }.count >= 5)
        #expect(cases.filter { $0.attached?.contains("audio") == true }.count >= 3)
        #expect(cases.filter { $0.statuses?.values.contains("needsDownload") == true }.count >= 1)   // the install-needed case
        #expect(cases.filter { $0.group == "non-english" }.count >= 3)
        for testCase in cases where testCase.group == "non-english" { #expect(testCase.language != nil && testCase.knownPolicyGap != nil) }
    }

    @Test func everyExampleValidatesAgainstTheToolSchemaAndMatchesItsOwnKeyArguments() throws {
        let defaults = Dictionary(ModelRunner.defaultTools().map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
        for testCase in try suite().cases {
            guard let name = testCase.expectedTool else { continue }
            let tool: any AgentTool
            if let spec = ModelToolSpec.spec(named: name) { tool = ModelTool(spec: spec, advertisesAssets: true) }
            else if let found = defaults[name] { tool = found }
            else { Issue.record("\(testCase.id): no tool named \(name)"); continue }
            let example = testCase.example ?? [:]
            #expect(RouterEvalValidator.problem(example, for: tool) == nil, "\(testCase.id) example is invalid: \(RouterEvalValidator.problem(example, for: tool) ?? "")")
            #expect(RouterEvalValidator.keyArgsMatch(example, expected: testCase.expected.args), "\(testCase.id) example misses its own key arguments")
        }
    }

    @Test func theValidatorRejectsWhatARouterShouldNotSend() throws {
        let ocr = ModelTool(spec: try #require(ModelToolSpec.spec(named: "ocr_image")), advertisesAssets: true)
        #expect(RouterEvalValidator.problem([:], for: ocr) != nil)                                  // no input
        #expect(RouterEvalValidator.problem(["asset": .string("a1"), "colour": .string("red")], for: ocr) != nil)   // unknown argument
        let image = ModelTool(spec: try #require(ModelToolSpec.spec(named: "generate_image")), advertisesAssets: true)
        #expect(RouterEvalValidator.problem(["text": .string("fox"), "steps": .int(500)], for: image) != nil)       // range
        #expect(RouterEvalValidator.problem(["text": .string("fox"), "steps": .int(20)], for: image) == nil)
        let translate = ModelTool(spec: try #require(ModelToolSpec.spec(named: "translate")), advertisesAssets: true)
        #expect(RouterEvalValidator.problem(["text": .string("hi")], for: translate) != nil)        // `to` is required
        #expect(RouterEvalValidator.problem(["expression": .string("1+1")], for: CalculatorTool()) == nil)
        #expect(RouterEvalValidator.problem([:], for: CalculatorTool()) != nil)
    }

    // MARK: the policy check (CI)

    /// For every case the policy must offer the expected tool, router by router. The cases marked `knownPolicyGap` are
    /// measurements of the English-only cue heuristic: they must MISS under the gate (Apple Intelligence) and be offered
    /// without it (MLX today). A fix to the cues makes this test fail on purpose: update the case file in the same change.
    @Test func thePolicyOffersTheExpectedToolForEveryCaseItShould() throws {
        let cases = try suite().cases
        for setup in [RouterEvalSetup.appleIntelligence, .mlxToday] {
            for testCase in cases {
                let plan = RouterEvalPlanner.plan(testCase, setup: setup)
                if testCase.knownPolicyGap != nil, setup.gateAmbiguousTools || testCase.gapWithoutGate == true {
                    #expect(!plan.expectedOffered, "\(testCase.id) is now offered under \(setup.name); the known gap is closed, update the case")
                } else {
                    #expect(plan.expectedOffered, "\(setup.name): \(testCase.id) expected \(testCase.expectedTool ?? "none") but offered \(plan.offered)")
                }
            }
        }
    }

    @Test func theOfferedSetRespectsTheRouterBudget() throws {
        for testCase in try suite().cases {
            #expect(RouterEvalPlanner.plan(testCase, setup: .appleIntelligence).offered.count <= AppleFoundationToolBudget.maxTools)
            #expect(RouterEvalPlanner.plan(testCase, setup: .mlxToday).offered.count <= ToolExposurePolicy.mlxToolBudget)
        }
    }

    // MARK: the harness scores correctly

    private func policyCeiling(_ cases: [RouterEvalCase], _ setup: RouterEvalSetup) -> Int {
        cases.filter { RouterEvalPlanner.plan($0, setup: setup).expectedOffered }.count
    }

    @Test func theOracleThroughTheHarnessMatchesThePolicyCeiling() async throws {
        let cases = try suite().cases
        for setup in [RouterEvalSetup.appleIntelligence, .mlxToday, .mlxGated] {
            let outcomes = await RouterEvalRunner.run(router: ScriptedEvalRouter.oracle(cases: cases), setup: setup, cases: cases)
            let report = RouterEvalReport(routerName: setup.name, outcomes: outcomes)
            #expect(report.toolChoice.hits == policyCeiling(cases, setup), "\(setup.name): \(report.render())")
            #expect(report.validArguments.hits == report.validArguments.total)
            #expect(report.keyArguments.hits == report.keyArguments.total)
            #expect(report.answeredWithoutTool.hits == report.answeredWithoutTool.total)
            #expect(report.unneededInstallAsks == 0)
        }
    }

    @Test func aRouterThatNeverCallsAToolScoresOnlyTheNoToolCases() async throws {
        let cases = try suite().cases
        let mute = ScriptedEvalRouter { _ in nil }
        let report = RouterEvalReport(routerName: "mute", outcomes: await RouterEvalRunner.run(router: mute, setup: .mlxToday, cases: cases))
        // Right on the no-tool cases and on the case that accepts answering directly (g08, summarize).
        let rightWithoutATool = cases.filter { $0.expectedTool == nil || $0.expected.accept?.contains("none") == true }
        #expect(report.toolChoice.hits == rightWithoutATool.count)
        #expect(report.validArguments.total == 0)
        #expect(report.askedBeforeDownload.hits == 0 && report.askedBeforeDownload.total >= 1)
    }

    @Test func aRouterThatSendsBadArgumentsIsCaughtAndAWrongToolIsCounted() async throws {
        let cases = try suite().cases
        let bad = ScriptedEvalRouter { _ in ("list_capabilities", ["colour": .string("red")]) }
        let report = RouterEvalReport(routerName: "bad", outcomes: await RouterEvalRunner.run(router: bad, setup: .mlxToday, cases: cases))
        #expect(report.validArguments.total == cases.count && report.validArguments.hits == 0)
        #expect(report.toolChoice.hits == cases.filter { $0.expectedTool == "list_capabilities" }.count)
        #expect(report.answeredWithoutTool.hits == 0)
    }

    @Test func callingADownloadToolAsksTheHostAndADeclineIsReportedBack() async throws {
        let cases = try suite().cases
        let testCase = try #require(cases.first { $0.id == "m02" })                  // describe_image needs a download
        let outcome = await RouterEvalRunner.runOne(router: ScriptedEvalRouter.oracle(cases: cases), setup: .mlxToday, testCase: testCase)
        #expect(outcome.installAsks == ["describe_image"] && outcome.toolChoiceOK && outcome.expectedNeedsDownload)
        #expect(outcome.text.contains("The user chose not to download"))
        var accepting = RouterEvalSetup.mlxToday; accepting.acceptInstalls = true
        let installed = await RouterEvalRunner.runOne(router: ScriptedEvalRouter.oracle(cases: cases), setup: accepting, testCase: testCase)
        #expect(installed.installAsks == ["describe_image"] && installed.text.contains("busy street market"))
    }

    // MARK: through the real router classes (fixture sessions)

    @Test func theOracleThroughTheRealAppleRouterMatchesThePolicyCeiling() async throws {
        let cases = try suite().cases
        let byPrompt = Dictionary(cases.map { ($0.prompt, $0) }, uniquingKeysWith: { first, _ in first })
        let chatter = AppleFoundationRouterTests.FakeChatter(script: { prompt, handler in
            guard let testCase = byPrompt[prompt], let tool = testCase.expectedTool else { return [.snapshot("answered directly")] }
            let result = await handler(tool, testCase.example ?? [:])
            return [.snapshot("got: \(result)")]
        })
        let router = AppleFoundationRouter(chatter: chatter)
        let outcomes = await RouterEvalRunner.run(router: router, setup: .appleIntelligence, cases: cases)
        let report = RouterEvalReport(routerName: "Apple router, fixture session", outcomes: outcomes)
        // The Apple router refuses a tool that was not offered with a corrective sentence and the turn then ends with no
        // `.toolStarted`; so the ceiling is the policy's.
        #expect(report.toolChoice.hits >= policyCeiling(cases, .appleIntelligence) - 0, "\(report.render())")
        #expect(report.validArguments.hits == report.validArguments.total, "\(report.render())")
        #expect(report.keyArguments.hits == report.keyArguments.total, "\(report.render())")
        #expect(outcomes.allSatisfy { $0.error == nil }, "\(report.render())")
    }

    @Test func theOracleThroughTheRealMLXRouterMatchesThePolicyCeiling() async throws {
        let cases = try suite().cases
        let byPrompt = Dictionary(cases.map { ($0.prompt, $0) }, uniquingKeysWith: { first, _ in first })
        final class FixtureSession: ChatStreamingSession, @unchecked Sendable {
            let turn: RouterTurn
            let onEvent: @Sendable (AgentToolEvent) -> Void
            let byPrompt: [String: RouterEvalCase]
            init(turn: RouterTurn, onEvent: @escaping @Sendable (AgentToolEvent) -> Void, byPrompt: [String: RouterEvalCase]) {
                self.turn = turn; self.onEvent = onEvent; self.byPrompt = byPrompt
            }
            func streamResponse(to prompt: String) -> AsyncThrowingStream<String, Error> {
                let turn = turn, onEvent = onEvent, byPrompt = byPrompt
                return AsyncThrowingStream { continuation in
                    let task = Task {
                        if let testCase = byPrompt[prompt], let name = testCase.expectedTool,
                           let tool = turn.tools.first(where: { $0.name == name }) {
                            let arguments = testCase.example ?? [:]
                            onEvent(.started(name: name, arguments: arguments))
                            let result = (try? await tool.execute(arguments: arguments)) ?? "error"
                            onEvent(.finished(name: name, result: result))
                            continuation.yield("got: \(result.prefix(30))")
                        } else {
                            continuation.yield("answered directly")
                        }
                        continuation.finish()
                    }
                    continuation.onTermination = { _ in task.cancel() }
                }
            }
        }
        let router = MLXChatRouter(identity: RouterIdentity(displayName: "Fixture MLX", kind: .mlx, contextSize: 8192)) { turn, onEvent in
            FixtureSession(turn: turn, onEvent: onEvent, byPrompt: byPrompt)
        }
        for setup in [RouterEvalSetup.mlxToday, .mlxGated] {
            let report = RouterEvalReport(routerName: setup.name, outcomes: await RouterEvalRunner.run(router: router, setup: setup, cases: cases))
            #expect(report.toolChoice.hits == policyCeiling(cases, setup), "\(report.render())")
            #expect(report.validArguments.hits == report.validArguments.total)
        }
    }

    // MARK: findings (measurements; none of these changes the policy)

    /// What the policy offers for a prompt, by name.
    private func offered(_ prompt: String, _ setup: RouterEvalSetup, attached: [String] = [], earlier: [String] = [],
                         statuses: [String: String] = [:]) -> [String] {
        let testCase = RouterEvalCase(id: "probe", group: "probe", prompt: prompt, attached: attached, earlier: earlier,
                                      statuses: statuses, expected: .init(tool: nil))
        return RouterEvalPlanner.plan(testCase, setup: setup).offered
    }

    /// Finding F1 (inherited, MT4-2b rule 4): after an image turn, every later turn offers all five image tools again, whatever
    /// the user types. With Apple Intelligence's budget of 6 that is the whole set: list_capabilities and the five image tools.
    @Test func findingF1TheImageToolsReturnOnEveryLaterTurn() {
        let later = offered("Thanks, that is helpful.", .appleIntelligence, earlier: ["image"])
        print("MT46 F1 turn 2, no image attached, prompt 'Thanks...': Apple offers", later)
        #expect(later == ["list_capabilities", "ocr_image", "describe_image", "upscale_image", "segment_image", "estimate_depth"])
        let plain = offered("Thanks, that is helpful.", .appleIntelligence)
        print("MT46 F1 same prompt in a chat that never had an image: Apple offers", plain)
    }

    /// Finding F2 (inherited): the readiness ordering of rule 2 only reorders inside one rule's group, and the default image
    /// group is the single tool ocr_image. With ocr_image a download away and describe_image installed, a "describe this"
    /// turn lists the not-installed ocr_image ahead of the installed describe_image.
    @Test func findingF2AReadyCuedToolCanSitBehindADownloadDefault() {
        let list = offered("Describe this photo for me.", .appleIntelligence, attached: ["image"],
                           statuses: ["ocr_image": "needsDownload", "describe_image": "ready"])
        print("MT46 F2 image + 'Describe this photo', ocr_image needsDownload, describe_image ready: Apple offers", list)
        let ocr = list.firstIndex(of: "ocr_image"), describe = list.firstIndex(of: "describe_image")
        #expect(ocr != nil && describe != nil)
        if let ocr, let describe { #expect(ocr < describe) }
    }

    /// Finding F3 (inherited): the cue heuristic is broad. Ordinary prompts that arm an ambiguous tool they do not want.
    @Test func findingF3TheCueHeuristicArmsToolsOnOrdinaryPrompts() {
        let probes: [(String, String)] = [
            ("What is the time complexity of quicksort?", "datetime"),
            ("How do I improve my time management?", "datetime"),
            ("What day did the Second World War start?", "datetime"),
            ("Explain how a database index speeds up queries.", "semantic_search"),
            ("Give me a summary of Hamlet.", "summarize"),
            ("Is there a script that writes better emails?", "run_javascript"),
            ("Write a short story about a robot named Js.", "run_javascript"),
            ("Which do you prefer, 3x4 or 4x3 shelves?", "calculator"),
        ]
        var armed = 0
        for (prompt, tool) in probes {
            let gated = offered(prompt, .appleIntelligence)
            if gated.contains(tool) { armed += 1 }
            print("MT46 F3 '\(prompt)' arms \(tool):", gated.contains(tool))
        }
        print("MT46 F3 \(armed) of \(probes.count) ordinary prompts arm an ambiguous tool they do not need")
        // The four eval no-tool prompts arm nothing: the "capital of France" trap is closed.
        for testCase in (try? suite().cases.filter({ $0.expectedTool == nil })) ?? [] {
            let names = Set(offered(testCase.prompt, .appleIntelligence)).intersection(ToolExposurePolicy.ambiguousTools)
            #expect(names.isEmpty, "\(testCase.id) arms \(names)")
        }
        #expect(armed == probes.count, "the cue breadth changed (finding F3): update the measurement")
    }

    /// Finding F4 (inherited): the cues are English-only. The non-English eval cases miss under the gate and not without it.
    @Test func findingF4TheCuesAreEnglishOnly() throws {
        let cases = try suite().cases.filter { $0.group == "non-english" }
        for testCase in cases {
            let apple = RouterEvalPlanner.plan(testCase, setup: .appleIntelligence).expectedOffered
            let mlx = RouterEvalPlanner.plan(testCase, setup: .mlxToday).expectedOffered
            print("MT46 F4 [\(testCase.language ?? "?")] '\(testCase.prompt)' expected \(testCase.expectedTool ?? "none"): Apple (gate on) offered=\(apple), MLX today offered=\(mlx)")
            #expect(!apple)
            #expect(mlx == (testCase.gapWithoutGate != true))
        }
    }

    /// What `gateAmbiguousTools` would do for the MLX router, against today's off. A measurement for the owner's decision.
    @Test func measureGatingTheMLXRouter() throws {
        let cases = try suite().cases
        func tally(_ setup: RouterEvalSetup) -> (hit: Int, exposed: Int, distractors: Int, ambiguousOffered: Int, mean: Double) {
            var hit = 0, exposed = 0, distractors = 0, ambiguousOffered = 0, total = 0
            for testCase in cases {
                let plan = RouterEvalPlanner.plan(testCase, setup: setup)
                if plan.expectedOffered { hit += 1 }
                total += plan.offered.count
                let ambiguous = plan.offered.filter(ToolExposurePolicy.ambiguousTools.contains)
                ambiguousOffered += ambiguous.count
                distractors += ambiguous.filter { $0 != testCase.expectedTool }.count
                if !ambiguous.isEmpty { exposed += 1 }
            }
            return (hit, exposed, distractors, ambiguousOffered, Double(total) / Double(cases.count))
        }
        func missed(_ setup: RouterEvalSetup) -> [String] { cases.filter { !RouterEvalPlanner.plan($0, setup: setup).expectedOffered }.map(\.id) }
        var gatedDistractors: [String] = []
        for testCase in cases {
            let plan = RouterEvalPlanner.plan(testCase, setup: .mlxGated)
            for name in plan.offered where ToolExposurePolicy.ambiguousTools.contains(name) && name != testCase.expectedTool {
                gatedDistractors.append("\(testCase.id):\(name)")
            }
        }
        print("MT46 GATE missed (off):", missed(.mlxToday), "missed (on):", missed(.mlxGated), "distractors with the gate on:", gatedDistractors)
        let off = tally(.mlxToday), on = tally(.mlxGated), apple = tally(.appleIntelligence)
        func line(_ name: String, _ t: (hit: Int, exposed: Int, distractors: Int, ambiguousOffered: Int, mean: Double)) -> String {
            String(format: "MT46 GATE %@: expected tool offered %d/30, cases with an ambiguous tool offered %d, ambiguous tools offered that the case did not call for %d, mean tools offered %.1f",
                   name, t.hit, t.exposed, t.distractors, t.mean)
        }
        print(line("MLX gate off (today)", off)); print(line("MLX gate on", on)); print(line("Apple (gate on)", apple))
        #expect(missed(.mlxToday) == ["l03"])                     // off withholds only what a generator cue misses (finding F4)
        #expect(missed(.mlxGated) == ["l01", "l02", "l03"])       // on adds the two English-only gated cues
        #expect(on.distractors < off.distractors)                 // the gate removes tools the turn did not call for
        #expect(on.hit < off.hit)                                 // ...and costs the cases the cue heuristic misses (the English-only ones)
    }

    /// Finding F5 (new): with no image or audio attached, the legacy-named enabled tools `ocr_image` and `transcribe_audio` are
    /// not model-tool rule 2 offers: they come through rule 5 (the user's enabled tools) with no status check, and `ChatToolSet`
    /// maps them to `ModelTool`. So a plain turn offers them even when the model tool cannot run here, and they take budget.
    @Test func findingF5LegacyNamedToolsBypassTheStatusCheckWhenNothingIsAttached() {
        let plain = offered("What is the capital of France?", .appleIntelligence,
                            statuses: ["ocr_image": "cantRun", "transcribe_audio": "none"])
        print("MT46 F5 plain prompt, ocr_image cantRun, transcribe_audio unavailable: Apple offers", plain)
        #expect(plain.contains("ocr_image") && plain.contains("transcribe_audio"))
    }
}
