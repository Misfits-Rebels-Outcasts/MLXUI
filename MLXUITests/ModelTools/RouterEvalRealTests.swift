import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// MT4-6 — the real-router passes of the eval. OPT-IN: both tests skip in the normal suite.
///
/// * Apple Intelligence (the default router on macOS 27): `TEST_RUNNER_MLXUI_TEST_REAL_AFM=1` and Apple Intelligence ready.
/// * MLX chat models (owner-run, MT4-7): `TEST_RUNNER_MLXUI_TEST_ROUTER_EVAL_MLX_DIRS=<dir1>:<dir2>` names copies of installed
///   checkpoints (the test host is sandboxed and cannot read the app's store; copy with `cp -Rc`). Optional
///   `TEST_RUNNER_MLXUI_TEST_ROUTER_EVAL_GATE=off|on|both` (default `both`: today's MLX policy and the gated one, as a measurement).
///   One model at a time through the machine-wide smoke lock (`RSI/evals/model-smoke.sh` holds it; nothing else should be loading
///   a model while this runs).
///
/// The workers are mocked in both: only the router's choice of tool and arguments is measured. The tests assert only that no turn
/// failed; the 85% bar is reported, not asserted (a result below it is a finding for the owner, journal 2026-731).
@MainActor
@Suite(.serialized)
struct RouterEvalRealTests {

    private nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var mlxDirectories: [String] {
        (env["MLXUI_TEST_ROUTER_EVAL_MLX_DIRS"] ?? "").split(separator: ":").map(String.init).filter { !$0.isEmpty }
    }

    private func announce(_ report: RouterEvalReport) {
        print(report.render())
        let bar = report.toolChoice.percent >= 85 ? "MET" : "BELOW"
        print("ROUTER-EVAL-RESULT \(report.routerName) tool-choice \(report.toolChoice.text) bar(85%) \(bar)")
        for outcome in report.outcomes where !outcome.toolChoiceOK {
            print("ROUTER-EVAL-FAIL \(report.routerName) \(outcome.id): expected \(outcome.expectedTool ?? "none"), got \(outcome.firstTool ?? "no tool"), offered \(outcome.offered)")
        }
    }

    // MARK: Apple Intelligence

    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realAppleIntelligencePass() async throws {
        let cases = try RouterEvalSuite.load().cases
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let router = AppleFoundationRouter(chatter: chatter)
        let outcomes = await RouterEvalRunner.run(router: router, setup: .appleIntelligence, cases: cases) { outcome in
            print("ROUTER-EVAL-CASE Apple \(outcome.id) \(String(format: "%.1f", outcome.seconds)) s: expected \(outcome.expectedTool ?? "none"), got \(outcome.firstTool ?? "no tool")")
        }
        let report = RouterEvalReport(routerName: "Apple Intelligence", outcomes: outcomes)
        announce(report)
        // MT4-6b: the same pass counting ANY call (not only the first) as the case's tool, for the journal.
        let anyCall = outcomes.filter { o in
            guard let expected = o.expectedTool else { return o.calls.isEmpty }
            let accept = cases.first { $0.id == o.id }?.expected.accept ?? []
            return o.calls.contains { $0.name == expected || accept.contains($0.name) } || (o.calls.isEmpty && accept.contains("none"))
        }
        print("ROUTER-EVAL-ANYCALL Apple Intelligence any-call tool-choice \(anyCall.count)/\(outcomes.count); first-call misses that an any-call count rescues: \(outcomes.filter { !$0.toolChoiceOK }.map(\.id).filter { id in anyCall.contains { $0.id == id } })")
        #expect(report.errors.isEmpty, "turns failed: \(report.errors.map { "\($0.id): \($0.error ?? "")" })")
    }

    /// MT4-7-FIX-2: after the user declines a download, what does Apple Intelligence say? Prints `DECLINE-REPLY`; asserts only
    /// that the decline reached the tool result and the turn did not fail. A reply that still claims success is a recorded known
    /// v1 limit (owner, 2026-11-12), not a failure.
    @Test(.enabled(if: AppleFoundationRouterTests.realEnabled, "needs Apple Intelligence ready and MLXUI_TEST_REAL_AFM=1"))
    func realAppleIntelligenceAfterADeclinedDownload() async throws {
        let cases = try RouterEvalSuite.load().cases
        let testCase = try #require(cases.first { $0.id == "m02" })          // describe_image needs a download; the mocked host declines
        let chatter = try #require(AppleFoundationAvailability.makeToolChatter(forceReal: true))
        let outcome = await RouterEvalRunner.runOne(router: AppleFoundationRouter(chatter: chatter), setup: .appleIntelligence, testCase: testCase)
        print("DECLINE-REPLY asks=\(outcome.installAsks) tool=\(outcome.firstTool ?? "none"): \(outcome.text.replacingOccurrences(of: "\n", with: " "))")
        #expect(outcome.error == nil)
    }

    // MARK: MLX chat models

    @Test(.enabled(if: !RouterEvalRealTests.mlxDirectories.isEmpty, "needs MLXUI_TEST_ROUTER_EVAL_MLX_DIRS=<dir>[:<dir>]"))
    func realMLXPass() async throws {
        let cases = try RouterEvalSuite.load().cases
        let gate = Self.env["MLXUI_TEST_ROUTER_EVAL_GATE"] ?? "both"
        var setups: [RouterEvalSetup] = []
        if gate == "off" || gate == "both" { setups.append(.mlxToday) }
        if gate == "on" || gate == "both" { setups.append(.mlxGated) }
        for path in Self.mlxDirectories {
            let directory = URL(fileURLWithPath: path)
            let name = directory.lastPathComponent
            let container = try await ModelContainerPool.loadFromDisk(directory: directory, architecture: nil)
            let router = Self.router(container: container, name: name)
            for setup in setups {
                let named = RouterEvalSetup(name: "\(name) \(setup.name)", router: setup.router, gateAmbiguousTools: setup.gateAmbiguousTools)
                let outcomes = await RouterEvalRunner.run(router: router, setup: named, cases: cases) { outcome in
                    print("ROUTER-EVAL-CASE \(named.name) \(outcome.id) \(String(format: "%.1f", outcome.seconds)) s: expected \(outcome.expectedTool ?? "none"), got \(outcome.firstTool ?? "no tool")")
                }
                let report = RouterEvalReport(routerName: named.name, outcomes: outcomes)
                announce(report)
                #expect(report.errors.isEmpty, "turns failed: \(report.errors.map { "\($0.id): \($0.error ?? "")" })")
            }
        }
    }

    /// Plan section 5: the cost of bringing an MLX router back after a worker evicted it. Load, one warm turn, release,
    /// load again, one turn. Run it alone (no other model loaded): the second load is page-cache warm unless the owner runs
    /// `sudo purge` between the two halves, which is a separate, owner-read cold figure.
    @Test(.enabled(if: !RouterEvalRealTests.mlxDirectories.isEmpty, "needs MLXUI_TEST_ROUTER_EVAL_MLX_DIRS=<dir>[:<dir>]"))
    func realMLXRouterReloadTime() async throws {
        let probe = try #require(try RouterEvalSuite.load().cases.first { $0.id == "n01" })
        for path in Self.mlxDirectories {
            let directory = URL(fileURLWithPath: path)
            let name = directory.lastPathComponent
            let clock = ContinuousClock()
            func seconds(_ d: Duration) -> Double { Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18 }

            let t0 = clock.now
            var container: ModelContainer? = try await ModelContainerPool.loadFromDisk(directory: directory, architecture: nil)
            let first = seconds(t0.duration(to: clock.now))
            let warm = await RouterEvalRunner.runOne(router: Self.router(container: try #require(container), name: name),
                                                     setup: .mlxToday, testCase: probe)
            container = nil                                              // the worker evicted the router

            let t1 = clock.now
            let reloaded = try await ModelContainerPool.loadFromDisk(directory: directory, architecture: nil)
            let reload = seconds(t1.duration(to: clock.now))
            let after = await RouterEvalRunner.runOne(router: Self.router(container: reloaded, name: name), setup: .mlxToday, testCase: probe)
            let m = RouterReloadMeasurement(directory: path, firstLoadSeconds: first, warmFirstEventSeconds: warm.firstEventSeconds,
                                            reloadSeconds: reload, firstEventAfterReloadSeconds: after.firstEventSeconds)
            print(m.line)
            #expect(reload > 0)
        }
    }

    private static func router(container: ModelContainer, name: String) -> MLXChatRouter {
        MLXChatRouter(
            identity: RouterIdentity(displayName: name, kind: .mlx, contextSize: nil),
            container: container, fileScope: nil, modelToolRuntime: nil, maxToolCalls: 6)
    }
}
