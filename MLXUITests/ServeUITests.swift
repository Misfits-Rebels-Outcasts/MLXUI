import Testing
import Foundation
@testable import MLXUI

// S1-5 — the Local Server UI's logic: gate-driven policy (pure), the controller's
// toggle → settings → server intent, the Connect snippets.

private let sampleModel = "mlx-community/Qwen3-4B-4bit"

private func freshDefaults() -> UserDefaults {
    UserDefaults(suiteName: "serve-ui-tests-\(UUID().uuidString)")!
}

@MainActor
private func environment() -> ServeEnvironment {
    ServeEnvironment(
        servedModelIDs: { [sampleModel] },
        installed: { InstalledModelIndex(entries: []) },
        appleFoundationReadiness: { nil },
        created: 1)
}

/// An index with these HF ids installed as chat models.
private func installedIndex(_ ids: [String]) -> InstalledModelIndex {
    InstalledModelIndex(entries: ids.map { .init(id: $0, hfModelId: $0, kind: .llm, ramGB: 4) })
}

/// Since S1-5d the controller counts *reachable* models (served AND installed), so the helper's
/// default Mac has the ids these tests serve installed, and AFM ready.
@MainActor
private func makeController(listenPort: UInt16 = 0, served: Set<String> = [], defaults: UserDefaults = freshDefaults(),
                            installed: @escaping () -> InstalledModelIndex = {
                                installedIndex([sampleModel, "b", "mlx-community/other"])
                            },
                            evict: @escaping (String) async -> Void = { _ in }) -> LocalServerController {
    let server = LocalServer(settings: ServeSettings(servedModelIDs: served), defaults: nil,
                             listenPort: listenPort, environment: environment())
    return LocalServerController(server: server, defaults: defaults, installed: installed,
                                 appleFoundation: { .ready }, evict: evict)
}

private actor Evicted {
    private(set) var ids: [String] = []
    func add(_ id: String) { ids.append(id) }
}

// MARK: - Pure policy

@MainActor
struct LocalServerPolicyTests {
    @Test func servingAModelStartsAStoppedServerAndUnservingTheLastStopsIt() {
        let go = { (served: Int, status: LocalServer.Status) in
            LocalServerPolicy.action(gateAvailable: true, enabled: true, servedCount: served, status: status)
        }
        #expect(go(1, .stopped) == .start)
        #expect(go(1, .failed(reason: "port in use")) == .start)        // a retry
        #expect(go(2, .running(port: 1212)) == .none)
        #expect(go(1, .starting) == .none)
        #expect(go(0, .running(port: 1212)) == .stop)                   // un-served the last one
        #expect(go(0, .stopped) == .none)
    }

    @Test func switchingTheServerOffStopsItEvenWithModelsServed() {
        #expect(LocalServerPolicy.action(gateAvailable: true, enabled: false, servedCount: 3, status: .running(port: 1212)) == .stop)
        #expect(LocalServerPolicy.action(gateAvailable: true, enabled: false, servedCount: 3, status: .stopped) == .none)
    }

    @Test func aClosedGateNeverStartsAndStopsAnythingRunning() {
        #expect(LocalServerPolicy.action(gateAvailable: false, enabled: true, servedCount: 5, status: .stopped) == .none)
        #expect(LocalServerPolicy.action(gateAvailable: false, enabled: true, servedCount: 5, status: .running(port: 1212)) == .stop)
    }

    @Test func autoStartNeedsTheGateTheSwitchAndServedModels() {
        #expect(LocalServerPolicy.shouldAutoStart(gateAvailable: true, enabled: true, servedCount: 1))
        #expect(!LocalServerPolicy.shouldAutoStart(gateAvailable: false, enabled: true, servedCount: 1))   // hidden: skipped
        #expect(!LocalServerPolicy.shouldAutoStart(gateAvailable: true, enabled: false, servedCount: 1))
        #expect(!LocalServerPolicy.shouldAutoStart(gateAvailable: true, enabled: true, servedCount: 0))
    }

    @Test(arguments: [
        (true, LocalServer.Status.running(port: 1212), false),   // serving: stay running
        (true, LocalServer.Status.starting, false),
        (true, LocalServer.Status.stopped, true),                // not serving: quit as before
        (true, LocalServer.Status.failed(reason: "x"), true),
        (false, LocalServer.Status.running(port: 1212), true),   // hidden: always quit as before
        (false, LocalServer.Status.stopped, true),
    ])
    func theTerminateDecisionIsAPureFunctionOfGateAndStatus(gate: Bool, status: LocalServer.Status, terminates: Bool) {
        #expect(LocalServerPolicy.shouldTerminateAfterLastWindowClosed(gateAvailable: gate, status: status) == terminates)
    }

    @Test func theLocalServerPaneFollowsTheGateAndOnlyTheGate() {
        let hidden = LocalServerPolicy.visibleSettingsPanes(gateAvailable: false, hideProvidersPrivacy: false)
        #expect(hidden == [.models, .providers, .agentTools, .privacy])                       // today's four
        let shown = LocalServerPolicy.visibleSettingsPanes(gateAvailable: true, hideProvidersPrivacy: false)
        #expect(shown == [.models, .providers, .agentTools, .privacy, .localServer])          // a fifth
        // R12: not behind hideProvidersPrivacy.
        let providersHidden = LocalServerPolicy.visibleSettingsPanes(gateAvailable: true, hideProvidersPrivacy: true)
        #expect(providersHidden == [.models, .agentTools, .localServer])
        #expect(LocalServerPolicy.visibleSettingsPanes(gateAvailable: false, hideProvidersPrivacy: true) == [.models, .agentTools])
    }

    @Test func aStalePaneSelectionLandsOnModels() {
        let hidden = LocalServerPolicy.visibleSettingsPanes(gateAvailable: false, hideProvidersPrivacy: false)
        #expect(LocalServerPolicy.resolvedPane(.localServer, visible: hidden) == .models)
        #expect(LocalServerPolicy.resolvedPane(.privacy, visible: hidden) == .privacy)
    }

    @Test func theStatusSentences() {
        func s(_ status: LocalServer.Status, _ n: Int, _ enabled: Bool = true) -> String {
            LocalServerPolicy.statusSentence(status: status, servedCount: n, enabled: enabled)
        }
        #expect(s(.running(port: 1212), 2) == "Serving 2 models at 127.0.0.1:1212")
        #expect(s(.running(port: 1212), 1) == "Serving 1 model at 127.0.0.1:1212")
        #expect(s(.failed(reason: "Port 1212 is already in use."), 1) == "Port 1212 is already in use.")
        #expect(s(.stopped, 0) == "Not serving — turn on Serve for a model to start.")
        #expect(s(.stopped, 2, false) == "The server is off.")
        #expect(s(.starting, 1) == "Starting…")
    }
}

// MARK: - Controller

@Suite(.serialized)
@MainActor
struct LocalServerControllerTests {
    @Test func theFirstServeEverStartsTheServerAndOpensConnectOnce() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let defaults = freshDefaults()
        let controller = makeController(defaults: defaults)
        defer { Task { await controller.server.stop() } }

        await controller.setServed(sampleModel, true)
        guard case .running = controller.server.status else { Issue.record("not running: \(controller.server.status)"); return }
        #expect(controller.isServing(sampleModel))
        #expect(controller.showConnect && controller.connectModelID == sampleModel)
        #expect(defaults.bool(forKey: "localServerConnectShown"))                  // the first-run flag persisted

        controller.showConnect = false
        await controller.setServed("mlx-community/other", true)                    // a second serve: no sheet
        #expect(!controller.showConnect)

        let relaunched = LocalServerController(server: LocalServer(settings: ServeSettings(), defaults: nil, listenPort: 0,
                                                                   environment: environment()), defaults: defaults)
        #expect(relaunched.hasShownConnect)                                         // the flag survives a relaunch
    }

    @Test func unservingTheLastModelStopsTheServer() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController()
        await controller.setServed(sampleModel, true)
        await controller.setServed("b", true)
        await controller.setServed(sampleModel, false)
        #expect(controller.isRunning)                                              // one still served
        await controller.setServed("b", false)
        #expect(controller.server.status == .stopped)
        #expect(controller.servedIDs.isEmpty)
    }

    @Test func withTheGateClosedNothingChangesAndNothingStarts() async {
        LocalServerGate.overrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController()
        await controller.setServed(sampleModel, true)
        #expect(!controller.isServing(sampleModel))
        #expect(controller.server.status == .stopped)
        #expect(!controller.showConnect)
        controller.presentConnect(for: sampleModel)
        #expect(!controller.showConnect)
    }

    @Test func autoStartResumesWhenModelsWereLeftServedAndIsSkippedWhenHidden() async {
        let controller = makeController(served: [sampleModel])
        LocalServerGate.overrideForTesting = false
        await controller.autoStart()
        #expect(controller.server.status == .stopped)                              // hidden: skipped

        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        defer { Task { await controller.server.stop() } }
        await controller.autoStart()
        #expect(controller.isRunning)
    }

    @Test func autoStartDoesNothingWithNothingServedOrWhenSwitchedOff() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let empty = makeController()
        await empty.autoStart()
        #expect(empty.server.status == .stopped)

        let off = makeController(served: [sampleModel])
        await off.setServerEnabled(false)
        await off.autoStart()
        #expect(off.server.status == .stopped)
        #expect(!off.enabled)
    }

    @Test func switchingOffKeepsTheListAndSwitchingOnResumes() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(served: [sampleModel])
        defer { Task { await controller.server.stop() } }
        await controller.setServerEnabled(true)
        #expect(controller.isRunning)
        await controller.setServerEnabled(false)
        #expect(controller.server.status == .stopped)
        #expect(controller.servedIDs == [sampleModel])                             // the list is kept
        await controller.setServerEnabled(true)
        #expect(controller.isRunning)
    }

    @Test func aPortInUseIsAClearFailureAndTheNextServeRetries() async throws {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let first = makeController()
        await first.setServed(sampleModel, true)
        defer { Task { await first.server.stop() } }
        guard case .running(let taken) = first.server.status else { Issue.record("first not running"); return }

        let second = makeController(listenPort: taken)
        defer { Task { await second.server.stop() } }
        await second.setServed(sampleModel, true)
        guard case .failed(let reason) = second.server.status else { Issue.record("expected failure: \(second.server.status)"); return }
        #expect(reason.contains("\(taken)"))                                       // names the port
        #expect(second.statusSentence == reason)

        await first.server.stop()                                                  // the port frees up…
        await second.setServed("again", true)                                      // …and the next change retries
        #expect(second.isRunning)
    }

    @Test func aNewPortIsSavedAndClampedAndARunningServerRestarts() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(served: [sampleModel])
        defer { Task { await controller.server.stop() } }
        await controller.setServerEnabled(true)
        await controller.setPort(80)
        #expect(controller.server.settings.port == 1024)                           // clamped (S1-2 bounds)
        #expect(controller.isRunning)                                              // restarted (port override keeps it free)
    }
}

// MARK: - Connect snippets

struct ConnectSnippetsTests {
    @Test func theBaseURLIsTheIPv4Form() {
        #expect(ConnectSnippets.baseURL(port: 1212) == "http://127.0.0.1:1212/v1")
    }

    @Test func everyHarnessCarriesTheBaseURLOrPortAndTheModel() {
        for harness in ConnectSnippets.Harness.allCases {
            let text = ConnectSnippets.text(for: harness, port: 4321, modelID: sampleModel)
            #expect(text.contains(sampleModel), "\(harness)")
            #expect(text.contains("4321"), "\(harness)")
        }
    }

    @Test func curlStreamsAndPointsAtChatCompletions() {
        let text = ConnectSnippets.text(for: .curl, port: 1212, modelID: sampleModel)
        #expect(text.hasPrefix("curl -N http://127.0.0.1:1212/v1/chat/completions"))
        #expect(text.contains(#""stream": true"#))
    }

    @Test func openCodeIsAnOpenAICompatibleProviderWithABaseURLUnderOptions() throws {
        let text = ConnectSnippets.text(for: .openCode, port: 1212, modelID: sampleModel)
        let object = try #require(try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])   // valid JSON
        let provider = try #require((object["provider"] as? [String: Any])?["mlxui"] as? [String: Any])
        #expect(provider["npm"] as? String == "@ai-sdk/openai-compatible")
        #expect((provider["options"] as? [String: Any])?["baseURL"] as? String == "http://127.0.0.1:1212/v1")
        #expect((provider["models"] as? [String: Any])?[sampleModel] != nil)
    }

    @Test func xcodeIsTheLocallyHostedPortRecipe() {
        let text = ConnectSnippets.text(for: .xcode, port: 1212, modelID: sampleModel)
        #expect(text.contains("Intelligence") && text.contains("Locally Hosted") && text.contains("Port to 1212"))
    }

    @Test func pythonUsesTheOpenAIClientWithABaseURL() {
        let text = ConnectSnippets.text(for: .python, port: 1212, modelID: sampleModel)
        #expect(text.contains(#"OpenAI(base_url="http://127.0.0.1:1212/v1", api_key="unused")"#))
    }

    @Test func aModelIDWithQuotesIsEscaped() throws {
        let text = ConnectSnippets.text(for: .openCode, port: 1212, modelID: #"we"ird\id"#)
        #expect(try JSONSerialization.jsonObject(with: Data(text.utf8)) is [String: Any])
    }

    @Test func contextLengthComesFromTheConfigAndFallsBackHonestly() {
        func length(_ json: String?) -> Int? { ConnectSnippets.contextLength(configJSON: json.map { Data($0.utf8) }) }
        #expect(length(#"{"max_position_embeddings": 40960}"#) == 40960)
        #expect(length(#"{"text_config": {"max_position_embeddings": 131072}}"#) == 131072)
        #expect(length(#"{"hidden_size": 8}"#) == nil)
        #expect(length(nil) == nil)
        #expect(ConnectSnippets.fitnessLine(contextLength: 40960).hasPrefix("Context length: 40,960 tokens."))
        #expect(ConnectSnippets.fitnessLine(contextLength: nil).contains("not stated"))
        #expect(ConnectSnippets.fitnessLine(contextLength: 8192).contains("plain text"))
    }

    @Test func theRequestLogLineHoldsMetadataOnly() {
        let entry = RequestLogEntry(time: Date(timeIntervalSince1970: 0), userAgent: "ua/1", model: "m", promptTokens: 3,
                                    completionTokens: 4, durationSeconds: 1.25, status: 200, toolsIgnored: true)
        let line = RequestLogList.line(entry)
        #expect(line.contains("200") && line.contains("m") && line.contains("3→4 tok") && line.contains("1.2s") || line.contains("1.3s"))
        #expect(line.contains("ua/1") && line.contains("tools ignored"))
    }
}

// MARK: - Live wiring (S1-5's by-hand run found the model directory built from the unslugged HF id)

struct LiveMLXBackendWiringTests {
    @Test func theLiveBackendPointsAtTheSlugDirectoryInstallManagerWrites() {
        let entry = InstalledModelIndex.Entry(id: "mlx-community--Qwen3-4B-4bit", hfModelId: "mlx-community/Qwen3-4B-4bit",
                                              kind: .llm, ramGB: 3)
        let backend = ServeEnvironment.mlxBackend(for: entry)
        #expect(backend.directory == ModelStore.shared.directory(forModelID: "mlx-community--Qwen3-4B-4bit"))
        #expect(backend.directory.lastPathComponent == "mlx-community--Qwen3-4B-4bit")          // one component, no nested "mlx-community/"
        #expect(backend.directory.deletingLastPathComponent() == ModelStore.shared.modelsDirectory)
        #expect(backend.footprintBytes == Int64(3 * 1_073_741_824))
    }
}

// MARK: - The detail page's Serve toggle (S1-5b)

struct ServeToggleVisibilityTests {
    private func show(_ state: InstallState, disk: Bool = false, kind: RunnerKind = .llm, gate: Bool = true) -> Bool {
        ServeToggleVisibility.shouldShow(state: state, isInstalledOnDisk: disk, runnerKind: kind, gateAvailable: gate)
    }

    @Test func aModelInTheInstalledStateShowsTheToggle() {
        // The S1-B hand-check bug: `.installed` is a different branch of the page than "idle but on disk".
        #expect(show(.installed))
        #expect(show(.installed, disk: true))
    }

    @Test func anIdleModelShowsItOnlyWhenItsFilesAreOnDisk() {
        #expect(show(.idle, disk: true))
        #expect(!show(.idle, disk: false))
    }

    @Test func nothingShowsWhileInstallingOrFailedOrGated() {
        for state in [InstallState.resolving, .downloading(progress: 0.5, downloaded: 1, total: 2), .verifying,
                      .error("x", canRetry: true), .needsAuth("x")] {
            #expect(!show(state, disk: true))
        }
    }

    @Test func onlyMLXChatModelsCanBeServed() {
        for kind in [RunnerKind.asr, .tts, .vision, .embedding, .ocr, .image, .unsupported] {
            #expect(!show(.installed, kind: kind), "\(kind)")
        }
    }

    @Test func aClosedGateHidesItInEveryState() {
        #expect(!show(.installed, gate: false))
        #expect(!show(.idle, disk: true, gate: false))
    }
}

// MARK: - S1-5c: "reachable now" (owner ruling 1)

struct ServeReachTests {
    @Test(arguments: [(true, true, true), (true, false, false), (false, true, false), (false, false, false)])
    func reachableNeedsTheListAndTheMasterSwitch(inList: Bool, master: Bool, reachable: Bool) {
        let served: Set<String> = inList ? ["m"] : []
        #expect(ServeReach.isReachable(id: "m", served: served, masterEnabled: master) == reachable)
    }

    @Test func theRuleIsTheSameForAppleFoundationModels() {
        #expect(ServeReach.isReachable(id: ServedModels.appleFoundationID, served: [ServedModels.appleFoundationID], masterEnabled: true))
        #expect(!ServeReach.isReachable(id: ServedModels.appleFoundationID, served: [ServedModels.appleFoundationID], masterEnabled: false))
    }
}

@Suite(.serialized)
@MainActor
struct ServeReachControllerTests {
    @Test func switchingTheMasterOffKeepsTheListButNothingIsReachableAndOnRestoresEverything() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(served: [sampleModel, ServedModels.appleFoundationID, "mlx-community/other"])
        defer { Task { await controller.server.stop() } }
        await controller.setServerEnabled(true)
        #expect(controller.isReachable(sampleModel) && controller.isReachable(ServedModels.appleFoundationID))

        await controller.setServerEnabled(false)
        #expect(controller.servedIDs.count == 3)                                    // the list is kept
        #expect(!controller.isReachable(sampleModel))                               // …but none is reachable now
        #expect(!controller.isReachable(ServedModels.appleFoundationID))
        #expect(controller.isServing(sampleModel))                                  // still "in the list"

        await controller.setServerEnabled(true)                                     // restores every one
        #expect(["mlx-community/other", sampleModel, ServedModels.appleFoundationID].allSatisfy(controller.isReachable))
    }

    @Test func switchingServeOnFromAModelPageTurnsTheMasterSwitchOn() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(served: ["mlx-community/other"])
        defer { Task { await controller.server.stop() } }
        await controller.setServerEnabled(false)
        #expect(!controller.isReachable(sampleModel))
        await controller.setServed(sampleModel, true)                               // R24
        #expect(controller.enabled && controller.isRunning)
        #expect(controller.isReachable(sampleModel))
        #expect(controller.isReachable("mlx-community/other"))                      // the earlier one is back too
    }
}

// MARK: - S1-5c: the memory guard (owner ruling 2)

struct ServeMemoryTests {
    @Test func aModelThatExceedsThisMacsRAMBlocksServeWithTheReason() throws {
        let reason = try #require(ServeMemory.blockedReason(modelRAMGB: 24, systemRAMGB: 16))
        #expect(reason.contains("Needs 24 GB RAM") && reason.contains("16 GB"))
        #expect(ServeMemory.blockedReason(modelRAMGB: 16, systemRAMGB: 16) == nil)         // equal fits (ModelEntry.exceedsRAM is `>`)
        #expect(ServeMemory.blockedReason(modelRAMGB: 6.75, systemRAMGB: 18) == nil)
    }

    @Test func theBlockMatchesTheRuleThatReplacesRunWithNeedsNGB() {
        for ram in [8.0, 16.0, 16.5, 24.0] {
            let model = makeEntry(ramGB: ram)
            #expect((ServeMemory.blockedReason(modelRAMGB: model.ramGB, systemRAMGB: 16) != nil) == model.exceedsRAM(16), "\(ram)")
        }
    }

    @Test func aBlockedToggleIsDisabledUnlessTheModelIsAlreadyReachable() {
        #expect(ServeMemory.isToggleDisabled(blockedReason: "x", isReachable: false))
        #expect(!ServeMemory.isToggleDisabled(blockedReason: "x", isReachable: true))       // you can always turn it off
        #expect(!ServeMemory.isToggleDisabled(blockedReason: nil, isReachable: false))
    }

    @Test func theTooLargeSentenceNamesTheModelItsSizeAndTheBudget() {
        #expect(ServeMemory.tooLargeSentence(model: "mlx-community/Big-70B", modelRAMGB: 42, capacityBytes: Int64(10.8 * 1_073_741_824))
                == "mlx-community/Big-70B needs 42.0 GB; the Local Server can use 10.8 GB on this Mac.")
    }

    @Test func exceedsBudgetIsStrictlyLargerThanTheWholeBudget() {
        let ten = Int64(10 * 1_073_741_824)
        #expect(ServeMemory.exceedsBudget(modelRAMGB: 10.5, capacityBytes: ten))
        #expect(!ServeMemory.exceedsBudget(modelRAMGB: 10, capacityBytes: ten))
        #expect(!ServeMemory.exceedsBudget(modelRAMGB: 3, capacityBytes: ten))
    }

    private func index(_ entries: [(String, Double, RunnerKind)]) -> InstalledModelIndex {
        InstalledModelIndex(entries: entries.map { .init(id: $0.0.replacingOccurrences(of: "/", with: "--"), hfModelId: $0.0, kind: $0.2, ramGB: $0.1) })
    }

    @Test func theNoteAppearsOnlyWhenTheServedModelsTogetherExceedTheBudget() {
        let ten = Int64(10 * 1_073_741_824)
        let installed = index([("a/one", 6, .llm), ("a/two", 6, .llm), ("a/three", 3, .llm)])
        #expect(ServeMemory.note(servedIDs: ["a/one", "a/two"], installed: installed, capacityBytes: ten) == ServeMemory.doesNotAllFitNote)
        #expect(ServeMemory.note(servedIDs: ["a/one"], installed: installed, capacityBytes: ten) == nil)
        #expect(ServeMemory.note(servedIDs: ["a/one", "a/three"], installed: installed, capacityBytes: ten) == nil)   // 6 + 3 = 9 GB fits in 10
        #expect(ServeMemory.note(servedIDs: ["a/one", "a/two", "a/three"], installed: installed, capacityBytes: ten) == ServeMemory.doesNotAllFitNote)
    }

    @Test func theNoteCountsOnlyServedInstalledMLXChatModels() {
        let ten = Int64(10 * 1_073_741_824)
        let installed = index([("a/one", 6, .llm), ("a/embed", 8, .embedding), ("a/unserved", 8, .llm)])
        // an installed-but-unserved model, a non-chat model, a served-but-not-installed id and Apple Foundation Models don't count
        let served: Set<String> = ["a/one", "a/embed", "a/gone", ServedModels.appleFoundationID]
        #expect(ServeMemory.note(servedIDs: served, installed: installed, capacityBytes: ten) == nil)
    }

    @MainActor
    @Test func theControllerShowsTheNoteFromItsInjectedCatalogAndBudget() {
        let installed = index([("a/one", 6, .llm), ("a/two", 6, .llm)])
        let server = LocalServer(settings: ServeSettings(servedModelIDs: ["a/one", "a/two"]), defaults: nil, listenPort: 0,
                                 environment: environment())
        let tight = LocalServerController(server: server, defaults: freshDefaults(), installed: { installed },
                                          budgetBytes: Int64(10 * 1_073_741_824))
        #expect(tight.memoryNote == "These models don't all fit in memory at once — switching between them reloads each one.")
        let roomy = LocalServerController(server: server, defaults: freshDefaults(), installed: { installed },
                                          budgetBytes: Int64(32 * 1_073_741_824))
        #expect(roomy.memoryNote == nil)
    }
}

// The 503 for a served model larger than the whole budget (route, no socket).
struct ServeMemoryRouteTests {
    private static let body = #"{"model":"a/big","messages":[{"role":"user","content":"hi"}]}"#

    private func env(ramGB: Double, capacityGB: Double, backend: FakeBackend, kind: RunnerKind = .llm) -> ServeEnvironment {
        var env = ServeEnvironment(
            servedModelIDs: { ["a/big"] },
            installed: { InstalledModelIndex(entries: [.init(id: "a--big", hfModelId: "a/big", kind: kind, ramGB: ramGB)]) },
            appleFoundationReadiness: { nil }, created: 1,
            backendFor: { _ in backend })
        env.memoryCapacityBytes = Int64(capacityGB * 1_073_741_824)
        return env
    }

    private func call(_ env: ServeEnvironment) async -> ServeResponse? {
        let data = Data(Self.body.utf8)
        let head = ServeRequestHead(method: "POST", path: "/v1/chat/completions", host: "127.0.0.1:1212", origin: nil,
                                    transferEncoding: nil, contentLength: String(data.count), userAgent: nil)
        if case .full(let response) = await ServeAPI.handle(head: head, guard: RequestGuard(port: 1212), environment: env, body: { data }) {
            return response
        }
        return nil
    }

    @Test func aModelLargerThanTheWholeBudgetIs503WithTheSentenceAndIsNeverLoaded() async throws {
        let backend = FakeBackend([.text("never"), .finish(.stop)])
        let response = try #require(await call(env(ramGB: 42, capacityGB: 10.8, backend: backend)))
        #expect(response.status == 503)
        let error = try #require((try JSONSerialization.jsonObject(with: response.body) as? [String: Any])?["error"] as? [String: Any])
        #expect(error["message"] as? String == "a/big needs 42.0 GB; the Local Server can use 10.8 GB on this Mac.")
        #expect(error["code"] as? String == "model_too_large")
        #expect(response.headers["Retry-After"] == nil)                       // waiting won't make it fit
        #expect(backend.requests.isEmpty)                                     // refused before loading
    }

    @Test func aModelThatFitsTheBudgetIsServedNormally() async throws {
        let backend = FakeBackend([.text("ok"), .finish(.stop)])
        let response = try #require(await call(env(ramGB: 6.75, capacityGB: 10.8, backend: backend)))
        #expect(response.status == 200)
        #expect(backend.requests.count == 1)
    }

    @Test func theBoundaryIsTheWholeBudget() async throws {
        let at = try #require(await call(env(ramGB: 10, capacityGB: 10, backend: FakeBackend([.finish(.stop)]))))
        #expect(at.status == 200)
        let over = try #require(await call(env(ramGB: 10.01, capacityGB: 10, backend: FakeBackend([.finish(.stop)]))))
        #expect(over.status == 503)
    }
}


// MARK: - S1-5d: uninstall, one reachable set, launch reconcile

@Suite(.serialized)
@MainActor
struct ServeReachableSetTests {
    private final class Disk: @unchecked Sendable {
        var ids: [String]
        init(_ ids: [String]) { self.ids = ids }
    }

    @Test func uninstallingAServedModelUnservesItEvictsItAndLeavesTheOthersServing() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let disk = Disk(["mlx-community/Ministral", "mlx-community/Qwen3-8B"])
        let evicted = Evicted()
        let controller = makeController(installed: { installedIndex(disk.ids) }, evict: { await evicted.add($0) })
        defer { Task { await controller.server.stop() } }
        await controller.setServed("mlx-community/Ministral", true)
        await controller.setServed("mlx-community/Qwen3-8B", true)
        #expect(controller.statusSentence.hasPrefix("Serving 2 models"))

        disk.ids = ["mlx-community/Qwen3-8B"]                                       // Ministral deleted
        await controller.modelUninstalled("mlx-community/Ministral")
        #expect(controller.servedIDs == ["mlx-community/Qwen3-8B"])
        #expect(controller.reachableIDs == ["mlx-community/Qwen3-8B"])
        #expect(controller.statusSentence.hasPrefix("Serving 1 model at"))
        #expect(await evicted.ids == ["mlx-community/Ministral"])
        #expect(controller.isRunning)
    }

    @Test func uninstallingTheLastServedModelStopsTheServer() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let disk = Disk([sampleModel])
        let controller = makeController(installed: { installedIndex(disk.ids) })
        await controller.setServed(sampleModel, true)
        #expect(controller.isRunning)
        disk.ids = []
        await controller.modelUninstalled(sampleModel)
        #expect(controller.servedIDs.isEmpty)
        #expect(controller.server.status == .stopped)
    }

    @Test func aRepoStillInstalledUnderAnotherCardIsLeftServed() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let evicted = Evicted()
        let controller = makeController(installed: { installedIndex([sampleModel]) }, evict: { await evicted.add($0) })
        defer { Task { await controller.server.stop() } }
        await controller.setServed(sampleModel, true)
        await controller.modelUninstalled(sampleModel)                              // directory was kept
        #expect(controller.isServing(sampleModel))
        #expect(await evicted.ids.isEmpty)
    }

    @Test func theStatusSentenceListAndPolicyAllCountReachableNotStored() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        // Stored: an installed model, a deleted one, and AFM while Apple Intelligence isn't ready.
        let server = LocalServer(settings: ServeSettings(servedModelIDs: [sampleModel, "gone/model", ServedModels.appleFoundationID]),
                                 defaults: nil, listenPort: 0, environment: environment())
        let controller = LocalServerController(server: server, defaults: freshDefaults(),
                                               installed: { installedIndex([sampleModel]) },
                                               appleFoundation: { .needsSetup(reason: "off", action: nil) }, evict: { _ in })
        defer { Task { await controller.server.stop() } }
        #expect(controller.servedIDs.count == 3)
        #expect(controller.reachableIDs == [sampleModel])                           // == what /v1/models lists
        #expect(controller.reachableIDs == Set(ServedModels.ids(served: controller.servedIDs, installed: installedIndex([sampleModel]),
                                                                 appleFoundation: .needsSetup(reason: "off", action: nil))))
        #expect(controller.statusSentence == "Not serving — turn on Serve for a model to start."
                || controller.statusSentence == "The server is stopped.")
        await controller.autoStart()
        #expect(controller.statusSentence.hasPrefix("Serving 1 model at"))
        #expect(!controller.isReachable("gone/model"))
        #expect(!controller.isReachable(ServedModels.appleFoundationID))
    }

    @Test func nothingReachableMeansNoAutoStartEvenWithAStoredList() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(served: ["gone/model"], installed: { installedIndex([]) })
        await controller.autoStart()
        #expect(controller.server.status == .stopped)
    }

    @Test func launchDropsServedIdsThatAreNoLongerInstalledAndEvictsThem() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let evicted = Evicted()
        let controller = makeController(served: [sampleModel, "gone/model", ServedModels.appleFoundationID],
                                        installed: { installedIndex([sampleModel]) }, evict: { await evicted.add($0) })
        defer { Task { await controller.server.stop() } }
        await controller.autoStart()
        #expect(controller.servedIDs == [sampleModel, ServedModels.appleFoundationID])   // AFM is kept
        #expect(await evicted.ids == ["gone/model"])
        #expect(controller.isRunning)
    }

    @Test func withTheGateClosedLaunchLeavesThePersistedListAlone() async {
        LocalServerGate.overrideForTesting = false
        defer { LocalServerGate.overrideForTesting = nil }
        let controller = makeController(served: ["gone/model"], installed: { installedIndex([]) })
        await controller.autoStart()
        await controller.modelUninstalled("gone/model")
        #expect(controller.servedIDs == ["gone/model"])
        #expect(controller.server.status == .stopped)
    }
}
