import Testing
import Foundation
@testable import MLXUI

// SC-4c — Home's one-line Local Server card: visibility, both sentences, the link target, and no per-card labels.

@Suite(.serialized)
@MainActor
struct HomeServerCardTests {
    private let model = "mlx-community/Card-A-4bit"
    private let other = "mlx-community/Card-B-4bit"

    private func controller(served: Set<String>, defaults: UserDefaults = UserDefaults(suiteName: "home-\(UUID().uuidString)")!)
        -> LocalServerController {
        let index = InstalledModelIndex(entries: [model, other].map {
            .init(id: $0, hfModelId: $0, kind: .llm, ramGB: 4, downloadSizeGB: 2) })
        let server = LocalServer(settings: ServeSettings(servedModelIDs: served), defaults: nil, listenPort: 0,
                                 environment: ServeEnvironment(servedModelIDs: { [] }, installed: { InstalledModelIndex(entries: []) },
                                                               appleFoundationReadiness: { nil }, created: 1))
        return LocalServerController(server: server, defaults: defaults, installed: { index }, appleFoundation: { nil },
                                     catalogLoaded: { true }, evict: { _ in }, evictPromptCache: { _ in })
    }

    @Test func theCardFollowsIsEnabled() {
        defer { LocalServerGate.overrideForTesting = nil; LocalServerGate.optionOverrideForTesting = nil }
        LocalServerGate.overrideForTesting = true
        LocalServerGate.optionOverrideForTesting = false
        let c = controller(served: [model])
        #expect(c.homeCard == nil)                                      // option off: no card even though a model is served
        LocalServerGate.optionOverrideForTesting = true
        #expect(c.homeCard != nil)
        LocalServerGate.overrideForTesting = false                      // hideLocalServer
        #expect(c.homeCard == nil)
    }

    @Test func theCardNeedsSomethingToHaveBeenServed() {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        #expect(controller(served: []).homeCard == nil)                // nothing ever served
        #expect(controller(served: [model]).homeCard != nil)
        let defaults = UserDefaults(suiteName: "home-\(UUID().uuidString)")!
        defaults.set(true, forKey: "localServerConnectShown")          // served once, list emptied since
        #expect(controller(served: [], defaults: defaults).homeCard?.sentence == "Not serving")
    }

    @Test func theRunningSentenceCountsModelsAndShowsTheAddress() {
        func card(_ n: Int) -> String? {
            HomeServerCard.make(enabled: true, everServed: true, servedCount: n, reachableCount: n, switchOn: true,
                                status: .running(port: 1212))?.sentence
        }
        #expect(card(2) == "Serving 2 models · 127.0.0.1:1212")
        #expect(card(1) == "Serving 1 model · 127.0.0.1:1212")
    }

    @Test func theOffSentenceNeedsAServedListAndTheSwitchOff() async {
        LocalServerGate.overrideForTesting = true
        defer { LocalServerGate.overrideForTesting = nil }
        let c = controller(served: [model])
        await c.setServerEnabled(false)
        #expect(c.homeCard?.sentence == "Server off")
        #expect(HomeServerCard.make(enabled: true, everServed: true, servedCount: 1, reachableCount: 1, switchOn: false,
                                    status: .stopped)?.sentence == "Server off")
        #expect(HomeServerCard.make(enabled: true, everServed: true, servedCount: 1, reachableCount: 1, switchOn: true,
                                    status: .failed(reason: "Port 1212 is in use"))?.sentence == "Port 1212 is in use")
    }

    @Test func theLinkOpensTheLocalServerScreen() {
        #expect(HomeServerCard.linkTitle == "Open Local Server")
        #expect(HomeServerCard.linkTarget == .server)
    }

    @Test func homesModelCardsCarryNoServingLabels() throws {
        // R51 "Serve just serves": the card ruling (owner, 2026-10-13) is a status card only. ModelCard must not learn about serving.
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Views/Browse/ModelCard.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        for word in ["localServer", "LocalServer", "Serving", "isServing", "servedIDs"] {
            #expect(!source.contains(word), "ModelCard mentions \(word)")
        }
        let home = try String(contentsOf: url.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Home/HomeView.swift"), encoding: .utf8)
        #expect(!home.contains("isServing") && !home.contains("Serving ·"))      // the card is the only serving text on Home
    }
}
