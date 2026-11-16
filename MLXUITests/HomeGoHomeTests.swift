import Foundation
import Testing
@testable import MLXUI

/// HC-FIX-1 (rulings I, J): every way of landing on Home pops pushed model/flow/workspace pages; a flow editor stays
/// (known pre-existing gap, ruling J). "Open Chat" in `ChatView` and the sidebar's Home row call `goHome()` directly.
@MainActor
@Suite(.serialized)
struct HomeGoHomeTests {
    private let model = makeEntry(id: "model-a", displayName: "Alpha")

    private func editor() -> FlowRoute {
        .editor(FlowEditTarget(flowID: "f1", name: "F", document: nil, savedText: nil))
    }

    @Test func goHomeDropsAPushedModelPageEvenWhenHomeIsAlreadySelected() {
        let app = AppState()
        app.selectedSection = .home
        app.route = [.model(model)]
        app.goHome()
        #expect(app.selectedSection == .home)
        #expect(app.route.isEmpty)
    }

    @Test func goHomeFromAnotherSectionSelectsHomeAndClearsTheRoute() {
        let app = AppState()
        app.selectedSection = .aiWorkflows
        app.route = [.model(model)]
        app.goHome()
        #expect(app.selectedSection == .home)
        #expect(app.route.isEmpty)
    }

    @Test func aFlowEditorStaysPoppedPagesBeneathItGo() {
        #expect(HomeRoutePolicy.remaining([]) == [])
        #expect(HomeRoutePolicy.remaining([.model(model)]) == [])
        #expect(HomeRoutePolicy.remaining([editor()]) == [editor()])
        #expect(HomeRoutePolicy.remaining([.model(model), editor()]) == [editor()])
        let app = AppState()
        app.route = [.model(model), editor()]
        app.goHome()
        #expect(app.route == [editor()])
        #expect(app.selectedSection == .home)
    }

    @Test func homeIsUnhighlightedOnlyWhilePagePushedOverIt() {
        #expect(SidebarHighlight.highlighted(selected: .home, routeIsEmpty: true) == .home)
        #expect(SidebarHighlight.highlighted(selected: .home, routeIsEmpty: false) == nil)
        for other in [SidebarItem.aiWorkflows, .overview, .server, .browse("chat-text")] {
            #expect(SidebarHighlight.highlighted(selected: other, routeIsEmpty: false) == other)
            #expect(SidebarHighlight.highlighted(selected: other, routeIsEmpty: true) == other)
        }
    }

    @Test func openInChatAndAskInChatLandOnHomeOverAPushedModelPage() {
        let app = AppState()
        ChatGate.overrideForTesting = true; defer { ChatGate.overrideForTesting = nil }
        app.selectedSection = .home
        app.route = [.model(model)]
        app.chat.acquire(.bench, model: model)
        #expect(app.openBenchInChat())
        #expect(app.selectedSection == .home)
        #expect(app.route.isEmpty)

        app.route = [.model(model)]
        #expect(app.askInChat(.text("a transcript", name: "transcript")))
        #expect(app.selectedSection == .home)
        #expect(app.route.isEmpty)
    }
}
