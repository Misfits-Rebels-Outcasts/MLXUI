import Testing
@testable import MLXUI

/// CD-2 — the pure draft rule behind a card click.
struct ChatCapabilityGridTests {
    @Test func anEmptyDraftBecomesTheStarter() {
        #expect(ChatCardDraft.draft(afterPicking: "Draw ", current: "") == "Draw ")
        #expect(ChatCardDraft.draft(afterPicking: "Draw ", current: "  \n") == "Draw ")
    }

    @Test func aTypedDraftKeepsItsTextAndGetsTheStarterAfterASpace() {
        #expect(ChatCardDraft.draft(afterPicking: "Translate into French: ", current: "hello")
                == "hello Translate into French: ")
    }

    @Test func aDraftEndingInWhitespaceIsNotGivenAnotherSeparator() {
        #expect(ChatCardDraft.draft(afterPicking: "Draw ", current: "first\n") == "first\nDraw ")
        #expect(ChatCardDraft.draft(afterPicking: "Draw ", current: "first ") == "first Draw ")
    }
}

/// CD-4 — the "no chat model" copy.
struct ChatNoModelTextTests {
    @Test func copyIsTheBacklogs() {
        #expect(ChatNoModelText.title == "Chat needs one language model")
        #expect(ChatNoModelText.body == "It understands your request, then downloads and runs OCR, speech and image models for you as needed.")
    }
}

/// CD-FIX-1 — one grid, and text that follows the message size.
struct ChatGridLayoutTests {
    private func card(_ id: String) -> ChatCapabilityCard {
        ChatCapabilityCard(id: id, title: id, symbol: "x", starterPrompt: "", takesFile: nil, state: .cantRun(reason: ""))
    }

    @Test func visibleCardsAreOneListThatGrowsWhenExpanded() {
        let primary = ["a", "b", "c"].map(card), more = ["d", "e"].map(card)
        #expect(ChatGridLayout.visibleCards(primary: primary, more: more, expanded: false).map(\.id) == ["a", "b", "c"])
        #expect(ChatGridLayout.visibleCards(primary: primary, more: more, expanded: true).map(\.id) == ["a", "b", "c", "d", "e"])
        #expect(ChatGridLayout.visibleCards(primary: [], more: [], expanded: true).isEmpty)
    }

    @Test func everyGridSizeIsItsBasePlusTheMessageStep() {
        for size in ChatTextSize.allCases {
            let step = size.points(for: .message) - ChatTextSize.Role.message.standardPoints
            #expect(step == size.delta)
            for role in ChatGridText.Role.allCases {
                #expect(ChatGridText.points(role, size: size) == ChatGridText.basePoints(role) + size.delta)
            }
            // the card title is exactly message text at every step
            #expect(ChatGridText.points(.cardTitle, size: size) == size.points(for: .message))
        }
        #expect(ChatGridText.points(.status, size: .extraLarge) - ChatGridText.points(.status, size: .small) == 6)
    }
}
