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
