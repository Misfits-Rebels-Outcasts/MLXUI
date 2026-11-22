import Testing
import CoreGraphics
@testable import MLXUI

/// ER-FIX-1: the outbox list's height is definite and capped, so a print entry's "Print…" button is never squeezed to a sliver.
/// These pin the metrics only; the view itself (`FlowListView.outboxDisclosure`) is not rendered here, and the on-screen fix is
/// owner-checked (SM-ER-6).
struct OutboxLayoutTests {

    @Test func noEntriesTakeNoHeight() {
        #expect(OutboxLayout.listHeight(for: []) == 0)
        #expect(!OutboxLayout.scrolls(for: []))
    }

    @Test func onePrintEntryIsGivenRoomForItsButtonRow() {
        let h = OutboxLayout.listHeight(for: [("print", false)])
        #expect(h == OutboxLayout.printEntryHeight)
        #expect(h > OutboxLayout.plainEntryHeight, "a print entry carries a button row a send entry doesn't")
        // The old failure: the whole disclosure was ~2 pt. The list is now a definite, comfortably tall height.
        #expect(h >= 80)
    }

    @Test func aFailedAttemptAddsItsErrorLine() {
        #expect(OutboxLayout.listHeight(for: [("print", true)]) == OutboxLayout.printEntryHeight + OutboxLayout.printErrorLineHeight)
        #expect(OutboxLayout.listHeight(for: [("send", true)]) == OutboxLayout.plainEntryHeight, "only a print entry shows an error line")
    }

    @Test func aLongOutboxIsCappedAndScrolls() {
        let many = Array(repeating: ("print", false), count: 20)
        #expect(OutboxLayout.listHeight(for: many) == OutboxLayout.maxListHeight)
        #expect(OutboxLayout.scrolls(for: many))
        #expect(!OutboxLayout.scrolls(for: [("print", false), ("send", false)]))
    }

    @Test func theCapLeavesTheRowsAboveRoomOnADefaultWindow() {
        // The cap is a fraction of a default flow window's height (about 700 pt): the outbox never takes the screen.
        #expect(OutboxLayout.maxListHeight <= 700 / 3)
    }
}
