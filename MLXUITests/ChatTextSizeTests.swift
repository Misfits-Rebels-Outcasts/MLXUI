import SwiftUI
import Testing
@testable import MLXUI

struct ChatTextSizeTests {
    @Test func standardMatchesTodaysMacOSSizes() {
        #expect(ChatTextSize.standard.points(for: .message) == 13)      // .body
        #expect(ChatTextSize.standard.points(for: .toolResult) == 10)   // .caption
        #expect(ChatTextSize.standard.points(for: .listTitle) == 12)    // .callout
        #expect(ChatTextSize.standard.points(for: .listDetail) == 10)   // .caption2
    }

    @Test func standardOverridesNothing() {
        for role in [ChatTextSize.Role.message, .toolResult, .listTitle, .listDetail] {
            #expect(ChatTextSize.standard.overridePoints(for: role) == nil)
        }
    }

    @Test(arguments: [
        (ChatTextSize.small, 11.0), (.standard, 13.0), (.large, 15.0), (.extraLarge, 17.0),
    ])
    func messagePointsPerStep(size: ChatTextSize, points: Double) {
        #expect(size.points(for: .message) == points)
        #expect(size == .standard || size.overridePoints(for: .message) == points)
    }

    @Test func stepsClampAtBothEnds() {
        #expect(ChatTextSize.standard.steppedUp == .large)
        #expect(ChatTextSize.large.steppedUp == .extraLarge)
        #expect(ChatTextSize.extraLarge.steppedUp == .extraLarge)
        #expect(ChatTextSize.standard.steppedDown == .small)
        #expect(ChatTextSize.small.steppedDown == .small)
        #expect(ChatTextSize.reset == .standard)
    }

    @Test func rawValuesRoundTripAndUnknownIsStandard() {
        for size in ChatTextSize.allCases {
            #expect(ChatTextSize(stored: size.rawValue) == size)
        }
        #expect(ChatTextSize(stored: "huge") == .standard)
        #expect(ChatTextSize(stored: "") == .standard)
        #expect(ChatTextSize.allCases == [.small, .standard, .large, .extraLarge])
    }

    /// CP-5-FIX-1: Settings and ChatView read the same two `@AppStorage` keys, pinned by name — a rename
    /// on one side would silently disconnect the setting.
    @Test func storageKeysArePinned() {
        #expect(ChatTextSize.messageKey == "chatMessageTextSize")
        #expect(ChatTextSize.listKey == "chatListTextSize")
    }

    /// CP-5-FIX-1: the font a row gets. `.standard` returns the call site's own style (nothing shifts);
    /// any other step must return an explicit size, which is what makes the setting visible.
    @Test func fontFollowsTheStepAndStandardKeepsTheCallSitesStyle() {
        #expect(ChatTextSize.standard.font(.listTitle, standard: .callout) == .callout)
        #expect(ChatTextSize.standard.font(.listDetail, standard: .caption2) == .caption2)
        for step in [ChatTextSize.small, .large, .extraLarge] {
            #expect(step.font(.listTitle, standard: .callout) == .system(size: step.points(for: .listTitle)))
            #expect(step.font(.listTitle, standard: .callout) != .callout)
        }
    }

    @Test func grabWidthIsWiderThanTheOldStrip() {
        #expect(ChatSidebarWidth.grabWidth >= 8 && ChatSidebarWidth.grabWidth <= 10)
    }
}
