import Testing
@testable import MLXUI

struct ChatSidebarWidthTests {
    @Test(arguments: [
        (100.0, 160.0), (160.0, 160.0), (210.0, 210.0), (420.0, 420.0), (900.0, 420.0),
        (Double.nan, 210.0), (Double.infinity, 210.0), (-Double.infinity, 210.0),
    ])
    func clampGoldens(input: Double, expected: Double) {
        #expect(ChatSidebarWidth.clamp(input) == expected)
    }

    @Test func defaultIsInsideRange() {
        #expect(ChatSidebarWidth.defaultWidth == 210)
        #expect(ChatSidebarWidth.clamp(ChatSidebarWidth.defaultWidth) == 210)
    }
}
