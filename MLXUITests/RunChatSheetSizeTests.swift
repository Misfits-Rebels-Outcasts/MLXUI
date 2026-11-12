import Testing
import CoreGraphics
@testable import MLXUI

/// CP-3-SPIKE: the Run sheet's minimum, parse and screen clamp.
struct RunChatSheetSizeTests {
    private let screen = CGSize(width: 1512, height: 944)

    @Test func minimumIsTheViewsFloorAndTheKeyIsPinned() {
        #expect(RunChatSheetSize.minSize == CGSize(width: 840, height: 520))
        #expect(RunChatSheetSize.storageKey == "runChatSheetSize")
    }

    @Test(arguments: [
        (CGSize(width: 100, height: 100), CGSize(width: 840, height: 520)),     // below the minimum
        (CGSize(width: 840, height: 520), CGSize(width: 840, height: 520)),
        (CGSize(width: 1100, height: 620), CGSize(width: 1100, height: 620)),   // today's ideal is untouched
        (CGSize(width: 5000, height: 4000), CGSize(width: 1512, height: 944)),  // above the screen
        (CGSize(width: 900, height: 4000), CGSize(width: 900, height: 944)),
        (CGSize(width: Double.nan, height: Double.infinity), CGSize(width: 840, height: 520)),
    ])
    func clamp(input: CGSize, expected: CGSize) {
        #expect(RunChatSheetSize.clamp(input, screen: screen) == expected)
    }

    @Test func aScreenSmallerThanTheMinimumNeverShrinksTheSheetBelowIt() {
        let tiny = CGSize(width: 700, height: 400)
        #expect(RunChatSheetSize.clamp(CGSize(width: 2000, height: 2000), screen: tiny) == RunChatSheetSize.minSize)
    }

    @Test func parseAndEncodeRoundTripAndRejectGarbage() {
        #expect(RunChatSheetSize.parse(RunChatSheetSize.encode(CGSize(width: 1100.4, height: 619.6))) == CGSize(width: 1100, height: 620))
        for bad in ["", "1100", "a,b", "1100,", "-5,600", "0,0", "nan,600", "1,2,3"] {
            #expect(RunChatSheetSize.parse(bad) == nil, "\(bad)")
        }
    }

    /// CP-3-SPIKE-FIX-1: a size below the floor is never written.
    @Test func aSizeBelowTheMinimumIsNeverStored() {
        for small in [CGSize(width: 839, height: 900), CGSize(width: 900, height: 519), CGSize(width: 300, height: 0),
                      CGSize(width: 2, height: 0), CGSize(width: Double.nan, height: 600)] {
            #expect(RunChatSheetSize.storableValue(for: small) == nil, "\(small)")
        }
        #expect(RunChatSheetSize.storableValue(for: CGSize(width: 840, height: 520)) == "840,520")
        #expect(RunChatSheetSize.storableValue(for: CGSize(width: 1100, height: 620)) == "1100,620")
    }

    @Test func snapTargetOnlyWhenBelowTheFloor() {
        #expect(RunChatSheetSize.snapTarget(for: CGSize(width: 1100, height: 620)) == nil)
        #expect(RunChatSheetSize.snapTarget(for: CGSize(width: 840, height: 520)) == nil)
        #expect(RunChatSheetSize.snapTarget(for: CGSize(width: 600, height: 0)) == CGSize(width: 840, height: 520))
        #expect(RunChatSheetSize.snapTarget(for: CGSize(width: 1000, height: 300)) == CGSize(width: 1000, height: 520))
    }
}
