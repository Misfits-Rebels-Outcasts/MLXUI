import Testing
@testable import MLXUI

/// Covers `SortOrder.comparator` — the predicate driving the Browse list ordering.
struct SortOrderTests {

    @Test func ramLowToHighOrdersAscending() {
        let small = makeEntry(id: "s", ramGB: 4)
        let large = makeEntry(id: "l", ramGB: 16)
        let sorted = [large, small].sorted(by: SortOrder.ramLowToHigh.comparator)
        #expect(sorted.map(\.id) == ["s", "l"])
    }

    @Test func ramHighToLowOrdersDescending() {
        let small = makeEntry(id: "s", ramGB: 4)
        let large = makeEntry(id: "l", ramGB: 16)
        let sorted = [small, large].sorted(by: SortOrder.ramHighToLow.comparator)
        #expect(sorted.map(\.id) == ["l", "s"])
    }

    @Test func nameSortsCaseInsensitivelyAscending() {
        let zebra = makeEntry(id: "zebra", displayName: "Zebra")
        let apple = makeEntry(id: "apple", displayName: "apple")
        let sorted = [zebra, apple].sorted(by: SortOrder.name.comparator)
        #expect(sorted.map(\.id) == ["apple", "zebra"])
    }
}
