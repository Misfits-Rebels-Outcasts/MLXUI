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

    @Test func mostDownloadedTreatsNilAsZero() {
        let unknown = makeEntry(id: "unknown", communityDownloads: nil)
        let popular = makeEntry(id: "popular", communityDownloads: 1000)
        let sorted = [unknown, popular].sorted(by: SortOrder.mostDownloaded.comparator)
        #expect(sorted.first?.id == "popular")
    }

    @Test func qualitySortPlacesNullBenchmarkModelsLast() {
        let benched = makeEntry(
            id: "benched",
            benchmarks: ModelBenchmarks(mmlu: 80, humanEval: nil, gsm8k: nil,
                                        hellaswag: nil, arc: nil, truthfulQA: nil)
        )
        let unscored = makeEntry(id: "unscored", benchmarks: nil)
        let sorted = [unscored, benched].sorted(by: SortOrder.quality.comparator)
        #expect(sorted.map(\.id) == ["benched", "unscored"])
    }
}
