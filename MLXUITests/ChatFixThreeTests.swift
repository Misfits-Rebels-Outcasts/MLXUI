import Testing
@testable import MLXUI

/// CD-FIX-3 — the long-name composer label and the progress text before the total is known.
@MainActor
struct ChatFixThreeTests {
    private let longName = "Ministral-3-3B-Instruct-2512"

    // MARK: 1 — composer width

    @Test func theLongMinistralNameIsCutWithAnEllipsisAtTheMinimumComposerWidth() throws {
        let width = ChatCapabilityMenuText.minimumComposerWidth
        let menuMax = try #require(ChatCapabilityMenuText.menuMaxWidth(containerWidth: width))
        let fit = ChatCapabilityMenuText.fitLabel(name: longName, summary: "6 tools", maxWidth: menuMax)
        #expect(!fit.fitsWhole)
        #expect(fit.summary == nil)
        #expect(fit.name.hasSuffix("…"))
        #expect(fit.name.count < longName.count)
        #expect(longName.hasPrefix(String(fit.name.dropLast())))
        // the label's estimated width stays inside its allotment, so the text field keeps its ~20 characters
        let labelWidth = ChatCapabilityMenuText.chromeWidth + Double(fit.name.count) * ChatCapabilityMenuText.charWidth
        #expect(labelWidth <= menuMax)
        let rest = width - menuMax - (ChatCapabilityMenuText.composerReserved - ChatCapabilityMenuText.fieldMinWidth)
        #expect(rest >= ChatCapabilityMenuText.fieldMinWidth)                     // ~20 characters
        #expect(ChatCapabilityMenuText.fieldMinWidth / ChatCapabilityMenuText.charWidth >= 20)
    }

    @Test func aShortNameStillShowsTheSummaryAndAMidNameLosesIt() {
        #expect(ChatCapabilityMenuText.fitLabel(name: "Qwen3 4B", summary: "6 tools", maxWidth: 360).summary == "6 tools")
        let mid = ChatCapabilityMenuText.fitLabel(name: longName, summary: "6 tools", maxWidth: 300)
        #expect(mid.summary == nil && mid.fitsWhole && mid.name == longName)
    }

    // MARK: 3 — progress text

    @Test func noPercentageUntilTheTotalCouldBeTheRealOne() {
        let expected: Int64 = 2_780_000_000
        // the engine's initial placeholder, and the metadata files at the start ("0.1 GB of 0.1 GB")
        #expect(ChatNoModelDownload.presentation(for: .downloading(progress: 0, downloaded: 0, total: 1), expectedBytes: expected) == .starting)
        #expect(ChatNoModelDownload.presentation(for: .downloading(progress: 1, downloaded: 100_000_000, total: 100_000_000), expectedBytes: expected) == .starting)
        // the weights have begun: the total is near the expected size
        let near = ChatNoModelDownload.presentation(for: .downloading(progress: 0.1, downloaded: 280_000_000, total: 2_700_000_000), expectedBytes: expected)
        #expect(near == .progress(fraction: 0.1, detail: ModelInstallCard.progressText(downloaded: 280_000_000, total: 2_700_000_000)))
        // a catalog that overstates the size: a quarter of the way in still shows progress
        let over = ChatNoModelDownload.presentation(for: .downloading(progress: 0.5, downloaded: 1_000_000_000, total: 1_500_000_000), expectedBytes: 3_000_000_000)
        #expect(over == .progress(fraction: 0.5, detail: ModelInstallCard.progressText(downloaded: 1_000_000_000, total: 1_500_000_000)))
    }

    @Test func withoutAnExpectedSizeOnlyThePlaceholderIsSkipped() {
        #expect(ChatNoModelDownload.presentation(for: .downloading(progress: 0, downloaded: 0, total: 0)) == .starting)
        #expect(ChatNoModelDownload.presentation(for: .downloading(progress: 0, downloaded: 0, total: 1)) == .starting)
        #expect(ChatNoModelDownload.presentation(for: .downloading(progress: 0.5, downloaded: 50, total: 100))
                == .progress(fraction: 0.5, detail: ModelInstallCard.progressText(downloaded: 50, total: 100)))
    }
}
