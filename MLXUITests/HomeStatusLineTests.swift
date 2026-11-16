import Testing
@testable import MLXUI

struct HomeStatusLineTests {
    private let line = HomeStatusLine(chipName: "Apple M2 Max", formattedRAM: "32 GB", formattedDisk: "512 GB", macOSVersion: "15.1")

    @Test func badgesKeepHomeViewsWordingAndOrder() {
        #expect(line.badges.map(\.label) == ["Apple M2 Max", "32 GB", "512 GB free", "macOS 15.1"])
        #expect(line.badges.map(\.icon) == ["cpu.fill", "memorychip.fill", "internaldrive.fill", "macbook"])
    }

    @Test func sentenceIsChipRamFreeDisk() {
        #expect(line.sentence == "Apple M2 Max · 32 GB · 512 GB free")
    }

    @Test func footerAddsTheServerSentenceOnlyWhenThereIsOne() {
        #expect(HomeStatusLine.footer(system: line.sentence, serverSentence: nil) == line.sentence)
        #expect(HomeStatusLine.footer(system: line.sentence, serverSentence: "") == line.sentence)
        #expect(HomeStatusLine.footer(system: line.sentence, serverSentence: "Serving 2 models · 127.0.0.1:1212")
                == "Apple M2 Max · 32 GB · 512 GB free · Serving 2 models · 127.0.0.1:1212")
    }
}
