import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// BX-2 (`RSI/DelegateBasicExpansionBacklog.md`, R4): the sample inputs for Basic flows 14-18
/// materialise from the built bundle under their bare names, with the pinned bytes.
struct BasicExpansionInputsTests {

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @MainActor private func prepared(_ flowID: String) throws -> (dir: URL, ws: FlowWorkspace, cleanup: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("bx2-\(UUID().uuidString)")
        let ws = try #require(FlowWorkspace(root: root))
        let sourceDir = try #require(GalleryLoader.resourcesDirectory)
        try ws.prepare(flowID: flowID, sourceDir: sourceDir,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        return (ws.directory(for: flowID), ws, root)
    }

    // Owner ruling, Boo, 2026-11-14: the second sentence reads "There will be no water in Block C on Saturday, 14 November," (the
    // first draft said the supply would be "switched off", which TranslateGemma rendered as "disahkan" in Malay, confirmed). Rest unchanged.
    static let letterText = """
        Dear residents,

        There will be no water in Block C on Saturday, 14 November,
        from 9 in the morning until 1 in the afternoon, while the main pipe is repaired.
        Please store enough water for cooking and washing before then.
        The lifts will keep running as normal.
        If you have any questions, call the management office on 03-5550 1234.

        Thank you for your patience.
        Bayview Residences Management

        """

    static let salesText = """
        month,sales
        Jan,4200
        Feb,3900
        Mar,4600
        Apr,4800
        May,5100
        Jun,5000
        Jul,5400
        Aug,5600
        Sep,5300
        Oct,5900
        Nov,6400
        Dec,7200

        """

    static let kittenSHA = "36b9cb9e3eb45df2be4cdf642d633fcdbd2312942ac5cb8a4cb623c9d62d3043"
    static let policySHA = "5812a441c45397eb1881cca9cb4b478ffb46c69c82b1adb2b1b4d67856b339b6"

    @Test @MainActor func flow14MaterialisesLetterTxtExactly() throws {
        let p = try prepared("14-TranslateLetter"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let text = try String(contentsOf: p.dir.appendingPathComponent("letter.txt"), encoding: .utf8)
        #expect(text == Self.letterText)
    }

    @Test @MainActor func flows15And17MaterialiseThePhotoByteIdenticalToFlowSix() throws {
        let six = try prepared("6-CutOutSubject"); defer { try? FileManager.default.removeItem(at: six.cleanup) }
        let a = try Data(contentsOf: six.dir.appendingPathComponent("photo.jpg"))
        for flowID in ["15-DepthMap", "17-StoryFromPhoto"] {
            let p = try prepared(flowID); defer { try? FileManager.default.removeItem(at: p.cleanup) }
            let b = try Data(contentsOf: p.dir.appendingPathComponent("photo.jpg"))
            #expect(a == b, "\(flowID)")
            #expect(b.count == a.count && b.count == 100175, "\(flowID)")
            #expect(sha(b) == Self.kittenSHA, "\(flowID)")
        }
    }

    @Test @MainActor func flow16MaterialisesLeavePolicyPdfWithPinnedBytes() throws {
        let p = try prepared("16-AskADocument"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let data = try Data(contentsOf: p.dir.appendingPathComponent("leave-policy.pdf"))
        #expect(sha(data) == Self.policySHA)
    }

    @Test @MainActor func flow18MaterialisesSalesCsvExactly() throws {
        let p = try prepared("18-ChartSpreadsheet"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let text = try String(contentsOf: p.dir.appendingPathComponent("sales.csv"), encoding: .utf8)
        #expect(text == Self.salesText)
    }

    /// `Read PDF` extracts the policy's facts: the text is selectable (no OCR involved).
    @Test @MainActor func readPDFExtractsTheLeavePolicyFacts() async throws {
        let p = try prepared("16-AskADocument"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let tool = ReadPDFTool(workspace: p.ws, flowID: "16-AskADocument", settings: "leave-policy.pdf")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let text = try #require(out.items.first?.value)
        for needle in ["Riverstone Trading Sdn Bhd", "Annual Leave Policy (2026)", "14 days", "18 days",
                       "3-month probation", "31 March", "7 days", "medical certificate"] {
            #expect(text.contains(needle), "\(needle) missing from extracted text")
        }
    }

    /// No metadata leaks a username or local path.
    @Test func leavePolicyPdfCarriesNoLocalPathOrUsername() throws {
        let url = try #require(Bundle.main.url(forResource: "16-AskADocument-leave-policy", withExtension: "pdf"))
        let raw = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        #expect(!raw.contains("/Users/") && !raw.contains("chinpinboo"))
    }

    /// `FlowInputAdvisory` reports no missing input for any of the five once prepared.
    @Test @MainActor func noMissingInputBannerForAnyOfTheFiveAfterPrepare() throws {
        for flowID in ["14-TranslateLetter", "15-DepthMap", "16-AskADocument",
                       "17-StoryFromPhoto", "18-ChartSpreadsheet"] {
            let p = try prepared(flowID); defer { try? FileManager.default.removeItem(at: p.cleanup) }
            let doc = try GalleryLoader.loadDocument(flowID: flowID)
            let scope = FlowScope(identity: flowID, workspace: p.ws, locationID: flowID)
            let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
            #expect(advisory == nil, "\(flowID): \(advisory?.reason ?? "")")
        }
    }
}
