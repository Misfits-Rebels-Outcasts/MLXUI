import Testing
import Foundation
import CryptoKit
@testable import MLXUI

/// BS-2 (`RSI/DelegateBasicShowcaseBacklog.md`, R4): the sample inputs for Basic flows 9, 10, 12, 13
/// materialise from the built bundle under their bare names, with the pinned bytes.
struct BasicShowcaseInputsTests {

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Prepare a real flow folder from the built bundle (the way a Run does) and return it.
    @MainActor private func prepared(_ flowID: String) throws -> (dir: URL, ws: FlowWorkspace, cleanup: URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("bs2-\(UUID().uuidString)")
        let ws = try #require(FlowWorkspace(root: root))
        let sourceDir = try #require(GalleryLoader.resourcesDirectory)
        try ws.prepare(flowID: flowID, sourceDir: sourceDir,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: flowID))
        return (ws.directory(for: flowID), ws, root)
    }

    static let noteText = """
        Good morning. Here is your reminder for today.
        The team stand-up moves to ten thirty, in the small meeting room.
        The quarterly report is due to finance by Friday at five.
        Please send your section to Mei Ling before Thursday noon, so there is time to review it.
        The office will close early on Friday for the building's fire drill, so plan to leave by four.
        Finally, the new coffee machine is in the pantry. It takes the round pods, not the square ones.
        Have a good day.

        """

    static let emailText = """
        From: Daniel Tan <daniel@harbourlane.example>
        Subject: Delivery address wrong for order 4471 - truck leaves at 3pm

        Hi,

        The delivery for order 4471 is booked to our old warehouse on Jalan Kilang.
        We moved last month. The truck leaves at 3pm today, so could you confirm
        the new address below before then?

          12 Harbour Lane, Unit 05-02, Singapore 098632

        Thanks,
        Daniel

        """

    @Test @MainActor func flow9MaterialisesNoteTxtExactly() throws {
        let p = try prepared("9-ReadItAloud"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let text = try String(contentsOf: p.dir.appendingPathComponent("note.txt"), encoding: .utf8)
        #expect(text == Self.noteText)
    }

    @Test @MainActor func flow10MaterialisesThePhotoByteIdenticalToFlowSix() throws {
        let six = try prepared("6-CutOutSubject"); defer { try? FileManager.default.removeItem(at: six.cleanup) }
        let ten = try prepared("10-DescribePhoto"); defer { try? FileManager.default.removeItem(at: ten.cleanup) }
        let a = try Data(contentsOf: six.dir.appendingPathComponent("photo.jpg"))
        let b = try Data(contentsOf: ten.dir.appendingPathComponent("photo.jpg"))
        #expect(a == b)
        #expect(b.count == 100175)
        #expect(sha(b) == "36b9cb9e3eb45df2be4cdf642d633fcdbd2312942ac5cb8a4cb623c9d62d3043")
    }

    @Test @MainActor func flow12MaterialisesInvoicePdfWithPinnedBytes() throws {
        let p = try prepared("12-InvoiceToSpreadsheet"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let data = try Data(contentsOf: p.dir.appendingPathComponent("invoice.pdf"))
        #expect(sha(data) == "28d9cb6a5260bec152a8f42317080149384998b4631056d32de150a636b14222")
    }

    @Test @MainActor func flow13MaterialisesEmailTxtExactly() throws {
        let p = try prepared("13-SortAnEmail"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let text = try String(contentsOf: p.dir.appendingPathComponent("email.txt"), encoding: .utf8)
        #expect(text == Self.emailText)
    }

    /// `Read PDF` extracts the invoice's values: the text is selectable (no OCR involved).
    @Test @MainActor func readPDFExtractsTheInvoiceValues() async throws {
        let p = try prepared("12-InvoiceToSpreadsheet"); defer { try? FileManager.default.removeItem(at: p.cleanup) }
        let tool = ReadPDFTool(workspace: p.ws, flowID: "12-InvoiceToSpreadsheet", settings: "invoice.pdf")
        let out = try await tool.run(Asset(items: [])) { _ in }
        let text = try #require(out.items.first?.value)
        for needle in ["BPS-2026-0417", "370.60", "Bluebird Print Studio", "2026-10-06", "2026-11-05", "340.00", "30.60"] {
            #expect(text.contains(needle), "\(needle) missing from extracted text")
        }
    }

    /// No metadata leaks a username or local path.
    @Test func invoicePdfCarriesNoLocalPathOrUsername() throws {
        let url = try #require(Bundle.main.url(forResource: "12-InvoiceToSpreadsheet-invoice", withExtension: "pdf"))
        let raw = String(decoding: try Data(contentsOf: url), as: UTF8.self)
        #expect(!raw.contains("/Users/") && !raw.contains("chinpinboo"))
    }

    /// `FlowInputAdvisory` reports no missing input for any of the five once prepared.
    @Test @MainActor func noMissingInputBannerForAnyOfTheFiveAfterPrepare() throws {
        for flowID in ["9-ReadItAloud", "10-DescribePhoto", "11-PictureFromWords",
                       "12-InvoiceToSpreadsheet", "13-SortAnEmail"] {
            let p = try prepared(flowID); defer { try? FileManager.default.removeItem(at: p.cleanup) }
            let doc = try GalleryLoader.loadDocument(flowID: flowID)
            let scope = FlowScope(identity: flowID, workspace: p.ws, locationID: flowID)
            let advisory = FlowInputAdvisory.advisory(for: doc, scope: scope)
            #expect(advisory == nil, "\(flowID): \(advisory?.reason ?? "")")
        }
    }
}
