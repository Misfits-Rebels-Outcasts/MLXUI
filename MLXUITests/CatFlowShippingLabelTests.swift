import Testing
import Foundation
import PDFKit
@testable import MLXUI

/// ER-12 (owner ruling R12, 2026-11-22): the Basic Gallery flow "Shipping Label" — Template -> Make Barcode -> Make PDF -> Stage Print.
/// Gallery number 88 (the next free one in the Swift gallery; the Python gallery tops out at 73), shelf slot 19, file `19-ShippingLabel`.
struct CatFlowShippingLabelTests {

    private static let flowID = "19-ShippingLabel"

    private struct Fake: PrintPresenting {
        @MainActor func present(fileURL: URL, mime: String, title: String) async -> PrintPanelOutcome { .completed }
    }

    // MARK: - the shelf

    @Test func theFlowIsListedAsBasicShelfSlotNineteenWithGalleryNumberEightyEight() throws {
        let entries = GalleryLoader.loadMetadata()
        let entry = try #require(entries.first { $0.flowID == Self.flowID })
        #expect(entry.number == 88 && entry.title == "Shipping Label" && entry.isBasic)
        // PF (owner R2b, 2026-11-24) appended shelf slot 20 (Redact a Document), so this flow is no longer last: it is the 19th.
        #expect(entries.filter(\.isBasic).map(\.flowID).dropLast().last == Self.flowID)
        #expect(entries.filter(\.isBasic).count == 20)
        #expect(entries.count == Set(entries.map(\.number)).count, "gallery numbers stay unique")
    }

    @Test func theFlowKeepsItsData() throws {
        let text = try GalleryLoader.rawCatText(flowID: Self.flowID)
        // Basic and private: no `offdevice`, no `network` flag, no model row, and no setting for the person to configure.
        let doc = try CatParser.parse(text)
        #expect(doc.flags.isEmpty, "\(doc.flags)")
        #expect(doc.rows.map(\.task) == ["Template", "Make Barcode", "Make PDF", "Stage Print"])
        #expect(doc.rows.allSatisfy { $0.model == nil })
        #expect(doc.rows[0].settings == "\"ORDER-48213\"", "the one visible input is row 1's text")
    }

    @Test func itChecksCleanWithNoIssues() throws {
        let text = try GalleryLoader.rawCatText(flowID: Self.flowID)
        let found = FlowValidator.checkFlow(try CatParser.parseForValidation(text))
        #expect(found.isEmpty, "\(found)")
    }

    // MARK: - runs

    @Test @MainActor func itRunsUnderTheMock() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("er12-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(try GalleryLoader.loadDocument(flowID: Self.flowID), executor: mock)
        #expect(events.filter { $0.kind == .rowFailed }.isEmpty, "\(events.filter { $0.kind == .rowFailed }.map { $0.message ?? "" })")
        #expect(mock.lastStaged?.kind == "print")
    }

    @Test @MainActor func theRealRuntimeQueuesOnePendingOnePagePrintEntryAndThePersonApprovesIt() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("er12-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: Self.flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: Self.flowID))
        let real = RealExecutor(workspace: ws, flowID: Self.flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                makeModelStage: { _, _ in throw StageError.kindMismatch(expected: .text, got: .audio) },
                                installedModelIDs: [], catalog: [])
        let events = try await FlowInterpreter.run(try GalleryLoader.loadDocument(flowID: Self.flowID), executor: real)
        #expect(events.filter { $0.kind == .rowFailed }.isEmpty, "\(events.filter { $0.kind == .rowFailed }.map { $0.message ?? "" })")

        let entries = OutboxStore.entries(workspace: ws, flowID: Self.flowID)
        #expect(entries.count == 1)
        let entry = try #require(entries.first)
        #expect(entry.kind == "print" && entry.mime == "application/pdf")
        let box = OutboxStore.directory(workspace: ws, flowID: Self.flowID)
        let stored = box.appendingPathComponent("files/\(try #require(entry.storedFile))")
        let pdf = try #require(PDFDocument(url: stored))
        #expect(pdf.pageCount == 1, "a one-page label")

        // Q259 (owner ruling F1, 2026-11-22): the id ignores the PDF's /CreationDate, /ModDate and /ID, so running the flow again
        // (a second later, so Core Graphics writes a different creation date) regenerates the same label and is the same pending entry.
        try await Task.sleep(nanoseconds: 1_200_000_000)
        let again = try await FlowInterpreter.run(try GalleryLoader.loadDocument(flowID: Self.flowID), executor: real)
        #expect(again.filter { $0.kind == .rowFailed }.isEmpty)
        #expect(OutboxStore.entries(workspace: ws, flowID: Self.flowID).map(\.id) == [entry.id], "a re-run of the same order is one pending entry")

        // Approval (a fake panel: no real one opens) commits it.
        #expect(await PrintOutbox.approve(id: entry.id, outbox: box, presenter: Fake()) == .committed)
        #expect(OutboxStore.entries(workspace: ws, flowID: Self.flowID).isEmpty)
    }

    @Test func theBarcodeRoundTripsThroughReadBarcode() async throws {
        // The reason for `type=qr`: it carries the order number as-is and reads back exactly (Read Barcode, Vision).
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("er12-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let text = Asset(items: [Item(kind: .text, value: "ORDER-48213", path: nil, sourceText: nil)])
        let image = try MakeBarcodeTool.make(inputs: [text], settings: "type=qr", outDirectory: dir)
        let read = try ReadBarcodeTool.run(inputs: [image], settings: nil)
        #expect(read.items.first?.value == "ORDER-48213")
    }
}
