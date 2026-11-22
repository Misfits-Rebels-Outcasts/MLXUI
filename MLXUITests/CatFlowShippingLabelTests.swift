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
        #expect(entries.filter(\.isBasic).map(\.flowID).last == Self.flowID)
        #expect(entries.filter(\.isBasic).count == 19)
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

        // Deliberately NOT asserted: that running the flow a second time reuses this entry. `Stage Print`'s id is the content digest, and
        // `Make PDF` output is not byte-stable across runs in either runtime (a creation date and an /ID are embedded; measured in
        // Python's reportlab output and in Core Graphics'), so a re-run queues a second pending entry. Reported with ER-12 as a
        // Q258 follow-up; the person approves each entry, so nothing prints twice by itself.

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
