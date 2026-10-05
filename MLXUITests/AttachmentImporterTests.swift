import Foundation
import Testing
import AppKit
import PDFKit
@testable import MLXUI

/// CW-3 — the importer against real temp files: the copy (C10), the extractors (C12) and the
/// refusals. OCR is not tested here (it needs a model): CW-S5 covers it, and the importer's OCR
/// branch is driven through the injected `ocr` closure instead.
struct AttachmentImporterTests {

    private func makeTemp() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("attachments-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// A one-page PDF with a real text layer, written without new dependencies.
    private func writePDF(text: String, to url: URL) throws {
        var mediaBox = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(url: url as CFURL),
              let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        let attributed = NSAttributedString(
            string: text,
            attributes: [.font: NSFont.systemFont(ofSize: 14)])
        let framesetter = CTFramesetterCreateWithAttributedString(attributed)
        let path = CGPath(rect: mediaBox.insetBy(dx: 40, dy: 40), transform: nil)
        let frame = CTFramesetterCreateFrame(
            framesetter, CFRangeMake(0, attributed.length), path, nil)
        context.beginPDFPage(nil)
        CTFrameDraw(frame, context)
        context.endPDFPage()
        context.closePDF()
    }

    // MARK: - copy (C10)

    @Test func theFileIsCopiedIntoTheAttachmentsDirectory() async throws {
        let source = try makeTemp()
        let destination = try makeTemp()
        defer {
            try? FileManager.default.removeItem(at: source)
            try? FileManager.default.removeItem(at: destination)
        }
        let file = source.appendingPathComponent("notes.csv")
        try "a,b\n1,2\n".write(to: file, atomically: true, encoding: .utf8)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: destination, ordinal: 3, window: 8_192, inlineBytesSoFar: 0)

        #expect(attachment.name == "notes.csv")
        #expect(attachment.storedPath.hasSuffix("/3-notes.csv"))
        #expect(FileManager.default.fileExists(atPath: attachment.storedPath))
        #expect(attachment.mode == .inline)
        #expect(attachment.inlineText == "a,b\n1,2\n")
    }

    // MARK: - extracts (C12)

    @Test func aTextFileIsInlined() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("readme.md")
        try "# Title\n".write(to: file, atomically: true, encoding: .utf8)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0)
        #expect(attachment.mode == .inline)
        #expect(attachment.inlineText == "# Title\n")
    }

    @Test func aTextPDFIsInlined() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("page.pdf")
        try writePDF(text: "Invoice 42 total 17.99", to: file)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0)
        #expect(attachment.mode == .inline)
        #expect(attachment.inlineText?.contains("Invoice 42") == true)
    }

    @Test func anRTFIsInlined() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("note.rtf")
        let attributed = NSAttributedString(string: "Rich text body")
        let data = try attributed.data(
            from: NSRange(location: 0, length: attributed.length),
            documentAttributes: [.documentType: NSAttributedString.DocumentType.rtf])
        try data.write(to: file)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0)
        #expect(attachment.inlineText?.contains("Rich text body") == true)
    }

    /// A binary with NUL bytes is not text: it goes by reference, with no `inlineText`.
    @Test func aBinaryGoesByReference() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("blob.bin")
        try Data([0x00, 0x01, 0x02, 0xFF]).write(to: file)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0)
        #expect(attachment.mode == .reference)
        #expect(attachment.inlineText == nil)
    }

    @Test func aFileOverTheInlineBudgetGoesByReference() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("big.txt")
        try String(repeating: "x", count: 20_000).write(to: file, atomically: true, encoding: .utf8)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0)
        #expect(attachment.mode == .reference)
    }

    // MARK: - images (C12)

    @Test func anImageIsInlinedThroughTheOCRClosure() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("shot.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: file)   // content is irrelevant to the stub

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0,
            ocr: { _ in "text in the image" })
        #expect(attachment.mode == .ocr)
        #expect(attachment.inlineText == "text in the image")
    }

    @Test func anImageWithNoOCRModelGoesByReference() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("shot.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: file)

        let attachment = try await AttachmentImporter.importAttachment(
            from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
            window: 8_192, inlineBytesSoFar: 0, ocr: nil)
        #expect(attachment.mode == .reference)
    }

    @Test func isImageRecognisesTheCommonExtensions() {
        #expect(AttachmentImporter.isImage(URL(fileURLWithPath: "/a/shot.PNG")))
        #expect(AttachmentImporter.isImage(URL(fileURLWithPath: "/a/photo.heic")))
        #expect(!AttachmentImporter.isImage(URL(fileURLWithPath: "/a/notes.txt")))
        #expect(!AttachmentImporter.isImage(URL(fileURLWithPath: "/a/page.pdf")))
    }

    // MARK: - refusals

    @Test func aFileOver50MBIsRefused() async throws {
        let dir = try makeTemp()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("huge.bin")
        // A sparse file: 51 MB of zeroes costs no disk.
        FileManager.default.createFile(atPath: file.path, contents: nil)
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: UInt64(51 * 1024 * 1024))
        try handle.close()

        await #expect(throws: AttachmentImporter.Failure.tooLarge(ChatAttachment.maxFileBytes)) {
            try await AttachmentImporter.importAttachment(
                from: file, into: dir.appendingPathComponent("store"), ordinal: 0,
                window: 8_192, inlineBytesSoFar: 0)
        }
    }

    @Test func tooManyFilesMessageNamesTheLimit() {
        #expect(AttachmentImporter.Failure.tooMany(10).message.contains("10 files"))
        #expect(AttachmentImporter.Failure.tooLarge(50 * 1024 * 1024).message.contains("50 MB"))
    }
}
