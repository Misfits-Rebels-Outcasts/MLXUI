import Testing
import Foundation
import CoreGraphics
import PDFKit
@testable import MLXUI

/// ER-8 — `Make PDF` (SPEC-Q255). Structure only, read back through PDFKit: page count, page
/// size, recovered text, image placement (rendered, non-white bounds), table order, title.
/// Never byte-for-byte: a PDF is not byte-stable. Page size: reportlab reports A4 as
/// 595x842 (595.2756 x 841.8898); compared here after rounding.
struct CatFlowMakePDFTests {

    // MARK: - helpers

    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("catflow-er8-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func text(_ s: String) -> [Asset] { [Asset(items: [Item(kind: .text, value: s, path: nil, sourceText: nil)])] }

    private func make(_ inputs: [Asset], _ settings: String? = nil, in dir: URL) throws -> PDFDocument {
        let out = try MakePDFTool.make(inputs: inputs, settings: settings, outDirectory: dir)
        let item = try #require(out.items.first)
        #expect(out.items.count == 1 && item.kind == .file)
        let url = try #require(item.path)
        #expect(url.pathExtension == "pdf")
        return try #require(PDFDocument(url: url))
    }

    private func refusal(_ inputs: [Asset], _ settings: String? = nil, in dir: URL) -> String? {
        do { _ = try MakePDFTool.make(inputs: inputs, settings: settings, outDirectory: dir); return nil } catch let e as FlowError {
            if case .stageFailure(let row, let message) = e { return row == "Make PDF" ? message : "wrong row \(row): \(message)" }
            return "\(e)"
        } catch { return "\(error)" }
    }

    private func png(_ dir: URL, width: Int, height: Int, name: String = UUID().uuidString) throws -> Asset {
        let ctx = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                                         space: CGColorSpaceCreateDeviceRGB(),
                                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        ctx.setFillColor(CGColor(srgbRed: 1, green: 0, blue: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try #require(ctx.makeImage())
        let data = try #require(PNGEncoder.pngData(from: image))
        let url = dir.appendingPathComponent("\(name).png")
        try data.write(to: url)
        return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
    }

    private func table(_ dir: URL, columns: [String], rows: [[Any?]]) throws -> [Asset] {
        [Asset(items: [try TableTool.tableItem(columns: columns, rows: rows, in: dir)])]
    }

    private func size(_ page: PDFPage) -> String {
        let r = page.bounds(for: .mediaBox)
        return "\(Int(r.width.rounded()))x\(Int(r.height.rounded()))"
    }

    /// Non-white bounds of a page rendered at 1 pt = 1 px, in PDF points (origin bottom-left).
    private func inkBounds(_ page: PDFPage) throws -> CGRect {
        let box = page.bounds(for: .mediaBox)
        let w = Int(box.width.rounded(.up)), h = Int(box.height.rounded(.up))
        let ctx = try #require(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                         space: CGColorSpaceCreateDeviceRGB(),
                                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        page.draw(with: .mediaBox, to: ctx)
        let data = try #require(ctx.data).assumingMemoryBound(to: UInt8.self)
        var minX = w, minY = h, maxX = -1, maxY = -1
        for row in 0..<h {            // bitmap row 0 is the top of the page
            for x in 0..<w {
                let p = (row * w + x) * 4
                if data[p] < 250 || data[p + 1] < 250 || data[p + 2] < 250 {
                    minX = min(minX, x); maxX = max(maxX, x)
                    minY = min(minY, h - 1 - row); maxY = max(maxY, h - 1 - row)
                }
            }
        }
        return CGRect(x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1)
    }

    // MARK: - tables, settings, validator

    @Test func makePdfIsInTheFourTables() throws {
        let d = try #require(TaskCatalog.catalog["Make PDF"])
        #expect(d.accepts == Shape.anyKind && d.gives == Shape.single(Kind.file))
        #expect(d.taskClass == .instant)
        #expect(!CacheKey.neverCache.contains("Make PDF"))     // a pure tool: cacheable, as in the Python
        #expect(TaskAvailability.supportedInstantTools.contains("Make PDF"))
        #expect(CatParser.taskNames.contains("Make PDF"))
    }

    @Test func settingsParse() throws {
        let none = try MakePdfSettings.parse(nil)
        #expect(none.page == "a4" && none.title == nil)
        let both = try MakePdfSettings.parse("page=LETTER; title=\"Q3 Rechnung\"")
        #expect(both.page == "letter" && both.title == "Q3 Rechnung")
        for bad in ["page=a3", "margin=10", "\"letter\"", "page=\"\"", "format=markdown"] {
            #expect(throws: MakePdfSettingsError.self) { _ = try MakePdfSettings.parse(bad) }
        }
    }

    @Test func er2RefusalGoldenMatches() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/goldens/check/er2_refusals.json")
        struct Case: Decodable { struct I: Decodable { var row: String; var code: String; var message: String }; var name: String; var flow: String; var issues: [I] }
        let cases = try JSONDecoder().decode([String: [Case]].self, from: Data(contentsOf: url))["cases"] ?? []
        #expect(cases.count == 5)
        for c in cases {
            let got = FlowValidator.checkFlow(try CatParser.parseForValidation(c.flow))
            #expect(got.map(\.code) == c.issues.map(\.code), "\(c.name)")
            #expect(got.map(\.message) == c.issues.map(\.message), "\(c.name)")
        }
    }

    // MARK: - text

    @Test func textDrawsAndReadsBackOnA4() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let doc = try make(text("Rechnung für Müller, Straße 12"), in: dir)
        #expect(doc.pageCount == 1)
        let page = try #require(doc.page(at: 0))
        #expect(size(page) == "595x842")
        #expect(page.string?.contains("Rechnung für Müller, Straße 12") == true, "\(page.string ?? "nil")")
    }

    @Test func frenchTextWithCp1252SpecialsDraws() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let doc = try make(text("Reçu n° 7 — 12,50 €"), in: dir)
        let s = try #require(doc.page(at: 0)?.string)
        #expect(s.contains("Reçu n° 7") && s.contains("12,50 €") && s.contains("—"), "\(s)")
    }

    @Test func letterPageSizeAndTitleMetadata() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let doc = try make(text("hello"), "page=letter; title=\"Hallo Welt\"", in: dir)
        let first = try #require(doc.page(at: 0))
        #expect(size(first) == "612x792")
        #expect(doc.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String == "Hallo Welt")
        #expect(doc.page(at: 0)?.string?.contains("Hallo Welt") == false)   // metadata only, never drawn
    }

    @Test func paragraphsLineBreaksAndSpaceCollapse() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let doc = try make(text("one   two\nthree\n\nfour"), in: dir)
        let s = (try #require(doc.page(at: 0)?.string)).replacingOccurrences(of: "\r", with: "\n")
        #expect(s.contains("one two"), "\(s)")
        let lines = s.split(separator: "\n").map(String.init).filter { !$0.isEmpty }
        #expect(lines == ["one two", "three", "four"], "\(lines)")
    }

    @Test func longTextFlowsOntoMorePages() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let para = (1...400).map { "word\($0)" }.joined(separator: " ")
        let doc = try make(text(Array(repeating: para, count: 6).joined(separator: "\n\n")), in: dir)
        #expect(doc.pageCount > 1)
        let ink = try inkBounds(try #require(doc.page(at: 0)))
        #expect(ink.minX >= 56.69 - 2 && ink.maxX <= 595.28 - 56.69 + 2)      // inside the margins
    }

    @Test func undrawableCharactersAreRefusedByName() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let cases: [(String, String)] = [
            ("日本語 text 中", "can't draw '日' (U+65E5) in the text"),
            ("Привет", "can't draw 'П' (U+041F) in the text"),
            ("alpha α", "can't draw 'α' (U+03B1) in the text"),
            ("party 🎉", "can't draw '🎉' (U+1F389) in the text"),
            ("bell \u{7}", "can't draw '\\x07' (U+0007) in the text"),
            ("C1 \u{81}", "can't draw '\\x81' (U+0081) in the text"),
        ]
        for (input, expected) in cases {
            let msg = try #require(refusal(text(input), in: dir), "\(input) should refuse")
            #expect(msg.hasPrefix("Make PDF \(expected)"), "\(msg)")
            #expect(msg.hasSuffix("remove or replace that character"))
        }
    }

    @Test func glyphRuleIsExactlyCp1252() {
        let high: Set<UInt32> = [0x152, 0x153, 0x160, 0x161, 0x178, 0x17D, 0x17E, 0x192, 0x2C6, 0x2DC, 0x2013, 0x2014, 0x2018, 0x2019,
                                 0x201A, 0x201C, 0x201D, 0x201E, 0x2020, 0x2021, 0x2022, 0x2026, 0x2030, 0x2039, 0x203A, 0x20AC, 0x2122]
        for v in UInt32(0)..<0x2200 {
            guard let s = Unicode.Scalar(v) else { continue }
            let expected = [0x09, 0x0A, 0x0D].contains(v) || (v >= 0x20 && v <= 0x7F) || (v >= 0xA0 && v <= 0xFF) || high.contains(v)
            #expect(MakePDFTool.isDrawable(s) == expected, "U+\(String(v, radix: 16))")
        }
    }

    @Test func inputRefusals() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        func starts(_ m: String?, _ s: String) { #expect(m?.hasPrefix("Make PDF \(s)") == true, "\(m ?? "nil")") }
        starts(refusal(text(""), in: dir), "got empty text")
        starts(refusal(text("  \n\n \t "), in: dir), "got empty text")
        starts(refusal([], in: dir), "got nothing")
        starts(refusal([Asset(items: [])], in: dir), "got nothing")
        starts(refusal(text("a") + text("b"), in: dir), "got 2 text items")
        starts(refusal(text("a") + [try png(dir, width: 4, height: 4)], in: dir), "got mixed inputs (image, text)")
        starts(refusal(try table(dir, columns: [], rows: []), in: dir), "got a table with no columns")
        starts(refusal([Asset(items: [Item(kind: .audio, value: nil, path: dir, sourceText: nil)])], in: dir), "can't make a PDF from audio")
        starts(refusal(text("x"), "page=a3", in: dir), "has `page=a3`")
        let missing = Asset(items: [Item(kind: .image, value: nil, path: dir.appendingPathComponent("nope.png"), sourceText: nil)])
        starts(refusal([missing], in: dir), "can't read image 1")
        let junk = dir.appendingPathComponent("junk.png"); try Data("not an image".utf8).write(to: junk)
        starts(refusal([Asset(items: [Item(kind: .image, value: nil, path: junk, sourceText: nil)])], in: dir), "can't read image 1")
    }

    // MARK: - images

    @Test func threeImagesMakeThreePagesEachFittedWithinMargins() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let items = try [png(dir, width: 400, height: 100), png(dir, width: 100, height: 400), png(dir, width: 50, height: 50)]
            .flatMap(\.items)
        let doc = try make([Asset(items: items)], in: dir)
        #expect(doc.pageCount == 3)
        let m: CGFloat = 56.69
        for i in 0..<3 {
            let page = try #require(doc.page(at: i))
            let ink = try inkBounds(page)
            #expect(ink.minX >= m - 2 && ink.maxX <= 595.28 - m + 2, "page \(i) x \(ink)")
            #expect(ink.minY >= m - 2 && ink.maxY <= 841.89 - m + 2, "page \(i) y \(ink)")
            // top-aligned and horizontally centred
            #expect(abs(ink.maxY - (841.89 - m)) <= 2.5, "page \(i) top \(ink)")
            #expect(abs((ink.minX + ink.maxX) / 2 - 595.28 / 2) <= 2.5, "page \(i) centre \(ink)")
        }
        // the wide one fills the frame width, the tall one the frame height (scaled up, aspect kept)
        let wide = try inkBounds(try #require(doc.page(at: 0)))
        #expect(abs(wide.width - (595.28 - 2 * m)) <= 3)
        let tall = try inkBounds(try #require(doc.page(at: 1)))
        #expect(abs(tall.height - (841.89 - 2 * m)) <= 3)
        #expect(abs(tall.width / tall.height - 0.25) <= 0.01)
    }

    @Test func aSingleImageOnLetter() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let doc = try make([try png(dir, width: 30, height: 20)], "page=letter", in: dir)
        let only = try #require(doc.page(at: 0))
        #expect(doc.pageCount == 1 && size(only) == "612x792")
        let ink = try inkBounds(try #require(doc.page(at: 0)))
        #expect(ink.minX >= 54 - 2 && ink.maxX <= 612 - 54 + 2)
    }

    // MARK: - tables

    @Test func tableHeaderThenRowsInOrder() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let doc = try make(try table(dir, columns: ["Name", "Menge", "Preis"],
                                     rows: [["Äpfel", 3, 1.5], ["Straße", 12, nil], ["Crème", 0, 99.25]]), in: dir)
        #expect(doc.pageCount == 1)
        let page = try #require(doc.page(at: 0))
        let s = try #require(page.string)
        // Reading order of cells is PDFKit's; assert the geometry instead: header above rows, rows in order.
        func top(_ word: String) -> CGFloat? {
            doc.findString(word, withOptions: []).first.map { $0.bounds(for: page).maxY }
        }
        let ys = ["Name", "Äpfel", "Straße", "Crème"].compactMap(top)
        #expect(ys.count == 4 && ys == ys.sorted(by: >), "\(ys) \(s)")
        let headerY = ["Menge", "Preis"].compactMap(top)
        #expect(headerY.count == 2 && headerY.allSatisfy { abs($0 - ys[0]) < 1 })
        #expect(s.contains("1.5") && s.contains("99.25"))
    }

    @Test func tableHeaderRepeatsOnEveryPage() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let rows: [[Any?]] = (1...120).map { n -> [Any?] in let name: String = "row\(n)"; return [name, n] }
        let doc = try make(try table(dir, columns: ["Artikel", "Anzahl"], rows: rows), in: dir)
        #expect(doc.pageCount >= 2)
        for i in 0..<doc.pageCount {
            let s = try #require(doc.page(at: i)?.string)
            #expect(s.contains("Artikel") && s.contains("Anzahl"), "page \(i)")
        }
        let all = (0..<doc.pageCount).compactMap { doc.page(at: $0)?.string }.joined()
        #expect(all.contains("row1") && all.contains("row120"), "\(all.prefix(300)) ... \(all.suffix(200))")
    }

    @Test func headerOnlyTableIsAllowedAndWideCellsWrap() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let headerOnly = try make(try table(dir, columns: ["A", "B"], rows: []), in: dir)
        #expect(headerOnly.pageCount == 1 && headerOnly.page(at: 0)?.string?.contains("A") == true)
        let long = (1...60).map { "wort\($0)" }.joined(separator: " ")
        let wide = try make(try table(dir, columns: ["Spalte", "Text"], rows: [["x", long]]), in: dir)
        let ink = try inkBounds(try #require(wide.page(at: 0)))
        #expect(ink.maxX <= 595.28 - 56.69 + 2)
        #expect(wide.page(at: 0)?.string?.contains("wort60") == true)
    }

    @Test func tableCellsWithUndrawableCharsRefuseNamingTheTable() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let msg = try #require(refusal(try table(dir, columns: ["Name"], rows: [["李"]]), in: dir))
        #expect(msg.hasPrefix("Make PDF can't draw '李' (U+674E) in the table"), "\(msg)")
    }

    // MARK: - executors

    private func realExecutor(_ ws: FlowWorkspace, _ base: URL) -> RealExecutor {
        RealExecutor(workspace: ws, flowID: "f", blobDirectory: base,
                     makeModelStage: { _, _ in MakePDFStub() }, installedModelIDs: [], catalog: [])
    }

    @Test func realExecutorMakesAPdfFile() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ws = FlowWorkspace(root: dir.appendingPathComponent("flows"))
        let out = try await realExecutor(ws, dir).execute(path: "2", row: Row(task: "Make PDF", settings: "title=\"T\""),
                                                           inputs: text("Grüße aus Zürich"), transcript: nil, context: nil, usedFlowContent: nil)
        let item = try #require(out.items.first)
        #expect(item.kind == .file)
        let pdfURL = try #require(item.path)
        let doc = try #require(PDFDocument(url: pdfURL))
        #expect(doc.page(at: 0)?.string?.contains("Grüße aus Zürich") == true)
    }

    @Test func realExecutorRefusesByRowName() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ws = FlowWorkspace(root: dir.appendingPathComponent("flows"))
        do {
            _ = try await realExecutor(ws, dir).execute(path: "2", row: Row(task: "Make PDF"), inputs: text("日本"),
                                                        transcript: nil, context: nil, usedFlowContent: nil)
            Issue.record("expected a refusal")
        } catch let e as FlowError {
            #expect(String(describing: e).contains("Make PDF can't draw"))
        }
    }

    @Test func mockExecutorFabricatesAFile() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let mock = MockExecutor(blobDirectory: dir.appendingPathComponent("blobs"))
        let out = try await mock.execute(path: "2", row: Row(task: "Make PDF"), inputs: text("x"),
                                         transcript: nil, context: nil, usedFlowContent: nil)
        #expect(out.items.first?.kind == .file)
    }
}

private nonisolated struct MakePDFStub: PipelineStage {
    let id = "stub.er8"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}
