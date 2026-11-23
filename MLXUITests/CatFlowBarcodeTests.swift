import Testing
import Foundation
import CoreGraphics
import ImageIO
import Vision
@testable import MLXUI

/// ER-9 — `Read Barcode` (SPEC-Q256) and `Make Barcode` (SPEC-Q257). Sizes are exact and equal
/// the Python's for a given scale; reading is checked by round trip, never by bytes.
struct CatFlowBarcodeTests {

    // MARK: - helpers

    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("catflow-er9-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func text(_ s: String) -> [Asset] { [Asset(items: [Item(kind: .text, value: s, path: nil, sourceText: nil)])] }

    private func make(_ s: String, _ settings: String? = nil, in dir: URL) throws -> (url: URL, image: CGImage) {
        let out = try MakeBarcodeTool.make(inputs: text(s), settings: settings, outDirectory: dir)
        let item = try #require(out.items.first)
        #expect(out.items.count == 1 && item.kind == .image)
        let url = try #require(item.path)
        #expect(url.pathExtension == "png")
        return (url, try ImageTools.load(url))
    }

    private func imageAsset(_ url: URL) -> [Asset] { [Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])] }

    private func read(_ url: URL, _ settings: String? = nil) throws -> String {
        let out = try ReadBarcodeTool.run(inputs: imageAsset(url), settings: settings)
        #expect(out.items.count == 1 && out.items.first?.kind == .text)
        return out.items.first?.value ?? "<nil>"
    }

    private func refusal(_ s: String, _ settings: String? = nil, in dir: URL) -> String? {
        do { _ = try MakeBarcodeTool.make(inputs: text(s), settings: settings, outDirectory: dir); return nil } catch let e as FlowError {
            if case .stageFailure(let row, let message) = e { return row == "Make Barcode" ? message : "wrong row \(row)" }
            return "\(e)"
        } catch { return "\(error)" }
    }

    /// A white canvas with `parts` drawn at `origin` (top-left, y down).
    private func compose(_ parts: [(CGImage, CGPoint)], size: CGSize, to url: URL) throws {
        let ctx = try #require(CGContext(data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8,
                                         bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(),
                                         bitmapInfo: CGImageAlphaInfo.none.rawValue))
        ctx.setFillColor(gray: 1, alpha: 1)
        ctx.fill(CGRect(origin: .zero, size: size))
        ctx.interpolationQuality = .none
        for (image, origin) in parts {
            ctx.draw(image, in: CGRect(x: origin.x, y: size.height - origin.y - CGFloat(image.height),
                                       width: CGFloat(image.width), height: CGFloat(image.height)))
        }
        let image = try #require(ctx.makeImage())
        try #require(PNGEncoder.pngData(from: image)).write(to: url)
    }

    // MARK: - tables, settings, goldens

    @Test func bothRowsAreInTheFourTables() throws {
        let r = try #require(TaskCatalog.catalog["Read Barcode"])
        #expect(r.accepts == Shape.single(Kind.image) && r.gives == Shape.single(Kind.text) && r.taskClass == .instant)
        let m = try #require(TaskCatalog.catalog["Make Barcode"])
        #expect(m.accepts == Shape.single(Kind.text) && m.gives == Shape.single(Kind.image) && m.taskClass == .instant)
        for n in ["Read Barcode", "Make Barcode"] {
            #expect(!CacheKey.neverCache.contains(n))
            #expect(TaskAvailability.supportedInstantTools.contains(n))
            #expect(CatParser.taskNames.contains(n))
        }
    }

    @Test func er3RefusalGoldenMatches() throws {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/CatFlow/goldens/check/er3_refusals.json")
        struct Case: Decodable { struct I: Decodable { var row: String; var code: String; var message: String }; var name: String; var flow: String; var issues: [I] }
        let cases = try JSONDecoder().decode([String: [Case]].self, from: Data(contentsOf: url))["cases"] ?? []
        #expect(cases.count == 12)
        for c in cases {
            let got = FlowValidator.checkFlow(try CatParser.parseForValidation(c.flow))
            #expect(got.map(\.code) == c.issues.map(\.code), "\(c.name)")
            #expect(got.map(\.message) == c.issues.map(\.message), "\(c.name)")
        }
    }

    @Test func settingsParse() throws {
        #expect(try BarcodeSettings.parseRead(nil) == "any")
        #expect(try BarcodeSettings.parseRead("type=QR") == "qr")
        let m = try BarcodeSettings.parseMake("type=Code128; scale=64")
        #expect(m.kind == "code128" && m.scale == 64)
        let d = try BarcodeSettings.parseMake(nil)
        #expect(d.kind == "qr" && d.scale == 8)
        for bad in ["type=maxicode", "qr", "type=\"\"", "rotate=yes"] {
            #expect(throws: ReadBarcodeSettingsError.self) { _ = try BarcodeSettings.parseRead(bad) }
        }
        for bad in ["type=ean13", "type=upce", "scale=0", "scale=65", "scale=2.5", "scale=-4", "scale=big", "scale=\"\"", "ec=H", "code128"] {
            #expect(throws: MakeBarcodeSettingsError.self) { _ = try BarcodeSettings.parseMake(bad) }
        }
    }

    // MARK: - Make: exact sizes

    @Test func qrSizesEqualThePythonsForEveryScale() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        // "Hello" is version 1: 21 modules -> (21 + 8) * scale square (Python: 29 / 116 / 232).
        for (scale, side) in [(1, 29), (4, 116), (8, 232)] {
            let img = try make("Hello", "scale=\(scale)", in: dir).image
            #expect(img.width == side && img.height == side, "scale \(scale): \(img.width)x\(img.height)")
        }
        let default8 = try make("Hello", in: dir).image
        #expect(default8.width == 232)
    }

    @Test func scaleFourIsHalfOfScaleEight() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        for text in ["Hello", "Rechnung für Müller", "https://example.org/a"] {
            let a = try make(text, "scale=4", in: dir).image, b = try make(text, "scale=8", in: dir).image
            #expect(a.width * 2 == b.width && a.height * 2 == b.height)
        }
    }

    @Test func code128SizesEqualThePythonsForEveryScale() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        // B-set text: modules = 11 * chars + 35 ("Hello" = 90, "x" = 46); width (modules + 20) * scale, height 50 * scale.
        for scale in [1, 4, 8] {
            let hello = try make("Hello", "type=code128; scale=\(scale)", in: dir).image
            #expect(hello.width == 110 * scale && hello.height == 50 * scale, "\(hello.width)x\(hello.height)")
            let x = try make("x", "type=code128; scale=\(scale)", in: dir).image
            #expect(x.width == 66 * scale && x.height == 50 * scale)
        }
    }

    @Test func quietZoneIsWhiteAndBarsFillTheHeight() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        func pixel(_ img: CGImage, _ x: Int, _ y: Int) -> UInt8 {
            var p: UInt8 = 255
            let ctx = CGContext(data: &p, width: 1, height: 1, bitsPerComponent: 8, bytesPerRow: 1,
                                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            ctx?.draw(img, in: CGRect(x: -x, y: -(img.height - 1 - y), width: img.width, height: img.height))
            return p
        }
        let qr = try make("Hello", "scale=2", in: dir).image
        for y in 0..<qr.height { for x in 0..<8 { #expect(pixel(qr, x, y) == 255) } }      // 4 modules * 2
        #expect(pixel(qr, 8, 8) == 0)                                                       // finder pattern corner
        let bar = try make("Hello", "type=code128; scale=2", in: dir).image
        for y in 0..<bar.height { for x in 0..<20 { #expect(pixel(bar, x, y) == 255) } }    // 10 modules * 2
        #expect(pixel(bar, 20, 0) == 0 && pixel(bar, 20, bar.height - 1) == 0)             // bars span top to bottom
    }

    // MARK: - Make: refusals

    @Test func makeRefusalsNameTheRow() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        func starts(_ m: String?, _ s: String) { #expect(m?.hasPrefix("Make Barcode \(s)") == true, "\(m ?? "nil")") }
        starts(refusal("", in: dir), "got empty text")
        starts(refusal("Müller", "type=code128", in: dir), "can't encode 'ü' (U+00FC) as Code 128 -- it takes ASCII 32-126 only; remove that character, or use `type=qr`")
        starts(refusal("tab\there", "type=code128", in: dir), "can't encode '\\t' (U+0009) as Code 128")
        starts(refusal("Hi", "type=ean13", in: dir), "has `type=ean13` -- EAN and UPC can be read but not made in v1")
        starts(refusal("Hi", "scale=0", in: dir), "has `scale=0`")
        starts(refusal("Hi", "scale=65", in: dir), "has `scale=65`")
    }

    @Test func makeRefusesNothingTwoTextsAndNonText() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        func msg(_ inputs: [Asset]) -> String? {
            do { _ = try MakeBarcodeTool.make(inputs: inputs, settings: nil, outDirectory: dir); return nil } catch let e as FlowError {
                if case .stageFailure(_, let m) = e { return m }; return "\(e)"
            } catch { return "\(error)" }
        }
        #expect(msg([])?.hasPrefix("Make Barcode got nothing to encode") == true)
        #expect(msg(text("a") + text("b"))?.hasPrefix("Make Barcode takes one text item") == true)
        let img = Asset(items: [Item(kind: .image, value: nil, path: dir, sourceText: nil)])
        #expect(msg([img])?.hasPrefix("Make Barcode takes one text item") == true)
    }

    @Test func qrCapacityMatchesThePythonAtTheEdge() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ok = try make(String(repeating: "a", count: 2331), "scale=1", in: dir).image
        #expect(ok.width == (177 + 8))                                                      // version 40: 177 modules
        let msg = try #require(refusal(String(repeating: "a", count: 2332), in: dir))
        #expect(msg.hasPrefix("Make Barcode can't encode that text as qr (too much text for one symbol)"), "\(msg)")
        // Same edges as zxing-cpp: upper-case alphanumerics 3391, digits 5596.
        _ = try make(String(repeating: "A", count: 3391), "scale=1", in: dir)
        #expect(refusal(String(repeating: "A", count: 3392), in: dir) != nil)
        _ = try make(String(repeating: "1", count: 5596), "scale=1", in: dir)
        #expect(refusal(String(repeating: "1", count: 5597), in: dir) != nil)
    }

    @Test func code128CapMatchesZxingNotCoreImage() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        // zxing-cpp: 101 characters of B-set text encode (103 symbols), 102 refuse; digit runs: 200 ok, 201 refuse, 202 ok.
        _ = try make(String(repeating: "A", count: 101), "type=code128; scale=1", in: dir)
        let msg = try #require(refusal(String(repeating: "A", count: 102), "type=code128", in: dir))
        #expect(msg.hasPrefix("Make Barcode can't encode that text as code128 (too much text for one symbol)"), "\(msg)")
        _ = try make(String(repeating: "1", count: 200), "type=code128; scale=1", in: dir)
        #expect(refusal(String(repeating: "1", count: 201), "type=code128", in: dir) != nil)
        _ = try make(String(repeating: "1", count: 202), "type=code128; scale=1", in: dir)
        #expect(refusal(String(repeating: "1", count: 204), "type=code128", in: dir) != nil)
    }

    // MARK: - round trips

    @Test func qrRoundTripsAtScales() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        for text in ["Hello", "https://example.org/a", "Rechnung für Müller", "Reçu n° 7 — 12,50 €"] {
            for scale in [4, 8] {
                let made = try make(text, "scale=\(scale)", in: dir)
                #expect(try read(made.url) == text, "\(text) @\(scale)")
                #expect(try read(made.url, "type=qr") == text)
            }
        }
    }

    @Test func code128RoundTripsAtScales() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        for text in ["Hello", "INV-2026-0042", "a b~Z 123456"] {
            for scale in [4, 8] {
                let made = try make(text, "type=code128; scale=\(scale)", in: dir)
                #expect(try read(made.url) == text, "\(text) @\(scale)")
                #expect(try read(made.url, "type=code128") == text)
            }
        }
    }

    @Test func scaleOneIsMadeToSizeAndReadWhenVisionCan() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        // At one pixel per module Vision may not find a code; the size is the contract, the read is not asserted.
        let made = try make("Hello", "type=code128; scale=1", in: dir)
        #expect(made.image.width == 110 && made.image.height == 50)
    }

    // MARK: - order and the frame

    @Test func orderingIsPureAndTopThenLeft() {
        func c(_ t: CGFloat, _ l: CGFloat, _ h: CGFloat, _ s: String) -> ReadBarcodeTool.Code { .init(top: t, left: l, height: h, text: s) }
        #expect(ReadBarcodeTool.order([c(100, 0, 50, "below"), c(0, 80, 50, "right"), c(2, 0, 50, "left")]) == ["left", "right", "below"])
        // within 10 % of the taller box: one line, left to right
        #expect(ReadBarcodeTool.order([c(4, 100, 50, "b"), c(0, 0, 50, "a")]) == ["a", "b"])
        // beyond 10 %: the lower-left code is a new line even though it is further left
        #expect(ReadBarcodeTool.order([c(0, 100, 50, "first"), c(6, 0, 50, "second")]) == ["first", "second"])
        #expect(ReadBarcodeTool.order([]).isEmpty)
    }

    @Test func visionBottomLeftConvertsToTopLeftPixels() {
        // A box in the top-right of a 200x100 image: Vision y runs up from the bottom.
        let b = ReadBarcodeTool.box(corners: [CGPoint(x: 0.5, y: 1.0), CGPoint(x: 0.75, y: 1.0), CGPoint(x: 0.75, y: 0.8), CGPoint(x: 0.5, y: 0.8)],
                                    imageWidth: 200, imageHeight: 100)
        #expect(b.top == 0 && b.left == 100 && abs(b.height - 20) < 0.0001)
    }

    @Test func upcaRidesOnEan13WithALeadingZero() {
        #expect(ReadBarcodeTool.normalise(symbology: .ean13, payload: "0123456789012") == ("upca", "123456789012"))
        #expect(ReadBarcodeTool.normalise(symbology: .ean13, payload: "5901234123457") == ("ean13", "5901234123457"))
        #expect(ReadBarcodeTool.normalise(symbology: .ean8, payload: "96385074") == ("ean8", "96385074"))
        #expect(ReadBarcodeTool.normalise(symbology: .qr, payload: "0123456789012") == ("qr", "0123456789012"))
    }

    @Test func qrAboveCode128ReadsQrFirst() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let qr = try make("https://example.org/a", "scale=6", in: dir).image
        let bar = try make("INV-2026-0042", "type=code128; scale=3", in: dir).image
        let url = dir.appendingPathComponent("stack.png")
        try compose([(qr, CGPoint(x: 40, y: 20)), (bar, CGPoint(x: 40, y: CGFloat(qr.height) + 80))],
                    size: CGSize(width: max(qr.width, bar.width) + 80, height: qr.height + bar.height + 140), to: url)
        #expect(try read(url) == "https://example.org/a\nINV-2026-0042")
        #expect(try read(url, "type=code128") == "INV-2026-0042")
        #expect(try read(url, "type=qr") == "https://example.org/a")
    }

    @Test func sideBySideReadsLeftFirst() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let left = try make("left one", "scale=6", in: dir).image
        let right = try make("right two", "scale=6", in: dir).image
        let url = dir.appendingPathComponent("row.png")
        try compose([(right, CGPoint(x: left.width + 120, y: 30)), (left, CGPoint(x: 30, y: 30))],
                    size: CGSize(width: left.width + right.width + 180, height: left.height + 60), to: url)
        #expect(try read(url) == "left one\nright two")
    }

    // MARK: - nothing found

    @Test func blankImageAndTypeMismatchGiveOneEmptyTextItemNoError() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let blank = dir.appendingPathComponent("blank.png")
        try compose([], size: CGSize(width: 200, height: 120), to: blank)
        let out = try ReadBarcodeTool.run(inputs: imageAsset(blank), settings: nil)   // what OCR does on a blank page
        #expect(out.items.count == 1 && out.items[0].kind == .text && out.items[0].value == "")
        let made = try make("Hello", in: dir)
        #expect(try read(made.url, "type=ean13") == "")      // a type matching nothing in the image
        #expect(try read(made.url, "type=pdf417") == "")
    }

    @Test func readRefusals() throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        func msg(_ inputs: [Asset], _ settings: String? = nil) -> String? {
            do { _ = try ReadBarcodeTool.run(inputs: inputs, settings: settings); return nil } catch let e as FlowError {
                if case .stageFailure(_, let m) = e { return m }; return "\(e)"
            } catch { return "\(error)" }
        }
        #expect(msg([])?.hasPrefix("Read Barcode needs an image input") == true)
        #expect(msg(text("x"))?.hasPrefix("Read Barcode needs an image item, got a text item") == true)
        #expect(msg(imageAsset(dir.appendingPathComponent("nope.png")))?.hasPrefix("Read Barcode can't read") == true)
        #expect(msg(imageAsset(dir), "type=maxicode")?.hasPrefix("Read Barcode has `type=maxicode`") == true)
    }

    // MARK: - executors

    @Test func realExecutorMakesThenReads() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let ws = FlowWorkspace(root: dir.appendingPathComponent("flows"))
        let ex = RealExecutor(workspace: ws, flowID: "f", blobDirectory: dir,
                              makeModelStage: { _, _ in BarcodeStub() }, installedModelIDs: [], catalog: [])
        let made = try await ex.execute(path: "2", row: Row(task: "Make Barcode", settings: "scale=6"), inputs: text("Rechnung für Müller"),
                                        transcript: nil, context: nil, usedFlowContent: nil)
        #expect(made.items.first?.kind == .image)
        let read = try await ex.execute(path: "3", row: Row(task: "Read Barcode"), inputs: [made],
                                        transcript: nil, context: nil, usedFlowContent: nil)
        #expect(read.items.first?.value == "Rechnung für Müller")
    }

    @Test func mockExecutorFabricates() async throws {
        let dir = try scratch(); defer { try? FileManager.default.removeItem(at: dir) }
        let mock = MockExecutor(blobDirectory: dir.appendingPathComponent("blobs"))
        let made = try await mock.execute(path: "2", row: Row(task: "Make Barcode"), inputs: text("x"),
                                          transcript: nil, context: nil, usedFlowContent: nil)
        #expect(made.items.first?.kind == .image)
        let read = try await mock.execute(path: "3", row: Row(task: "Read Barcode"), inputs: [made],
                                          transcript: nil, context: nil, usedFlowContent: nil)
        #expect(read.items.first?.kind == .text)
    }
}

private nonisolated struct BarcodeStub: PipelineStage {
    let id = "stub.er9"
    let name = "Stub"
    var accepts: MediaKind { .text }
    var produces: MediaKind { .text }
    func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media { input }
}
