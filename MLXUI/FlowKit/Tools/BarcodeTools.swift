import Foundation
import CoreGraphics
import CoreImage
import Vision

/// ER-9 — `Read Barcode` (SPEC-Q256) and `Make Barcode` (SPEC-Q257), ported from catflow-mlx's
/// `tools/barcode.py` (owner rulings R6, R7, R9, Boo, 2026-11-22). Vision reads, Core Image
/// writes; no package. One settings parse per row, shared by the tool and the validator's
/// check 25 (E720 / E721), with the Python's detail strings byte for byte.
///
/// **Measured (ER-9, this Mac):** `CICode128BarcodeGenerator` accepts up to 4096 `A`s where
/// zxing-cpp refuses past 101 (102 symbols counting start and check), and its symbol counts
/// agree with zxing's for every sample tried (B-set text, digit runs, mixed). `Make Barcode`
/// therefore applies the Python's cap itself (see `code128MaxSymbols`) rather than inherit
/// Core Image's. `CIQRCodeGenerator` at level M matches zxing at the extremes: 2331 bytes of
/// lowercase, 3391 upper-case alphanumerics, 5596 digits, and `nil` one past each.

// MARK: - Settings

nonisolated struct ReadBarcodeSettingsError: Error, Equatable {
    /// Completes "Row N's `Read Barcode` …" (E720).
    var detail: String
}

nonisolated struct MakeBarcodeSettingsError: Error, Equatable {
    /// Completes "Row N's `Make Barcode` …" (E721).
    var detail: String
}

nonisolated enum BarcodeSettings {  // SPEC-Q256 / SPEC-Q257
    static let readTypes = ["any", "qr", "code128", "ean13", "ean8", "upca", "pdf417", "datamatrix", "aztec"]
    static let makeTypes = ["qr", "code128"]
    static let minScale = 1, maxScale = 64, defaultScale = 8

    /// `parse_read_barcode` — the lower-cased `type`.
    static func parseRead(_ raw: String?) throws -> String {
        let s = FlowSettings(raw)
        let listing = "`type=` (any, qr, code128, ean13, ean8, upca, pdf417, datamatrix, aztec)"
        if let bare = s.firstBare() {
            throw ReadBarcodeSettingsError(detail: "doesn't take the bare value `\(bare)` -- its settings are \(listing)")
        }
        for key in s.orderedKeys where key != "type" {
            throw ReadBarcodeSettingsError(detail: "doesn't know `\(key)=` -- its settings are \(listing); remove `\(key)=`")
        }
        let given = s.value(for: "type")
        let kind = (given ?? "any").lowercased()
        guard readTypes.contains(kind) else {
            throw ReadBarcodeSettingsError(detail: "has `type=\(given ?? "")` -- the types are \(readTypes.joined(separator: ", ")); use one of them")
        }
        return kind
    }

    /// `parse_make_barcode` — `(type, scale)`.
    static func parseMake(_ raw: String?) throws -> (kind: String, scale: Int) {
        let s = FlowSettings(raw)
        let listing = "`type=qr|code128` and `scale=1-64`"
        if let bare = s.firstBare() {
            throw MakeBarcodeSettingsError(detail: "doesn't take the bare value `\(bare)` -- its settings are \(listing)")
        }
        for key in s.orderedKeys where key != "type" && key != "scale" {
            throw MakeBarcodeSettingsError(detail: "doesn't know `\(key)=` -- its settings are \(listing); remove `\(key)=`")
        }
        let given = s.value(for: "type")
        let kind = (given ?? "qr").lowercased()
        if ["ean13", "ean8", "upca", "upce"].contains(kind) {
            throw MakeBarcodeSettingsError(detail: "has `type=\(given ?? "")` -- EAN and UPC can be read but not made in v1; use `qr` or `code128`")
        }
        guard makeTypes.contains(kind) else {
            throw MakeBarcodeSettingsError(detail: "has `type=\(given ?? "")` -- the types are qr and code128; use one of them")
        }
        let rawScale = s.value(for: "scale") ?? String(defaultScale)
        guard !rawScale.isEmpty, rawScale.unicodeScalars.allSatisfy({ $0.value >= 0x30 && $0.value <= 0x39 }),
              let scale = Int(rawScale), scale >= minScale, scale <= maxScale else {
            throw MakeBarcodeSettingsError(detail: "has `scale=\(rawScale)` -- scale is a whole number of pixels per module from \(minScale) to \(maxScale)")
        }
        return (kind, scale)
    }
}

// MARK: - Read Barcode

/// `Read Barcode` (image → text): every code found, one per line, in **one frame** — image
/// pixels, origin top-left, y down; the box is the bounds of the code's four corners; sorted by
/// top edge then left edge; tops within 10 % of the taller box are one line, read left to
/// right. Vision reports a normalised, bottom-left origin; `box(...)` converts. Nothing found
/// gives one empty text item and no error — exactly what `OCR` does on a blank page (its stage
/// returns `.text("")`, persisted as one empty text item).
nonisolated enum ReadBarcodeTool {  // SPEC-Q256
    static let lineTolerance: CGFloat = 0.10

    struct Code: Equatable {
        var top: CGFloat, left: CGFloat, height: CGFloat
        var text: String
    }

    /// Python's `order_codes` — pure, so the frame is testable without an image.
    static func order(_ codes: [Code]) -> [String] {
        let sorted = codes.sorted { ($0.top, $0.left) < ($1.top, $1.left) }
        var lines: [[Code]] = []
        for code in sorted {
            if let anchor = lines.last?.first,
               abs(code.top - anchor.top) <= lineTolerance * max(code.height, anchor.height) {
                lines[lines.count - 1].append(code)
            } else {
                lines.append([code])
            }
        }
        return lines.flatMap { $0.sorted { $0.left < $1.left }.map(\.text) }
    }

    /// Vision's normalised bottom-left corners → `(top, left, height)` in image pixels, y down.
    static func box(corners: [CGPoint], imageWidth: CGFloat, imageHeight: CGFloat) -> (top: CGFloat, left: CGFloat, height: CGFloat) {
        let ys = corners.map { (1 - $0.y) * imageHeight }
        let xs = corners.map { $0.x * imageWidth }
        let top = ys.min() ?? 0
        return (top, xs.min() ?? 0, (ys.max() ?? 0) - top)
    }

    /// Vision reports UPC-A as an EAN-13 with a leading `0`; zxing reports it as UPC-A (12
    /// digits). Mirror zxing: a 13-digit EAN-13 starting `0` is UPC-A and loses the `0`.
    /// (Implementer's call, pending owner confirmation — see Q256's Swift note.)
    static func normalise(symbology: VNBarcodeSymbology, payload: String) -> (type: String, text: String) {
        if symbology == .ean13, payload.count == 13, payload.hasPrefix("0"), payload.allSatisfy(\.isASCII) {
            return ("upca", String(payload.dropFirst()))
        }
        return (type(of: symbology), payload)
    }

    private static func type(of symbology: VNBarcodeSymbology) -> String {
        switch symbology {
        case .qr: return "qr"
        case .code128: return "code128"
        case .ean13: return "ean13"
        case .ean8: return "ean8"
        case .pdf417: return "pdf417"
        case .dataMatrix: return "datamatrix"
        case .aztec: return "aztec"
        default: return "other"
        }
    }

    private static func symbologies(for kind: String) -> [VNBarcodeSymbology]? {
        switch kind {
        case "qr": return [.qr]
        case "code128": return [.code128]
        case "ean13", "upca": return [.ean13]       // UPC-A rides on Vision's EAN-13
        case "ean8": return [.ean8]
        case "pdf417": return [.pdf417]
        case "datamatrix": return [.dataMatrix]
        case "aztec": return [.aztec]
        default: return nil                          // any: every symbology Vision reads
        }
    }

    static func read(image: CGImage, kind: String) throws -> String {
        let request = VNDetectBarcodesRequest()
        if let wanted = symbologies(for: kind) { request.symbologies = wanted }
        let handler = VNImageRequestHandler(cgImage: image, options: [:])
        try handler.perform([request])
        let w = CGFloat(image.width), h = CGFloat(image.height)
        var codes: [Code] = []
        for observation in request.results ?? [] {
            guard let payload = observation.payloadStringValue else { continue }
            let normalised = normalise(symbology: observation.symbology, payload: payload)
            if kind != "any" && normalised.type != kind { continue }
            let b = box(corners: [observation.topLeft, observation.topRight, observation.bottomRight, observation.bottomLeft],
                        imageWidth: w, imageHeight: h)
            codes.append(Code(top: b.top, left: b.left, height: b.height, text: normalised.text))
        }
        return order(codes).joined(separator: "\n")
    }

    static func run(inputs: [Asset], settings: String?) throws -> Asset {
        let kind: String
        do { kind = try BarcodeSettings.parseRead(settings) } catch let e as ReadBarcodeSettingsError {
            throw FlowError.stageFailure(row: "Read Barcode", message: "Read Barcode \(e.detail).")
        }
        guard let item = inputs.first?.items.first else {
            throw FlowError.stageFailure(row: "Read Barcode", message: "Read Barcode needs an image input")
        }
        guard item.kind == .image, let path = item.path else {
            throw FlowError.stageFailure(row: "Read Barcode", message: "Read Barcode needs an image item, got a \(item.kind.rawValue) item")
        }
        guard FileManager.default.fileExists(atPath: path.path), let image = try? ImageTools.load(path) else {
            throw FlowError.stageFailure(row: "Read Barcode", message: "Read Barcode can't read \(path.path) -- check the file exists")
        }
        let text = try read(image: image, kind: kind)
        return Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }
}

// MARK: - Make Barcode

/// `Make Barcode` (text → image, a PNG): QR (level M, quiet zone 4 modules) or Code 128 (quiet
/// zone 10). Core Image makes the module grid at one pixel per module; this builds the bitmap
/// from it by nearest-neighbour, so every module is exactly `scale` pixels and the size equals
/// the Python's for the same text and scale: QR `(modules + 8) * scale` square; Code 128
/// `(modules + 20) * scale` wide and `50 * scale` tall (the bars fill the full height).
nonisolated enum MakeBarcodeTool {  // SPEC-Q257
    static let code128HeightModules = 50
    static let qrQuiet = 4, code128Quiet = 10
    /// zxing-cpp's Code 128 limit: start + data + check symbols, stop excluded. Core Image's own
    /// limit is far higher (4096 characters); the Python's is applied so a flow means the same
    /// thing in both runtimes. # SPEC-Q257
    static let code128MaxSymbols = 103
    static let qrMaxBytes = 2331   // informational: Core Image returns nil past the real capacity.

    private static func refuse(_ detail: String) -> FlowError {
        .stageFailure(row: "Make Barcode", message: "Make Barcode \(detail)")
    }

    /// The dark/light module grid Core Image draws at one pixel per module.
    private static func grid(_ image: CIImage, cropBorder: Int) throws -> (modules: [[Bool]], width: Int, height: Int) {
        let context = CIContext(options: [.useSoftwareRenderer: true])
        let extent = image.extent.integral
        let w = Int(extent.width), h = Int(extent.height)
        var rgba = [UInt8](repeating: 255, count: w * h * 4)
        context.render(image, toBitmap: &rgba, rowBytes: w * 4, bounds: extent, format: .RGBA8,
                       colorSpace: CGColorSpaceCreateDeviceRGB())
        // CIImage rows run bottom-up in extent space; `render(toBitmap:)` writes top row first.
        var rows: [[Bool]] = []
        for y in 0..<h { rows.append((0..<w).map { rgba[(y * w + $0) * 4] < 128 }) }
        if cropBorder > 0 {
            guard w > 2 * cropBorder, h > 2 * cropBorder else { throw refuse("couldn't build the symbol") }
            rows = rows[cropBorder..<(h - cropBorder)].map { Array($0[cropBorder..<(w - cropBorder)]) }
        }
        return (rows, rows.first?.count ?? 0, rows.count)
    }

    private static func bitmap(from modules: [[Bool]], quietX: Int, quietY: Int, scale: Int,
                               heightModulesOverride: Int?) throws -> CGImage {
        let cols = modules.first?.count ?? 0
        let rowsN = heightModulesOverride ?? modules.count
        let wPx = (cols + 2 * quietX) * scale, hPx = (rowsN + 2 * quietY) * scale
        var pixels = [UInt8](repeating: 255, count: wPx * hPx)
        for ry in 0..<rowsN {
            let source = modules[heightModulesOverride == nil ? ry : modules.count / 2]
            for cx in 0..<cols where source[cx] {
                for dy in 0..<scale {
                    let y = (ry + quietY) * scale + dy
                    let base = y * wPx + (cx + quietX) * scale
                    for dx in 0..<scale { pixels[base + dx] = 0 }
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(width: wPx, height: hPx, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: wPx,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent) else {
            throw refuse("couldn't build the image")
        }
        return image
    }

    static func make(inputs: [Asset], settings: String?, outDirectory: URL) throws -> Asset {
        let parsed: (kind: String, scale: Int)
        do { parsed = try BarcodeSettings.parseMake(settings) } catch let e as MakeBarcodeSettingsError {
            throw refuse("\(e.detail).")
        }
        let items = inputs.flatMap(\.items)
        guard !items.isEmpty else { throw refuse("got nothing to encode -- feed it text") }
        guard items.count == 1, items[0].kind == .text else {
            throw refuse("takes one text item -- join several with Join Text first")
        }
        let text = items[0].value ?? ""
        guard !text.isEmpty else { throw refuse("got empty text -- there is nothing to encode") }

        let image: CGImage
        if parsed.kind == "code128" {
            for scalar in text.unicodeScalars where scalar.value < 32 || scalar.value > 126 {
                throw refuse("can't encode \(MakePDFTool.pythonRepr(scalar)) (U+\(String(format: "%04X", scalar.value))) as Code 128 -- it takes ASCII 32-126 only; remove that character, or use `type=qr`")
            }
            guard let filter = CIFilter(name: "CICode128BarcodeGenerator") else { throw refuse("couldn't build the image") }
            filter.setValue(Data(text.utf8), forKey: "inputMessage")
            filter.setValue(0, forKey: "inputQuietSpace")
            guard let out = filter.outputImage else {
                throw refuse("can't encode that text as code128 (too much text for one symbol) -- shorten it, or use a different type")
            }
            let symbols = (Int(out.extent.width) - 13) / 11
            guard symbols <= code128MaxSymbols else {
                throw refuse("can't encode that text as code128 (too much text for one symbol) -- shorten it, or use a different type")
            }
            let g = try grid(out, cropBorder: 0)
            image = try bitmap(from: g.modules, quietX: code128Quiet, quietY: 0, scale: parsed.scale,
                               heightModulesOverride: code128HeightModules)
        } else {
            guard let filter = CIFilter(name: "CIQRCodeGenerator") else { throw refuse("couldn't build the image") }
            filter.setValue(Data(text.utf8), forKey: "inputMessage")
            filter.setValue("M", forKey: "inputCorrectionLevel")
            guard let out = filter.outputImage else {
                throw refuse("can't encode that text as qr (too much text for one symbol) -- shorten it, or use a different type")
            }
            // Core Image's QR carries a one-module white border; the module grid is inside it.
            let g = try grid(out, cropBorder: 1)
            image = try bitmap(from: g.modules, quietX: qrQuiet, quietY: qrQuiet, scale: parsed.scale, heightModulesOverride: nil)
        }
        guard let data = PNGEncoder.pngData(from: image) else { throw refuse("couldn't encode the PNG") }
        try FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)
        let url = outDirectory.appendingPathComponent("barcode-\(UUID().uuidString).png")
        try data.write(to: url)
        return Asset(items: [Item(kind: .image, value: nil, path: url, sourceText: nil)])
    }
}
