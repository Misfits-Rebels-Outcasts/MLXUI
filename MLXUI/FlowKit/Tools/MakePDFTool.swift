import Foundation
import CoreGraphics
import CoreText
import ImageIO

/// ER-8 — `Make PDF` (SPEC-Q255; owner ruling R4, Boo, 2026-11-22), ported from catflow-mlx's
/// `tools/pdf.py`. `text | image | [image] | table → file` (a `.pdf`), drawn with Core
/// Graphics + CoreText only (R9: no package). The behaviour is the Python's, measured:
/// A4 margin 20 mm (56.69 pt) / Letter 0.75 in (54 pt) with zero padding; Helvetica 11 pt,
/// leading 14, 11 pt after a paragraph; a single newline is a line break, a blank line a
/// paragraph break, runs of spaces collapse; a table is a 0.5 pt grid with a bold grey header
/// repeated on each page and wrapping cells; each image is on its own page, scaled to fit
/// (aspect kept, up or down), centred, top-aligned; `title=` is PDF metadata only.
///
/// **Glyph rule (trap).** Helvetica is drawn through a WinAnsi lens: a character that does
/// not encode in cp1252 — CJK, Cyrillic, Greek, emoji — or is a control character other than
/// newline/tab is refused by name. CoreText font fallback is never relied on, because it
/// would silently draw them; the check runs before any drawing.

nonisolated struct MakePdfSettingsError: Error, Equatable {
    /// Completes "Row N's `Make PDF` …" (E719).
    var detail: String
}

nonisolated enum MakePdfSettings {  // SPEC-Q255
    static let keys: Set<String> = ["page", "title"]

    /// `parse_make_pdf` — `(page, title)`, page lower-cased (`a4` | `letter`).
    static func parse(_ raw: String?) throws -> (page: String, title: String?) {
        let s = FlowSettings(raw)
        if let bare = s.firstBare() {
            throw MakePdfSettingsError(detail: "doesn't take the bare value `\(bare)` -- its settings are `page=a4|letter` and `title=`")
        }
        for key in s.orderedKeys where !keys.contains(key) {
            throw MakePdfSettingsError(detail: "doesn't know `\(key)=` -- its settings are `page=a4|letter` and `title=`; remove `\(key)=`")
        }
        let page = s.value(for: "page") ?? "a4"
        guard ["a4", "letter"].contains(page.lowercased()) else {
            throw MakePdfSettingsError(detail: "has `page=\(page)` -- the page sizes are `a4` and `letter`; use one of them")
        }
        return (page.lowercased(), s.value(for: "title"))
    }
}

nonisolated enum MakePDFTool {  // SPEC-Q255
    static let fontSize: CGFloat = 11
    static let leading: CGFloat = 14
    private static let head = "Make PDF"

    /// A run-time refusal; the text starts with the task name, as the Python's does.
    private static func refuse(_ detail: String) -> FlowError {
        .stageFailure(row: head, message: "\(head) \(detail)")
    }

    // MARK: - Glyph rule

    /// The 27 cp1252 characters in 0x80–0x9F (the rest of that block is undefined there).
    private static let cp1252High: Set<UInt32> = [
        0x20AC, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0x017D,
        0x2018, 0x2019, 0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0x017E, 0x0178,
    ]

    /// Python's `ch.encode("cp1252")` succeeding and `ord(ch) >= 32`, newline/CR/tab excepted.
    static func isDrawable(_ scalar: Unicode.Scalar) -> Bool {
        let v = scalar.value
        if v == 0x0A || v == 0x0D || v == 0x09 { return true }
        if v < 0x20 { return false }
        if v <= 0x7F { return true }
        if v >= 0xA0 && v <= 0xFF { return true }
        return cp1252High.contains(v)
    }

    /// Python's `repr()` of one character, enough for the characters this check can meet.
    static func pythonRepr(_ scalar: Unicode.Scalar) -> String {
        let v = scalar.value
        switch v {
        case 0x0A: return "'\\n'"
        case 0x0D: return "'\\r'"
        case 0x09: return "'\\t'"
        case 0x27: return "\"'\""
        case 0x5C: return "'\\\\'"
        default: break
        }
        if v < 0x20 || (v >= 0x7F && v <= 0xA0) || v == 0xAD { return String(format: "'\\x%02x'", v) }
        switch scalar.properties.generalCategory {
        case .control, .format, .surrogate, .privateUse, .unassigned, .lineSeparator, .paragraphSeparator, .spaceSeparator:
            return v > 0xFFFF ? String(format: "'\\U%08x'", v) : String(format: "'\\u%04x'", v)
        default:
            return "'\(Character(scalar))'"
        }
    }

    static func requireDrawable(_ text: String, in place: String) throws {
        for scalar in text.unicodeScalars where !isDrawable(scalar) {
            throw refuse("can't draw \(pythonRepr(scalar)) (U+\(String(format: "%04X", scalar.value))) in \(place) -- the built-in font covers Western European text only; remove or replace that character")
        }
    }

    // MARK: - Inputs

    private enum Payload {
        case text(String)
        case table(URL)
        case images([URL?])
    }

    private static func collect(_ inputs: [Asset]) throws -> Payload {
        let items = inputs.flatMap(\.items)
        guard let first = items.first else {
            throw refuse("got nothing to put in a PDF -- feed it text, a table, or one or more images")
        }
        let kinds = Set(items.map(\.kind))
        if kinds.count > 1 {
            let names = kinds.map(\.rawValue).sorted().joined(separator: ", ")
            throw refuse("got mixed inputs (\(names)) -- give it one kind at a time: text, a table, or images")
        }
        if first.kind == .image { return .images(items.map(\.path)) }
        if items.count > 1 {
            throw refuse("got \(items.count) \(first.kind.rawValue) items -- it takes one text or one table; join them first (Join Text)")
        }
        switch first.kind {
        case .text: return .text(first.value ?? "")
        case .table:
            guard let path = first.path else { throw refuse("got a table with no file behind it") }
            return .table(path)
        default: throw refuse("can't make a PDF from \(first.kind.rawValue) -- feed it text, a table, or images")
        }
    }

    /// `_paragraphs` — CR/CRLF → LF, tab → 4 spaces, blank-line separated, empty blocks dropped.
    static func paragraphs(_ text: String) -> [String] {
        let t = text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\t", with: "    ")
        var out: [String] = []
        for block in t.components(separatedBy: "\n\n") {
            let stripped = block.trimmingCharacters(in: CharacterSet(charactersIn: "\n"))
            if !stripped.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { out.append(stripped) }
        }
        return out
    }

    /// A table cell as Python's `str(cell)` would print it.
    private static func cellString(_ cell: Any?) -> String {
        guard let cell else { return "" }
        if let s = cell as? String { return s }
        if let n = cell as? NSNumber {
            if CFGetTypeID(n) == CFBooleanGetTypeID() { return n.boolValue ? "True" : "False" }
            if CFNumberIsFloatType(n) { return "\(n.doubleValue)" }
            return "\(n.int64Value)"
        }
        return "\(cell)"
    }

    // MARK: - Run

    /// Builds the PDF in `outDirectory` and returns it as a `file` asset.
    static func make(inputs: [Asset], settings: String?, outDirectory: URL) throws -> Asset {
        let parsed: (page: String, title: String?)
        do { parsed = try MakePdfSettings.parse(settings) } catch let e as MakePdfSettingsError {
            throw refuse("\(e.detail).")
        }
        let payload = try collect(inputs)

        let pageSize = parsed.page == "a4" ? CGSize(width: 595.2756, height: 841.8898) : CGSize(width: 612, height: 792)
        let margin: CGFloat = parsed.page == "a4" ? 56.6929 : 54
        var layout = PDFLayout(pageSize: pageSize, margin: margin)

        // Everything refusable is refused before any file is created.
        var draws: [(inout PDFLayout) -> Void] = []
        switch payload {
        case .text(let text):
            let paras = paragraphs(text)
            if paras.isEmpty { throw refuse("got empty text -- there is nothing to put in a PDF") }
            try requireDrawable(paras.joined(), in: "the text")
            draws.append { $0.drawParagraphs(paras) }
        case .table(let url):
            let (columns, rows) = try TableTool.readTable(from: url)
            if columns.isEmpty { throw refuse("got a table with no columns -- there is nothing to put in a PDF") }
            let cells = rows.map { $0.map(cellString) }
            try requireDrawable(columns.joined() + cells.map { $0.joined() }.joined(), in: "the table")
            draws.append { $0.drawTable(columns: columns, rows: cells) }
        case .images(let urls):
            var images: [CGImage] = []
            for (i, url) in urls.enumerated() {
                guard let url, FileManager.default.fileExists(atPath: url.path),
                      let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let image = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                    throw refuse("can't read image \(i + 1) (\(url?.path ?? "None")) -- check the file exists")
                }
                images.append(image)
            }
            draws.append { l in for image in images { l.drawImagePage(image) } }
        }

        try FileManager.default.createDirectory(at: outDirectory, withIntermediateDirectories: true)
        let out = outDirectory.appendingPathComponent("makepdf-\(UUID().uuidString).pdf")
        var box = CGRect(origin: .zero, size: pageSize)
        var aux: [CFString: Any] = [:]
        if let title = parsed.title { aux[kCGPDFContextTitle] = title }
        guard let consumer = CGDataConsumer(url: out as CFURL),
              let ctx = CGContext(consumer: consumer, mediaBox: &box, aux as CFDictionary) else {
            throw refuse("couldn't create the PDF file -- the flow's folder may be read-only")
        }
        layout.ctx = ctx
        for draw in draws { draw(&layout) }
        layout.finish()
        return Asset(items: [Item(kind: .file, value: nil, path: out, sourceText: nil)])
    }
}

// MARK: - Layout

/// The page cursor and the three drawers. Origin bottom-left (PDF space); `y` is the top of
/// the next block.
private nonisolated let fontKey = NSAttributedString.Key(kCTFontAttributeName as String)

nonisolated private struct PDFLayout {
    var ctx: CGContext?
    let pageSize: CGSize
    let margin: CGFloat
    private var y: CGFloat = 0
    private var open = false

    init(pageSize: CGSize, margin: CGFloat) {
        self.pageSize = pageSize
        self.margin = margin
    }

    private var availW: CGFloat { pageSize.width - 2 * margin }
    private var top: CGFloat { pageSize.height - margin }
    private var bottom: CGFloat { margin }
    private static let leading = MakePDFTool.leading

    private let body = CTFontCreateWithName("Helvetica" as CFString, MakePDFTool.fontSize, nil)
    private let bold = CTFontCreateWithName("Helvetica-Bold" as CFString, MakePDFTool.fontSize, nil)

    private mutating func newPage() {
        if open { ctx?.endPDFPage() }
        ctx?.beginPDFPage(nil)
        open = true
        y = top
    }

    mutating func finish() {
        if open { ctx?.endPDFPage() }
        ctx?.closePDF()
        open = false
    }

    private mutating func ensure(_ height: CGFloat) {
        if !open || y - height < bottom - 0.001 { newPage() }
    }

    // MARK: text

    /// Wrapped lines of one hard line, whitespace collapsed (a Paragraph's `split()`).
    private func wrap(_ text: String, font: CTFont, width: CGFloat) -> [String] {
        let collapsed = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        if collapsed.isEmpty { return [""] }
        let attr = NSAttributedString(string: collapsed, attributes: [fontKey: font])
        let setter = CTTypesetterCreateWithAttributedString(attr)
        let ns = collapsed as NSString
        var lines: [String] = []
        var start = 0
        while start < ns.length {
            var count = CTTypesetterSuggestLineBreak(setter, start, Double(width))
            if count < 1 { count = 1 }
            lines.append(ns.substring(with: NSRange(location: start, length: count)).trimmingCharacters(in: .whitespaces))
            start += count
        }
        return lines
    }

    private func draw(_ line: String, font: CTFont, x: CGFloat, baseline: CGFloat) {
        guard let ctx, !line.isEmpty else { return }
        let attr = NSAttributedString(string: line, attributes: [
            fontKey: font, kCTForegroundColorFromContextAttributeName as NSAttributedString.Key: true,
        ])
        ctx.saveGState()
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.textPosition = CGPoint(x: x, y: baseline)
        CTLineDraw(CTLineCreateWithAttributedString(attr), ctx)
        ctx.restoreGState()
    }

    mutating func drawParagraphs(_ paras: [String]) {
        let lineHeight = MakePDFTool.leading
        for para in paras {
            var lines: [String] = []
            for hard in para.components(separatedBy: "\n") { lines += wrap(hard, font: body, width: availW) }
            for line in lines {
                ensure(lineHeight)
                draw(line, font: body, x: margin, baseline: y - MakePDFTool.fontSize)
                y -= lineHeight
            }
            y -= MakePDFTool.fontSize   // spaceAfter
        }
    }

    // MARK: table

    private static let padX: CGFloat = 6
    private static let padY: CGFloat = 3

    mutating func drawTable(columns: [String], rows: [[String]]) {
        let n = columns.count
        // Natural column width: the widest single-line cell + padding; squeezed to the frame.
        func width(_ s: String, _ f: CTFont) -> CGFloat {
            let line = CTLineCreateWithAttributedString(NSAttributedString(string: s.split(whereSeparator: \.isWhitespace).joined(separator: " "), attributes: [fontKey: f]))
            return CGFloat(CTLineGetTypographicBounds(line, nil, nil, nil))
        }
        var natural = (0..<n).map { c in
            ([width(columns[c], bold)] + rows.map { c < $0.count ? width($0[c], body) : 0 }).max() ?? 0
        }.map { $0 + 2 * Self.padX }
        let total = natural.reduce(0, +)
        if total > availW {
            let scale = availW / total
            natural = natural.map { $0 * scale }
        }
        let widths = natural

        let hdr = rowHeight(columns: columns, font: bold, widths: widths)
        ensure(hdr)
        drawTableRow(columns, font: bold, shaded: true, widths: widths)
        for row in rows {
            let h = rowHeight(columns: row, font: body, widths: widths, count: n)
            if y - h < bottom - 0.001 {
                newPage()                     // the header repeats on each page
                drawTableRow(columns, font: bold, shaded: true, widths: widths)
            }
            drawTableRow(row, font: body, shaded: false, widths: widths)
        }
    }

    private func cellLines(_ s: String, _ f: CTFont, _ w: CGFloat) -> [String] {
        wrap(s, font: f, width: max(w - 2 * Self.padX, 1))
    }

    private func rowHeight(columns cells: [String], font: CTFont, widths: [CGFloat], count: Int? = nil) -> CGFloat {
        let n = count ?? widths.count
        let counts = (0..<n).map { c in cellLines(c < cells.count ? cells[c] : "", font, widths[c]).count }
        return CGFloat(counts.max() ?? 1) * MakePDFTool.leading + 2 * Self.padY
    }

    private mutating func drawTableRow(_ cells: [String], font: CTFont, shaded: Bool, widths: [CGFloat]) {
        let n = widths.count
        let lines = (0..<n).map { c in cellLines(c < cells.count ? cells[c] : "", font, widths[c]) }
        let h = CGFloat(lines.map(\.count).max() ?? 1) * MakePDFTool.leading + 2 * Self.padY
        var x = margin
        let rowTop = y
        for c in 0..<n {
            let rect = CGRect(x: x, y: rowTop - h, width: widths[c], height: h)
            if shaded {
                ctx?.setFillColor(CGColor(gray: 0.827, alpha: 1))
                ctx?.fill(rect)
            }
            ctx?.setStrokeColor(CGColor(gray: 0, alpha: 1))
            ctx?.setLineWidth(0.5)
            ctx?.stroke(rect)
            for (i, line) in lines[c].enumerated() {
                draw(line, font: font, x: x + Self.padX,
                     baseline: rowTop - Self.padY - CGFloat(i) * MakePDFTool.leading - MakePDFTool.fontSize)
            }
            x += widths[c]
        }
        y -= h
    }

    // MARK: images

    mutating func drawImagePage(_ image: CGImage) {
        newPage()
        let w = CGFloat(image.width), h = CGFloat(image.height)
        // 1 pt of slack, as the Python: a flowable exactly the frame's size can be refused.
        let scale = min((availW - 1) / w, (pageSize.height - 2 * margin - 1) / h)
        let dw = w * scale, dh = h * scale
        let rect = CGRect(x: margin + (availW - dw) / 2, y: top - dh, width: dw, height: dh)
        ctx?.draw(image, in: rect)
    }
}
