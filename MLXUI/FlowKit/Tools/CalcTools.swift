import Foundation
import CoreGraphics
import AppKit

// CFM-R12-7 group a — the data tools: `Calculate` (a whitelist arithmetic parser, never
// `eval`), `Compare` (Calculate with a tag set — a deterministic branch), `Range` (a literal
// sequence), and `Chart` (table → image, CoreGraphics). Ported from `calc.py` / `compare.py`
// / `range_.py` / `data.py::chart`.

// MARK: - CalcEngine (the `calc.py` whitelist parser)

/// `calc.py` port: tokenizer + precedence-climbing parser over `+ - * / % ^`, parens, unary
/// minus, `sqrt abs round ln log min max`, constants `pi e`. Total and terminating — errors
/// are typed `CalcError`, never raised past the row except where the caller chooses.
nonisolated enum CalcEngine {
    enum CalcError: Error, Equatable {
        case message(String)
    }

    static let constants: [String: Double] = ["pi": Double.pi, "e": M_E]

    private static let tokenRegex = NSRegularExpression.compiled(#"\d+\.\d+|\.\d+|\d+|[A-Za-z_][A-Za-z0-9_]*|[-+*/%^(),]"#)
    private static let whitespaceRegex = NSRegularExpression.compiled(#"\s+"#)

    private enum TokenKind { static let eof = "EOF", num = "NUM", id = "ID" }

    /// `(kind, value)` — kind is the token text for operators/parens, else EOF/NUM/ID.
    private struct Token {
        let kind: String
        let value: Any
    }

    static func evaluate(_ expr: String) throws -> Double {
        let stripped = expr.trimmingCharacters(in: .whitespacesAndNewlines)
        if stripped.isEmpty { throw CalcError.message("empty expression") }
        let tokens = try tokenize(stripped)
        let parser = Parser(tokens: tokens)
        let result = try parser.parse()
        if result.isNaN { throw CalcError.message("math domain error") }
        if result.isInfinite { throw CalcError.message("overflow") }
        return result
    }

    /// `_format_number` — `round(x, 6)` (CPython's banker's rounding, applied to the exact
    /// binary value), then integral → `str(int(x))` (via `%.0f`, which never overflows —
    /// CFM-R12-FIX-8b) else `%.6f` trimmed.
    static func formatNumber(_ x: Double) -> String {
        // Already integral: print the exact integer straight from the double — a Decimal
        // round-trip would lose exact huge values like `2 ^ 100`.
        if x == x.rounded() {
            return String(format: "%.0f", x.rounded())
        }
        let rounded6 = roundBankers(x, places: 6)
        var s = String(format: "%.6f", rounded6)
        while s.hasSuffix("0") { s.removeLast() }
        if s.hasSuffix(".") { s.removeLast() }
        return s
    }

    /// CPython's `round(value, ndigits)`: round half to even on the **exact** decimal of the
    /// double — `Decimal(double)` alone is not enough (`2.675` round-trips to `2.675`, while
    /// the binary value is `2.67499…`), so the exact value comes from `%.17g` (CFM-R12-FIX-8d).
    static func roundBankers(_ x: Double, places: Int) -> Double {
        var exact = Decimal(string: String(format: "%.17g", x)) ?? Decimal(x)
        var rounded = Decimal()
        NSDecimalRound(&rounded, &exact, places, .bankers)
        return NSDecimalNumber(decimal: rounded).doubleValue
    }

    /// Exact integer-power by squaring — `2 ^ 100` is `1267650600228229401496703205376`,
    /// matching Python's `2.0**100.0` (which Apple's `pow` misses by one ULP).
    static func intPow(_ base: Double, _ n: Int) -> Double {
        var result = 1.0
        var b = base
        var e = abs(n)
        while e > 0 {
            if e & 1 == 1 { result *= b }
            e >>= 1
            if e > 0 { b *= b }
        }
        return n < 0 ? 1.0 / result : result
    }

    // MARK: Tokenizer

    private static func tokenize(_ expr: String) throws -> [Token] {
        var tokens: [Token] = []
        var pos = expr.startIndex
        while pos < expr.endIndex {
            if let ws = whitespaceRegex.firstMatch(in: expr, range: NSRange(pos..<expr.endIndex, in: expr)),
               let wsRange = Range(ws.range, in: expr), wsRange.lowerBound == pos {
                pos = wsRange.upperBound
                continue
            }
            guard let m = tokenRegex.firstMatch(in: expr, range: NSRange(pos..<expr.endIndex, in: expr)),
                  let range = Range(m.range, in: expr), range.lowerBound == pos else {
                throw CalcError.message("unexpected character '\(expr[pos])'")
            }
            let text = String(expr[range])
            pos = range.upperBound
            if text.first?.isNumber == true || text.hasPrefix(".") {
                tokens.append(Token(kind: TokenKind.num, value: Double(text) ?? 0))
            } else if text.first?.isLetter == true || text.hasPrefix("_") {
                tokens.append(Token(kind: TokenKind.id, value: text))
            } else {
                tokens.append(Token(kind: text, value: text))
            }
        }
        tokens.append(Token(kind: TokenKind.eof, value: ""))
        return tokens
    }

    private static func describe(_ tok: Token) -> String {
        switch tok.kind {
        case TokenKind.eof: return "end of input"
        case TokenKind.num: return formatNumber(tok.value as? Double ?? 0)
        default: return "\(tok.value)"
        }
    }

    // MARK: Parser

    private final class Parser {
        let tokens: [Token]
        var pos = 0

        init(tokens: [Token]) { self.tokens = tokens }

        func peek() -> Token { tokens[pos] }
        func advance() -> Token { defer { pos += 1 }; return tokens[pos] }

        func expect(_ kind: String) throws -> Token {
            let tok = advance()
            if tok.kind != kind {
                throw CalcError.message("expected '\(kind)', got '\(describe(tok))'")
            }
            return tok
        }

        func parse() throws -> Double {
            let value = try parseExpression()
            if peek().kind != TokenKind.eof {
                throw CalcError.message("unexpected token '\(describe(peek()))'")
            }
            return value
        }

        func parseExpression() throws -> Double {
            var value = try parseTerm()
            while ["+", "-"].contains(peek().kind) {
                let op = advance().kind
                let rhs = try parseTerm()
                value = op == "+" ? value + rhs : value - rhs
            }
            return value
        }

        func parseTerm() throws -> Double {
            var value = try parseUnary()
            while ["*", "/", "%"].contains(peek().kind) {
                let op = advance().kind
                let rhs = try parseUnary()
                if op == "*" {
                    value = value * rhs
                } else {
                    if rhs == 0 { throw CalcError.message("division by zero") }
                    if op == "/" {
                        value = value / rhs
                    } else {
                        // Python's `%` is floored (non-negative for a positive divisor):
                        // `-7 % 3 == 2`, not `-1` (CFM-R12-FIX-8a).
                        let m = value.truncatingRemainder(dividingBy: rhs)
                        value = m < 0 ? m + rhs : m
                    }
                }
            }
            return value
        }

        func parseUnary() throws -> Double {
            if peek().kind == "-" { advance(); return try -parseUnary() }
            if peek().kind == "+" { advance(); return try parseUnary() }
            return try parsePower()
        }

        func parsePower() throws -> Double {
            let base = try parseAtom()
            if peek().kind == "^" {
                advance()
                let exponent = try parseUnary()   // right-associative, allows `2^-2`
                // Python: `0 ^ -1` raises ZeroDivisionError -> "division by zero" (FIX-8).
                if base == 0 && exponent < 0 { throw CalcError.message("division by zero") }
                // Integral base+exponent: exact exponentiation by squaring, so `2 ^ 100`
                // lands on the exact power Apple's `pow` is one ULP below — the printed
                // integer matches Python's (CFM-R12-FIX-8b). Non-integral falls back.
                let result: Double
                if base == base.rounded(), exponent == exponent.rounded(),
                   base.isFinite, exponent.isFinite,
                   base >= -9_007_199_254_740_992, base <= 9_007_199_254_740_992,
                   exponent >= -4_096, exponent <= 4_096 {
                    result = CalcEngine.intPow(base, Int(exponent))
                } else {
                    result = pow(base, exponent)
                }
                if result.isNaN { throw CalcError.message("math domain error") }
                if result.isInfinite { throw CalcError.message("overflow") }
                return result
            }
            return base
        }

        func parseAtom() throws -> Double {
            let tok = peek()
            if tok.kind == TokenKind.num { advance(); return tok.value as? Double ?? 0 }
            if tok.kind == "(" {
                advance()
                let inner = try parseExpression()
                _ = try expect(")")
                return inner
            }
            if tok.kind == TokenKind.id {
                let name = tok.value as? String ?? ""
                advance()
                if peek().kind == "(" {
                    advance()
                    var args: [Double] = []
                    if peek().kind != ")" {
                        args.append(try parseExpression())
                        while peek().kind == "," {
                            advance()
                            args.append(try parseExpression())
                        }
                    }
                    _ = try expect(")")
                    return try callFunction(name, args: args)
                }
                guard let constant = constants[name] else {
                    throw CalcError.message("unknown identifier '\(name)'")
                }
                return constant
            }
            if tok.kind == TokenKind.eof { throw CalcError.message("unexpected end of input") }
            throw CalcError.message("unexpected token '\(describe(tok))'")
        }

        func callFunction(_ name: String, args: [Double]) throws -> Double {            func arity(_ expected: String) -> CalcError { .message("'\(name)' expects \(expected)") }
            switch name {
            case "sqrt":
                if args.count != 1 { throw arity("1 argument") }
                if args[0] < 0 { throw CalcError.message("math domain error") }
                return args[0].squareRoot()
            case "abs":
                if args.count != 1 { throw arity("1 argument") }
                return abs(args[0])
            case "round":
                if args.count == 1 { return args[0].rounded(.toNearestOrEven) }
                if args.count == 2 { return CalcEngine.roundBankers(args[0], places: Int(args[1])) }
                throw arity("1 or 2 arguments")
            case "ln":
                if args.count != 1 { throw arity("1 argument") }
                if args[0] <= 0 { throw CalcError.message("math domain error") }
                return log(args[0])
            case "log":
                if args.count != 1 { throw arity("1 argument") }
                if args[0] <= 0 { throw CalcError.message("math domain error") }
                return log10(args[0])
            case "min":
                if args.isEmpty { throw arity("at least 1 argument") }
                return args.min()!
            case "max":
                if args.isEmpty { throw arity("at least 1 argument") }
                return args.max()!
            default:
                throw CalcError.message("unknown identifier '\(name)'")
            }
        }
    }
}

// MARK: - The tools

/// `Calculate` (text → text): evaluate the input's math expression; every `CalcError` is
/// returned as the output text (the plan's "errors return as text observations" rule).
nonisolated struct CalculateTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let expr = input.items.first?.value ?? ""
        let text: String
        do {
            text = CalcEngine.formatNumber(try CalcEngine.evaluate(expr))
        } catch CalcEngine.CalcError.message(let message) {
            text = message
        }
        progress(1.0)
        return Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }
}

/// `Range` (any → text list): a literal `a..b` sequence, `step=` for the rarer case.
nonisolated struct RangeTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .anyKind }
    var produces: Shape { .listOf(.text) }

    static func parseBounds(_ settingsRaw: String?) throws -> (start: Int, stop: Int, step: Int) {
        let s = FlowSettings(settingsRaw)
        let literal = s.firstBare() ?? ""
        let pattern = NSRegularExpression.compiled(#"^(-?\d+)\.\.(-?\d+)$"#)
        let ns = NSRange(literal.startIndex..<literal.endIndex, in: literal)
        guard let m = pattern.firstMatch(in: literal, range: ns),
              let startRange = Range(m.range(at: 1), in: literal),
              let stopRange = Range(m.range(at: 2), in: literal),
              let start = Int(literal[startRange]), let stop = Int(literal[stopRange]) else {
            throw FlowError.invalidSettings(row: "Range", setting: "bounds",
                                            detail: "isn't `a..b` — write an inclusive integer range, e.g. `0..99`")
        }
        let stepRaw = s.value(for: "step") ?? "1"
        guard let step = Int(stepRaw), step > 0 else {
            throw FlowError.invalidSettings(row: "Range", setting: "step",
                                            detail: "must be a positive integer")
        }
        guard start <= stop else {
            throw FlowError.invalidSettings(row: "Range", setting: "bounds",
                                            detail: "descends — write the ascending range you meant")
        }
        return (start, stop, step)
    }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let bounds = try Self.parseBounds(settings)
        var values: [Item] = []
        var v = bounds.start
        while v <= bounds.stop {
            values.append(Item(kind: .text, value: String(v), path: nil, sourceText: nil))
            v += bounds.step
        }
        progress(1.0)
        return Asset(items: values)
    }
}

/// `Compare` (text → sameAsInput): "Calculate with a tag set" — a deterministic branch. The
/// output is the row's own input, unchanged; the fired tag (first declared tag when the
/// comparison holds, second when it doesn't) is set on the executor for the interpreter's
/// edge resolution, exactly like a decider.
nonisolated enum CompareTool {
    static let comparatorRegex = NSRegularExpression.compiled(#"^\s*(>=|<=|==|!=|>|<)\s*(.*)$"#)

    static func declaredTags(_ row: Row) -> [String] {
        if let tags = row.tags, !tags.isEmpty { return tags }
        if case .decide(let edges)? = row.clause { return edges.map(\.tag) }
        return []
    }

    static func run(row: Row, inputs: [Asset], path: String) throws -> (output: Asset, firedTag: String) {
        let output = inputs.first ?? Asset(items: [])
        let criterion = FlowSettings(row.settings).firstBare() ?? ""
        let ns = NSRange(criterion.startIndex..<criterion.endIndex, in: criterion)
        guard let m = comparatorRegex.firstMatch(in: criterion, range: ns),
              let opRange = Range(m.range(at: 1), in: criterion),
              let rhsRange = Range(m.range(at: 2), in: criterion) else {
            throw FlowError.stageFailure(row: path, message: "no comparator (>, <, >=, <=, ==, !=)")
        }
        let op = String(criterion[opRange])
        let rhsExpr = String(criterion[rhsRange])
        let leftText = inputs.first?.items.first?.value ?? ""
        let left: Double
        let right: Double
        do { left = try CalcEngine.evaluate(leftText) }
        catch { throw FlowError.stageFailure(row: path, message: "non-numeric input '\(leftText)'") }
        do { right = try CalcEngine.evaluate(rhsExpr) }
        catch { throw FlowError.stageFailure(row: path, message: "non-numeric right operand '\(rhsExpr)'") }

        let tags = declaredTags(row)
        let trueTag = tags.count > 0 ? tags[0] : "true"
        let falseTag = tags.count > 1 ? tags[1] : "false"
        let holds: Bool
        switch op {
        case ">":  holds = left > right
        case "<":  holds = left < right
        case ">=": holds = left >= right
        case "<=": holds = left <= right
        case "==": holds = left == right
        default:   holds = left != right
        }
        return (output, holds ? trueTag : falseTag)
    }
}

/// `Chart` (table → image): bar / line / scatter from the table's x/y columns, drawn with
/// CoreGraphics — the `matplotlib`-free analogue of `data.py::chart`.
nonisolated struct ChartTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.table) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let tablePath = input.items.first?.path else {
            throw FlowError.badInputCardinality(row: "Chart", expected: "a table input", got: 0)
        }
        let (columns, rows) = try TableTool.readTable(from: tablePath)
        let s = FlowSettings(settings)
        let kind = s.value(for: "kind") ?? "bar"
        let xCol = s.value(for: "x") ?? columns[0]
        let yCol = s.value(for: "y") ?? (columns.count > 1 ? columns[1] : columns[0])
        guard let xIdx = columns.firstIndex(of: xCol), let yIdx = columns.firstIndex(of: yCol) else {
            throw FlowError.stageFailure(row: "Chart",
                                         message: "column '\(xCol)'/'\(yCol)' not in \(columns)")
        }
        let xs = rows.map { $0[xIdx].map { "\($0)" } ?? "" }
        let ys = rows.map { row -> Double in
            guard let cell = row[yIdx] else { return 0 }
            if let n = cell as? NSNumber { return n.doubleValue }
            if let str = cell as? String { return Double(str) ?? 0 }
            return 0
        }
        guard !xs.isEmpty else {
            throw FlowError.stageFailure(row: "Chart", message: "the table has no rows")
        }
        let image = try Self.render(xs: xs, ys: ys, kind: kind, xLabel: xCol, yLabel: yCol)
        guard let data = PNGEncoder.pngData(from: image) else {
            throw FlowError.stageFailure(row: "Chart", message: "couldn't render the chart")
        }
        let url = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let out = url.appendingPathComponent("chart-\(UUID().uuidString).png")
        try data.write(to: out)
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }

    /// A compact CoreGraphics bar/line/scatter renderer: 800×400, axes, x labels.
    static func render(xs: [String], ys: [Double], kind: String, xLabel: String, yLabel: String) throws -> CGImage {
        let width = 800, height = 400, margin = 60
        let space = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(data: nil, width: width, height: height,
                                  bitsPerComponent: 8, bytesPerRow: width * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
            throw FlowError.stageFailure(row: "Chart", message: "couldn't create the canvas")
        }
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))

        let plotW = width - 2 * margin, plotH = height - 2 * margin
        let minY = (ys.min() ?? 0), maxY = (ys.max() ?? 1)
        let lo = min(minY, 0), hi = max(maxY, lo + 1)
        let span = max(hi - lo, 1)

        func xPos(_ i: Int) -> CGFloat {
            let n = max(xs.count - 1, 1)
            return CGFloat(margin) + CGFloat(i) * CGFloat(plotW) / CGFloat(n)
        }
        func yPos(_ v: Double) -> CGFloat {
            CGFloat(height - margin) - CGFloat((v - lo) / span) * CGFloat(plotH)
        }

        // Axes.
        ctx.setStrokeColor(CGColor(gray: 0.25, alpha: 1))
        ctx.setLineWidth(1)
        ctx.stroke(CGRect(x: CGFloat(margin), y: CGFloat(margin),
                          width: CGFloat(plotW), height: CGFloat(plotH)))
        // Y gridlines + labels.
        for i in 0...4 {
            let v = lo + Double(i) * span / 4.0
            let y = yPos(v)
            ctx.setStrokeColor(CGColor(gray: 0.85, alpha: 1))
            ctx.strokeLineSegments(between: [CGPoint(x: CGFloat(margin), y: y),
                                             CGPoint(x: CGFloat(width - margin), y: y)])
            ctx.setFillColor(CGColor(gray: 0.4, alpha: 1))
            let label = NSString(format: "%.1f", v)
            label.draw(at: CGPoint(x: 4, y: y - 6), withAttributes: [.font: NSFont.systemFont(ofSize: 9)])
        }
        // X labels.
        for (i, x) in xs.enumerated() {
            ctx.setFillColor(CGColor(gray: 0.4, alpha: 1))
            let label = (x as NSString)
            label.draw(at: CGPoint(x: xPos(i) - 12, y: CGFloat(margin) - 14),
                       withAttributes: [.font: NSFont.systemFont(ofSize: 9)])
        }
        ctx.setFillColor(CGColor(gray: 0.3, alpha: 1))
        (xLabel as NSString).draw(at: CGPoint(x: CGFloat(width) / 2 - 20, y: 4),
                                  withAttributes: [.font: NSFont.systemFont(ofSize: 10)])
        (yLabel as NSString).draw(at: CGPoint(x: 6, y: CGFloat(height) / 2),
                                  withAttributes: [.font: NSFont.systemFont(ofSize: 10)])

        let barColor = CGColor(red: 0.2, green: 0.5, blue: 0.9, alpha: 0.85)
        let lineColor = CGColor(red: 0.9, green: 0.3, blue: 0.2, alpha: 1)
        let baseline = yPos(0)
        switch kind {
        case "line":
            ctx.setStrokeColor(lineColor)
            ctx.setLineWidth(2)
            var pts: [CGPoint] = []
            for i in xs.indices { pts.append(CGPoint(x: xPos(i), y: yPos(ys[i]))) }
            ctx.strokeLineSegments(between: pts)
        case "scatter":
            ctx.setFillColor(lineColor)
            for i in xs.indices {
                let p = CGPoint(x: xPos(i), y: yPos(ys[i]))
                ctx.fillEllipse(in: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6))
            }
        default:   // bar
            ctx.setFillColor(barColor)
            let step = xs.count > 1 ? CGFloat(plotW) / CGFloat(xs.count - 1) : 20
            let barW = max(min(step * 0.6, 40), 4)
            for i in xs.indices {
                let cx = xPos(i)
                let y = yPos(ys[i])
                let top = min(y, baseline), bottom = max(y, baseline)
                ctx.fill(CGRect(x: cx - barW / 2, y: top, width: barW, height: max(bottom - top, 1)))
            }
        }
        guard let image = ctx.makeImage() else {
            throw FlowError.stageFailure(row: "Chart", message: "couldn't render the chart")
        }
        return image
    }
}
