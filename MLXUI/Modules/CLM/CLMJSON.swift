import Foundation

/// A copy of `LayaJSON`'s shape (`Modules/Laya/LayaPrompt.swift`), except `number` keeps the
/// **literal JSON token** rather than converting to a `Double`.
///
/// Corrected 2026-09-28 (`RSI/DelegateCLMBacklog.md` CL-4a): `LayaJSON.number(Double)` can't
/// tell `101050` from `101050.0`, and Python's `str()` renders them differently. An earlier
/// version of this port rendered every integral double without its `.0`, which fed the
/// encoder **different text** than Python for any JSON state with a float field — 6 of the 7
/// `answers.json` disagreements in CL-4b traced back to exactly this. Do not change
/// `LayaJSON`; CLM gets its own type instead.
nonisolated indirect enum CLMJSON: Equatable, Sendable {
    case string(String)
    case bool(Bool)
    /// The literal token exactly as written in the source JSON (`"101050.0"`, `"-5"`,
    /// `"1e16"`, a 20-digit integer) — never parsed into a fixed-width numeric type, so
    /// arbitrarily large integers round-trip exactly.
    case number(String)
    case null
    case array([CLMJSON])
    case object([CLMJSONField])
}

/// An ordered `key: value` pair — same reason `LayaJSONField` is ordered (Python dicts
/// preserve insertion order, and `json.dumps`'s output order depends on it).
nonisolated struct CLMJSONField: Equatable, Sendable {
    let key: String
    let value: CLMJSON
}

nonisolated enum CLMJSONError: Error, CustomStringConvertible, Equatable {
    case unexpectedCharacter(Character, at: Int)
    case unexpectedEnd
    case invalidEscape(at: Int)

    var description: String {
        switch self {
        case .unexpectedCharacter(let c, let i): return "Unexpected character '\(c)' at position \(i)"
        case .unexpectedEnd: return "Unexpected end of JSON"
        case .invalidEscape(let i): return "Invalid escape sequence at position \(i)"
        }
    }
}

nonisolated extension CLMJSON {
    /// Renders a `.number` token the way Python's `str()` would after `json.loads` parsed it:
    /// classify by the literal token (no `.`/`e`/`E` → int; otherwise float), matching
    /// Python's own `json` scanner, which decides `int(...)` vs `float(...)` the same way
    /// (checked against `id`, not re-derived: `json.loads('-0')` → Python `int` `0`, and
    /// `str(0) == "0"` — the sign is lost for integer zero, unlike `-0.0` as a float, which
    /// keeps it. `Double.description` matches Python's `repr` for finite doubles, confirmed
    /// for `101050.0`, `1.5`, `0.1`, `1e+16`, `1e-05`, `-0.0`, `1e+300`.)
    static func numberText(_ token: String) -> String {
        let isFloat = token.contains(".") || token.contains("e") || token.contains("E")
        if isFloat {
            guard let value = Double(token) else { return token }
            return value.description
        }
        if token == "-0" { return "0" }
        return token
    }
}

/// A minimal recursive-descent JSON parser whose only job `JSONSerialization`/`JSONDecoder`
/// can't do: preserve a number's literal token instead of converting it to `NSNumber`/`Double`
/// (`RSI/DelegateCLMBacklog.md` CL-4a: "Parse request JSON with a decoder that keeps the
/// number text").
nonisolated enum CLMJSONParser {
    static func parse(_ text: String) throws -> CLMJSON {
        var chars = Array(text)
        var i = 0
        skipWhitespace(&chars, &i)
        let value = try parseValue(&chars, &i)
        return value
    }

    private static func skipWhitespace(_ chars: inout [Character], _ i: inout Int) {
        while i < chars.count, chars[i] == " " || chars[i] == "\t" || chars[i] == "\n" || chars[i] == "\r" {
            i += 1
        }
    }

    private static func current(_ chars: [Character], _ i: Int) -> Character {
        i < chars.count ? chars[i] : " "
    }

    private static func parseValue(_ chars: inout [Character], _ i: inout Int) throws -> CLMJSON {
        skipWhitespace(&chars, &i)
        guard i < chars.count else { throw CLMJSONError.unexpectedEnd }
        switch chars[i] {
        case "{": return try parseObject(&chars, &i)
        case "[": return try parseArray(&chars, &i)
        case "\"": return .string(try parseString(&chars, &i))
        case "t": try expect("true", &chars, &i); return .bool(true)
        case "f": try expect("false", &chars, &i); return .bool(false)
        case "n": try expect("null", &chars, &i); return .null
        case "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9":
            return .number(try parseNumberToken(&chars, &i))
        default:
            throw CLMJSONError.unexpectedCharacter(chars[i], at: i)
        }
    }

    private static func expect(_ literal: String, _ chars: inout [Character], _ i: inout Int) throws {
        for expected in literal {
            guard i < chars.count, chars[i] == expected else {
                throw CLMJSONError.unexpectedCharacter(current(chars, i), at: i)
            }
            i += 1
        }
    }

    private static func parseObject(_ chars: inout [Character], _ i: inout Int) throws -> CLMJSON {
        i += 1
        var fields: [CLMJSONField] = []
        skipWhitespace(&chars, &i)
        if current(chars, i) == "}" { i += 1; return .object(fields) }
        while true {
            skipWhitespace(&chars, &i)
            guard current(chars, i) == "\"" else { throw CLMJSONError.unexpectedCharacter(current(chars, i), at: i) }
            let key = try parseString(&chars, &i)
            skipWhitespace(&chars, &i)
            guard current(chars, i) == ":" else { throw CLMJSONError.unexpectedCharacter(current(chars, i), at: i) }
            i += 1
            let value = try parseValue(&chars, &i)
            fields.append(CLMJSONField(key: key, value: value))
            skipWhitespace(&chars, &i)
            guard i < chars.count else { throw CLMJSONError.unexpectedEnd }
            if chars[i] == "," { i += 1; continue }
            if chars[i] == "}" { i += 1; break }
            throw CLMJSONError.unexpectedCharacter(chars[i], at: i)
        }
        return .object(fields)
    }

    private static func parseArray(_ chars: inout [Character], _ i: inout Int) throws -> CLMJSON {
        i += 1
        var items: [CLMJSON] = []
        skipWhitespace(&chars, &i)
        if current(chars, i) == "]" { i += 1; return .array(items) }
        while true {
            items.append(try parseValue(&chars, &i))
            skipWhitespace(&chars, &i)
            guard i < chars.count else { throw CLMJSONError.unexpectedEnd }
            if chars[i] == "," { i += 1; continue }
            if chars[i] == "]" { i += 1; break }
            throw CLMJSONError.unexpectedCharacter(chars[i], at: i)
        }
        return .array(items)
    }

    private static func parseString(_ chars: inout [Character], _ i: inout Int) throws -> String {
        i += 1
        var result = ""
        while true {
            guard i < chars.count else { throw CLMJSONError.unexpectedEnd }
            let c = chars[i]
            if c == "\"" { i += 1; break }
            if c == "\\" {
                i += 1
                guard i < chars.count else { throw CLMJSONError.unexpectedEnd }
                switch chars[i] {
                case "\"": result.append("\""); i += 1
                case "\\": result.append("\\"); i += 1
                case "/": result.append("/"); i += 1
                case "b": result.append("\u{08}"); i += 1
                case "f": result.append("\u{0C}"); i += 1
                case "n": result.append("\n"); i += 1
                case "r": result.append("\r"); i += 1
                case "t": result.append("\t"); i += 1
                case "u":
                    i += 1
                    guard i + 4 <= chars.count, let code = UInt32(String(chars[i ..< i + 4]), radix: 16) else {
                        throw CLMJSONError.invalidEscape(at: i)
                    }
                    i += 4
                    if (0xD800...0xDBFF).contains(code), i + 5 < chars.count, chars[i] == "\\", chars[i + 1] == "u",
                       let low = UInt32(String(chars[i + 2 ..< i + 6]), radix: 16), (0xDC00...0xDFFF).contains(low) {
                        let combined = 0x10000 + (code - 0xD800) * 0x400 + (low - 0xDC00)
                        if let scalar = Unicode.Scalar(combined) { result.unicodeScalars.append(scalar) }
                        i += 6
                    } else if let scalar = Unicode.Scalar(code) {
                        result.unicodeScalars.append(scalar)
                    }
                default:
                    throw CLMJSONError.invalidEscape(at: i)
                }
            } else {
                result.append(c)
                i += 1
            }
        }
        return result
    }

    /// `number = -? int frac? exp?`. Captures the literal substring verbatim.
    private static func parseNumberToken(_ chars: inout [Character], _ i: inout Int) throws -> String {
        let start = i
        if current(chars, i) == "-" { i += 1 }
        guard i < chars.count, chars[i].isASCII, chars[i].isNumber else {
            throw CLMJSONError.unexpectedCharacter(current(chars, i), at: i)
        }
        while i < chars.count, chars[i].isASCII, chars[i].isNumber { i += 1 }
        if current(chars, i) == "." {
            i += 1
            while i < chars.count, chars[i].isASCII, chars[i].isNumber { i += 1 }
        }
        if current(chars, i) == "e" || current(chars, i) == "E" {
            i += 1
            if current(chars, i) == "+" || current(chars, i) == "-" { i += 1 }
            while i < chars.count, chars[i].isASCII, chars[i].isNumber { i += 1 }
        }
        return String(chars[start ..< i])
    }
}
