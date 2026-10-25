import Foundation
@testable import MLXUI

// WW4-2-1 -- the `{columns, rows}` table the two paths are compared on: its canonical serialization (owner ruling 2026-10-25:
// "identical" = the table matches after canonical serialization) and the comparison that names the first differing cell.

/// One table cell, canonicalized on construction: strings in Unicode NFC (the same text however it was composed), an
/// integral `Double` as an `Int` (`5.0` is `5`), so `==` is the canonical equality. A string `"5"` and the number `5` stay
/// different: the Q35 loop writes strings, and a path that returns a number where the other returns text has differed.
nonisolated enum ParityCell: Sendable, Equatable, Encodable, CustomStringConvertible {
    case null
    case string(String)
    case int(Int)
    case double(Double)

    /// From a JSON-decoded cell (`TableTool.Row`'s element type).
    init(_ value: Any?) {
        switch value {
        case nil: self = .null
        case let text as String: self = .string(text.precomposedStringWithCanonicalMapping)
        case let number as Int: self = .int(number)
        case let number as Double:
            if number.isFinite, number == number.rounded(), abs(number) < 9e15 {
                self = .int(Int(number))
            } else {
                self = .double(number)
            }
        case let other?: self = .string("\(other)".precomposedStringWithCanonicalMapping)
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .string(let text): try container.encode(text)
        case .int(let number): try container.encode(number)
        case .double(let number): try container.encode(number)
        }
    }

    var description: String {
        switch self {
        case .null: "null"
        case .string(let text): "\"\(text)\""
        case .int(let number): "\(number)"
        case .double(let number): "\(number)"
        }
    }
}

/// A table in canonical form. Column order and row order are significant; nothing else about how it was produced is.
nonisolated struct ParityTable: Sendable, Equatable, Encodable {
    var columns: [String]
    var rows: [[ParityCell]]

    init(columns: [String], rows: [[Any?]]) {
        self.columns = columns.map { $0.precomposedStringWithCanonicalMapping }
        self.rows = rows.map { $0.map(ParityCell.init) }
    }

    /// Reads a `.table` item the way every table tool does (`TableTool.readTable`, SPEC-Q16).
    init(item: Item) throws {
        guard item.kind == .table, let url = item.path else {
            throw FlowError.stageFailure(row: "parity", message: "not a file-backed table item")
        }
        let table = try TableTool.readTable(from: url)
        self.init(columns: table.columns, rows: table.rows)
    }

    /// The canonical serialization: compact JSON, keys sorted, no slash escaping, `{"columns":[...],"rows":[[...]]}`.
    var canonicalJSON: String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(self) else { return "" }
        return String(decoding: data, as: UTF8.self)
    }
}

/// Where two tables first differ, in the order a reader would look: the columns, then cells row by row (left to right), then
/// the row count. A path that failed outright is a difference of its own.
nonisolated enum ParityDifference: Sendable, Equatable, CustomStringConvertible {
    case pathFailed(path: ParityPath, reason: String)
    case columns(a: [String], b: [String])
    /// `nil` = that table's row has no cell at this position.
    case cell(row: Int, column: Int, name: String, a: ParityCell?, b: ParityCell?)
    case rowCount(a: Int, b: Int)

    var description: String {
        switch self {
        case .pathFailed(let path, let reason): "\(path.label) failed: \(reason)"
        case .columns(let a, let b): "columns differ: A \(a) vs B \(b)"
        case .cell(let row, let column, let name, let a, let b):
            "row \(row) column \(column) (\(name)): A \(a.map(String.init(describing:)) ?? "<none>") vs B \(b.map(String.init(describing:)) ?? "<none>")"
        case .rowCount(let a, let b): "row count differs: A \(a) vs B \(b)"
        }
    }
}

nonisolated struct ParityComparison: Sendable, Equatable {
    var identical: Bool { first == nil }
    var first: ParityDifference?

    static func compare(_ a: ParityTable, _ b: ParityTable) -> ParityComparison {
        if a.columns != b.columns { return ParityComparison(first: .columns(a: a.columns, b: b.columns)) }
        for row in 0..<min(a.rows.count, b.rows.count) {
            let left = a.rows[row], right = b.rows[row]
            for column in 0..<max(left.count, right.count) {
                let x = column < left.count ? left[column] : nil
                let y = column < right.count ? right[column] : nil
                if x != y {
                    let name = column < a.columns.count ? a.columns[column] : "#\(column)"
                    return ParityComparison(first: .cell(row: row, column: column, name: name, a: x, b: y))
                }
            }
        }
        if a.rows.count != b.rows.count { return ParityComparison(first: .rowCount(a: a.rows.count, b: b.rows.count)) }
        return ParityComparison(first: nil)
    }
}
