import Foundation

/// CFM-R10-Store — the table-data pipeline ported from `catflow-mlx/src/catflow/tools/data.py`
/// and `tools/entity.py::set_field`: the `{columns, rows}` on-disk table format (SPEC-Q16),
/// Read CSV / Read JSON, Query Table (filter/group/sort/columns), Set Field, and Table to
/// Text. `Store Query` reuses `filterRows` verbatim so a filter string means the same thing
/// on a stored or an in-memory table (SPEC-Q165).
nonisolated enum TableTool {

    /// A table's rows — JSON-decoded cells (Int / Double / String / nil).
    typealias Row = [Any?]

    // MARK: - The on-disk format (SPEC-Q16)

    /// `_write_table`/`_read_table` — `{"columns": [...], "rows": [[...]]}`.
    static func writeTable(columns: [String], rows: [Row], to url: URL) throws {
        let payload: [String: Any] = ["columns": columns, "rows": rows.map { $0.map { $0 ?? NSNull() } }]
        let data = try JSONSerialization.data(withJSONObject: payload)
        try data.write(to: url)
    }

    static func readTable(from url: URL) throws -> (columns: [String], rows: [Row]) {
        let data = try Data(contentsOf: url)
        let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let columns = obj?["columns"] as? [String],
              let rawRows = obj?["rows"] as? [[Any]] else {
            throw FlowError.stageFailure(row: url.lastPathComponent,
                                         message: "isn't a table file ({columns, rows})")
        }
        return (columns, rawRows.map { $0.map { $0 is NSNull ? nil : $0 } })
    }

    /// `table_item` — a file-backed `.table` item in the scratch dir.
    static func tableItem(columns: [String], rows: [Row], in blobDirectory: URL) throws -> Item {
        try FileManager.default.createDirectory(at: blobDirectory, withIntermediateDirectories: true)
        let url = blobDirectory.appendingPathComponent("catflow-table-\(UUID().uuidString).table.json")
        try writeTable(columns: columns, rows: rows, to: url)
        return Item(kind: .table, value: nil, path: url, sourceText: nil)
    }

    // MARK: - `_coerce`

    /// `_coerce` — int, then float, then a currency-prefixed number (`$5,000` → 5000), else
    /// the string unchanged.
    static func coerce(_ value: String) -> Any {
        if let i = Int(value) { return i }
        if let d = Double(value) { return d }
        let stripped = value.trimmingCharacters(in: .whitespaces)
        if let first = stripped.first, "$€£¥".contains(first) {
            let numeric = String(stripped.dropFirst()).replacingOccurrences(of: ",", with: "")
            if let i = Int(numeric) { return i }
            if let d = Double(numeric) { return d }
        }
        return value
    }

    // MARK: - `_cell_matches`

    /// A cell's numeric value, nil for a non-number.
    static func number(_ cell: Any?) -> Double? {
        if let i = cell as? Int { return Double(i) }
        if let d = cell as? Double { return d }
        return nil
    }

    static func matches(cell: Any?, op: String, target: Any) -> Bool {
        guard let cell else { return false }
        if let a = number(cell), let b = number(target) {
            switch op {
            case "=": return a == b
            case "!=": return a != b
            case ">": return a > b
            case ">=": return a >= b
            case "<": return a < b
            case "<=": return a <= b
            default: break
            }
        }
        switch op {
        case "=": return "\(cell)" == "\(target)"
        case "!=": return "\(cell)" != "\(target)"
        default: return false
        }
    }

    /// `filter_rows` — `filter=col op value`, shared by Query Table and Store Query.
    static func filterRows(rowName: String, columns: [String], rows: [Row], filter: String) throws -> [Row] {
        let pattern = NSRegularExpression.compiled(#"(\w+)\s*(>=|<=|!=|>|<|=)\s*(.+)"#)
        let ns = NSRange(filter.startIndex..<filter.endIndex, in: filter)
        guard let match = pattern.firstMatch(in: filter, range: ns),
              let colRange = Range(match.range(at: 1), in: filter),
              let opRange = Range(match.range(at: 2), in: filter),
              let targetRange = Range(match.range(at: 3), in: filter) else {
            throw FlowError.stageFailure(row: rowName, message: "bad filter expression '\(filter)'")
        }
        let col = String(filter[colRange])
        let op = String(filter[opRange])
        guard let idx = columns.firstIndex(of: col) else {
            throw FlowError.stageFailure(row: rowName, message: "unknown column '\(col)' in filter")
        }
        let target = coerce(String(filter[targetRange]).trimmingCharacters(in: .whitespaces))
        return rows.filter { matches(cell: $0[idx], op: op, target: target) }
    }

    // MARK: - Read CSV

    /// A minimal `csv.reader` — quoted fields, `""` escapes, the configured delimiter.
    static func parseCSV(_ text: String, delimiter: Character) -> [[String]] {
        var rows: [[String]] = []
        var row: [String] = []
        var field = ""
        var inQuotes = false
        let chars = Array(text)
        var i = 0
        while i < chars.count {
            let c = chars[i]
            if inQuotes {
                if c == "\"" {
                    if i + 1 < chars.count, chars[i + 1] == "\"" {
                        field.append("\"")
                        i += 1
                    } else {
                        inQuotes = false
                    }
                } else {
                    field.append(c)
                }
            } else if c == "\"" {
                inQuotes = true
            } else if c == delimiter {
                row.append(field)
                field = ""
            } else if c == "\n" || c == "\r" {
                if !row.isEmpty || !field.isEmpty {
                    row.append(field)
                    rows.append(row)
                }
                row = []
                field = ""
                if c == "\r", i + 1 < chars.count, chars[i + 1] == "\n" { i += 1 }
            } else {
                field.append(c)
            }
            i += 1
        }
        if !row.isEmpty || !field.isEmpty {
            row.append(field)
            rows.append(row)
        }
        return rows
    }

    /// `read_csv` — the header row names the columns; data cells are coerced.
    static func readCSV(settings: String?, from url: URL) throws -> Asset {
        let s = FlowSettings(settings)
        let delimiter: Character = s.value(for: "delimiter").flatMap { $0.first } ?? ","
        let text = try String(contentsOf: url, encoding: .utf8)
        let raw = parseCSV(text, delimiter: delimiter)
        guard let header = raw.first else {
            throw FlowError.stageFailure(row: "Read CSV", message: "found no rows in \(url.lastPathComponent)")
        }
        let rows = raw.dropFirst().map { row in row.map { coerce($0) } }
        return Asset(items: [try tableItem(columns: header, rows: rows, in: scratchDir(url))])
    }

    // MARK: - Read JSON

    static func readJSON(settings: String?, from url: URL) throws -> Asset {
        let s = FlowSettings(settings)
        let at = s.value(for: "at")
        let data = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        var value: Any = data
        if let at {
            for key in at.split(separator: ".") {
                if let list = value as? [Any], let i = Int(key) {
                    guard i >= 0, i < list.count else {
                        throw FlowError.stageFailure(row: "Read JSON", message: "`at=\(at)` doesn't resolve inside \(url.lastPathComponent)")
                    }
                    value = list[i]
                } else if let dict = value as? [String: Any], let k = dict[String(key)] {
                    value = k
                } else {
                    throw FlowError.stageFailure(row: "Read JSON", message: "`at=\(at)` doesn't resolve inside \(url.lastPathComponent)")
                }
            }
        }
        guard let list = value as? [[String: Any]] else {
            throw FlowError.stageFailure(row: "Read JSON",
                                         message: "needs a list of objects at the selected location — use `at=a.b.c` to point at the array of rows")
        }
        var columns: [String] = []
        for row in list {
            for key in row.keys where !columns.contains(key) { columns.append(key) }
        }
        let rows = list.map { row in columns.map { row[$0] } }
        return Asset(items: [try tableItem(columns: columns, rows: rows, in: scratchDir(url))])
    }

    // MARK: - Query Table

    /// `query_table` — filter, then `group=` (sum over numeric columns, else count), else
    /// `sort=` + `columns=`.
    static func queryTable(settings: String?, from item: Item) throws -> Asset {
        guard let path = item.path else {
            throw FlowError.stageFailure(row: "Query Table", message: "needs a table input")
        }
        var (columns, rows) = try readTable(from: path)
        let s = FlowSettings(settings)

        if let filter = s.value(for: "filter") {
            rows = try filterRows(rowName: "Query Table", columns: columns, rows: rows, filter: filter)
        }

        if let groupCol = s.value(for: "group") {
            guard let idx = columns.firstIndex(of: groupCol) else {
                throw FlowError.stageFailure(row: "Query Table", message: "unknown column '\(groupCol)' in group")
            }
            var candidateCols: [String]
            if let colsSetting = s.value(for: "columns") {
                candidateCols = colsSetting.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                for col in candidateCols where !columns.contains(col) {
                    throw FlowError.stageFailure(row: "Query Table", message: "unknown column '\(col)' in columns")
                }
                candidateCols = candidateCols.filter { $0 != groupCol }
            } else {
                candidateCols = columns.filter { $0 != groupCol }
            }
            let sumCols = candidateCols.filter { col in
                let c = columns.firstIndex(of: col)!
                return rows.allSatisfy { number($0[c]) != nil }
            }
            if !sumCols.isEmpty {
                var sums: [String: [Double]] = [:]
                var order: [String] = []
                for row in rows {
                    let key = cellText(row[idx])
                    if sums[key] == nil { order.append(key); sums[key] = Array(repeating: 0, count: sumCols.count) }
                    for (i, col) in sumCols.enumerated() {
                        sums[key]![i] += number(row[columns.firstIndex(of: col)!]) ?? 0
                    }
                }
                columns = [groupCol] + sumCols
                rows = order.map { key in [key] + (sums[key] ?? []) }
            } else {
                var counts: [String: Int] = [:]
                var order: [String] = []
                for row in rows {
                    let key = cellText(row[idx])
                    if counts[key] == nil { order.append(key); counts[key] = 0 }
                    counts[key]! += 1
                }
                columns = [groupCol, "count"]
                rows = order.map { [counts[$0]!] }
            }
        } else {
            if let sortCol = s.value(for: "sort") {
                guard let idx = columns.firstIndex(of: sortCol) else {
                    throw FlowError.stageFailure(row: "Query Table", message: "unknown column '\(sortCol)' in sort")
                }
                let reverse = s.value(for: "direction", default: "asc") == "desc"
                rows = rows.sorted { a, b in
                    let an = number(a[idx]) ?? Double.greatestFiniteMagnitude
                    let bn = number(b[idx]) ?? Double.greatestFiniteMagnitude
                    return reverse ? an > bn : an < bn
                }
            }
            if let colsSetting = s.value(for: "columns") {
                let selected = colsSetting.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                for col in selected where !columns.contains(col) {
                    throw FlowError.stageFailure(row: "Query Table", message: "unknown column '\(col)' in columns")
                }
                let idxs = selected.compactMap { columns.firstIndex(of: $0) }
                rows = rows.map { row in idxs.map { row[$0] } }
                columns = selected
            }
        }
        return Asset(items: [try tableItem(columns: columns, rows: rows, in: path.deletingLastPathComponent())])
    }

    // MARK: - Set Field

    /// `set_field` — every `field=value` setting coerced and written into (or appended to)
    /// each row.
    static func setField(settings: String?, from item: Item) throws -> Asset {
        guard let path = item.path else {
            throw FlowError.stageFailure(row: "Set Field", message: "needs a table input")
        }
        var (columns, rows) = try readTable(from: path)
        let s = FlowSettings(settings)
        guard !s.testKeys.isEmpty else {
            throw FlowError.stageFailure(row: "Set Field", message: "needs at least one `field=value` setting")
        }
        for field in s.testKeys {
            let raw = s.value(for: field) ?? ""
            let value = coerce(raw)
            if let idx = columns.firstIndex(of: field) {
                for i in rows.indices { rows[i][idx] = value }
            } else {
                columns.append(field)
                for i in rows.indices { rows[i].append(value) }
            }
        }
        return Asset(items: [try tableItem(columns: columns, rows: rows, in: path.deletingLastPathComponent())])
    }

    // MARK: - CFM-R12-7 group c: Append Row / Merge Record (entity.py)

    /// `append_row` — add a new row from `field=value` settings; unknown fields widen the
    /// table (padding existing rows with nil).
    static func appendRow(settings: String?, from item: Item) throws -> Asset {
        guard let path = item.path, item.kind == .table else {
            throw FlowError.stageFailure(row: "Append Row", message: "needs a table input")
        }
        var (columns, rows) = try readTable(from: path)
        let s = FlowSettings(settings)
        guard !s.testKeys.isEmpty else {
            throw FlowError.stageFailure(row: "Append Row", message: "needs at least one `field=value` setting")
        }
        for field in s.testKeys where !columns.contains(field) {
            columns.append(field)
            for i in rows.indices { rows[i].append(nil) }
        }
        let newRow: Row = columns.map { c in
            guard let raw = s.value(for: c) else { return nil }
            return coerce(raw)
        }
        rows.append(newRow)
        return Asset(items: [try tableItem(columns: columns, rows: rows, in: path.deletingLastPathComponent())])
    }

    /// `merge_record` — a key join of two tables; the second table's last row wins on a
    /// duplicate key, and its unmatched rows append.
    static func mergeRecord(settings: String?, inputs: [Asset]) throws -> Asset {
        let allItems = inputs.flatMap { $0.items }.filter { $0.kind == .table }
        guard allItems.count >= 2, let pathA = allItems[0].path, let pathB = allItems[1].path else {
            throw FlowError.stageFailure(row: "Merge Record", message: "needs two table inputs")
        }
        var (columnsA, rowsA) = try readTable(from: pathA)
        let (columnsB, rowsB) = try readTable(from: pathB)
        let s = FlowSettings(settings)
        guard let key = s.value(for: "key") else {
            throw FlowError.stageFailure(row: "Merge Record", message: "needs a `key=` setting naming the join column")
        }
        guard let keyA = columnsA.firstIndex(of: key), let keyB = columnsB.firstIndex(of: key) else {
            throw FlowError.stageFailure(row: "Merge Record", message: "key column '\(key)' isn't in both tables")
        }
        let columns = columnsA + columnsB.filter { !columnsA.contains($0) }

        // Last row wins on a duplicate key within table B (the Python's dict build).
        var bByKey: [AnyHashable: Row] = [:]
        for row in rowsB { if let k = row[keyB] { bByKey[AnyHashable(String(describing: k))] = row } }

        var merged: [Row] = []
        var matched = Set<AnyHashable>()
        for row in rowsA {
            var record: [String: Any] = [:]
            for (i, c) in columnsA.enumerated() { record[c] = row[i] }
            if let k = row[keyA] {
                let keyHash = AnyHashable(String(describing: k))
                if let bRow = bByKey[keyHash] {
                    matched.insert(keyHash)
                    for (i, c) in columnsB.enumerated() { record[c] = bRow[i] }
                }
            }
            merged.append(columns.map { record[$0] })
        }
        for (keyHash, bRow) in bByKey where !matched.contains(keyHash) {
            var record: [String: Any] = [:]
            for (i, c) in columnsB.enumerated() { record[c] = bRow[i] }
            merged.append(columns.map { record[$0] })
        }
        return Asset(items: [try tableItem(columns: columns, rows: merged, in: pathA.deletingLastPathComponent())])
    }

    // MARK: - Table to Text

    /// `table_to_text` — `format=markdown` (default) or `format=csv`.
    static func tableToText(settings: String?, from item: Item) throws -> Asset {        guard let path = item.path else {
            throw FlowError.stageFailure(row: "Table to Text", message: "needs a table input")
        }
        let (columns, rows) = try readTable(from: path)
        let fmt = FlowSettings(settings).value(for: "format", default: "markdown")
        let text: String
        if fmt == "csv" {
            // Python's `csv.writer` uses CRLF line endings; the trailing newline is stripped.
            var lines = [columns.joined(separator: ",")]
            for row in rows {
                lines.append(row.map { cellText($0) }.joined(separator: ","))
            }
            text = lines.joined(separator: "\r\n")
        } else {
            let header = "| " + columns.joined(separator: " | ") + " |"
            let sep = "| " + Array(repeating: "---", count: columns.count).joined(separator: " | ") + " |"
            let body = rows.map { row in
                "| " + row.map { cellText($0) }.joined(separator: " | ") + " |"
            }
            text = ([header, sep] + body).joined(separator: "\n")
        }
        return Asset(items: [Item(kind: .text, value: text, path: nil, sourceText: nil)])
    }

    // MARK: - Cell display

    /// `str(cell)` — the Python's rendering (an Int stays "5", a Double shows as Python
    /// would, a nil cell is the empty string).
    static func cellText(_ cell: Any?) -> String {
        if let cell {
            if let i = cell as? Int { return String(i) }
            if let d = cell as? Double { return formatDouble(d) }
            return "\(cell)"
        }
        return ""
    }

    static func formatDouble(_ d: Double) -> String {
        if d == d.rounded() && abs(d) < 1e15 {
            return String(Int(d))
        }
        return "\(d)"
    }

    /// The scratch dir for table items — the flow's blob directory (the Python's `_scratch_path`).
    private static func scratchDir(_ url: URL) -> URL {
        url.deletingLastPathComponent().appendingPathComponent(".table-scratch", isDirectory: true)
    }
}
