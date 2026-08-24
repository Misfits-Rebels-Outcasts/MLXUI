import Foundation
import SQLite3

/// CFM-R10-Store — the store door ported from `catflow-mlx/src/catflow/tools/store.py`: Store
/// Query / Store Read / Store Write over a local SQLite (`.db`) or JSON (`.json`) store file.
///
/// - **Never SQL passthrough**: the only settings-controlled SQL fragment is `SELECT * FROM
///   <validated-identifier>`; `table=`/`key=`/column names are validated against an
///   identifier regex before any interpolation (the store is opened read-only for reads).
/// - **Store Write** mutates an *existing* table in an *existing* file (no schema inference),
///   `mode=upsert|insert|replace` with a `key=` identity column, one SQLite transaction, and a
///   snapshot copy to the flow's `.trash` before the first mutation (SPEC-Q166). **SPEC-Q
///   divergence:** the Python's trash is `~/Library/Application Support/catflow/trash`; the
///   sandboxed app keeps it in the flow workspace's `.trash`.
nonisolated enum StoreTool {

    static let identifierRegex = NSRegularExpression.compiled(#"[A-Za-z_][A-Za-z0-9_]*"#)
    static let writeModes = ["upsert", "insert", "replace"]

    // MARK: - Paths and validation

    /// `_resolve_path` — the store file: an upstream `.file` input, else `path=`/bare setting,
    /// resolved against the flow workspace.
    static func resolveStorePath(inputs: [Asset], settings: String?, workspace: FlowWorkspace, flowID: String) throws -> URL {
        if let first = inputs.first?.items.first, first.kind == .file, let path = first.path {
            return path
        }
        let s = FlowSettings(settings)
        guard let raw = s.value(for: "path") ?? s.firstBare() else {
            throw FlowError.stageFailure(row: "Store", message: "no store file given — pass one via an upstream file input or a `path=`/bare setting")
        }
        return try workspace.resolve(raw, flowID: flowID)
    }

    /// `_table_setting` — `table=` naming a valid identifier.
    static func tableSetting(settings: String?, rowName: String) throws -> String {
        let table = FlowSettings(settings).value(for: "table")
        guard let table, identifierRegex.fullMatch(table) else {
            throw FlowError.stageFailure(row: rowName, message: table.map { "`\($0)` isn't a valid table name" } ?? "needs a `table=` setting naming which table to read")
        }
        return table
    }

    // MARK: - Reading

    static func isJSONStore(_ url: URL) -> Bool {
        url.pathExtension.lowercased() == "json"
    }

    /// `_read_sqlite_table` — read-only (`mode=ro`), `SELECT * FROM <validated-table>`.
    static func readSQLiteTable(rowName: String, path: URL, table: String) throws -> (columns: [String], rows: [[Any?]]) {
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw FlowError.fileReadFailed(row: rowName, path: path.path)
        }
        var db: OpaquePointer?
        let flags = SQLITE_OPEN_READONLY
        guard sqlite3_open_v2(path.path, &db, flags, nil) == SQLITE_OK, let db else {
            throw FlowError.stageFailure(row: rowName, message: "couldn't open the store file \(path.lastPathComponent)")
        }
        defer { sqlite3_close(db) }

        var tableExists: OpaquePointer?
        let existsSQL = "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?"
        guard sqlite3_prepare_v2(db, existsSQL, -1, &tableExists, nil) == SQLITE_OK else {
            throw FlowError.stageFailure(row: rowName, message: "couldn't inspect the store file")
        }
        sqlite3_bind_text(tableExists, 1, table, -1, sqliteTransient)
        let exists = sqlite3_step(tableExists) == SQLITE_ROW
        sqlite3_finalize(tableExists)
        guard exists else {
            throw FlowError.stageFailure(row: rowName, message: "no table named '\(table)' in \(path.lastPathComponent)")
        }

        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT * FROM \(table)", -1, &stmt, nil) == SQLITE_OK, let stmt else {
            throw FlowError.stageFailure(row: rowName, message: "couldn't read table '\(table)'")
        }
        defer { sqlite3_finalize(stmt) }

        let colCount = sqlite3_column_count(stmt)
        var columns: [String] = []
        for i in 0..<colCount {
            if let name = sqlite3_column_name(stmt, i) {
                columns.append(String(cString: name))
            }
        }
        var rows: [[Any?]] = []
        while sqlite3_step(stmt) == SQLITE_ROW {
            var row: [Any?] = []
            for i in 0..<colCount {
                row.append(sqliteColumnValue(stmt, i))
            }
            rows.append(row)
        }
        return (columns, rows)
    }

    private static func sqliteColumnValue(_ stmt: OpaquePointer, _ index: Int32) -> Any? {
        switch sqlite3_column_type(stmt, index) {
        case SQLITE_INTEGER: return Int(sqlite3_column_int64(stmt, index))
        case SQLITE_FLOAT: return sqlite3_column_double(stmt, index)
        case SQLITE_TEXT:
            if let text = sqlite3_column_text(stmt, index) { return String(cString: text) }
            return nil
        case SQLITE_NULL: return nil
        default:
            if let text = sqlite3_column_text(stmt, index) { return String(cString: text) }
            return nil
        }
    }

    /// `_read_json_table` — a JSON store is `{"<table>": [{...}, ...], ...}`. **SPEC-Q
    /// divergence:** Swift dictionaries are unordered, so the Python's insertion-order column
    /// discovery can't be reproduced — columns are sorted alphabetically, which is
    /// deterministic and matches the corpus (id/status/total).
    static func readJSONTable(rowName: String, path: URL, table: String) throws -> (columns: [String], rows: [[Any?]]) {
        let doc = try readJSONDocument(rowName: rowName, path: path)
        guard let rowsDict = doc[table] as? [[String: Any]] else {
            throw FlowError.stageFailure(row: rowName, message: "no table named '\(table)' in \(path.lastPathComponent)")
        }
        let columns = rowsDict.reduce(into: Set<String>()) { $0.formUnion($1.keys) }.sorted()
        let rows = rowsDict.map { row in columns.map { row[$0] } }
        return (columns, rows)
    }

    static func readJSONDocument(rowName: String, path: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: path)
        guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw FlowError.stageFailure(row: rowName, message: "\(path.lastPathComponent) must be a JSON object of table name → rows")
        }
        return doc
    }

    static func readStoreTable(rowName: String, path: URL, table: String) throws -> (columns: [String], rows: [[Any?]]) {
        if isJSONStore(path) {
            return try readJSONTable(rowName: rowName, path: path, table: table)
        }
        return try readSQLiteTable(rowName: rowName, path: path, table: table)
    }

    // MARK: - Store Read / Store Query

    static func storeRead(inputs: [Asset], settings: String?, workspace: FlowWorkspace, flowID: String) throws -> Asset {
        let path = try resolveStorePath(inputs: inputs, settings: settings, workspace: workspace, flowID: flowID)
        let table = try tableSetting(settings: settings, rowName: "Store Read")
        let (columns, rows) = try readStoreTable(rowName: "Store Read", path: path, table: table)
        return Asset(items: [try TableTool.tableItem(columns: columns, rows: rows, in: path.deletingLastPathComponent())])
    }

    static func storeQuery(inputs: [Asset], settings: String?, workspace: FlowWorkspace, flowID: String) throws -> Asset {
        let path = try resolveStorePath(inputs: inputs, settings: settings, workspace: workspace, flowID: flowID)
        let table = try tableSetting(settings: settings, rowName: "Store Query")
        var (columns, rows) = try readStoreTable(rowName: "Store Query", path: path, table: table)
        if let filter = FlowSettings(settings).value(for: "filter") {
            rows = try TableTool.filterRows(rowName: "Store Query", columns: columns, rows: rows, filter: filter)
        }
        return Asset(items: [try TableTool.tableItem(columns: columns, rows: rows, in: path.deletingLastPathComponent())])
    }

    // MARK: - Store Write

    /// `SQLITE_TRANSIENT` is a C macro — Swift needs the function-pointer sentinel explicitly.
    private static let sqliteTransient = unsafeBitCast(-1, to: sqlite3_destructor_type.self)

    // MARK: - Store Write

    /// `_snapshot_to_trash` — a copy of the store file into the flow's `.trash`, taken before
    /// the first mutation (SPEC-Q166's coarse undo). A UUID suffix keeps same-second writes
    /// from colliding.
    static func snapshotToTrash(path: URL, workspace: FlowWorkspace, flowID: String) throws {
        let trashDir = workspace.directory(for: flowID).appendingPathComponent(".trash", isDirectory: true)
        try FileManager.default.createDirectory(at: trashDir, withIntermediateDirectories: true)
        let stamp = "\(Int(Date().timeIntervalSince1970 * 1000))-\(UUID().uuidString.prefix(6))"
        try FileManager.default.copyItem(at: path,
                                         to: trashDir.appendingPathComponent("\(path.lastPathComponent).\(stamp)"))
    }

    /// `store_write` — `mode=upsert|insert|replace`, `key=` identity column, one transaction.
    static func storeWrite(inputs: [Asset], settings: String?, workspace: FlowWorkspace, flowID: String) throws -> Asset {
        guard let first = inputs.first?.items.first else {
            throw FlowError.stageFailure(row: "Store Write", message: "needs a table input to write")
        }
        guard first.kind == .table, let tablePath = first.path else {
            throw FlowError.stageFailure(row: "Store Write", message: "needs a table input, got \(first.kind.rawValue)")
        }
        let (columns, rows) = try TableTool.readTable(from: tablePath)
        try validateIdentifiers(columns, rowName: "Store Write")

        let path = try resolveStorePath(inputs: inputs, settings: settings, workspace: workspace, flowID: flowID)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw FlowError.stageFailure(row: "Store Write", message: "no such store file: \(path.lastPathComponent)")
        }
        let s = FlowSettings(settings)
        let table = try tableSetting(settings: settings, rowName: "Store Write")
        let mode = s.value(for: "mode", default: "upsert") ?? "upsert"
        guard writeModes.contains(mode) else {
            throw FlowError.stageFailure(row: "Store Write", message: "mode='\(mode)' isn't one of \(writeModes.joined(separator: ", "))")
        }
        let key = s.value(for: "key")
        if mode != "insert" && key == nil {
            throw FlowError.stageFailure(row: "Store Write", message: "mode=\(mode) needs a `key=` setting naming the row-identity column")
        }
        if let key, !columns.contains(key) {
            throw FlowError.stageFailure(row: "Store Write", message: "key column '\(key)' isn't in the input table")
        }

        try snapshotToTrash(path: path, workspace: workspace, flowID: flowID)

        if isJSONStore(path) {
            try storeWriteJSON(path: path, table: table, mode: mode, key: key, columns: columns, rows: rows)
        } else {
            try storeWriteSQLite(path: path, table: table, mode: mode, key: key, columns: columns, rows: rows)
        }
        return Asset(items: [Item(kind: .status,
                                  value: "wrote \(rows.count) row(s) to '\(table)' in \(path.lastPathComponent) (mode=\(mode))",
                                  path: nil, sourceText: nil)])
    }

    private static func validateIdentifiers(_ names: [String], rowName: String) throws {
        for name in names where !identifierRegex.fullMatch(name) {
            throw FlowError.stageFailure(row: rowName, message: "'\(name)' isn't a valid column name")
        }
    }

    /// `_store_write_sqlite` — one transaction, committed or rolled back as a unit.
    private static func storeWriteSQLite(path: URL, table: String, mode: String, key: String?,
                                         columns: [String], rows: [[Any?]]) throws {
        var db: OpaquePointer?
        guard sqlite3_open(path.path, &db) == SQLITE_OK, let db else {
            throw FlowError.stageFailure(row: "Store Write", message: "couldn't open the store file")
        }
        defer { sqlite3_close(db) }

        let tableExists = try! prepare(db, "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", table)
        let exists = sqlite3_step(tableExists) == SQLITE_ROW
        sqlite3_finalize(tableExists)
        guard exists else {
            throw FlowError.stageFailure(row: "Store Write", message: "no table named '\(table)' in \(path.lastPathComponent)")
        }

        sqlite3_exec(db, "BEGIN", nil, nil, nil)
        let colList = columns.joined(separator: ", ")
        let placeholders = Array(repeating: "?", count: columns.count).joined(separator: ", ")

        if mode == "insert" {
            let sql = "INSERT INTO \(table) (\(colList)) VALUES (\(placeholders))"
            for row in rows {
                let stmt = try! prepare(db, sql)
                bindRow(stmt, row)
                if sqlite3_step(stmt) != SQLITE_DONE {
                    rollback(db); sqlite3_finalize(stmt)
                    throw FlowError.stageFailure(row: "Store Write", message: "insert failed")
                }
                sqlite3_finalize(stmt)
            }
        } else {
            let keyIdx = columns.firstIndex(of: key!)!
            let nonKeyOffsets = columns.enumerated().compactMap { $0.offset == keyIdx ? nil : $0.offset }
            let setClause = nonKeyOffsets.map { "\(columns[$0]) = ?" }.joined(separator: ", ")
            for row in rows {
                let keyValue = row[keyIdx]
                let matched = rowExists(db, table: table, key: key!, keyValue: keyValue)
                if matched && mode == "replace" {
                    deleteRow(db, table: table, key: key!, keyValue: keyValue)
                }
                if matched && mode != "replace" {
                    let stmt = try! prepare(db, "UPDATE \(table) SET \(setClause) WHERE \(key!) = ?")
                    for (i, offset) in nonKeyOffsets.enumerated() {
                        bindCell(stmt, Int32(i + 1), row[offset])
                    }
                    bindCell(stmt, Int32(nonKeyOffsets.count + 1), keyValue)
                    if sqlite3_step(stmt) != SQLITE_DONE {
                        rollback(db); sqlite3_finalize(stmt)
                        throw FlowError.stageFailure(row: "Store Write", message: "update failed")
                    }
                    sqlite3_finalize(stmt)
                } else {
                    let stmt = try! prepare(db, "INSERT INTO \(table) (\(colList)) VALUES (\(placeholders))")
                    bindRow(stmt, row)
                    if sqlite3_step(stmt) != SQLITE_DONE {
                        rollback(db); sqlite3_finalize(stmt)
                        throw FlowError.stageFailure(row: "Store Write", message: "insert failed")
                    }
                    sqlite3_finalize(stmt)
                }
            }
        }
        sqlite3_exec(db, "COMMIT", nil, nil, nil)
    }

    private static func prepare(_ db: OpaquePointer, _ sql: String, _ bind: String? = nil) throws -> OpaquePointer {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else {
            throw FlowError.stageFailure(row: "Store Write", message: "couldn't prepare SQL")
        }
        if let bind {
            sqlite3_bind_text(stmt, 1, bind, -1, sqliteTransient)
        }
        return stmt
    }

    private static func rowExists(_ db: OpaquePointer, table: String, key: String, keyValue: Any?) -> Bool {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "SELECT 1 FROM \(table) WHERE \(key) = ?", -1, &stmt, nil) == SQLITE_OK, let stmt else { return false }
        bindCell(stmt, 1, keyValue)
        let result = sqlite3_step(stmt) == SQLITE_ROW
        sqlite3_finalize(stmt)
        return result
    }

    private static func deleteRow(_ db: OpaquePointer, table: String, key: String, keyValue: Any?) {
        var stmt: OpaquePointer?
        guard sqlite3_prepare_v2(db, "DELETE FROM \(table) WHERE \(key) = ?", -1, &stmt, nil) == SQLITE_OK, let stmt else { return }
        bindCell(stmt, 1, keyValue)
        sqlite3_step(stmt)
        sqlite3_finalize(stmt)
    }

    private static func bindRow(_ stmt: OpaquePointer, _ row: [Any?]) {
        for (i, cell) in row.enumerated() {
            bindCell(stmt, Int32(i + 1), cell)
        }
    }

    private static func bindCell(_ stmt: OpaquePointer, _ index: Int32, _ cell: Any?) {
        if let cell {
            if let i = cell as? Int { sqlite3_bind_int64(stmt, index, Int64(i)); return }
            if let d = cell as? Double { sqlite3_bind_double(stmt, index, d); return }
            sqlite3_bind_text(stmt, index, "\(cell)", -1, sqliteTransient)
        } else {
            sqlite3_bind_null(stmt, index)
        }
    }

    private static func rollback(_ db: OpaquePointer) {
        sqlite3_exec(db, "ROLLBACK", nil, nil, nil)
    }

    /// `_store_write_json` — whole-document read + fold + write; nothing touches disk until
    /// every input row has folded in without error.
    private static func storeWriteJSON(path: URL, table: String, mode: String, key: String?,
                                      columns: [String], rows: [[Any?]]) throws {
        var doc = try readJSONDocument(rowName: "Store Write", path: path)
        var existing = try (doc[table] as? [[String: Any]]) ?? {
            throw FlowError.stageFailure(row: "Store Write", message: "no table named '\(table)' in \(path.lastPathComponent)")
        }()

        if mode == "insert" {
            for row in rows {
                existing.append(Dictionary(uniqueKeysWithValues: zip(columns, row)))
            }
        } else {
        let keyIdx = columns.firstIndex(of: key!)!
        for row in rows {
            let keyValue = row[keyIdx]
            let newRecord = Dictionary(uniqueKeysWithValues: zip(columns, row))
            let matchPos = existing.firstIndex { "\($0[key!] ?? "")" == "\(keyValue ?? "")" }
                if let matchPos, mode == "replace" {
                    existing.remove(at: matchPos)
                }
                if let matchPos {
                    var merged = existing[matchPos]
                    for (k, v) in newRecord { merged[k] = v }
                    existing[matchPos] = merged
                } else {
                    existing.append(newRecord)
                }
            }
        }

        doc[table] = existing
        let data = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
        try (data + Data("\n".utf8)).write(to: path)
    }
}

private extension NSRegularExpression {
    func fullMatch(_ text: String) -> Bool {
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        return firstMatch(in: text, range: ns)?.range == ns
    }
}
