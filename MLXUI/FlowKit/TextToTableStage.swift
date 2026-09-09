import Foundation

/// Text to Table (P4-01, **SPEC-Q112**) — "model-assisted" in the catalog, but every corpus
/// occurrence (gallery 09 / 28 / 30, the evals guide) feeds it text a sibling
/// `Table to Text format=csv` row already produced, its header matching `expected=` exactly,
/// which is why none of those rows names a model.
///
/// Ported verbatim from `catflow-mlx/src/catflow/engines/llm.py::text_to_table` and
/// `tools/data.py::parse_delimited_table` at commit `3600a4a` (DA-6).
///
/// **The asymmetry, verbatim (SPEC-Q112 / SPEC-Q95).** A row that names **no** model first
/// tries the deterministic fast path (`TableTool.parseDelimitedTable` — the exact inverse of
/// a format this project's own tools write) and raises only if that genuinely fails. A row
/// that **does** name a model skips the fast path entirely and always calls the model —
/// explicit beats implicit — reusing `ExtractStructuredStage.extractRows`, the row-by-row
/// machinery DA-3a already built (SPEC-Q112 *is* that sharing).
///
/// `_parse_expected` reads the `expected=` **key** (a `key=value` pair), unlike Extract
/// Structured's bare quoted schema string — the two tasks spell their columns differently and
/// that asymmetry is deliberate.
nonisolated enum TextToTableStage {

    // MARK: - `_parse_expected`

    static func parseExpected(_ settings: String?) throws -> [String] {
        let value = FlowSettings(settings).value(for: "expected")
        guard let value, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw FlowError.stageFailure(
                row: "Text to Table",
                message: "Text to Table needs an `expected=` column list -- e.g. `Text to Table "
                    + "expected=\"name, date, amount\"`")
        }
        let columns = value.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }
        guard !columns.isEmpty else {
            throw FlowError.stageFailure(
                row: "Text to Table",
                message: "Text to Table couldn't find any column names in \(settings ?? "")")
        }
        return columns
    }

    // MARK: - `text_to_table`

    /// - Parameter model: `nil` for a model-less row (fast path only); the injected
    ///   `(gate, field)` calls when a model is named (fast path skipped).
    static func toTable(
        inputs: [Asset],
        settings: String?,
        model: (gate: @Sendable (_ prompt: String) async throws -> String,
                field: @Sendable (_ prompt: String) async throws -> String)?,
        onPrompt: (@Sendable (String) -> Void)? = nil,
        in blobDirectory: URL
    ) async throws -> Asset {
        let columns = try parseExpected(settings)
        let texts = DeciderFrame.flatTexts(inputs)
        guard !texts.isEmpty else {
            throw FlowError.stageFailure(
                row: "Text to Table",
                message: "Text to Table needs an input text asset to parse")
        }
        let source = texts.joined(separator: "\n")

        guard let model else {
            guard let rows = TableTool.parseDelimitedTable(source: source, columns: columns) else {
                throw FlowError.stageFailure(
                    row: "Text to Table",
                    message: "Text to Table needs a model for text that isn't already a clean table "
                        + "matching its `expected=` columns -- e.g. `Text to Table Ministral 3B; "
                        + "expected=\"name, date, amount\"`")
            }
            return Asset(items: [try TableTool.tableItem(columns: columns, rows: rows,
                                                         in: blobDirectory)])
        }

        let rows = try await ExtractStructuredStage.extractRows(
            source: source, columns: columns, gate: model.gate, field: model.field,
            onPrompt: onPrompt)
        return Asset(items: [try TableTool.tableItem(columns: columns,
                                                     rows: rows.map { $0.map { $0 as Any? } },
                                                     in: blobDirectory)])
    }
}
