import Foundation

/// Extract Structured — schema mode ("Generate (schema mode)," whitepaper §M12-02).
///
/// Ported verbatim from `catflow-mlx/src/catflow/engines/llm.py`
/// (`_parse_schema`, `_extracted_so_far`, `_more_records_prompt`, `_field_prompt`,
/// `_normalize_field_value`, `_extract_rows`, `extract_structured`) at commit
/// `3600a4a`. **The three prompt builders are character-for-character copies** — their
/// wording encodes SPEC-Q35 (short, direct phrasing beat a longer instruction against a
/// real tiny model that otherwise echoed the field label or ignored the text) and
/// SPEC-Q90 (the two-shape `_more_records_prompt`: "does the text mention another record"
/// read as "is this data present at all," trivially true once `rows` quote values pulled
/// from the text, so the model never stopped — the explicit "ALREADY extracted, don't
/// recount, answer no if nothing new remains" wording fixed an always-empty-or-runaway
/// table on both `Qwen2.5-0.5B` and `Qwen3-8B`). Paraphrasing either silently degrades
/// small-model output and no test in this phase would catch it.
///
/// The guarantee-by-construction (SPEC-Q35): the model is **never** asked to produce the
/// table's structure — only "is there another record?" (a yes/no tag decision, real
/// constrained decoding via `RealExecutor.fireTag`) and "what is the X value?" (one bounded
/// single line). Swift owns every comma, key name and row boundary; the result is written
/// through `TableTool.tableItem`, the same on-disk `{columns, rows}` format Query Table /
/// Table to Text / Chart already read (SPEC-Q16).
///
/// `_extract_rows`'s two real model calls are injected as closures — the same split
/// `RerankStage` uses — so this orchestration is unit-tested without weights, matching the
/// Python tests' `monkeypatch` of `decide` / `_mlx_complete_line`. `RealExecutor.runModel`'s
/// `engines.llm.extract_structured` branch supplies the real ones.
///
/// Placed flat in `FlowKit/` next to `RerankStage.swift` (the backlog names
/// `FlowKit/Stages/…`, but there is no `Stages/` subdir — the tree keeps stage files flat).
///
/// See DA-3a, `RSI/DelegateDeciderBacklog.md`. Cross-references (DA-4,
/// `catflow-mlx/SPEC_QUESTIONS.md`): **SPEC-Q35** (the schema-mode design + the field-prompt
/// and normalization wording), **SPEC-Q90** (the two-shape `_more_records_prompt`),
/// **SPEC-Q112** (`text_to_table` shares this loop as its model fallback — not ported here),
/// **SPEC-Q21** (the original Phase-2 stub this closes), **SPEC-Q213** (why offering this task
/// is a Swift-only fix, DA-3b).
nonisolated enum ExtractStructuredStage {

    /// Fixed, conservative runaway-loop safety bounds — not something a row can override (the
    /// whole settings string is the schema, so there's no `max_tokens=`/`temp=` room to read
    /// out of it). SPEC-Q35's `temp=0` (CLAUDE.md determinism) is enforced **on the executor
    /// side** (DA-5): `RealExecutor`'s `engines.llm.extract_structured` branch builds this
    /// task's stage with `StageConfig(temperature: 0)`, which `ChatSDK.makeStage` threads into
    /// `LLMEngine.generate`. The gate and the field calls both run greedy; a sampled gate
    /// would make the loop's *termination* non-deterministic, not just its wording.
    static let maxRows = 20            // DEFAULT_MAX_ROWS
    static let maxFieldTokens = 32     // DEFAULT_MAX_FIELD_TOKENS
    static let continueTags = ["yes", "no"]   // _CONTINUE_TAGS

    /// SPEC-Q35: an instruct model told to "answer with an empty string" sometimes answers
    /// with the literal words instead of leaving its answer blank. Canonicalizing these to
    /// `""` is a formatting fix for an observed model quirk, not a semantic guess at content.
    static let emptyValueSentinels: Set<String> =
        ["empty string", "n/a", "na", "none", "not present", "missing"]

    // MARK: - `_parse_schema`

    static func parseSchema(_ settings: String?) throws -> [String] {
        let trimmed = settings?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else {
            throw FlowError.stageFailure(
                row: "Extract Structured",
                message: "Extract Structured needs a schema in its settings -- e.g. `Extract Structured "
                    + "Ministral 3B; \"name, date, amount\"`")
        }
        // The row's settings string is captured verbatim, quotes kept — `_unquote` strips the
        // one wrapping pair catflow's own examples always use (`"a, b, c"`) before splitting
        // on comma. Without it the quote chars leak into the first/last field name and from
        // there into every prompt built from `columns`.
        let schema = FlowSettings.unquote(trimmed)
        let columns = schema.split(separator: ",").map {
            $0.trimmingCharacters(in: .whitespaces)
        }.filter { !$0.isEmpty }
        guard !columns.isEmpty else {
            throw FlowError.stageFailure(
                row: "Extract Structured",
                message: "Extract Structured couldn't find any field names in \(trimmed)")
        }
        return columns
    }

    // MARK: - prompt builders (character-for-character from `engines/llm.py`)

    static func extractedSoFar(columns: [String], rows: [[String]]) -> String {
        if rows.isEmpty { return "(none yet)" }
        return rows.map { row in
            zip(columns, row).map { "\($0)=\($1)" }.joined(separator: ", ")
        }.joined(separator: "\n")
    }

    /// SPEC-Q35 / SPEC-Q90 — two distinct shapes. First call (empty `rows`) keeps the
    /// original SPEC-Q35 wording; second-and-later calls carry the SPEC-Q90 exclusion wording.
    static func moreRecordsPrompt(source: String, columns: [String], rows: [[String]]) -> String {
        let fields = columns.joined(separator: ", ")
        if rows.isEmpty {
            return "Text:\n\(source)\n\n"
                + "Does the text mention a record with values for: \(fields)? Answer yes or no.\n\n"
                + "(Records already extracted: \(extractedSoFar(columns: columns, rows: rows)))"
        }
        return "Text:\n\(source)\n\n"
            + "You are extracting every record with values for: \(fields).\n"
            + "Records already listed below have ALREADY been extracted -- do not count them again.\n"
            + "Records already extracted:\n\(extractedSoFar(columns: columns, rows: rows))\n\n"
            + "Is there one more record in the text, distinct from all records already listed above? "
            + "Answer no if every matching record in the text is already listed above.\n"
            + "Answer yes or no."
    }

    /// SPEC-Q35 — the short "What is the X value…" phrasing, verbatim.
    static func fieldPrompt(source: String, columns: [String], rows: [[String]], field: String) -> String {
        return "Text:\n\(source)\n\n"
            + "Records already extracted:\n\(extractedSoFar(columns: columns, rows: rows))\n\n"
            + "What is the \"\(field)\" value for the next record in the text, if any? Answer with "
            + "just the value, nothing else. If there is none, answer with an empty string.\n\n"
            + "\(field):"
    }

    // MARK: - `_normalize_field_value`

    /// Strips a `"{field}: "` echo the model sometimes prepends to its own answer (an
    /// instruct-chat artifact — it restates the field name), then canonicalizes an
    /// empty-value sentinel phrase to a real empty string.
    static func normalizeFieldValue(_ raw: String, field: String) -> String {
        let pattern = "^\\s*\(NSRegularExpression.escapedPattern(for: field))\\s*:\\s*(.*)$"
        let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive])
        let range = NSRange(raw.startIndex..<raw.endIndex, in: raw)
        var value = raw
        if let match = regex?.firstMatch(in: raw, options: [], range: range),
           match.numberOfRanges > 1,
           let captured = Range(match.range(at: 1), in: raw) {
            value = String(raw[captured])
        }
        value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        let stripped = value.trimmingCharacters(in: CharacterSet(charactersIn: "\"'.")).lowercased()
        if emptyValueSentinels.contains(stripped) { return "" }
        return value
    }

    // MARK: - `_extract_rows` + `extract_structured`

    /// The row-by-row loop: for up to `maxRows` records, ask "is there another record?"
    /// through the yes/no tag path; on `no`, stop; on `yes`, ask one bounded single-line
    /// question per column, normalize, append. Swift owns every row boundary.
    /// - Parameters:
    ///   - gate: `decide(…, tags=("yes","no"))` — returns the fired tag.
    ///   - field: `_mlx_complete_line` — one bounded line of free text for one field.
    ///   - onPrompt: fired with each sub-call's base prompt (the Python's `on_prompt`).
    static func extractRows(
        source: String,
        columns: [String],
        gate: @Sendable (_ prompt: String) async throws -> String,
        field: @Sendable (_ prompt: String) async throws -> String,
        onPrompt: (@Sendable (String) -> Void)? = nil
    ) async throws -> [[String]] {
        var rows: [[String]] = []
        for _ in 0..<maxRows {
            let morePrompt = moreRecordsPrompt(source: source, columns: columns, rows: rows)
            onPrompt?(morePrompt)
            let cont = try await gate(morePrompt)
            if cont == "no" { break }
            var row: [String] = []
            for name in columns {
                let prompt = fieldPrompt(source: source, columns: columns, rows: rows, field: name)
                onPrompt?(prompt)
                let raw = try await field(prompt)
                row.append(normalizeFieldValue(raw, field: name))
            }
            rows.append(row)
        }
        return rows
    }

    /// `extract_structured` — parse the schema, gather the source text, run the loop, and
    /// write the result through `TableTool.tableItem` (the shared `{columns, rows}` format).
    static func extract(
        inputs: [Asset],
        settings: String?,
        gate: @Sendable (_ prompt: String) async throws -> String,
        field: @Sendable (_ prompt: String) async throws -> String,
        onPrompt: (@Sendable (String) -> Void)? = nil,
        in blobDirectory: URL
    ) async throws -> Asset {
        let columns = try parseSchema(settings)
        let texts = DeciderFrame.flatTexts(inputs)
        guard !texts.isEmpty else {
            throw FlowError.stageFailure(
                row: "Extract Structured",
                message: "Extract Structured needs an input text asset to extract from")
        }
        let source = texts.joined(separator: "\n")
        let rows = try await extractRows(source: source, columns: columns,
                                         gate: gate, field: field, onPrompt: onPrompt)
        let item = try TableTool.tableItem(columns: columns,
                                           rows: rows.map { $0.map { $0 as Any? } },
                                           in: blobDirectory)
        return Asset(items: [item])
    }
}
