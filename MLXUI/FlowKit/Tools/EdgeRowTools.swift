import Foundation

/// ER-7 — `Save File` (SPEC-Q253) and `Replace` (SPEC-Q254), ported from the Python
/// `tools/files.py::save_file` / `tools/text.py::replace` (catflow-mlx `er-stream`, owner
/// rulings R3 and R5, 2026-11-22). One settings parse each, shared by the tool and the
/// validator's check 25 (E717 / E718) — the same arrangement the Python has — and the
/// `detail` strings are the Python's, byte for byte (the check goldens pin them).

// MARK: - Save File

nonisolated struct SaveFileSettingsError: Error, Equatable {
    /// Completes "Row N's `Save File` …" (E717).
    var detail: String
}

nonisolated enum SaveFileSettings {  // SPEC-Q253
    /// `to=` is required and is the only setting. Python: `parse_save_file_to`.
    static func parseTo(_ raw: String?) throws -> String {
        let s = FlowSettings(raw)
        if let bare = s.firstBare() {
            throw SaveFileSettingsError(detail: "doesn't take the bare value `\(bare)` -- name the destination: `to=\"\(bare)\"`")
        }
        // Source order, so the first offending key is the same one the Python names.
        for key in s.orderedKeys where key != "to" {
            throw SaveFileSettingsError(detail: "doesn't know `\(key)=` -- its only setting is `to=`; remove `\(key)=`")
        }
        guard let to = s.value(for: "to") else {
            throw SaveFileSettingsError(detail: "has no `to=` -- say where the file goes, e.g. `to=\"out/report.pdf\"`")
        }
        if to.isEmpty {
            throw SaveFileSettingsError(detail: "has an empty `to=` -- give it the destination path, e.g. `to=\"out/report.pdf\"`")
        }
        return to
    }
}

/// `Save File` (file → status): the input file's bytes to `to=`, through `SaveCopy` — the
/// very write `Save Image` uses. **Known divergence (Q253 note):** Python's `Save Image`
/// has no workdir confinement; Swift's does, so a `..` or absolute `to=` is refused here
/// by row name rather than written.
nonisolated struct SaveFileTool: AssetStage {  // SPEC-Q253
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let to: String
        do { to = try SaveFileSettings.parseTo(settings) } catch let e as SaveFileSettingsError {
            throw FlowError.stageFailure(row: "Save File", message: "Save File \(e.detail).")
        }
        guard let src = input.items.first?.path else {
            throw FlowError.missingInlineValue(row: "Save File", kind: .file)
        }
        do {
            _ = try workspace.resolve(to, flowID: flowID)
        } catch let e as FlowWorkspaceError {
            throw FlowError.stageFailure(row: "Save File", message: "Save File refused `to=\(to)` -- \(e.description)")
        }
        return try SaveCopy.copy(src: src, rawPath: to, row: "Save File",
                                 workspace: workspace, flowID: flowID, progress: progress)
    }
}

// MARK: - Replace

nonisolated struct ReplaceSettingsError: Error, Equatable {
    /// Completes "Row N's `Replace` …" (E718).
    var detail: String
}

nonisolated enum ReplaceSettings {  // SPEC-Q254
    /// `NAMED_PATTERN_DEFAULTS` — the `with=` default of a named pattern.
    static let namedDefaults: [String: String] = ["emails": "<EMAIL>", "dates": "<DATE>", "urls": "<URL>"]
    static let knownKeys: Set<String> = ["find", "pattern", "with"]

    /// The compiled expression and the **literal** replacement. Python: `parse_replace`.
    /// The named patterns are `ExtractTool.presets` itself — never a copy — so `Replace` and
    /// `Extract` can never disagree about what an email is.
    static func parse(_ raw: String?) throws -> (regex: NSRegularExpression, replacement: String) {
        let s = FlowSettings(raw)
        if let bare = s.firstBare() {
            throw ReplaceSettingsError(detail: "doesn't take the bare value `\(bare)` -- name it `find=` (literal text) or `pattern=` (a regex)")
        }
        for key in s.orderedKeys where !knownKeys.contains(key) {
            throw ReplaceSettingsError(detail: "doesn't know `\(key)=` -- its settings are `find=`, `pattern=` and `with=`; remove `\(key)=`")
        }
        let hasFind = s.has("find"), hasPattern = s.has("pattern")
        if hasFind && hasPattern {
            throw ReplaceSettingsError(detail: "has both `find=` and `pattern=` -- keep exactly one of them")
        }
        if !hasFind && !hasPattern {
            throw ReplaceSettingsError(detail: "needs exactly one of `find=` (literal text) or `pattern=` (a regex, or `emails`, `dates`, `urls`) -- add one")
        }
        if hasFind {
            let find = s.value(for: "find") ?? ""
            if find.isEmpty {
                throw ReplaceSettingsError(detail: "has an empty `find=` -- there is nothing to look for; give it the text to replace")
            }
            if !s.has("with") {
                throw ReplaceSettingsError(detail: "has `find=` but no `with=` -- add `with=\"...\"` (an empty `with=\"\"` deletes the match)")
            }
            return (try compile(NSRegularExpression.escapedPattern(for: find)), s.value(for: "with") ?? "")
        }
        let pattern = s.value(for: "pattern") ?? ""
        if pattern.isEmpty {
            throw ReplaceSettingsError(detail: "has an empty `pattern=` -- give it a regex, or `emails`, `dates`, `urls`")
        }
        let named = pattern.lowercased()
        if let preset = ExtractTool.presets[named] {
            let replacement = s.has("with") ? (s.value(for: "with") ?? "") : (namedDefaults[named] ?? "")
            return (try compile(preset), replacement)
        }
        if !s.has("with") {
            throw ReplaceSettingsError(detail: "has a regex `pattern=` but no `with=` -- add `with=\"...\"` (an empty `with=\"\"` deletes the match)")
        }
        return (try compile(pattern), s.value(for: "with") ?? "")
    }

    private static func compile(_ pattern: String) throws -> NSRegularExpression {
        do { return try NSRegularExpression(pattern: pattern) } catch {
            // The regex libraries word the failure differently, so it is not quoted.
            throw ReplaceSettingsError(detail: "has a `pattern=` that isn't a valid regex -- fix the expression, or use `find=` for literal text")
        }
    }
}

/// `Replace` (text → text): exact, deterministic find-and-replace. `with=` is literal — a
/// backslash or `$1` in it is never a group reference (`escapedTemplate` keeps both literal).
nonisolated struct ReplaceTool: AssetStage {  // SPEC-Q254
    let settings: String

    init(settings: String?) { self.settings = settings ?? "" }

    var accepts: Shape { .single(.text) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let text = TextTools.singleText([input])
        let parsed: (regex: NSRegularExpression, replacement: String)
        do { parsed = try ReplaceSettings.parse(settings) } catch let e as ReplaceSettingsError {
            throw FlowError.stageFailure(row: "Replace", message: "Replace \(e.detail).")
        }
        let template = NSRegularExpression.escapedTemplate(for: parsed.replacement)
        let ns = NSRange(text.startIndex..<text.endIndex, in: text)
        let out = parsed.regex.stringByReplacingMatches(in: text, range: ns, withTemplate: template)
        progress(1.0)
        return Asset(items: [Item(kind: .text, value: out, path: nil, sourceText: nil)])
    }
}
