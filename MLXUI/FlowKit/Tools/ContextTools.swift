import Foundation

// CFM-R12-7 group b — the context tools (`tools/context.py`, 37 lines). A `context`-kind
// item's inline value is a JSON array of `{label, content}` entries (the run journal);
// Save/Read move that text to/from a file, Count Context counts its entries. The interpreter
// is the only producer/consumer — Read Context's output hydrates the run journal (R7 FIX-11,
// dead until this landed).

/// `Save Context` (any → status): write the input's inline JSON text to the resolved path.
nonisolated struct SaveContextTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .anyKind }
    var produces: Shape { .single(.status) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let item = input.items.first, let value = item.value else {
            throw FlowError.missingInlineValue(row: "Save Context", kind: .context)
        }
        guard let path = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Save Context", kind: .file)
        }
        let url = try workspace.resolve(path, flowID: flowID)
        do {
            try value.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw FlowError.writeFailed(row: "Save Context", path: path)
        }
        progress(1.0)
        return Asset(items: [Item(kind: .status, value: "saved to \(path)", path: nil, sourceText: nil)])
    }
}

/// `Read Context` (file → context): read the journal JSON file as a `context` item.
nonisolated struct ReadContextTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.context) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        // FIX-11: an upstream file input wins (the Python `_resolve_path` reads it first) —
        // `Read Context (1)` fed by an upstream `Read Files` row must not fail.
        let url: URL
        if let upstream = input.items.first, upstream.kind == .file, let path = upstream.path {
            url = path
        } else if let raw = FlowSettings(settings).pathValue() {
            url = try workspace.resolve(raw, flowID: flowID)
        } else {
            throw FlowError.missingInlineValue(row: "Read Context", kind: .file)
        }
        do {
            let raw = try String(contentsOf: url, encoding: .utf8)
            progress(1.0)
            return Asset(items: [Item(kind: .context, value: raw, path: nil, sourceText: nil)])
        } catch {
            throw FlowError.fileReadFailed(row: "Read Context", path: url.lastPathComponent)
        }
    }
}

/// `Count Context` (context → text): the number of entries in the journal JSON array.
nonisolated struct CountContextTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.context) }
    var produces: Shape { .single(.text) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let item = input.items.first, let value = item.value else {
            throw FlowError.missingInlineValue(row: "Count Context", kind: .context)
        }
        guard let data = value.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw FlowError.invalidSettings(row: "Count Context", setting: "entries",
                                            detail: "the context isn't a valid journal JSON array")
        }
        progress(1.0)
        return Asset(items: [Item(kind: .text, value: String(array.count), path: nil, sourceText: nil)])
    }
}
