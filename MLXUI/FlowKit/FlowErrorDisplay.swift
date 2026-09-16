import Foundation

/// Maps every `StageError` and `FlowError` case to a plain, row-naming sentence that implies
/// the fix — CFM-R2-8's error voice. A `switch` with **no `default`**: a new error case is a
/// compile error, never a silent "Unknown error". See `RSI/DelegateMergeBacklog.md` CFM-R2-8.
nonisolated enum FlowErrorDisplay {
    static func sentence(for error: any Error) -> String {
        if let flow = error as? FlowError { return sentence(for: flow) }
        if let stage = error as? StageError { return sentence(for: stage) }
        // M1: the remaining FlowKit errors conform to `CustomStringConvertible` (not
        // `LocalizedError`) — surface the written sentence, never "couldn't be completed".
        if let convertible = error as? CustomStringConvertible {
            return convertible.description
        }
        return "This step failed: \(error.localizedDescription)"
    }

    static func sentence(for error: FlowError) -> String {
        switch error {
        case .badInputCardinality(let row, let expected, let got):
            return "Row \(row) needs \(expected), but got \(got) — check what feeds it."
        case .unsupportedKind(let row, let kind):
            return "Row \(row) produces \(kind.rawValue), which this version of Flows doesn't run yet."
        case .missingInlineValue(let row, let kind):
            return "Row \(row) is missing its \(kind.rawValue) content — the input wasn't produced."
        case .fileReadFailed(let row, let path):
            return "Row \(row) couldn't read '\(path)' — make sure it's in the flow's folder."
        case .writeFailed(let row, let path):
            return "Row \(row) couldn't write '\(path)' — the flow's folder may be read-only."
        case .frameInputOutOfRange(let index, let count):
            return "A prompt references input[\(index)], but only \(count) item(s) were given."
        case .unknownTask(let row):
            return "Row \(row) uses a task this version of Flows doesn't know — update the flow."
        case .unsupportedReference(let row):
            return "Row \(row) uses a reference this version of Flows can't follow."
        case .referenceNotFound(let row):
            return "Row \(row) references a row that hasn't run yet — check the flow's order."
        case .stageFailure(let row, let message):
            return "Row \(row) failed: \(message)"
        case .modelNotRunnable(let row, let display, let reason):
            return "Row \(row) needs \(display), which can't run: \(reason)"
        case .unsupportedTask(let row, let task):
            return "Row \(row) uses \(task), which this version of Flows doesn't run yet."
        case .invalidSettings(let row, let setting, let detail):
            return "Row \(row)'s \(setting) setting is malformed — \(detail), then run again."
        case .budgetExceeded(let row, let visitsLeq):
            return "Row \(row) hit its budget of \(visitsLeq) visits with `on_budget=fail` — no forced edge to take."
        case .missingRerankQuery(let row):
            return "Row \(row) needs a query — e.g. Rerank BGE Reranker; query=\"...\"."
        case .emptyFolder(let row, let path):
            return "Row \(row) found nothing to read in '\(path)' — add files there, or point the row at a different folder."
        case .missingInput(_, let message):
            return message
        }
    }

    static func sentence(for error: StageError) -> String {
        switch error {
        case .kindMismatch(let expected, let got):
            return "This step expected \(expected.rawValue) but got \(got.rawValue) — check what feeds it."
        case .unsupportedModel(let id, let kind):
            return "The model '\(id)' isn't runnable as \(kind.rawValue) in this version — install a supported build."
        case .modelNotInstalled(let id):
            return "The model '\(id)' isn't installed yet — install it, then run again."
        case .insufficientRAM(let required, let available):
            return String(format: "This model needs %.2f GB of RAM but this Mac has %.2f GB available.",
                          required, available)
        case .engineFailure(let stage, let underlying):
            // Name the stage and the fix; surface the underlying cause when it has a real
            // description (the bridge to the engine's own error voice).
            let cause = (underlying as? CustomStringConvertible)?.description
                ?? (underlying as NSError).localizedDescription
            if !cause.isEmpty && !cause.contains("couldn't be completed") {
                return "The \(stage) engine failed — \(cause). Check the model files and try again."
            }
            return "The \(stage) engine failed — check the model files and try again."
        case .unsupportedSetting(let setting):
            // CFM-R16-1: a row named a setting (width/height/steps) the engine can't honour.
            return "This step's \(setting) setting isn't supported by this engine yet — remove it or use a model that supports it."
        }
    }
}
