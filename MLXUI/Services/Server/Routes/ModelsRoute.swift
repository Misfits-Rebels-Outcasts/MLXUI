import Foundation

/// Which models `GET /v1/models` lists (design §3 "Model ids", backlog S1-2):
/// *served ∩ installed MLX chat models* (the HF repo id), plus `apple-foundation` when it is
/// served **and** Apple Intelligence reports `.ready`.
///
/// **Remote-provider models can never appear** (design P3, implementer's call pending owner
/// confirmation): they are not in the installed catalog and are not `apple-foundation`, so a
/// served id naming one is simply ignored — a test pins this, because serving "Claude through
/// localhost" with the user's API key is the one thing this server must never do (readme T13).
///
/// `kind == .llm` stands for "MLX chat model": every `llm` entry in the bundled catalog is
/// `source: mlx` today (checked 2026-10-01); if a non-MLX `llm` entry is ever added, filter here.
nonisolated enum ServedModels {
    /// The id `/v1/models` reports for Apple Foundation Models (design §3).
    static let appleFoundationID = "apple-foundation"

    static func ids(served: Set<String>, installed: InstalledModelIndex, appleFoundation: Readiness?) -> [String] {
        var result: Set<String> = []
        for entry in installed.entries where entry.kind == .llm && served.contains(entry.hfModelId) {
            result.insert(entry.hfModelId)
        }
        if served.contains(appleFoundationID), appleFoundation == .ready {
            result.insert(appleFoundationID)
        }
        return result.sorted()
    }
}

nonisolated enum ModelsRoute {
    /// The OpenAI list body:
    /// `{"object":"list","data":[{"id","object":"model","created","owned_by":"mlxui","context_length"}]}`.
    ///
    /// `context_length` is **additive** (S2-4) — OpenAI's list doesn't define one, so it is ignored
    /// by a strict client and read by the ones that look for it. The two shapes followed are
    /// OpenRouter's (`/api/v1/models` carries `context_length` per model) and LM Studio's
    /// (`/v1/models` entries carry `max_context_length`); we use OpenRouter's spelling. The value is
    /// the **same** number the Connect sheet's fitness line and the S2-6 pre-count use
    /// (`ModelContextWindow`), and 8,192 for `apple-foundation` — so a client that sizes its
    /// requests from this field is never refused by the pre-count for a prompt it was told would fit.
    static func body(ids: [String], created: Int) -> Data {
        body(ids: ids, created: created, contextLength: { _ in nil })
    }

    /// `contextLength` is injected (it reads a checkpoint's `config.json`, i.e. the disk) so this
    /// stays a pure function of its inputs in tests.
    static func body(ids: [String], created: Int, contextLength: @Sendable (String) -> Int?) -> Data {
        struct Model: Encodable {
            let id: String
            let object = "model"
            let created: Int
            let owned_by = "mlxui"
            let context_length: Int?
        }
        struct List: Encodable { let object = "list"; let data: [Model] }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let list = List(data: ids.map { Model(id: $0, created: created, context_length: contextLength($0)) })
        return (try? encoder.encode(list)) ?? Data()
    }
}
