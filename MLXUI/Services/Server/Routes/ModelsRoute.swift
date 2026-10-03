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
    /// The OpenAI list body: `{"object":"list","data":[{"id","object":"model","created","owned_by":"mlxui"}]}`.
    static func body(ids: [String], created: Int) -> Data {
        struct Model: Encodable { let id: String; let object = "model"; let created: Int; let owned_by = "mlxui" }
        struct List: Encodable { let object = "list"; let data: [Model] }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let list = List(data: ids.map { Model(id: $0, created: created) })
        return (try? encoder.encode(list)) ?? Data()
    }
}
