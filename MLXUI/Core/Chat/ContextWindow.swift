//
//  ContextWindow.swift
//  MLXUI — chat workspace, CW-3 (`RSI/DelegateChatWorkspaceBacklog.md`, C11).
//
//  How many tokens the chat model can hold. `ModelEntry.contextWindow` comes from the catalog, but
//  it is `nil` for several installed models — including `mlx-community--Qwen3-8B-4bit`, the owner's
//  test model — so the installed `config.json` is the second source and 8 192 the last resort.
//

import Foundation

nonisolated enum ContextWindow {

    /// The fallback when neither the catalog nor the config file says.
    static let fallback = 8_192

    /// Catalog first, then the model's `config.json`, then `fallback`.
    static func tokens(for entry: ModelEntry?, modelDirectory: URL) -> Int {
        if let fromCatalog = entry?.contextWindow, fromCatalog > 0 { return fromCatalog }
        if let fromConfig = configTokens(in: modelDirectory) { return fromConfig }
        return fallback
    }

    /// `max_position_embeddings` from an installed `config.json`, or `nil` when the file is
    /// missing, unreadable or the key is absent/zero. Also reads the nested
    /// `text_config.max_position_embeddings` some VLM checkpoints use.
    static func configTokens(in directory: URL) -> Int? {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let direct = positiveInt(object["max_position_embeddings"]) { return direct }
        if let text = object["text_config"] as? [String: Any],
           let nested = positiveInt(text["max_position_embeddings"]) {
            return nested
        }
        return nil
    }

    private static func positiveInt(_ value: Any?) -> Int? {
        let number: Int?
        switch value {
        case let int as Int: number = int
        case let double as Double: number = Int(double)
        case let string as String: number = Int(string)
        default: number = nil
        }
        guard let number, number > 0 else { return nil }
        return number
    }
}
