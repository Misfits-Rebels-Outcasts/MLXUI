import Foundation

struct DomainNode: Codable, Identifiable {
    let id: String
    let name: String
    let sfSymbol: String
    let children: [DomainNode]?
    let models: [ModelEntry]?

    var totalModelCount: Int {
        if let models = models { return models.count }
        return children?.reduce(0) { $0 + $1.totalModelCount } ?? 0
    }

    var allModels: [ModelEntry] {
        if let models = models { return models }
        return children?.flatMap { $0.allModels } ?? []
    }

    /// All descendant nodes that directly carry models. Handles both the nested
    /// schema (models live on leaf nodes two levels down) and the flat schema
    /// where models hang off the domain node itself (`children == nil`).
    var leafDomains: [DomainNode] {
        if models != nil { return [self] }
        return children?.flatMap { $0.leafDomains } ?? []
    }
}

struct DomainIndexEntry: Codable {
    let icon: String
    let description: String
}
