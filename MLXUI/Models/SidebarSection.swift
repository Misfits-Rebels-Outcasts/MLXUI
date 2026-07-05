import Foundation

struct SidebarSection: Codable, Identifiable {
    let id: String
    let name: String
    let sfSymbol: String
    let domainIds: [String]

    func modelCount(in data: BrowserData) -> Int {
        data.domains
            .filter { domainIds.contains($0.id) }
            .reduce(0) { $0 + $1.totalModelCount }
    }
}
