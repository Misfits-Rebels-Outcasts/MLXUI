import Foundation

struct BrowserData: Codable {
    let version: String
    let generatedAt: String
    let sidebarSections: [SidebarSection]
    let domainIndex: [String: DomainIndexEntry]
    let domains: [DomainNode]
}
