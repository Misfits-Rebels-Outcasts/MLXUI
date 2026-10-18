import Testing
import Foundation
@testable import MLXUI

/// DSB-1 — the bundled `browser.json` lays every domain out in exactly one Browse section,
/// and the decision models have a section of their own ("Deciders"), not Infrastructure.
struct BrowseSectionsTests {

    private func browser() throws -> BrowserData {
        let url = try #require(Bundle.main.url(forResource: "browser", withExtension: "json"))
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
    }

    /// The invariant a stale `regenerate_catalog.py` would break: a domain listed by no
    /// section is unreachable in Browse; one listed twice shows its models twice.
    @Test func everyDomainIsInExactlyOneSection() throws {
        let data = try browser()
        for domain in data.domains {
            let owners = data.sidebarSections.filter { $0.domainIds.contains(domain.id) }.map(\.id)
            #expect(owners.count == 1, "domain \(domain.id) is in sections \(owners)")
        }
        let known = Set(data.domains.map(\.id))
        for section in data.sidebarSections {
            for id in section.domainIds {
                #expect(known.contains(id), "section \(section.id) lists unknown domain \(id)")
            }
        }
    }

    @Test func decisionDomainLivesInTheDecidersSection() throws {
        let data = try browser()
        let owners = data.sidebarSections.filter { $0.domainIds.contains("decision") }
        #expect(owners.map(\.id) == ["decision"])
        #expect(owners.first?.name == "Deciders")
        let infrastructure = try #require(data.sidebarSections.first { $0.id == "infrastructure" })
        #expect(!infrastructure.domainIds.contains("decision"))
    }

    @Test func decidersSectionIsVisibleWithModels() throws {
        let state = AppState()
        state.loadBrowserData()
        #expect(state.loadError == nil, "browser.json must decode: \(state.loadError ?? "")")
        let data = try #require(state.browserData)
        let section = try #require(state.visibleSections.first { $0.id == "decision" })
        #expect(section.modelCount(in: data) >= 2)
    }
}
