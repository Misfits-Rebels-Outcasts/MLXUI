import Testing
import Foundation
@testable import MLXUI

/// KC-5.2 — the Providers pane splits credential names by manifest `kind`.
struct CredentialGroupsTests {

    private func manifest(_ id: String, kind: String, credentials: String?) throws -> CuratedManifest {
        let creds = credentials.map { "\"\($0)\"" } ?? "null"
        let json = """
        {"id":"\(id)","display":"\(id)","kind":"\(kind)","engine":"e","credentials":\(creds),
         "settings":{},"resources":{"disk_gb":0,"ram_gb":0},"manifest_version":1,"pinned":false}
        """
        return try JSONDecoder().decode(CuratedManifest.self, from: Data(json.utf8))
    }

    @Test func splitsSearchFromProviderAndSortsEach() throws {
        let groups = CuratedManifest.credentialGroups(manifests: [
            try manifest("a", kind: "provider", credentials: "openai"),
            try manifest("b", kind: "search", credentials: "tavily"),
            try manifest("c", kind: "provider", credentials: "anthropic"),
            try manifest("d", kind: "search", credentials: "brave"),
            try manifest("e", kind: "provider", credentials: "openai"),   // duplicate name
            try manifest("f", kind: "provider", credentials: nil),        // credential-less LAN case
        ])
        #expect(groups.search == ["brave", "tavily"])
        #expect(groups.provider == ["anthropic", "openai"])
    }

    /// The shipped bundle: five credentials, Brave/Tavily under search, three AI providers.
    @Test func shippedBundleGroupsAsDocumented() {
        let host = CuratedManifest.credentialGroups()
        #expect(host.search == ["brave", "tavily"])
        #expect(host.provider == ["anthropic", "deepseek", "openai"])
    }

}
