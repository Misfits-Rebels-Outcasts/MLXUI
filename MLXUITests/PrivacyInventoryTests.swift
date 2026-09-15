import Testing
import Foundation
@testable import MLXUI

/// SET-5's G2 seam (`RSI/DelegateSettingsBacklog.md`) — this is the test that keeps the
/// Privacy pane's disclosure honest as the catalog grows: every off-machine bundled manifest
/// appears in the inventory, every manifest naming `credentials` has a matching Providers
/// row, and a brand-new provider manifest shows up without anyone touching `PrivacyView`.
struct PrivacyInventoryTests {

    /// The real, shipping `Resources/CatFlow/models/` — read from source, same pattern as
    /// `CatFlowModelSlotTests.bundledCatalog()`, since the test bundle isn't the app bundle.
    private func bundledManifestURLs() throws -> [URL] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
        let dir = repoRoot.appendingPathComponent("MLXUI/Resources/CatFlow/models")
        return try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }
    }

    // MARK: - Every off-machine bundled manifest appears

    @Test func everyOffMachineBundledManifestAppearsInTheInventory() throws {
        let manifests = CuratedManifest.installedManifests(manifestURLs: try bundledManifestURLs())
        let offMachine = manifests.filter(PrivacyInventory.isOffMachine)
        #expect(!offMachine.isEmpty)  // the catalog does ship provider/search manifests today

        let rows = PrivacyInventory.rows(from: manifests)
        #expect(rows.count == offMachine.count)
        for manifest in offMachine {
            #expect(rows.contains { $0.destination == manifest.display })
        }
    }

    @Test func onDeviceManifestsNeverAppear() throws {
        let manifests = CuratedManifest.installedManifests(manifestURLs: try bundledManifestURLs())
        let onDevice = manifests.filter { !PrivacyInventory.isOffMachine($0) }
        #expect(!onDevice.isEmpty)  // most of the catalog is local MLX models

        let rows = PrivacyInventory.rows(from: manifests)
        for manifest in onDevice {
            #expect(!rows.contains { $0.destination == manifest.display })
        }
    }

    // MARK: - Every credentialed row has a matching Providers row

    @Test func everyRowNamingCredentialsHasAMatchingProvidersRow() throws {
        let urls = try bundledManifestURLs()
        let manifests = CuratedManifest.installedManifests(manifestURLs: urls)
        let rows = PrivacyInventory.rows(from: manifests)
        let providerNames = Set(CuratedManifest.installedCredentialNames(manifestURLs: urls))

        let named = rows.compactMap(\.credentialsName)
        #expect(!named.isEmpty)  // Tavily/Brave/Anthropic/OpenAI/DeepSeek all name one today
        for name in named {
            #expect(providerNames.contains(name))
        }
    }

    @Test func aCredentialLessLanRowHasNoCredentialsName() throws {
        // The shape `macstudio-qwen3-32b.json` used to ship (kind "provider", egress "lan",
        // no credentials) — constructed inline now that no bundled manifest has this shape;
        // the app no longer ships an example LAN endpoint.
        let lan = CuratedManifest(id: "lanbox/some-model", display: "some-model @ http://lanbox.local:8080",
                                  kind: "provider", engine: "openai-compatible", egress: "lan",
                                  baseURL: "http://lanbox.local:8080/v1", settings: [:], resources: nil)
        let row = try #require(PrivacyInventory.rows(from: [lan]).first)
        #expect(row.credentialsName == nil)
        #expect(row.account == "none — the user's own machine")
    }

    // MARK: - A fixture manifest for a new provider shows up untouched

    @Test func aNewProviderManifestShowsUpWithoutTouchingTheView() {
        let fixture = CuratedManifest(
            id: "newprovider/model-x",
            display: "model-x @ newprovider",
            kind: "provider",
            credentials: "newprovider",
            settings: [:],
            resources: nil
        )
        let rows = PrivacyInventory.rows(from: [fixture])
        #expect(rows.count == 1)
        #expect(rows[0].destination == "model-x @ newprovider")
        #expect(rows[0].credentialsName == "newprovider")
        #expect(rows[0].account == "the user's own API key")
        #expect(rows[0].carrying == "the row's prompt — the user's actual document content")
    }

    @Test func aNewSearchManifestCarriesOnlyTheQueryText() {
        let fixture = CuratedManifest(
            id: "newsearch/engine",
            display: "NewSearch",
            kind: "search",
            credentials: "newsearch",
            settings: [:],
            resources: nil
        )
        let rows = PrivacyInventory.rows(from: [fixture])
        #expect(rows.count == 1)
        #expect(rows[0].carrying == "the query text")
    }

    @Test func rowsAreSortedByDestination() {
        let a = CuratedManifest(id: "a", display: "Zebra", kind: "provider", credentials: "z", settings: [:], resources: nil)
        let b = CuratedManifest(id: "b", display: "Alpha", kind: "search", credentials: "a", settings: [:], resources: nil)
        let rows = PrivacyInventory.rows(from: [a, b])
        #expect(rows.map(\.destination) == ["Alpha", "Zebra"])
    }
}
