import Foundation
import Testing
@testable import MLXUI

/// Round-trips the Codable layer that decodes `browser.json`. Uses inline fixtures so the
/// test never depends on the bundled catalog resource being reachable from the test bundle.
struct DecodingTests {

    @Test func decodesModelEntryWithEnumsAndOptionals() throws {
        let json = """
        {
            "id": "mlx-community--Qwen3-4B-4bit",
            "family": "Qwen3",
            "displayName": "Qwen3 4B",
            "paramSize": "4B",
            "paramCountB": 4.0,
            "modelType": "llm",
            "source": "mlx",
            "format": "mlx",
            "platforms": ["macos"],
            "hfRepo": "mlx-community/Qwen3-4B",
            "hfModelId": "mlx-community/Qwen3-4B-4bit",
            "ramGB": 4.5,
            "downloadSizeGB": 2.3,
            "speedTokensPerSec": 60,
            "speedEstimated": true
        }
        """
        let entry = try JSONDecoder().decode(ModelEntry.self, from: Data(json.utf8))
        #expect(entry.modelType == .llm)
        #expect(entry.source == .mlx)
        #expect(entry.ramGB == 4.5)
        #expect(entry.contextWindow == nil)        // omitted optional decodes to nil
        #expect(entry.speedEstimated == true)
    }

    @Test func decodesEmptyBrowserDataRoot() throws {
        let json = """
        {
            "version": "5",
            "generatedAt": "2026-01-01T00:00:00Z",
            "sidebarSections": [],
            "domainIndex": {},
            "domains": []
        }
        """
        let data = try JSONDecoder().decode(BrowserData.self, from: Data(json.utf8))
        #expect(data.version == "5")
        #expect(data.domains.isEmpty)
    }
}
