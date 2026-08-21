import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R2-7: model preflight — resolve a flow's model rows through `CatalogBridge`
/// into installed / to-download / blocked buckets, and the RAM check (max, not sum). See
/// `RSI/DelegateMergeBacklog.md` CFM-R2-7.
struct CatFlowPreflightTests {

    private func decode(_ flowID: String) throws -> FlowDocument {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/Gallery/\(flowID).cat")
        return try CatParser.parse(try String(contentsOf: url, encoding: .utf8))
    }

    private func loadCatalog() throws -> [ModelEntry] {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let url = repoRoot.appendingPathComponent("MLXUI/Resources/browser.json")
        let catalog = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return catalog.domains.flatMap { $0.allModels }
    }

    // MARK: - Spoken Summary preflight (real catalog)

    @Test func spokenSummaryPreflightBuckets() throws {
        let doc = try decode("01-SpokenSummary")
        let catalog = try loadCatalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16)

        // Three distinct model rows: Whisper Large v3, Qwen3 8B, Kokoro 82M.
        let displays = Set(result.needs.map(\.display))
        #expect(displays == ["Whisper Large v3", "Qwen3 8B", "Kokoro 82M"])
        #expect(result.needs.count == 3)
        // Nothing installed → all to download.
        #expect(result.installed.isEmpty)
        #expect(result.toDownload.count == 3)
        #expect(result.blocked.isEmpty)
        // Download set dedupes to three catalog entries.
        #expect(result.downloadSet.count == 3)
    }

    @Test func spokenSummaryInstalledBuckets() throws {
        let doc = try decode("01-SpokenSummary")
        let catalog = try loadCatalog()
        let whisper = try #require(catalog.first { $0.hfModelId == "mlx-community/whisper-large-v3-asr-fp16" })
        let result = FlowPreflight.run(doc, catalog: catalog,
                                       installedModelIDs: [whisper.id], totalRAMGB: 16)
        #expect(result.installed.count == 1)
        #expect(result.installed.first?.display == "Whisper Large v3")
        #expect(result.toDownload.count == 2)
    }

    // MARK: - RAM check: max, not sum

    @Test func ramCheckUsesLargestRowNotSum() throws {
        let doc = try decode("01-SpokenSummary")
        let catalog = try loadCatalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16)

        // The three rows' ramGB are 4.62 / 6.75 / 0.25. Sum ≈ 11.62, max = 6.75.
        // A 16 GB machine fits the largest row, so the flow is not RAM-blocked.
        let max = try #require(catalog.first { $0.hfModelId == "mlx-community/Qwen3-8B-4bit" })!.ramGB
        #expect(result.largestRowRAMGB == max)
        #expect(result.largestRowRAMGB < 7)
        #expect(FlowPreflight.fitsRAM(result, totalRAMGB: 16))
        #expect(FlowPreflight.blockedReason(result, totalRAMGB: 16) == nil)
    }

    @Test func ramCheckBlocksWhenLargestRowExceedsRAM() throws {
        // A flow whose largest row (Qwen3 8B, 6.75 GB) exceeds a 6 GB machine. The sum of
        // the two rows (6.75 + 4.62 = 11.37) is irrelevant — the max rules.
        let rows = [
            Row(task: "Summarize", model: "Qwen3 8B", settings: "\"x\""),
            Row(task: "Transcribe", model: "Whisper Large v3"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let catalog = try loadCatalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 6)
        #expect(result.largestRowRAMGB == 6.75)
        #expect(FlowPreflight.fitsRAM(result, totalRAMGB: 6) == false)
        let reason = FlowPreflight.blockedReason(result, totalRAMGB: 6)
        #expect(reason?.contains("GB") == true)
    }

    // MARK: - No-candidate model blocks with the model named

    @Test func noCandidateModelBlocksNamingTheModel() throws {
        let rows = [Row(task: "Transcribe", model: "A Model That Doesn't Exist")]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let catalog = try loadCatalog()
        let result = FlowPreflight.run(doc, catalog: catalog, installedModelIDs: [], totalRAMGB: 16)
        #expect(result.isBlocked)
        #expect(result.blocked.count == 1)
        #expect(result.blocked.first?.display == "A Model That Doesn't Exist")
        let reason = FlowPreflight.blockedReason(result, totalRAMGB: 16)
        #expect(reason?.contains("A Model That Doesn't Exist") == true)
    }

    // MARK: - Non-model rows don't create model needs

    @Test func instantRowsProduceNoModelNeeds() throws {
        let rows = [
            Row(task: "Read Audio", settings: "a.m4a"),
            Row(task: "Save Text", settings: "out.txt"),
        ]
        let doc = FlowDocument(version: "0.8", rows: rows)
        let result = FlowPreflight.run(doc, catalog: [], installedModelIDs: [], totalRAMGB: 16)
        #expect(result.needs.isEmpty)
        #expect(result.isBlocked == false)
    }
}
