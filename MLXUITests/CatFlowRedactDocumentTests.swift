import Testing
import Foundation
@testable import MLXUI

/// PF follow-up (owner ruling R2b, Boo, 2026-11-24): the Basic Gallery flow "Redact a Document" — Read Text → Redact → Save Text.
/// Gallery number 89 (the next free one), shelf slot 20, file `20-RedactDocument`. Also pins E725 (SPEC-Q252), the check-25 refusal
/// of a bad `Redact` row that the synced conformance cases `pf_redact_*` exercise.
@MainActor
struct CatFlowRedactDocumentTests {

    private static let flowID = "20-RedactDocument"
    private static let hfID = "openai/privacy-filter"

    private func loadCatalog() throws -> [ModelEntry] {
        let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("MLXUI/Resources/browser.json")
        return try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url)).domains.flatMap { $0.allModels }
    }

    private nonisolated struct StubRedactStage: PipelineStage {
        let id = "privacy-filter.stub"
        let name = "Privacy Filter"
        var accepts: MediaKind { .text }
        var produces: MediaKind { .text }
        let text: String
        func run(_ input: Media, progress: @Sendable @escaping (Double) -> Void) async throws -> Media {
            guard case .text = input else { throw StageError.kindMismatch(expected: .text, got: input.kind) }
            progress(1)
            return .text(text)
        }
    }

    // MARK: - the shelf

    @Test func theFlowIsBasicShelfSlotTwentyWithGalleryNumberEightyNine() throws {
        let entries = GalleryLoader.loadMetadata()
        let entry = try #require(entries.first { $0.flowID == Self.flowID })
        #expect(entry.number == 89 && entry.title == "Redact a Document" && entry.isBasic)
        #expect(entries.filter(\.isBasic).map(\.flowID).last == Self.flowID)
        #expect(entries.filter(\.isBasic).count == 20)
        #expect(entries.count == Set(entries.map(\.number)).count, "gallery numbers stay unique")
        #expect(!AppState.hiddenFlowNumbers.contains(89))
    }

    @Test func theFlowKeepsItsData() throws {
        let doc = try CatParser.parse(try GalleryLoader.rawCatText(flowID: Self.flowID))
        // Private by construction: the model is the on-device Privacy Filter, so no `offdevice` and no `network` flag.
        #expect(doc.flags.isEmpty, "\(doc.flags)")
        #expect(doc.rows.map(\.task) == ["Read Text", "Redact", "Save Text"])
        #expect(doc.rows[1].model == "Privacy Filter")
        #expect(doc.rows[1].settings == nil, "all eight categories: the one setting, only=, stays at its default")
        #expect(doc.rows[0].settings == "document.txt")
        #expect(doc.models["Privacy Filter"] == Self.hfID)
    }

    @Test func itChecksCleanWithNoIssues() throws {
        let text = try GalleryLoader.rawCatText(flowID: Self.flowID)
        let found = FlowValidator.checkFlow(try CatParser.parseForValidation(text))
        #expect(found.isEmpty, "\(found)")
    }

    @Test func theSampleDocumentShipsAndHoldsOnlySyntheticPersonalData() throws {
        let assets = GalleryLoader.bundledAssets(flowID: Self.flowID)
        #expect(assets.map(\.destination) == ["document.txt"])
        let name = (assets[0].source as NSString).deletingPathExtension
        let url = try #require(Bundle.main.url(forResource: name, withExtension: "txt"))
        let text = try String(contentsOf: url, encoding: .utf8)
        // Decision 10: invented names, example.com addresses, 555 phone numbers; nothing real.
        #expect(text.contains("example.com") && text.contains("555"))
        #expect(!text.contains("/Users/") && !text.contains("chinpinboo"))
        for real in ["gmail.com", "outlook.com", "yahoo.com", "icloud.com"] { #expect(!text.contains(real)) }
    }

    // MARK: - runs

    @Test func itRunsUnderTheMock() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("pf-gal-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: Self.flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: Self.flowID))
        let mock = MockExecutor(blobDirectory: base.appendingPathComponent("blobs"))
        let events = try await FlowInterpreter.run(try GalleryLoader.loadDocument(flowID: Self.flowID), executor: mock)
        #expect(events.filter { $0.kind == .rowFailed }.isEmpty, "\(events.filter { $0.kind == .rowFailed }.map { $0.message ?? "" })")
    }

    /// The real executor with a stub stage: the flow reads the bundled sample, hands it to the Privacy Filter stage and saves what
    /// that returns as `document-redacted.txt` in the flow's folder.
    @Test func theRealRuntimeSavesTheRedactedCopy() async throws {
        let catalog = try loadCatalog()
        let entry = try #require(catalog.first { $0.hfModelId == Self.hfID })
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("pf-gal-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let ws = try #require(FlowWorkspace(root: base.appendingPathComponent("flows")))
        try ws.prepare(flowID: Self.flowID, sourceDir: GalleryLoader.resourcesDirectory,
                       bundledAssets: GalleryLoader.bundledAssets(flowID: Self.flowID))
        let redacted = "Hello <PRIVATE_PERSON>, we will refund the account ending <ACCOUNT_NUMBER>."
        let real = RealExecutor(workspace: ws, flowID: Self.flowID, blobDirectory: base.appendingPathComponent("blobs"),
                                makeModelStage: { _, _ in StubRedactStage(text: redacted) },
                                installedModelIDs: [entry.id], catalog: catalog)
        let events = try await FlowInterpreter.run(try GalleryLoader.loadDocument(flowID: Self.flowID), executor: real)
        #expect(events.filter { $0.kind == .rowFailed }.isEmpty, "\(events.filter { $0.kind == .rowFailed }.map { $0.message ?? "" })")
        let saved = try ws.resolve("document-redacted.txt", flowID: Self.flowID)
        #expect(try String(contentsOf: saved, encoding: .utf8) == redacted)
    }

    // MARK: - E725 (SPEC-Q252): the check refuses a bad Redact row before it runs

    private func issues(_ settings: String) throws -> [FlowIssue] {
        let text = "mlxflow 0.8\n1. Read Text  a.txt\n2. Redact  Privacy Filter; \(settings)\n\nmodels:\n  Privacy Filter = openai/privacy-filter\n"
        return FlowValidator.checkFlow(try CatParser.parseForValidation(text))
    }

    @Test func aBadRedactRowIsRefusedByNameAtCheckTime() throws {
        let list = "person, email, phone, url, address, date, account and secret"
        let cases: [(String, String)] = [
            ("threshold=0.5", "takes only `only=` (a comma-separated subset of \(list)); remove `threshold=`."),
            ("only=\"person,ssn\"", "doesn't know the category \"ssn\" -- use a comma-separated subset of \(list)."),
            ("only=\"\"", "has an empty category in `only=` -- use a comma-separated subset of \(list)."),
            ("strict", "takes only `only=` (a comma-separated subset of \(list)); remove `strict`."),
        ]
        for (settings, detail) in cases {
            let found = try issues(settings)
            #expect(found.map(\.code) == ["E725"], "\(settings): \(found)")
            #expect(found.first?.message == "Row 2's `Redact` \(detail)", "\(settings): \(found)")
        }
    }

    @Test func aGoodRedactRowChecksClean() throws {
        for settings in ["only=person", "only=\"email,phone\"", "only=\"Person, EMAIL\""] {
            #expect(try issues(settings).isEmpty, "\(settings)")
        }
        #expect(try RedactRowSettings.parse(nil) == nil)
        #expect(try RedactRowSettings.parse("only=\"secret,email,email,person\"") == [.person, .email, .secret])
    }
}
