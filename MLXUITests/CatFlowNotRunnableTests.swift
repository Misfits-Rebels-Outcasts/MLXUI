import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R4-4 + CFM-R7-FIX-2/3: the not-runnable rendering surface. Every gallery flow
/// ships in the list; runnable ones have no badge, the rest carry an honest reason naming
/// what they need. The R7 interpreter brought blocks/continuations/deciders into scope, so
/// the once-"needs blocks" flows are runnable now; only flows a *real* engine gap still
/// refuses keep a badge.
struct CatFlowNotRunnableTests {

    private var repoRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test func everyMetadataEntryHasAReasonUnlessRunnable() throws {
        let metadata = GalleryLoader.loadMetadata()
        #expect(metadata.count == 69)
        for flow in metadata {
            // Exactly one of the two: runnable (no reason) or has an honest reason.
            #expect(flow.isRunnable == (flow.notRunnableReason == nil))
        }
    }

    @Test func runnableFlowsAreTheShippedR7Set() {
        // CFM-R7-FIX-2: blocks/continuations/deciders run under the interpreter, so the
        // R7-runnable set is these 43 — the pinned answer to "what runs from the UI".
        let runnable = GalleryLoader.loadMetadata().filter(\.isRunnable).map(\.flowID).sorted()
        #expect(runnable == [
            "01-SpokenSummary", "02-MeetingMinutes", "03-ShowNotes", "04-InterviewArticle",
            "05-VoiceJournal", "06-ContractScan", "07-PaperDigest", "08-PolicyDiff",
            "09-InvoiceLedger", "10-BookSummary", "11-ReflexionWriter", "12-ChannelBlast",
            "13-TranslateThree", "14-FrontmatterBot", "15-HouseStyle", "16-IngestFolder",
            "17-AskYourDocs", "18-DocChat", "19-CorrectiveRag", "20-TwoIndexAnalyst",
            "21-PhotoWebPrep", "22-ScreenshotHowTo", "23-AltText", "24-ClipNotes",
            "25-PhotoCull", "26-CsvInsights", "27-SurveyThemes", "28-TicketTrends",
            "29-LogTriage", "30-ReceiptsExpense", "36-SupportAutoDraft",
            "37-InboxTriage", "39-MinutesNameFix", "41-ReactAgent", "42-PlanExecute",
            "43-TinyFirstBatch", "44-FrontierEscalate", "45-RouterDesk", "46-IndexSelfTest",
            "47-GoldenRegression", "48-AbJudge", "50-NightlyDrift", "51-InboxIngest",
            "52-AutoTranscribe", "55-ReceiptsLedger", "56-OrderTriageQueue",
            "65-VoiceoverBed", "66-SeedSweep",
        ])
    }

    @Test func notRunnableReasonsNameTheGap() throws {
        let metadata = GalleryLoader.loadMetadata()

        // CFM-R7-FIX-2: these block/decider/continuation flows now run — no badge.
        for id in ["21-PhotoWebPrep", "23-AltText", "17-AskYourDocs", "24-ClipNotes",
                   "26-CsvInsights", "11-ReflexionWriter", "41-ReactAgent"] {
            let flow = try #require(metadata.first { $0.flowID == id })
            #expect(flow.isRunnable, "\(id) should be runnable under the interpreter")
        }

        // GenerateProductShot needs a model this version can't run.
        let shot = try #require(metadata.first { $0.flowID == "60-GenerateProductShot" })
        #expect(shot.notRunnableReason?.contains("model") == true)

        // CFM-R10-Human/Store/Events: human, store, and trigger flows now run.
        let docChat = try #require(metadata.first { $0.flowID == "18-DocChat" })
        #expect(docChat.isRunnable)
        let research = try #require(metadata.first { $0.flowID == "31-ResearchBrief" })
        #expect(research.notRunnableReason?.contains("net") == true)
        // CFM-R10-Events: 51-InboxIngest (a trigger flow) is now runnable; a trigger flow
        // still refused carries its *real* blocker — 54-PipelineAlarm's staged row.
        let inbox = try #require(metadata.first { $0.flowID == "51-InboxIngest" })
        #expect(inbox.isRunnable)
        let alarm = try #require(metadata.first { $0.flowID == "54-PipelineAlarm" })
        #expect(alarm.notRunnableReason?.contains("staged") == true)
        let migration = try #require(metadata.first { $0.flowID == "49-ModelMigration" })
        #expect(migration.notRunnableReason?.contains("RunSuite") == true)

        // The .catpipeline flows name the file kind.
        let latent = try #require(metadata.first { $0.flowID == "67-DriveTheLatent" })
        #expect(latent.notRunnableReason?.contains("catpipeline") == true)
    }

    @Test func notRunnableFlowsStillShipRawCatText() throws {
        // A not-runnable flow's raw `.cat` is available for the read-only disclosure.
        let raw = try GalleryLoader.rawCatText(flowID: "51-InboxIngest")
        #expect(raw.contains("On File"))
    }

    @Test func everyBundledCatHasMetadata() throws {
        let metadata = GalleryLoader.loadMetadata()
        let flowIDs = Set(metadata.map(\.flowID))
        for flow in metadata {
            // Parse documents ship for the runnable five; the rest have raw .cat only.
            if flow.isRunnable {
                let doc = try? GalleryLoader.loadDocument(flowID: flow.flowID)
                #expect(doc != nil, "\(flow.flowID) should have a parse document")
            }
        }
        // Spot check a few not-runnable flows have .cat text.
        for id in ["02-MeetingMinutes", "31-ResearchBrief", "51-InboxIngest"] {
            #expect(flowIDs.contains(id))
            let raw = try? GalleryLoader.rawCatText(flowID: id)
            #expect(raw != nil)
        }
    }
}
