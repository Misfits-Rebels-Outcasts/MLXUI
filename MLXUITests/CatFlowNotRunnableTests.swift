import Testing
import Foundation
@testable import MLXUI

/// Covers CFM-R4-4: the not-runnable rendering surface. Every gallery flow ships in the
/// list; runnable ones have no badge, the rest carry an honest reason naming what they need
/// ("needs blocks", "needs a model this version can't run", …). See
/// `RSI/DelegateMergeBacklog.md` CFM-R4-4.
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

    @Test func runnableFlowsAreTheShippedFive() {
        let runnable = GalleryLoader.loadMetadata().filter(\.isRunnable).map(\.flowID).sorted()
        #expect(runnable == ["01-SpokenSummary", "08-PolicyDiff", "15-HouseStyle",
                             "29-LogTriage", "65-VoiceoverBed"])
    }

    @Test func notRunnableReasonsNameTheGap() throws {
        let metadata = GalleryLoader.loadMetadata()

        // PhotoWebPrep needs blocks (the `<each>`).
        let photoWeb = try #require(metadata.first { $0.flowID == "21-PhotoWebPrep" })
        #expect(photoWeb.notRunnableReason?.contains("blocks") == true)

        // AltText needs blocks.
        let altText = try #require(metadata.first { $0.flowID == "23-AltText" })
        #expect(altText.notRunnableReason?.contains("blocks") == true)

        // AskYourDocs needs index tooling.
        let askDocs = try #require(metadata.first { $0.flowID == "17-AskYourDocs" })
        #expect(askDocs.notRunnableReason?.contains("index") == true)

        // ClipNotes needs video tools.
        let clip = try #require(metadata.first { $0.flowID == "24-ClipNotes" })
        #expect(clip.notRunnableReason?.contains("video") == true)

        // CsvInsights needs table tools.
        let csv = try #require(metadata.first { $0.flowID == "26-CsvInsights" })
        #expect(csv.notRunnableReason?.contains("table") == true)

        // GenerateProductShot needs a model this version can't run.
        let shot = try #require(metadata.first { $0.flowID == "60-GenerateProductShot" })
        #expect(shot.notRunnableReason?.contains("model") == true)

        // A decider/continuation flow names those.
        let reflex = try #require(metadata.first { $0.flowID == "11-ReflexionWriter" })
        #expect(reflex.notRunnableReason?.contains("continuations") == true)
        #expect(reflex.notRunnableReason?.contains("deciders") == true)

        // The .catpipeline flows name the file kind.
        let latent = try #require(metadata.first { $0.flowID == "67-DriveTheLatent" })
        #expect(latent.notRunnableReason?.contains("catpipeline") == true)
    }

    @Test func notRunnableFlowsStillShipRawCatText() throws {
        // A not-runnable flow's raw `.cat` is available for the read-only disclosure.
        let raw = try GalleryLoader.rawCatText(flowID: "21-PhotoWebPrep")
        #expect(raw.contains("<each"))
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
