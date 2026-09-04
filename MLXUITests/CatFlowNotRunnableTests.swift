import Testing
import Foundation
@testable import MLXUI

/// CFM-R12-FIX-1 — the not-runnable surface, now driven by the **live** gates
/// (`FlowRunnability.refusalReason` = `FlowRunner.canRun` + the model preflight), never a
/// hand-written `_metadata.json` string. This is the exact function the flow detail's load
/// path calls, so a green assertion here is a flow the app can actually open and run.
struct CatFlowNotRunnableTests {

    /// The real catalog (the app's `browser.json`) and installed set, so the preflight
    /// verdicts are the app's.
    private func catalog() throws -> [ModelEntry] {
        let url = try #require(Bundle.main.url(forResource: "browser", withExtension: "json"))
        let browser = try JSONDecoder().decode(BrowserData.self, from: Data(contentsOf: url))
        return browser.domains.flatMap { $0.allModels }
    }

    private func refusal(_ flowID: String) throws -> String? {
        let doc = try GalleryLoader.loadDocument(flowID: flowID)
        return FlowRunnability.refusalReason(for: doc, catalog: try catalog(),
                                             installed: [], totalRAMGB: 32)
    }

    /// CFM-R12-FIX-1's missing test: the six flows the stale metadata locked out now reach
    /// the runnable state (the same function the view calls, not `canRun` directly).
    @Test func theSixStaleFlowsAreRunnableNow() throws {
        for flowID in ["32-FeedWatch", "33-PageDiff", "35-WebIngest", "38-ReplyApproval", "40-StagedPost", "54-PipelineAlarm"] {
            let reason = try refusal(flowID)
            #expect(reason == nil, "\(flowID) should be runnable after R12 (reason: \(reason ?? ""))")
        }
    }

    @Test func flowsThatRunStayRunnable() throws {
        for flowID in ["01-SpokenSummary", "02-MeetingMinutes", "18-DocChat", "51-InboxIngest",
                       "21-PhotoWebPrep", "65-VoiceoverBed", "1-TranscribeAudio", "2-SummaryFromAudio"] {
            #expect(try refusal(flowID) == nil, "\(flowID) should be runnable")
        }
    }

    @Test func flowsWithRealGapsRefuseByName() throws {
        // 31/34/53 — Web Search, no provider.
        let research = try refusal("31-ResearchBrief")
        #expect(research?.contains("Web Search") == true)
        // 60 — Generate Image has no bridge model.
        #expect(try refusal("60-GenerateProductShot") != nil)
        // 67-69 — .catpipeline model-space flows have no runnable models.
        #expect(try refusal("67-DriveTheLatent") != nil)
        #expect(try refusal("69-PickALook") != nil)
    }

    /// CFM-R15-1 — `63-CutOutSubject` runs now that the owner ruled hazard H2 option (1)
    /// (2026-08-27): the executor serves `engines.diffusion.segment`, `SAM Base` bridges onto
    /// `mlx-community/sam3-4bit` as a `.substitute`, and the preflight buckets it to-download
    /// instead of blocking.
    @Test func cutOutSubjectRunsSinceSegmentIsServed() throws {
        #expect(try refusal("63-CutOutSubject") == nil,
                "63-CutOutSubject should run after CFM-R15-1")
    }

    @Test func notRunnableFlowsStillShipRawCatText() throws {
        let raw = try GalleryLoader.rawCatText(flowID: "51-InboxIngest")
        #expect(raw.contains("On File"))
    }
}
