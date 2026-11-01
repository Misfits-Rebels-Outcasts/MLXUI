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
                                             installed: [], totalRAMGB: 32, claimableModelIDs: [])
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
                       "21-PhotoWebPrep", "8-VoiceoverBed", "1-TranscribeAudio", "2-SummaryFromAudio",
                       "3-ExtractTableFromImage", "4-WebPageSummary", "5-ExtractWebPageLinks",
                       // BG-3 (RSI/DelegateBasicImageFlowsBacklog.md): 63/64/65 moved to the
                       // Basic Gallery as 6/7/8 — same contents, new flow ids.
                       "6-CutOutSubject", "7-UpscaleSmall"] {
            #expect(try refusal(flowID) == nil, "\(flowID) should be runnable")
        }
    }

    @Test func flowsWithRealGapsRefuseByName() throws {
        // MP7-W-f (owner ruling, Boo, 2026-11-01): the exact display "Z-Image Turbo (4-bit)" is a `.substitute` bridge row onto FLUX.2 Klein 4B, so 60 (and 66, 69: they write
        // the same name) are no longer refused for "no bridge model"; `Flux2KleinWorkflowTests` pins them runnable with the substitute note. 61 (Edit Image) stays refused.
        #expect(try refusal("60-GenerateProductShot") == nil)
        #expect(try refusal("69-PickALook") == nil)
        #expect(try refusal("61-EditInPlace") != nil)
        // 67 — .catpipeline model-space flow: the latent tasks are unserved.
        #expect(try refusal("67-DriveTheLatent") != nil)
    }

    /// Phase WS: 31/34/53 (Web Search, no provider) are no longer a `canRun` refusal — a
    /// missing key is `.needsSetup`, not a channel dead end, so these flows are structurally
    /// runnable now (`canRun`'s job — "does the runtime have any implementation for this
    /// task"), same as a `.needsSetup` model row already doesn't block `canRun`/preflight
    /// either. Pasting a Tavily/Brave key is what makes them actually *complete*; that's a
    /// Settings step, not a language gap `refusalReason` reports.
    @Test func webSearchFlowsAreRunnableNowNeedingOnlyAKey() throws {
        for flowID in ["31-ResearchBrief", "34-TopicMonitor", "53-MorningBriefing"] {
            #expect(try refusal(flowID) == nil, "\(flowID) should be runnable now that Web Search is ported")
        }
    }

    /// CFM-R15-1 — `6-CutOutSubject` (was `63-CutOutSubject`; BG-3 renamed it) runs now that the owner ruled hazard H2 option (1)
    /// (2026-08-27): the executor serves `engines.diffusion.segment`, `SAM Base` bridges onto
    /// `mlx-community/sam3-4bit` as a `.substitute`, and the preflight buckets it to-download
    /// instead of blocking.
    @Test func cutOutSubjectRunsSinceSegmentIsServed() throws {
        #expect(try refusal("6-CutOutSubject") == nil,
                "6-CutOutSubject should run after CFM-R15-1")
    }

    @Test func notRunnableFlowsStillShipRawCatText() throws {
        let raw = try GalleryLoader.rawCatText(flowID: "51-InboxIngest")
        #expect(raw.contains("On File"))
    }
}
