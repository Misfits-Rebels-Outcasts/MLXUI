import Foundation

/// Metadata for one gallery flow — title/category/description only. The refusal to run is
/// derived from the live gates (`FlowRunnability`) at load time, never a hand-written string
/// that can go stale (CFM-R12-FIX-1).
nonisolated struct GalleryFlowMetadata: Codable, Identifiable, Hashable, Sendable {
    /// The gallery number (1, 8, 21, …).
    var number: Int
    /// Friendly title, e.g. "Spoken Summary".
    var title: String
    /// The `.cat` filename, e.g. "01-SpokenSummary.cat".
    var filename: String
    /// Category, e.g. "Voice & Meetings".
    var category: String
    /// One-line description.
    var description: String
    /// The gallery shelf this flow lives on: `"basic"` → "Basic Gallery", anything
    /// else/nil → "Advance Gallery". Decoded additively — a missing key is an advance
    /// flow, so pre-existing metadata entries stay untouched.
    var tier: String?

    var id: Int { number }

    /// Whether this flow belongs in the "Basic Gallery" shelf.
    var isBasic: Bool { tier == "basic" }

    /// The flow id — the filename minus the `.cat` extension (e.g. `01-SpokenSummary`).
    var flowID: String {
        (filename as NSString).deletingPathExtension
    }
}

/// Loads the bundled gallery: `_metadata.json` for the list, and a `FlowDocument` + the raw
/// `.cat` text on demand.
///
/// **Resource layout note.** The Xcode `PBXFileSystemSynchronizedRootGroup` flattens
/// `MLXUI/Resources/Gallery/**` into the app bundle's `Contents/Resources/` root (no
/// `Gallery/` subdirectory survives — verified on the built `.app`). The gallery's
/// filenames are globally unique (`01-SpokenSummary…`, `08-PolicyDiff…`, `21-PhotoWebPrep…`),
/// so lookups by name work fine without a subdirectory.
nonisolated enum GalleryLoader {
    /// The three (or however many ship) gallery flows, in gallery order.
    static func loadMetadata() -> [GalleryFlowMetadata] {
        guard let url = Bundle.main.url(forResource: "_metadata", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode(GalleryMetadataRoot.self, from: data) else {
            return []
        }
        return decoded.entries
    }

    /// Load the parsed `FlowDocument` for a flow id — **parsed at runtime from the bundled
    /// `.cat` text** (R5's parser retirement: the pre-parsed `*.parse.json` resources are
    /// gone; the Swift `CatParser` reproduces the Python trees byte-for-byte, pinned by
    /// `CatFlowParserTests`).
    static func loadDocument(flowID: String) throws -> FlowDocument {
        let raw = try rawCatText(flowID: flowID)
        return try CatParser.parse(raw)
    }

    /// The raw `.cat` file text, verbatim — what the user reads in the disclosure.
    /// Resolves both `.cat` and `.catpipeline` (the three model-space flows ship as the
    /// latter; the flattened bundle keeps both extensions).
    static func rawCatText(flowID: String) throws -> String {
        for ext in ["cat", "catpipeline"] {
            if let url = Bundle.main.url(forResource: flowID, withExtension: ext) {
                return try String(contentsOf: url, encoding: .utf8)
            }
        }
        throw GalleryError.missingResource(flowID, kind: ".cat")
    }

    /// The bundled input assets for a flow: `(flat source name, relative destination)` pairs.
    /// Sources are the flat filenames in `Contents/Resources/` (the synchronized root group
    /// flattens the `Gallery/` tree); destinations are the paths the flow's `.cat` expects
    /// (e.g. `vacation/photo-01.png` for `21-PhotoWebPrep`'s `Read Images vacation/`).
    /// Copied from `catflow-mlx/gallery_fixtures/`.
    ///
    /// **Flat names are globally unique across the whole bundle.** Two flows both read
    /// `cases.txt` and `tickets/ticket-*.txt`, and seven share the prebuilt `library.index`
    /// — so colliding files ship under a `flowID-`-prefixed flat name (`47-GoldenRegression-
    /// cases.txt`, `library-chunks.jsonl`…) and are copied to the path the flow expects.
    static func bundledAssets(flowID: String) -> [(source: String, destination: String)] {
        switch flowID {
        case "01-SpokenSummary":
            return [("memo.m4a", "memo.m4a")]
        case "02-MeetingMinutes":
            return [("all-hands.m4a", "all-hands.m4a")]
        case "03-ShowNotes":
            return [("episode.mp3", "episode.mp3")]
        case "04-InterviewArticle":
            return [("interview.m4a", "interview.m4a")]
        case "05-VoiceJournal":
            return [
                ("day1.m4a", "memos/day1.m4a"),
                ("day2.m4a", "memos/day2.m4a"),
                ("day3.m4a", "memos/day3.m4a")
            ]
        case "06-ContractScan":
            return [("contract.pdf", "contract.pdf")]
        case "07-PaperDigest":
            return [("paper.pdf", "paper.pdf")]
        case "08-PolicyDiff":
            return [("policy-2025.md", "policy-2025.md"), ("policy-2026.md", "policy-2026.md")]
        case "09-InvoiceLedger":
            return [
                ("invoice-01.pdf", "invoices/invoice-01.pdf"),
                ("invoice-02.pdf", "invoices/invoice-02.pdf"),
                ("invoice-03.pdf", "invoices/invoice-03.pdf")
            ]
        case "10-BookSummary":
            return [("book.txt", "book.txt")]
        case "12-ChannelBlast":
            return [("launch-notes.txt", "launch-notes.txt")]
        case "13-TranslateThree":
            return [("announcement.md", "announcement.md")]
        case "14-FrontmatterBot":
            return [("post.md", "post.md")]
        case "15-HouseStyle":
            return [("draft.md", "draft.md"), ("style-guide.md", "style-guide.md")]
        case "16-IngestFolder":
            return [
                ("doc-a.pdf", "docs/doc-a.pdf"),
                ("doc-b.pdf", "docs/doc-b.pdf"),
                ("doc-c.pdf", "docs/doc-c.pdf")
            ]
        case "17-AskYourDocs":
            return [
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "18-DocChat":
            return [
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "19-CorrectiveRag":
            return [
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "20-TwoIndexAnalyst":
            return [
                ("hr-chunks.jsonl", "hr.index/chunks.jsonl"),
                ("hr-manifest.json", "hr.index/manifest.json"),
                ("hr-vectors.bin", "hr.index/vectors.bin"),
                ("eng-chunks.jsonl", "eng.index/chunks.jsonl"),
                ("eng-manifest.json", "eng.index/manifest.json"),
                ("eng-vectors.bin", "eng.index/vectors.bin")
            ]
        case "21-PhotoWebPrep":
            return [
                ("logo.png", "logo.png"),
                ("photo-01.png", "vacation/photo-01.png"),
                ("photo-02.png", "vacation/photo-02.png"),
                ("photo-03.png", "vacation/photo-03.png")
            ]
        case "22-ScreenshotHowTo":
            return [
                ("step-01.png", "steps/step-01.png"),
                ("step-02.png", "steps/step-02.png"),
                ("step-03.png", "steps/step-03.png")
            ]
        case "23-AltText":
            return [
                ("img-01.png", "site-images/img-01.png"),
                ("img-02.png", "site-images/img-02.png"),
                ("img-03.png", "site-images/img-03.png")
            ]
        case "24-ClipNotes":
            return [("webinar.mp4", "webinar.mp4")]
        case "25-PhotoCull":
            return [
                ("shot-01.jpg", "shoot/shot-01.jpg"),
                ("shot-02.jpg", "shoot/shot-02.jpg"),
                ("shot-03.jpg", "shoot/shot-03.jpg"),
                ("shot-04.jpg", "shoot/shot-04.jpg"),
                ("shot-05.jpg", "shoot/shot-05.jpg")
            ]
        case "26-CsvInsights":
            return [("sales.csv", "sales.csv")]
        case "27-SurveyThemes":
            return [("survey.csv", "survey.csv")]
        case "28-TicketTrends":
            return [
                ("28-TicketTrends-ticket-01.txt", "tickets/ticket-01.txt"),
                ("28-TicketTrends-ticket-02.txt", "tickets/ticket-02.txt"),
                ("28-TicketTrends-ticket-03.txt", "tickets/ticket-03.txt")
            ]
        case "29-LogTriage":
            return [("app.log", "app.log")]
        case "30-ReceiptsExpense":
            return [
                ("receipt-01.png", "receipts/receipt-01.png"),
                ("receipt-02.png", "receipts/receipt-02.png"),
                ("receipt-03.png", "receipts/receipt-03.png")
            ]
        case "36-SupportAutoDraft":
            return [
                ("36-SupportAutoDraft-ticket-01.txt", "tickets/ticket-01.txt"),
                ("36-SupportAutoDraft-ticket-02.txt", "tickets/ticket-02.txt"),
                ("36-SupportAutoDraft-ticket-03.txt", "tickets/ticket-03.txt")
            ]
        case "37-InboxTriage":
            return [
                ("mail-01.txt", "inbox/mail-01.txt"),
                ("mail-02.txt", "inbox/mail-02.txt"),
                ("mail-03.txt", "inbox/mail-03.txt")
            ]
        case "39-MinutesNameFix":
            return [("meeting.m4a", "meeting.m4a")]
        case "41-ReactAgent":
            return [
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "42-PlanExecute":
            return [
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "43-TinyFirstBatch":
            return [
                ("clip1.m4a", "memos/clip1.m4a"),
                ("clip2.m4a", "memos/clip2.m4a"),
                ("clip3.m4a", "memos/clip3.m4a")
            ]
        case "44-FrontierEscalate":
            return [
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "45-RouterDesk":
            return [
                ("ticket.txt", "ticket.txt"),
                ("billing-kb-chunks.jsonl", "billing-kb.index/chunks.jsonl"),
                ("billing-kb-manifest.json", "billing-kb.index/manifest.json"),
                ("billing-kb-vectors.bin", "billing-kb.index/vectors.bin"),
                ("docs-kb-chunks.jsonl", "docs-kb.index/chunks.jsonl"),
                ("docs-kb-manifest.json", "docs-kb.index/manifest.json"),
                ("docs-kb-vectors.bin", "docs-kb.index/vectors.bin")
            ]
        case "46-IndexSelfTest":
            return [
                ("handbook.txt", "handbook.txt"),
                ("library-chunks.jsonl", "library.index/chunks.jsonl"),
                ("library-manifest.json", "library.index/manifest.json"),
                ("library-vectors.bin", "library.index/vectors.bin")
            ]
        case "47-GoldenRegression":
            return [("47-GoldenRegression-cases.txt", "cases.txt"), ("run-baseline.md", "run-baseline.md")]
        case "48-AbJudge":
            return [("prompts.txt", "prompts.txt")]
        case "50-NightlyDrift":
            return [("eval-history.ctx", "eval-history.ctx"), ("50-NightlyDrift-cases.txt", "cases.txt")]
        case "51-InboxIngest":
            return [("doc1.pdf", "inbox/doc1.pdf"), ("doc2.pdf", "inbox/doc2.pdf")]
        case "52-AutoTranscribe":
            return [("note1.m4a", "memos/note1.m4a"), ("note2.m4a", "memos/note2.m4a")]
        case "55-ReceiptsLedger":
            return [
                ("ledger.ctx", "ledger.ctx"),
                ("receipt-01.jpg", "receipts/receipt-01.jpg"),
                ("receipt-02.jpg", "receipts/receipt-02.jpg")
            ]
        case "56-OrderTriageQueue":
            return [("new_orders.csv", "new_orders.csv"), ("orders.db", "orders.db")]
        case "1-TranscribeAudio":
            return [("canond.wav", "canond.wav")]
        case "2-SummaryFromAudio":
            return [("2-SummaryFromAudio-canond.wav", "canond.wav")]
        case "3-ExtractTableFromImage":
            return [("3-ExtractTableFromImage-budget.png", "budget.png")]
        default:
            return []
        }
    }

    /// The app bundle's `Contents/Resources/` directory, where the flattened gallery
    /// resources live (or `nil` if the bundle is malformed).
    static var resourcesDirectory: URL? {
        Bundle.main.resourceURL
    }

    private struct GalleryMetadataRoot: Decodable {
        var entries: [GalleryFlowMetadata]
    }
}

/// Gallery-loading failures. Error voice: one plain sentence implying the fix.
nonisolated enum GalleryError: Error, CustomStringConvertible, Equatable {
    case missingResource(String, kind: String)

    var description: String {
        switch self {
        case .missingResource(let flowID, let kind):
            return "The gallery flow '\(flowID)' is missing its \(kind) — reinstall the app to restore it."
        }
    }
}
