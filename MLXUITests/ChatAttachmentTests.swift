import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// CW-3 — the attachment model, the C11 budget decision and the model-facing rendering. Pure: no
/// model, no files. The importer's file work is `AttachmentImporterTests`; OCR needs a model and is
/// covered by smoke CW-S5.
struct ChatAttachmentTests {

    private func attachment(
        _ name: String, mode: ChatAttachmentMode = .inline, text: String? = "contents"
    ) -> ChatAttachment {
        ChatAttachment(
            name: name, storedPath: "/container/attachments/\(name)", mode: mode,
            byteCount: (text?.utf8.count ?? 12), inlineText: mode == .reference ? nil : text)
    }

    // MARK: - estimation

    @Test func estimatedTokensRoundsUpQuarters() {
        #expect(ChatAttachment.estimatedTokens(forUTF8Count: 0) == 0)
        #expect(ChatAttachment.estimatedTokens(forUTF8Count: 1) == 1)
        #expect(ChatAttachment.estimatedTokens(forUTF8Count: 4) == 1)
        #expect(ChatAttachment.estimatedTokens(forUTF8Count: 5) == 2)
    }

    // MARK: - budget (C11)

    @Test func aSmallFileIsInlined() {
        #expect(ChatAttachment.mode(byteCount: 500, window: 8_192, inlineBytesSoFar: 0) == .inline)
    }

    @Test func aFileOverTheWindowFractionGoesByReference() {
        // 8192/4 tokens = 2048 tokens = 8192 bytes is the cap; one byte more is by reference.
        #expect(ChatAttachment.mode(byteCount: 8_192, window: 8_192, inlineBytesSoFar: 0) == .inline)
        #expect(ChatAttachment.mode(byteCount: 8_193, window: 8_192, inlineBytesSoFar: 0) == .reference)
    }

    @Test func thePerMessageTotalCapsWhatIsInlined() {
        // 8192/3 tokens = 2730 tokens = 10920 bytes. A 6 000-byte file fits; the next 6 000 does not.
        #expect(ChatAttachment.mode(byteCount: 6_000, window: 8_192, inlineBytesSoFar: 0) == .inline)
        #expect(ChatAttachment.mode(byteCount: 6_000, window: 8_192, inlineBytesSoFar: 6_000) == .reference)
    }

    @Test func aTinyWindowStillAllowsTheMinimum() {
        #expect(ChatAttachment.mode(byteCount: 2_000, window: 2_000, inlineBytesSoFar: 0) == .inline)
        #expect(ChatAttachment.mode(byteCount: 9_000, window: 2_000, inlineBytesSoFar: 0) == .reference)
    }

    // MARK: - the absolute caps (B2)

    /// The reviewer's own correction: a 262 144-token window would otherwise inline ~256 KB.
    @Test func a60KbFileGoesByReferenceEvenWithA262144Window() {
        #expect(ChatAttachment.maxInlineBytesPerFile == 48 * 1024)
        #expect(ChatAttachment.mode(byteCount: 60 * 1024, window: 262_144, inlineBytesSoFar: 0)
            == .reference)
    }

    /// …while a window whose own share is already the smaller number still inlines.
    @Test func a30KbFileIsInlinedWithA40960Window() {
        #expect(ChatAttachment.maxInlineBytesPerMessage == 96 * 1024)
        #expect(ChatAttachment.mode(byteCount: 30 * 1024, window: 40_960, inlineBytesSoFar: 0)
            == .inline)
    }

    /// The per-message cap binds before the window's own share would.
    @Test func thePerMessageCapStopsTheSecondBigFile() {
        // 40 KB each: each is under the 48 KB per-file cap, but two exceed the 96 KB message cap.
        #expect(ChatAttachment.mode(byteCount: 40 * 1024, window: 262_144, inlineBytesSoFar: 0)
            == .inline)
        #expect(ChatAttachment.mode(byteCount: 40 * 1024, window: 262_144, inlineBytesSoFar: 60 * 1024)
            == .reference)
    }

    @Test func limitsAreTheContract() {
        #expect(ChatAttachment.maxFileBytes == 50 * 1024 * 1024)
        #expect(ChatAttachment.maxFilesPerMessage == 10)
    }

    // MARK: - rendering (C13)

    @Test func inlineRenderingGolden() {
        let rendered = ChatAttachment.modelFacing(
            text: "Summarise this.",
            attachments: [attachment("rates.csv")])
        #expect(rendered == """
        Summarise this.

        <attachment name="rates.csv">
        contents
        </attachment>
        """)
    }

    @Test func referenceRenderingGolden() {
        let rendered = ChatAttachment.modelFacing(
            text: "Summarise this.",
            attachments: [attachment("big.txt", mode: .reference, text: nil)])
        #expect(rendered == """
        Summarise this.

        <attachment name="big.txt" path="/container/attachments/big.txt" mode="reference"/>
        """)
    }

    @Test func ocrRenderingUsesTheSameBlockAsInline() {
        let rendered = ChatAttachment.modelFacing(
            text: "What does it say?", attachments: [attachment("shot.png", mode: .ocr, text: "hello")])
        #expect(rendered.contains("<attachment name=\"shot.png\">\nhello\n</attachment>"))
    }

    // MARK: - what the model is actually sent (CW-C-FIX-1)

    /// The report: a small `Name.txt` went into the bubble and `chatHistory` but **not** into the
    /// turn being sent, so the model said "attach it first" while looking at a message that already
    /// had it. `prompt(text:attachments:)` is what `send` now passes to the session.
    @Test func anInlineAttachmentsContentIsInThePromptForThisTurn() {
        let prompt = ChatAttachment.prompt(
            text: "What is the content of the file?",
            attachments: [attachment("Name.txt", text: "hello from the file")])
        #expect(prompt.contains("hello from the file"))
        #expect(prompt.contains("What is the content of the file?"))
        #expect(prompt.contains("<attachment name=\"Name.txt\">"))
    }

    /// The bug, in one line, so a regression is unmistakable: the bare text is what the old call
    /// site passed, and it carries none of the file.
    @Test func theBareTextIsNotEnoughWhichIsWhyTheCallSiteChanged() {
        let attachments = [attachment("Name.txt", text: "hello from the file")]
        let bare = "What is the content of the file?"
        #expect(!bare.contains("hello from the file"))
        #expect(ChatAttachment.prompt(text: bare, attachments: attachments)
            != bare)
    }

    /// The regression that started this: `ChatAttachment.prompt` was correct and unused. The
    /// call site is what has to be pinned, so this asserts the source of the one function that
    /// hands the prompt to the model — a text-only assertion, deliberately.
    ///
    /// This is the only test in the suite that reads source. Say so plainly: no unit test can see
    /// a wrong call site through a `ModelContainer`, and the alternative was to ship the bug again.
    @Test func sendFeedsTheModelTheRenderedPrompt() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("MLXUI/Services/ModelRunner.swift"),
            encoding: .utf8)
        #expect(source.contains(
            "let modelPrompt = ChatAttachment.prompt(text: trimmed, attachments: readyAttachments)"))
        #expect(source.contains("session.streamResponse(to: modelPrompt)"))
        // …and nothing hands the bare text to the model.
        #expect(!source.contains("streamResponse(to: trimmed)"))
    }

    /// A by-reference file is still not inlined — the model is given the path and told to read it.
    @Test func aByReferenceAttachmentIsAPathAndNotTheContent() {
        let prompt = ChatAttachment.prompt(
            text: "Summarise this.",
            attachments: [attachment("big.txt", mode: .reference, text: nil)])
        #expect(prompt.contains("path=\"/container/attachments/big.txt\""))
        #expect(!prompt.contains("contents"))
    }

    /// The prompt and the replayed history are the same rendering, so a relaunch replays byte-for-
    /// byte what the model saw the first time.
    @Test func thePromptMatchesWhatTheTranscriptReplays() {
        let attachments = [
            attachment("rates.csv", text: "a,b"),
            attachment("big.txt", mode: .reference, text: nil),
        ]
        let prompt = ChatAttachment.prompt(text: "look", attachments: attachments)
        let replayed = Conversation.chatHistory(from: [
            .message(ChatMessage(role: .user, text: "look", attachments: attachments)),
        ])
        #expect(replayed.count == 1)
        #expect(replayed[0].content == prompt)
    }

    @Test func aMessageWithNoAttachmentsIsJustItsText() {
        #expect(ChatAttachment.modelFacing(text: "hi", attachments: []) == "hi")
    }

    @Test func environmentNoteNamesOnlyByReferenceFiles() {
        let note = ChatAttachment.environmentNote(for: [
            attachment("small.csv"),
            attachment("big.txt", mode: .reference, text: nil),
            attachment("scan.pdf", mode: .reference, text: nil),
            attachment("shot.png", mode: .ocr, text: "x"),
        ])
        #expect(note?.contains("big.txt") == true)
        #expect(note?.contains("scan.pdf") == true)
        #expect(note?.contains("small.csv") == false)
        #expect(note?.contains("shot.png") == false)
        #expect(note?.contains("read it with read_file") == true)
    }

    /// B3: an image with no OCR model must not be described as readable — `read_file` on a PNG
    /// only ever answers "not UTF-8 text".
    @Test func aReferenceImageGetsTheNoOCRNote() {
        let note = ChatAttachment.environmentNote(for: [
            attachment("shot.png", mode: .reference, text: nil),
        ])
        #expect(note == "Attached image shot.png could not be read: no OCR model is installed."
            + " Tell the user to install one from Browse.")
        #expect(note?.contains("read_file") == false)
    }

    @Test func aReferenceDocumentStillSaysToReadIt() {
        let note = ChatAttachment.environmentNote(for: [
            attachment("big.txt", mode: .reference, text: nil),
        ])
        #expect(note?.contains("read it with read_file") == true)
        #expect(note?.contains("could not be read") == false)
    }

    /// The chip badge: a reference image that failed OCR reads "no OCR", not "reference".
    @Test func aReferenceImageBadgesAsNoOCR() {
        #expect(attachment("shot.png", mode: .reference, text: nil).badgeLabel == "no OCR")
        #expect(attachment("shot.png", mode: .reference, text: nil).isUnreadImage)
        #expect(attachment("big.txt", mode: .reference, text: nil).badgeLabel == "reference")
        #expect(!attachment("big.txt", mode: .reference, text: nil).isUnreadImage)
        #expect(attachment("shot.png", mode: .ocr, text: "x").badgeLabel == "ocr")
        #expect(!attachment("shot.png", mode: .ocr, text: "x").isUnreadImage)
    }

    /// B6: a quote in the file name must not close the tag early.
    @Test func aQuoteInTheNameIsEscaped() {
        let rendered = ChatAttachment.modelFacing(
            text: "see this", attachments: [attachment("a\"b.txt")])
        #expect(rendered.contains("<attachment name=\"a&quot;b.txt\">"))
        #expect(!rendered.contains("name=\"a\"b.txt\""))
    }

    @Test func anAmpersandInTheNameIsEscapedFirst() {
        #expect(ChatAttachment.escapedAttribute("a&b") == "a&amp;b")
        #expect(ChatAttachment.escapedAttribute("a\"b") == "a&quot;b")
        #expect(ChatAttachment.escapedAttribute("&\"") == "&amp;&quot;")
    }

    @Test func environmentNoteIsNilWithNothingByReference() {
        #expect(ChatAttachment.environmentNote(for: [attachment("a.csv")]) == nil)
        #expect(ChatAttachment.environmentNote(for: []) == nil)
    }
}
