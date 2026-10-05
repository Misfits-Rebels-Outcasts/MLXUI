import Foundation
import Testing
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

    @Test func environmentNoteIsNilWithNothingByReference() {
        #expect(ChatAttachment.environmentNote(for: [attachment("a.csv")]) == nil)
        #expect(ChatAttachment.environmentNote(for: []) == nil)
    }
}
