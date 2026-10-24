import Foundation
import Testing

@testable import MLXUI

/// S-FIX-1: a decode that merges a space into a following comma (swift-transformers' `clean_up_tokenization_spaces`) shrinks the
/// decoded window, and the delta — sliced by the OLD window's length — used to swallow the first scalar of the new text. Found as a
/// JSON comma that vanished from a model's output (the model had emitted the token `,"`); it affects any server reply.
struct IncrementalDetokenizerCleanupTests {
    /// Each id is a string piece; the decode joins them and then applies the " ," → "," clean-up, as the tokenizer does.
    private static let pieces = ["\"]", "\n", "    ", ",\"", "steps", "\":"]
    private static func decode(_ ids: [Int]) -> String {
        ids.map { pieces[$0] }.joined().replacingOccurrences(of: " ,", with: ",")
    }

    @Test func aCommaAfterWhitespaceIsNotSwallowedWhenTheDecodeCleansUpTheSpace() {
        var detokenizer = IncrementalDetokenizer(decode: Self.decode)
        var out = ""
        for id in 0 ..< Self.pieces.count { out += detokenizer.push(id) }
        out += detokenizer.flush()
        #expect(out.contains(",\"steps"), "the comma survived: \(out.debugDescription)")
        #expect(out.replacingOccurrences(of: " ", with: "") == "\"]\n,\"steps\":", "every non-space scalar of the stream is emitted once")
    }

    @Test func anOrdinaryStreamIsUnchanged() {
        var detokenizer = IncrementalDetokenizer(decode: { $0.map { "<\($0)>" }.joined() })
        var out = ""
        for id in [1, 2, 3] { out += detokenizer.push(id) }
        #expect(out == "<1><2><3>")
    }

    @Test func aTrailingIncompleteSequenceIsStillHeldBack() {
        // U+FFFD at the end of the window means "incomplete UTF-8"; it is held until the next token completes it.
        var detokenizer = IncrementalDetokenizer(decode: { ids in ids == [1] ? "a\u{FFFD}" : ids.map { $0 == 1 ? "a" : "é" }.joined() })
        #expect(detokenizer.push(1) == "")
        #expect(detokenizer.push(2) == "aé", "the held text is released together with the completing token")
    }
}
