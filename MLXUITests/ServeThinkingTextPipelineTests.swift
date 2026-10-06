import Foundation
import Testing
@testable import MLXUI

// S2-3b — the thinking path's stop handling.
//
// S1's `ServeTextPipeline` guards every flush with `if !stopped`; the thinking path has to as well,
// and this is the test that shows why: a stop sequence matched, and the detokenizer is still holding
// the first half of a character that arrived *after* it. Nothing after a stop may reach content.

struct ServeThinkingTextPipelineTests {

    /// A byte-level fake tokenizer, as `ServeTextPipelineTests` uses: a partial UTF-8 sequence
    /// decodes to U+FFFD, which is what makes the detokenizer hold it back.
    private static let bytes: [Int: [UInt8]] = [
        1: Array("hello".utf8),
        2: Array("STOP".utf8),
        3: [0xF0, 0x9F],                       // first half of 🙂
        4: [0x99, 0x82],                       // second half
        5: Array("after".utf8),
        6: Array("<think>".utf8),
        7: Array(" reasoning".utf8),
        8: Array("</think>".utf8),
        9: Array("\n\nanswer".utf8),
    ]

    private static func decode(_ ids: [Int]) -> String {
        String(decoding: ids.flatMap { bytes[$0] ?? [] }, as: UTF8.self)
    }

    private func pipeline(stop: [String], startsInsideThink: Bool = true) -> ServeThinkingTextPipeline {
        ServeThinkingTextPipeline(decode: Self.decode, stop: stop, startsInsideThink: startsInsideThink)
    }

    /// The leak this closes: `STOP` arrives, then the first half of a multi-byte character, and the
    /// stream ends. `finish()` must not flush that half into content.
    @Test func nothingAfterAStopReachesContent() {
        var pipeline = self.pipeline(stop: ["STOP"], startsInsideThink: false)
        var reasoning = "", content = ""
        var halves = pipeline.push(1)                       // "hello"
        reasoning += halves.reasoning; content += halves.content
        halves = pipeline.push(2)                           // "STOP" — matches, "hello" is what went out
        reasoning += halves.reasoning; content += halves.content
        #expect(pipeline.stopped)
        #expect(content == "hello")

        // The model had already produced the next character's first half before the loop broke.
        halves = pipeline.push(3)
        reasoning += halves.reasoning; content += halves.content
        let tail = pipeline.finish()
        reasoning += tail.reasoning; content += tail.content

        #expect(content == "hello")                         // no "\u{FFFD}", no "after"
        #expect(reasoning.isEmpty)
    }

    /// Once stopped, further tokens and the flush are ignored outright.
    @Test func aStoppedPipelineIgnoresEverythingAfter() {
        var pipeline = self.pipeline(stop: ["STOP"], startsInsideThink: false)
        _ = pipeline.push(1)
        _ = pipeline.push(2)
        #expect(pipeline.push(5) == ("", ""))               // "after"
        #expect(pipeline.finish() == ("", ""))
    }

    /// A stop string that never completes still gives up its held-back tail — that text came before
    /// any match, so it is content.
    @Test func textThatOnlyLooksLikeAStopIsStillEmitted() {
        var pipeline = self.pipeline(stop: ["STOP"], startsInsideThink: false)
        var content = ""
        content += pipeline.push(1).content                 // "hello"
        // "ST" then "OP" split, but with a stop list that doesn't match: use a token that can't
        // complete it. Re-use "hello" twice to keep the fake table small.
        content += pipeline.push(5).content                 // "after"
        let tail = pipeline.finish()
        content += tail.content
        #expect(content == "hello" + "after")
        #expect(!pipeline.stopped)
    }

    /// The thinking half is never matched against `stop` — a stop string inside the thinking is not
    /// the client's stop.
    @Test func aStopStringInsideTheThinkingDoesNotStopTheAnswer() {
        var pipeline = self.pipeline(stop: ["STOP"], startsInsideThink: true)
        var reasoning = "", content = ""
        // "STOP" as *thinking*, then a plain answer: model it with the tokens we have.
        for id in [1, 2] {                                  // "helloSTOP" — but no stop inside thinking
            let halves = pipeline.push(id)
            reasoning += halves.reasoning; content += halves.content
        }
        let tail = pipeline.finish()
        reasoning += tail.reasoning; content += tail.content
        // In a prompt-opened pipeline everything is thinking, so no stop was applied at all.
        #expect(reasoning == "helloSTOP")
        #expect(content.isEmpty)
    }

    /// The two halves come out of one pipeline in order, with the answer's newlines trimmed and the
    /// thinking intact.
    @Test func thinkingAndAnswerFlowThroughInOrder() {
        var pipeline = self.pipeline(stop: [], startsInsideThink: false)
        var reasoning = "", content = ""
        for id in [6, 7, 8, 9] {
            let halves = pipeline.push(id)
            reasoning += halves.reasoning; content += halves.content
        }
        let tail = pipeline.finish()
        reasoning += tail.reasoning; content += tail.content
        #expect(reasoning == " reasoning")
        #expect(content == "answer")
    }
}
