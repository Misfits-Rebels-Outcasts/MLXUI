import Testing
@testable import MLXUI

/// BX-4 follow-up: flow 17's story check (owner ruling, Boo, 2026-11-14) proven on fixtures, no model needed.
struct BasicExpansionStoryCheckTests {
    typealias C = BasicExpansionStoryCheck

    @Test func failsOnAGemmaThoughtChannel() {
        #expect(C.violations(in: "<|channel>thought\nThe user wants a story.<channel|>Once upon a time.").contains("<|channel"))
    }

    @Test func failsOnAChannelCloseTag() {
        #expect(C.violations(in: "Once upon a time.<channel|>").contains("<channel|>"))
    }

    @Test func failsOnAThinkTag() {
        #expect(C.violations(in: "<think>plan</think>Once upon a time.").contains("<think"))
    }

    @Test func failsOnALeadingLineOfOnlyThoughtInAnyCase() {
        for lead in ["thought\nOnce upon a time.", "Thought\nOnce upon a time.", "THOUGHT\nOnce upon a time.",
                     "\n  thought  \nOnce upon a time."] {
            #expect(!C.violations(in: lead).isEmpty, "should fail: \(lead.debugDescription)")
        }
    }

    @Test func stillRejectsHashAndStar() {
        #expect(!C.violations(in: "# The Kitten\nOnce upon a time.").isEmpty)
        #expect(!C.violations(in: "Once upon a **time**.").isEmpty)
    }

    @Test func passesTheWordThoughtInsideProse() {
        #expect(C.violations(in: "She thought about the stars.").isEmpty)
        #expect(C.violations(in: "Once upon a time, a kitten thought.\nThought was a gentle thing.").isEmpty)
        #expect(C.violations(in: "The kitten curled up and fell asleep.").isEmpty)
    }
}
