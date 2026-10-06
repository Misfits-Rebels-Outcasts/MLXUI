import Foundation
import Testing
@testable import MLXUI

// S2-1 — R22: the server's message assembly hoists (or folds) `system` messages for the families
// whose own chat template cannot take them where the client put them. The per-family answer is
// `ToolCallFormatTable.systemPlacement(forModelType:)`; this suite exercises the assembly rule
// with placements directly, then once end-to-end after `OpenAIChatDialect.decode`.
//
// Implementer's call, pending owner confirmation (see the S2-1 journal).

struct ServeMessageAssemblyTests {

    private func msg(_ role: ServeMessage.Role, _ text: String) -> ServeMessage {
        ServeMessage(role: role, text: text)
    }

    // MARK: strict families (qwen3_5, mistral3, gemma3_text, gpt_oss)

    @Test func systemInTheMiddleMovesToTheFrontForAStrictFamily() {
        let messages = [msg(.user, "a"), msg(.assistant, "b"), msg(.system, "s"), msg(.user, "c")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly)
                == [msg(.system, "s"), msg(.user, "a"), msg(.assistant, "b"), msg(.user, "c")])
    }

    @Test func everySystemMessageIsMergedIntoOneLeadingTurn() {
        let messages = [msg(.system, "first"), msg(.user, "u"), msg(.system, "second")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly)
                == [msg(.system, "first\n\nsecond"), msg(.user, "u")])
    }

    /// Moving a system message out of the middle can make two same-role turns adjacent that were
    /// not before — a strict-alternation template rejects that too, so the merge is re-applied.
    @Test func hoistingReMergesTurnsItMakesAdjacent() {
        let messages = [msg(.user, "a"), msg(.system, "s"), msg(.user, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly)
                == [msg(.system, "s"), msg(.user, "a\n\nb")])
    }

    @Test func aSingleLeadingSystemMessageIsLeftExactlyAsItIs() {
        let messages = [msg(.system, "s"), msg(.user, "a"), msg(.assistant, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly) == messages)
    }

    /// A single leading system message is already where a `.firstOnly` family needs it, empty text
    /// and all — nothing is rebuilt, so nothing else changes either.
    @Test func aLeadingEmptySystemMessageIsLeftAsItIs() {
        let messages = [msg(.system, ""), msg(.user, "a")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly) == messages)
    }

    /// Where the list *is* rebuilt to hoist, an empty system message leaves no empty turn behind.
    @Test func anEmptySystemMessageDoesNotSurviveHoisting() {
        let messages = [msg(.user, "a"), msg(.system, ""), msg(.user, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly)
                == [msg(.user, "a\n\nb")])
    }

    @Test func aStrictFamilyWithNoSystemMessageIsUntouched() {
        let messages = [msg(.user, "a"), msg(.assistant, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .firstOnly) == messages)
    }

    // MARK: families that allow `system` anywhere (llama, qwen3, qwen2, unknown)

    @Test func aSystemMessageInTheMiddleStaysPutForAFamilyThatAllowsIt() {
        let messages = [msg(.user, "a"), msg(.assistant, "b"), msg(.system, "s"), msg(.user, "c")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .anywhere) == messages)
    }

    @Test func theDefaultPlacementForAnUnknownFamilyLeavesTheOrderAlone() {
        let placement = ToolCallFormatTable.systemPlacement(forModelType: "phi3")
        #expect(placement == .anywhere)
        let messages = [msg(.user, "a"), msg(.system, "s"), msg(.user, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: placement) == messages)
    }

    // MARK: a family with no `system` role at all (gemma2)

    @Test func foldingPutsTheSystemTextAtTheFrontOfTheFirstUserTurn() {
        let messages = [msg(.system, "s"), msg(.user, "a"), msg(.assistant, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .unsupported)
                == [msg(.user, "s\n\na"), msg(.assistant, "b")])
    }

    @Test func foldingUsesTheFirstUserTurnEvenWhenItIsNotFirst() {
        let messages = [msg(.assistant, "b"), msg(.system, "s"), msg(.user, "a")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .unsupported)
                == [msg(.assistant, "b"), msg(.user, "s\n\na")])
    }

    @Test func foldingCreatesAUserTurnWhenThereIsNoneToFoldInto() {
        let messages = [msg(.system, "s"), msg(.assistant, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .unsupported)
                == [msg(.user, "s"), msg(.assistant, "b")])
    }

    @Test func foldingReMergesUserTurnsItMakesAdjacent() {
        let messages = [msg(.user, "a"), msg(.system, "s"), msg(.user, "b")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .unsupported)
                == [msg(.user, "s\n\na\n\nb")])
    }

    @Test func foldingWithNoSystemMessageIsUntouched() {
        let messages = [msg(.user, "a")]
        #expect(ServeMessageAssembly.hoistingSystemMessages(messages, placement: .unsupported) == messages)
    }

    // MARK: the whole assembly, in the order R22 describes

    /// `decode` flattens and merges; the backend then applies the family's placement. This is the
    /// pipeline a request actually takes, with the model's `model_type` standing in for the
    /// checkpoint the backend reads.
    @Test func decodeThenHoistMatchesWhatTheBackendHandsTheTemplate() throws {
        let body = Data(#"""
        {"model":"m","messages":[
          {"role":"user","content":"first"},
          {"role":"assistant","content":"ok"},
          {"role":"system","content":"be terse"},
          {"role":"user","content":"second"},
          {"role":"user","content":"and third"}
        ]}
        """#.utf8)
        let decoded = try OpenAIChatDialect.decode(body)
        // S1-3c's merge has already run inside decode.
        #expect(decoded.messages.map(\.role) == [.user, .assistant, .system, .user])

        let forQwen35 = ServeMessageAssembly.hoistingSystemMessages(
            decoded.messages, placement: ToolCallFormatTable.systemPlacement(forModelType: "qwen3_5"))
        #expect(forQwen35.map(\.role) == [.system, .user, .assistant, .user])
        #expect(forQwen35[0].text == "be terse")
        #expect(forQwen35[3].text == "second\n\nand third")

        // Llama 3.1 renders a later system header inline and never raises, so R22 does not apply.
        let forLlama = ServeMessageAssembly.hoistingSystemMessages(
            decoded.messages, placement: ToolCallFormatTable.systemPlacement(forModelType: "llama"))
        #expect(forLlama == decoded.messages)

        // Gemma 2 has no system role to hoist into.
        let forGemma2 = ServeMessageAssembly.hoistingSystemMessages(
            decoded.messages, placement: ToolCallFormatTable.systemPlacement(forModelType: "gemma2"))
        #expect(forGemma2.map(\.role) == [.user, .assistant, .user])
        #expect(forGemma2[0].text == "be terse\n\nfirst")
        #expect(forGemma2[2].text == "second\n\nand third")
    }

    /// The dialect's own name for the merge rule still forwards to the neutral one.
    @Test func theDialectsMergeEntryPointStillMatchesTheSharedRule() {
        let messages = [msg(.assistant, ""), msg(.assistant, "x"), msg(.user, "y"), msg(.user, "")]
        #expect(OpenAIChatDialect.mergingAdjacentRoles(messages)
                == ServeMessageAssembly.mergingAdjacentRoles(messages))
        #expect(OpenAIChatDialect.mergingAdjacentRoles(messages)
                == [msg(.assistant, "x"), msg(.user, "y")])
    }
}
