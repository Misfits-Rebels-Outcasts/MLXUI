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

    // MARK: - S2-8: one-call-per-turn history

    private func call(_ id: String?, _ name: String) -> ServeToolCall {
        ServeToolCall(id: id, name: name, argumentsJSON: "{}")
    }

    private func assistant(_ text: String, _ calls: [ServeToolCall]) -> ServeMessage {
        ServeMessage(role: .assistant, text: text, toolCalls: calls)
    }

    private func result(_ id: String?, _ name: String?, _ text: String) -> ServeMessage {
        ServeMessage(role: .tool, text: text, toolCallID: id, toolName: name)
    }

    /// The SV-6 shape: one assistant turn, two calls, two results — split and interleaved.
    @Test func aTwoCallTurnSplitsAndInterleavesItsResults() {
        let messages = [
            msg(.user, "do it"),
            assistant("on it", [call("a", "read"), call("b", "write")]),
            result("a", "read", "file body"),
            result("b", "write", "ok"),
        ]
        #expect(ServeMessageAssembly.splittingMultiCallTurns(messages) == [
            msg(.user, "do it"),
            assistant("on it", [call("a", "read")]),      // the turn's text rides the first piece
            result("a", "read", "file body"),
            assistant("", [call("b", "write")]),
            result("b", "write", "ok"),
        ])
    }

    /// Three calls (what this Mac's Llama emitted) and a result that arrived *before* its turn.
    @Test func threeCallsSplitAndResultsAreMatchedWhereverTheySit() {
        let messages = [
            msg(.user, "go"),
            result("c", "glob", "1 match"),                // out of order: names a later call
            assistant("", [call("a", "read"), call("b", "write"), call("c", "glob")]),
            result("a", "read", "body"),
            result("b", "write", "ok"),
        ]
        #expect(ServeMessageAssembly.splittingMultiCallTurns(messages) == [
            msg(.user, "go"),
            assistant("", [call("a", "read")]),
            result("a", "read", "body"),
            assistant("", [call("b", "write")]),
            result("b", "write", "ok"),
            assistant("", [call("c", "glob")]),
            result("c", "glob", "1 match"),
        ])
    }

    /// A result whose id matches no call keeps the client's order — after the split pieces, not
    /// guessed into one of them.
    @Test func anUnmatchedResultKeepsTheClientsOrder() {
        let messages = [
            msg(.user, "go"),
            assistant("", [call("a", "read"), call("b", "write")]),
            result("zz", nil, "from nowhere"),
            result("a", "read", "body"),
        ]
        #expect(ServeMessageAssembly.splittingMultiCallTurns(messages) == [
            msg(.user, "go"),
            assistant("", [call("a", "read")]),
            result("a", "read", "body"),
            assistant("", [call("b", "write")]),
            result("zz", nil, "from nowhere"),
        ])
    }

    /// Calls **without ids** are still split (otherwise the template refuses), and their results
    /// keep the client's order after the pieces.
    @Test func callsWithoutIdsStillSplitAndTheirResultsKeepOrder() {
        let messages = [
            msg(.user, "go"),
            assistant("t", [call(nil, "read"), call(nil, "write")]),
            result(nil, "read", "body"),
            result(nil, "write", "ok"),
        ]
        #expect(ServeMessageAssembly.splittingMultiCallTurns(messages) == [
            msg(.user, "go"),
            assistant("t", [call(nil, "read")]),
            assistant("", [call(nil, "write")]),
            result(nil, "read", "body"),
            result(nil, "write", "ok"),
        ])
    }

    /// A single call — every family's normal shape — is untouched, text and result position alike.
    @Test func aSingleCallTurnIsUntouched() {
        let messages = [msg(.user, "go"), assistant("t", [call("a", "read")]), result("a", "read", "body")]
        #expect(ServeMessageAssembly.splittingMultiCallTurns(messages) == messages)
    }

    /// A family **without** the limit keeps the client's packed turn: the wire history is already
    /// what its template loops over, and splitting would change a prompt that works.
    @Test func aFamilyWithoutTheLimitKeepsThePackedTurn() {
        let messages = [
            msg(.user, "go"),
            assistant("", [call("a", "read"), call("b", "write")]),
            result("a", "read", "body"),
            result("b", "write", "ok"),
        ]
        #expect(ServeMessageAssembly.assembling(messages, acceptsTools: true, oneCallPerTurn: false,
                                                placement: .anywhere) == messages)
        // …and with the flag, the same input comes out split.
        let split = ServeMessageAssembly.assembling(messages, acceptsTools: true, oneCallPerTurn: true,
                                                    placement: .anywhere)
        #expect(split.map(\.role) == [.user, .assistant, .tool, .assistant, .tool])
    }

    /// The split pieces survive the assembly's own same-role merge (S2-21's rule) — which is exactly
    /// why the interleaving order matters: two assistant turns in a row *would* be glued.
    @Test func theSplitPiecesAreNotGluedByTheMerge() {
        let messages = [
            msg(.user, "go"),
            assistant("", [call("a", "read"), call("b", "write")]),
            result("a", "read", "body"),
            result("b", "write", "ok"),
        ]
        let assembled = ServeMessageAssembly.assembling(messages, acceptsTools: true, oneCallPerTurn: true,
                                                        placement: .firstOnly)
        #expect(assembled.count == 5)
        #expect(assembled.map(\.role) == [.user, .assistant, .tool, .assistant, .tool])
        #expect(assembled[1].toolCalls.map(\.name) == ["read"])
        #expect(assembled[3].toolCalls.map(\.name) == ["write"])
    }

    /// The per-family flag, cited from the templates (S2-8): only `llama` refuses a packed turn.
    @Test func onlyTheLlamaFamilyCarriesTheOneCallLimit() {
        #expect(ToolCallHistoryTable.oneCallPerTurn(forModelType: "llama"))
        #expect(ToolCallHistoryTable.oneCallPerTurn(forModelType: "Llama"))
        #expect(!ToolCallHistoryTable.oneCallPerTurn(forModelType: "llama4"))       // unread template
        for family in ["qwen3_5", "qwen3_5_text", "qwen3", "qwen2", "mistral3", "ministral3",
                       "gemma2", "gemma3_text", "gpt_oss", "some_new_arch"] {
            #expect(!ToolCallHistoryTable.oneCallPerTurn(forModelType: family), "\(family)")
        }
        // The silent variant is recorded too (llama drops the rest, gpt-oss never reaches the path).
        #expect(ToolCallHistoryTable.rendersFirstCallOnly(forModelType: "llama"))
        #expect(ToolCallHistoryTable.rendersFirstCallOnly(forModelType: "gpt_oss"))
        #expect(!ToolCallHistoryTable.rendersFirstCallOnly(forModelType: "qwen3"))
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
