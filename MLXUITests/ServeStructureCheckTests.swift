import Foundation
import MLXGuidedGeneration
import MLXLMCommon
import Testing

@testable import MLXUI

// WW4-1-2 — ServeStructure, ServeStructureCheck: the three 400s, the GrammarError mapping, and ruling AO's no-op for
// {"type": "text"}. Pure: no model, no server wiring, no dialect parsing (WW4-1-4).

/// One token per character after a BOS, so the count of a minimal reply is its length (+1 for the special token `encode` adds).
private struct CharacterTokenizer: MLXLMCommon.Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        (addSpecialTokens ? [1] : []) + text.unicodeScalars.map { Int($0.value) + 10 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
                           additionalContext: [String: any Sendable]?) throws -> [Int] { [] }
}

struct ServeStructureCheckTests {
    private static let param = "response_format.json_schema.schema"
    private static let valid = #"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#     // smallest reply: {"a":0} = 7 tokens
    private static func schema(_ text: String = valid) -> ServeStructure { .jsonSchema(name: "x", schema: text, strict: true) }

    // MARK: - ServeStructure

    @Test func aRequestIsUnconstrainedByDefault() {
        let request = ServeRequest(model: "m", messages: [])
        #expect(request.structure == .none)
        #expect(!request.structure.isConstrained)
        #expect(request.structure.grammarFormat == nil)
    }

    @Test func theStructureKindsMapToWhatTheGrammarCacheCompiles() {
        #expect(ServeStructure.jsonObject.isConstrained)
        #expect(ServeStructure.jsonObject.grammarFormat == .jsonObject)
        #expect(Self.schema().grammarFormat == .jsonSchema(Self.valid))
        #expect(Self.schema().isConstrained)
    }

    // MARK: - ruling AO: {"type": "text"} is a no-op

    @Test func aTextFormatWithToolsIsNeverRefusedAndNeverConstrained() {
        // `{"type": "text"}` decodes to `.none` (WW4-1-4). With tools honoured it must behave exactly as a request without the field.
        let outcome = ServeStructureCheck.validate(structure: .none, param: Self.param, honouredTools: true, effectiveMaxTokens: 1,
                                                   minimumTokens: { Issue.record("a text format must not even measure the schema"); return 99 })
        #expect(outcome == nil)
        #expect(ServeStructureCheck.combinationRefusal(structure: .none, honouredTools: true) == nil)
        #expect(!ServeStructure.none.isConstrained)
        #expect(ServeStructure.none.grammarFormat == nil)
    }

    @Test func toolsPlusAJSONFormatIsAnUnsupportedCombination() {
        for structure in [ServeStructure.jsonObject, Self.schema()] {
            let error = ServeStructureCheck.combinationRefusal(structure: structure, honouredTools: true)
            #expect(error?.status == 400)
            #expect(error?.code == "unsupported_combination")
            #expect(error?.type == "invalid_request_error")
            #expect(error?.message.contains("tool_choice") == true, "says how to fix it")
        }
    }

    @Test func toolsThatAreNotHonouredDoNotConflict() {
        // A family that ignores `tools`, or `tool_choice: "none"`, renders none: nothing to conflict with.
        for structure in [ServeStructure.jsonObject, Self.schema()] {
            #expect(ServeStructureCheck.combinationRefusal(structure: structure, honouredTools: false) == nil)
        }
    }

    // MARK: - the schema refusal and its messages

    @Test func aGoodSchemaPassesAndABadOneIsA400InvalidJSONSchema() {
        #expect(ServeStructureCheck.schemaRefusal(Self.valid, param: Self.param) == nil)
        let error = ServeStructureCheck.schemaRefusal(#"{"type":"object","properties":{"q":{"type":"integer","multipleOf":5}}}"#, param: Self.param)
        #expect(error?.status == 400)
        #expect(error?.code == "invalid_json_schema")
        #expect(error?.type == "invalid_request_error")
        #expect(error?.message.contains("response_format.json_schema.schema/properties/q/multipleOf") == true)
        #expect(error?.message.contains("`multipleOf`") == true)
    }

    @Test func malformedSchemasNameTheParam() {
        let error = ServeStructureCheck.schemaRefusal("{nope", param: "text.format.schema")
        #expect(error?.status == 400)
        #expect(error?.code == "invalid_json_schema")
        #expect(error?.message.hasPrefix("text.format.schema ") == true)
    }

    @Test func aSlowConstructIsA400NamingTheConstructThePointerAndTheBound() {
        let error = ServeStructureCheck.schemaRefusal(#"{"type":"object","properties":{"title":{"type":"string","maxLength":200}}}"#, param: Self.param)
        #expect(error?.status == 400)
        #expect(error?.message.contains("response_format.json_schema.schema/properties/title/maxLength") == true)
        #expect(error?.message.contains("`maxLength` is 200") == true)
        #expect(error?.message.contains("maximum of \(ServeSchemaCheck.Bounds.maxLength)") == true)
    }

    @Test func manyFindingsAreSummarisedNotDumped() {
        let properties = (0 ..< 10).map { #""p\#($0)":{"type":"integer","multipleOf":2}"# }.joined(separator: ",")
        let error = ServeStructureCheck.schemaRefusal(#"{"type":"object","properties":{\#(properties)}}"#, param: Self.param)
        #expect(error?.message.contains("(and 4 more)") == true)
        #expect(error?.message.components(separatedBy: "can't be enforced").count == ServeStructureCheck.sentencesShown + 1, "six sentences shown")
        #expect((error?.message.count ?? 0) < 2_000)
    }

    // MARK: - GrammarError → ServeError

    @Test func aHardRejectionByXGrammarIsA400CarryingItsMessage() {
        let error = ServeStructureCheck.error(from: .invalidJSONSchema("Unsupported type \"foo\""), param: Self.param)
        #expect(error.status == 400)
        #expect(error.code == "invalid_json_schema")
        #expect(error.type == "invalid_request_error")
        #expect(error.message.contains("Unsupported type \"foo\""))
        #expect(error.message.hasPrefix(Self.param))
        #expect(ServeStructureCheck.error(from: .invalidJSONSchema("  "), param: Self.param).message.contains("not a valid JSON Schema"))
    }

    @Test func everyOtherGrammarErrorIsA500WithNoLibraryText() {
        let errors: [GrammarError] = [.tokenizerCreationFailed("secret tokens"), .constraintCompilationFailed("secret compile"),
                                      .maskComputationFailed("secret mask"), .commitFailed("secret commit"),
                                      .rollbackFailed("secret rollback"), .forkFailed("secret fork")]
        for grammarError in errors {
            let error = ServeStructureCheck.error(from: grammarError, param: Self.param)
            #expect(error.status == 500)
            #expect(error.type == "server_error")
            #expect(error.code == ServeStructureCheck.unavailableCode)
            #expect(!error.message.contains("secret"), "a backend failure doesn't leak its internals")
        }
    }

    @Test func aRealXGrammarRejectionMapsToA400() async throws {
        // End to end through the real compiler: a schema `ServeSchemaCheck` lets through that xgrammar throws on.
        let vocabulary = try ServeGrammarVocabulary(key: "k", vocab: ["a", "b", "<eos>"], vocabType: .raw, eosTokenID: 2, whitespaceTokenIDs: [])
        let schema = #"{"type":"object","properties":{"t":{"type":"nonsense"}}}"#
        #expect(ServeSchemaCheck.check(schema) == .ok, "the pre-check doesn't know every hard error — xgrammar does")
        do {
            _ = try await ServeGrammarCache().lease(vocabulary: vocabulary, format: .jsonSchema(schema))
            Issue.record("xgrammar should reject an unknown type")
        } catch let error as GrammarError {
            let mapped = ServeStructureCheck.error(from: error, param: Self.param)
            #expect(mapped.status == 400)
            #expect(mapped.code == "invalid_json_schema")
            #expect(mapped.message.count > Self.param.count + 10)
        }
    }

    // MARK: - max_tokens_too_small_for_schema

    @Test func theSmallestReplyIsCountedWithoutTheSpecialToken() {
        // {"a":0} is 7 characters; the fake tokenizer's encode() adds a BOS that must not be counted.
        #expect(ServeStructureCheck.minimumReplyTokens(schema: Self.valid, tokenizer: CharacterTokenizer()) == 7)
        #expect(ServeStructureCheck.minimumReplyTokens(schema: #"{"type":"object"}"#, tokenizer: CharacterTokenizer()) == 2)   // {}
    }

    @Test func aSchemaTheEstimatorCannotSynthesiseHasNoKnownMinimumNotAMadeUpOne() {
        // `CompletionReserve.estimate` answers its default (64) for these; that is not a reserve.
        #expect(ServeStructureCheck.minimumReplyTokens(schema: "{}", tokenizer: CharacterTokenizer()) == nil)
        #expect(ServeStructureCheck.minimumReplyTokens(schema: "not json", tokenizer: CharacterTokenizer()) == nil)
        #expect(ServeStructureCheck.replyBudgetRefusal(effectiveMaxTokens: 1, minimumTokens: nil) == nil)
    }

    @Test func aReplyBudgetBelowTheSmallestReplyIsA400NamingBothNumbers() {
        let refused = ServeStructureCheck.replyBudgetRefusal(effectiveMaxTokens: 6, minimumTokens: 7)
        #expect(refused?.status == 400)
        #expect(refused?.code == "max_tokens_too_small_for_schema")
        #expect(refused?.type == "invalid_request_error")
        #expect(refused?.message.contains("7 tokens") == true)
        #expect(refused?.message.contains("allows 6") == true)
        #expect(ServeStructureCheck.replyBudgetRefusal(effectiveMaxTokens: 7, minimumTokens: 7) == nil, "exactly enough is enough")
        #expect(ServeStructureCheck.replyBudgetRefusal(effectiveMaxTokens: 8, minimumTokens: 7) == nil)
    }

    // MARK: - all three, in order

    private static func validate(_ structure: ServeStructure, tools: Bool = false, maxTokens: Int = 4096, minimum: Int? = 7) -> ServeError? {
        ServeStructureCheck.validate(structure: structure, param: param, honouredTools: tools, effectiveMaxTokens: maxTokens, minimumTokens: { minimum })
    }

    @Test func validateAppliesTheRefusalsInOrder() {
        let bad = #"{"type":"object","properties":{"q":{"type":"integer","multipleOf":5}}}"#
        #expect(Self.validate(Self.schema(bad), tools: true, maxTokens: 1)?.code == "invalid_json_schema", "schema first")
        #expect(Self.validate(Self.schema(), tools: true, maxTokens: 1)?.code == "unsupported_combination", "then tools")
        #expect(Self.validate(Self.schema(), tools: false, maxTokens: 1)?.code == "max_tokens_too_small_for_schema", "then the budget")
        #expect(Self.validate(Self.schema()) == nil)
    }

    @Test func jsonObjectIsOnlyRefusedForToolsNeverForItsBudget() {
        #expect(Self.validate(.jsonObject, tools: true)?.code == "unsupported_combination")
        #expect(Self.validate(.jsonObject, maxTokens: 1, minimum: 99) == nil, "{} is two tokens: nothing worth refusing")
        #expect(Self.validate(.jsonObject) == nil)
    }

    @Test func theTokenizerIsOnlyWalkedWhenEarlierChecksPass() {
        let walked = LockedBox(false)
        _ = ServeStructureCheck.validate(structure: Self.schema(#"{"not":{}}"#), param: Self.param, honouredTools: false, effectiveMaxTokens: 1,
                                         minimumTokens: { walked.value = true; return 7 })
        _ = ServeStructureCheck.validate(structure: Self.schema(), param: Self.param, honouredTools: true, effectiveMaxTokens: 1,
                                         minimumTokens: { walked.value = true; return 7 })
        #expect(!walked.value)
        _ = ServeStructureCheck.validate(structure: Self.schema(), param: Self.param, honouredTools: false, effectiveMaxTokens: 1,
                                         minimumTokens: { walked.value = true; return 7 })
        #expect(walked.value)
    }
}

private final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: T
    init(_ value: T) { stored = value }
    var value: T { get { lock.lock(); defer { lock.unlock() }; return stored } set { lock.lock(); stored = newValue; lock.unlock() } }
}
