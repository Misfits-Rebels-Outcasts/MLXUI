import CryptoKit
import Foundation
import Jinja
import Testing
@testable import MLXUI

/// MC2-3 — the language table and the chat template, against the pinned files in `Fixtures/TranslateGemma/`.
///
/// Three things are proven:
///  1. **The table is the template's.** Every one of the 581 codes in the pinned `chat_template.jinja` appears in
///     `TranslateGemmaLanguages`, in the template's order, with the template's name — and nothing else does.
///  2. **swift-jinja renders the template byte for byte.** The three reference requests (EN→MS, ZH-Hans→EN, and the
///     regional `en_GB` case whose underscore the template turns into a hyphen) match Python jinja2's output exactly
///     (`Fixtures/TranslateGemma/prompt_goldens.json`, written with the provenance recorded inside it).
///  3. **One BOS.** The rendered string starts with exactly one `<bos>` and contains no second one — the template
///     emits it itself, so `add_bos_token` must not be applied again (MC2 Trap 4).
struct TranslateGemmaTemplateTests {

    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
    }

    private func templateText() throws -> String {
        try String(contentsOf: root.appendingPathComponent("Fixtures/TranslateGemma/chat_template.jinja"),
                   encoding: .utf8)
    }

    private func goldens() throws -> [String: Any] {
        let data = try Data(contentsOf: root.appendingPathComponent("Fixtures/TranslateGemma/prompt_goldens.json"))
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: - 1. The table is the template's own

    /// The template's `languages` dict, parsed the way the fixture generator parses it: the dict literal sits between
    /// `{%- set languages = {` and the closing `-%}`, and is JSON-shaped apart from the trailing comma.
    private func templateLanguages() throws -> [(code: String, name: String)] {
        let text = try templateText()
        guard let start = text.range(of: "{%- set languages = {")?.upperBound,
              let end = text.range(of: "-%}", range: start..<text.endIndex)?.lowerBound else {
            return []
        }
        let literal = String(text[start..<end])
        let regex = try NSRegularExpression(pattern: #""([^"]+)"\s*:\s*"([^"]+)""#)
        let range = NSRange(literal.startIndex..., in: literal)
        return regex.matches(in: literal, range: range).compactMap { match in
            guard let codeRange = Range(match.range(at: 1), in: literal),
                  let nameRange = Range(match.range(at: 2), in: literal) else { return nil }
            return (String(literal[codeRange]), String(literal[nameRange]))
        }
    }

    @Test func templateFixtureIsThePinnedRevision() throws {
        let g = try goldens()
        let provenance = try #require(g["provenance"] as? [String: Any])
        let expectedSHA = try #require(provenance["templateSHA256"] as? String)
        let actualSHA = SHA256.hash(data: Data(try templateText().utf8))
            .map { String(format: "%02x", $0) }.joined()
        #expect(actualSHA == expectedSHA, "chat_template.jinja drifted from the pinned revision")
        #expect(expectedSHA == "5e85514635b80a62b64c9982810bfdfbdd482be9a59ed12c25ded2bf3aabddaf")
        #expect(provenance["model"] as? String
                == "mlx-community/translategemma-4b-it-4bit @ 5788ec08c047f3f2e17808101b8d9566ac930d58")
        #expect(provenance["jinja2"] as? String == "3.1.6")
        #expect(provenance["python"] as? String == "3.13.3")
    }

    @Test func theTableIsExactlyTheTemplatesLanguages() throws {
        let template = try templateLanguages()
        #expect(template.count == 581)
        #expect(TranslateGemmaLanguages.templateCodes.count == 581)
        #expect(TranslateGemmaLanguages.templateCodes.map(\.code) == template.map(\.code))
        #expect(TranslateGemmaLanguages.templateCodes.map(\.name) == template.map(\.name))
        // The reverse direction is the same assertion, spelled as a set so a duplicated code cannot hide.
        #expect(Set(TranslateGemmaLanguages.templateCodes.map(\.code)) == Set(template.map(\.code)))
    }

    @Test func oneListEntryPerNameWithChineseSplit() throws {
        let template = try templateLanguages()
        let templateNames = Set(template.map(\.name))
        #expect(templateNames.count == 161)
        #expect(TranslateGemmaLanguages.all.count == 162)
        // Chinese (the template's single name for all 13 Chinese codes) is replaced by the two script entries in place.
        #expect(!TranslateGemmaLanguages.all.contains { $0.name == "Chinese" })
        let names = TranslateGemmaLanguages.all.map(\.name)
        let simplified = try #require(names.firstIndex(of: "Chinese (Simplified)"))
        let traditional = try #require(names.firstIndex(of: "Chinese (Traditional)"))
        #expect(traditional == simplified + 1)
        #expect(names[simplified - 1] == "Zhuang")
        #expect(names[traditional + 1] == "Zulu")
        // Every other name is the template's own.
        #expect(Set(names).subtracting([TranslateGemmaLanguages.simplifiedChineseName,
                                       TranslateGemmaLanguages.traditionalChineseName]) == templateNames.subtracting(["Chinese"]))
        // One entry per name, no duplicate base code.
        let codes = TranslateGemmaLanguages.all.map(\.code)
        #expect(Set(codes).count == codes.count)
        // Every alias is a template code; the two split Chinese entries carry only their own code, and the other 11
        // Chinese codes are the `chineseCodes` field. Nothing else is duplicated or missing.
        let aliases = TranslateGemmaLanguages.all.flatMap(\.aliases)
        #expect(aliases.count == 581 - TranslateGemmaLanguages.chineseCodes.count + 2)   // 570
        #expect(Set(aliases).union(TranslateGemmaLanguages.chineseCodes) == Set(template.map(\.code)))
        #expect(Set(aliases).count == aliases.count, "no code appears twice across the entries")
        #expect(TranslateGemmaLanguages.chineseCodes.count == 13)
        #expect(Set(TranslateGemmaLanguages.chineseCodes).isSubset(of: Set(template.map(\.code))))
    }

    @Test func favouritesAreTheOrderedShortList() throws {
        #expect(TranslateGemmaLanguages.favourites.map(\.name)
                == ["English", "Malay", "Chinese (Simplified)", "Chinese (Traditional)",
                    "Indonesian", "Tamil", "Japanese", "Korean"])
        #expect(TranslateGemmaLanguages.favourites.map(\.code)
                == ["en", "ms", "zh-Hans", "zh-Hant", "id", "ta", "ja", "ko"])
    }

    @Test func baseCodesAndTheFilipinoFallback() throws {
        #expect(TranslateGemmaLanguages.language(named: "English")?.code == "en")
        #expect(TranslateGemmaLanguages.language(named: "Malay")?.code == "ms")
        #expect(TranslateGemmaLanguages.language(named: "tamil")?.code == "ta")
        // Filipino is the one template name with no bare code: fil-PH is its only code.
        #expect(TranslateGemmaLanguages.language(named: "Filipino")?.code == "fil-PH")
        // `Chinese` alone is deliberately not an entry: the split exists so the output script is never a guess.
        #expect(TranslateGemmaLanguages.language(named: "Chinese") == nil)
    }

    // MARK: - 2. swift-jinja renders the reference requests byte for byte

    @Test func rendersThePinnedGoldensByteForByte() throws {
        let goldens = try goldens()
        let cases = try #require(goldens["cases"] as? [[String: Any]])
        #expect(cases.count == 3)
        let template = try Template(try templateText(), with: .init(lstripBlocks: true, trimBlocks: true))
        for testCase in cases {
            let name = try #require(testCase["name"] as? String)
            let expected = try #require(testCase["expected"] as? String)
            let rawContext = try #require(testCase["context"] as? [String: Any])
            let context = try rawContext.mapValues { try Value(any: $0 as Any? ?? ()) }
            let rendered = try template.render(context)
            #expect(rendered == expected, "\(name) rendered differently from Python jinja2")
        }
    }

    @Test func theThreeCasesAreTheRequestedOnes() throws {
        let cases = try #require(try goldens()["cases"] as? [[String: Any]])
        let sources = cases.compactMap { testCase in
            (testCase["context"] as? [String: Any])
                .flatMap { $0["messages"] as? [[String: Any]] }?
                .first
                .flatMap { $0["content"] as? [[String: Any]] }?
                .first.flatMap { $0["source_lang_code"] as? String }
        }
        #expect(sources == ["en", "zh-Hans", "en_GB"])
        // The underscore survives into the rendered prompt as a hyphen: the template does `.replace("_", "-")`.
        let expected = cases.compactMap { $0["expected"] as? String }.joined()
        #expect(expected.contains("(en-GB)"))
        #expect(expected.contains("Chinese (zh-Hans)"))
    }

    @Test func theTemplateEmitsExactlyOneBOS() throws {
        let goldens = try goldens()
        for testCase in try #require(goldens["cases"] as? [[String: Any]]) {
            let expected = try #require(testCase["expected"] as? String)
            #expect(expected.hasPrefix("<bos>"), "the rendered prompt must start with the template's own BOS")
            #expect(expected.components(separatedBy: "<bos>").count == 2, "exactly one <bos> in the rendered prompt")
            #expect(!expected.contains("<start_of_turn>user\nYou are a professional <"))  // no Jinja undefined leaked
        }
    }

    // MARK: - 3. `to=` and detection resolution

    @Test func toAcceptsANameOrACode() {
        #expect(TranslateGemmaLanguages.requestCode(for: "Malay") == "ms")
        #expect(TranslateGemmaLanguages.requestCode(for: "malay") == "ms")          // case-insensitive name
        #expect(TranslateGemmaLanguages.requestCode(for: " Indonesian ") == "id")
        #expect(TranslateGemmaLanguages.requestCode(for: "Chinese (Simplified)") == "zh-Hans")
        #expect(TranslateGemmaLanguages.requestCode(for: "Chinese (Traditional)") == "zh-Hant")
        #expect(TranslateGemmaLanguages.requestCode(for: "ms") == "ms")
        #expect(TranslateGemmaLanguages.requestCode(for: "MS") == "ms")             // case-insensitive code
        #expect(TranslateGemmaLanguages.requestCode(for: "en_GB") == "en-GB")       // `_` ≡ `-`
        #expect(TranslateGemmaLanguages.requestCode(for: "fil-PH") == "fil-PH")
        // A regional code is kept as asked: the template names the variant in the prompt.
        #expect(TranslateGemmaLanguages.requestCode(for: "zh-TW") == "zh-TW")
    }

    @Test func anUnknownTargetIsRefused() {
        for token in ["Klingon", "xx", "", "  ", "Chinese", "tlh", "not a language"] {
            #expect(TranslateGemmaLanguages.requestCode(for: token) == nil, "\(token) must not resolve to a code")
            #expect(TranslateGemmaLanguages.language(matching: token) == nil, "\(token) must not resolve to a list entry")
        }
    }

    @Test func uiListEntryForACodeUsesTheScriptSplitForChinese() {
        #expect(TranslateGemmaLanguages.language(matching: "zh-TW")?.name == "Chinese (Traditional)")
        #expect(TranslateGemmaLanguages.language(matching: "zh-Hans-SG")?.name == "Chinese (Simplified)")
        #expect(TranslateGemmaLanguages.language(matching: "en-GB")?.name == "English")
        #expect(TranslateGemmaLanguages.name(forTemplateCode: "zh-Hant-HK") == "Chinese (Traditional)")
        #expect(TranslateGemmaLanguages.name(forTemplateCode: "en") == "English")
        #expect(TranslateGemmaLanguages.name(forTemplateCode: "xx") == nil)
    }

    @Test func detectionCodesMapOntoTheTemplate() {
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "en") == "en")
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "ms") == "ms")
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "zh-Hans") == "zh-Hans")
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "zh-Hant") == "zh-Hant")
        // Python's recogniser can answer a script-only tag; the split still settles the script.
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "zh") == "zh-Hans")
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "zh-TW") == "zh-Hant")
        // No exact template code for these: the base language / the script in the tag decides.
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "fil") == "fil-PH")
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "de-AT") == "de-AT")   // a table code, kept as asked
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "zh-Hant-TW") == "zh-Hant")
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "xx") == nil)
        #expect(TranslateGemmaLanguages.code(forDetectedLanguage: "") == nil)
    }
}
