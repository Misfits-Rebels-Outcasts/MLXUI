import Foundation

/// MC2-4 — the request TranslateGemma's chat template accepts, built in the one place the engine, the Properties-tab
/// preview and the tests read (MC2 Trap 2; FV-2).
///
/// The template is strict about user content: it must be an iterable with **exactly one** item, that item a mapping
/// with `type`, `source_lang_code`, `target_lang_code` and `text`. It looks both codes up in its own 581-code table
/// **without raising on a miss** (Jinja undefined renders a broken prompt), so both codes are validated here before
/// anything is rendered. The rendered text below is the template's own output for that request, byte for byte
/// (`TranslateGemmaTemplateTests` proves the fixture render equals Python jinja2's, and the gated real-weights test
/// proves the tokenizer's `applyChatTemplate` produces the same token ids).
nonisolated enum TranslateGemmaPrompt {

    /// The structured user message the template expects (Trap 2).
    static func message(sourceCode: String, targetCode: String, text: String) throws -> [String: any Sendable] {
        let codes = try validatedCodes(sourceCode: sourceCode, targetCode: targetCode)
        return [
            "role": "user",
            "content": [[
                "type": "text",
                "source_lang_code": codes.source,
                "target_lang_code": codes.target,
                "text": text,
            ] as [String: any Sendable]],
        ]
    }

    /// `UserInput(prompt: .messages(…))` for the processor, which renders this through the checkpoint's own
    /// `chat_template.jinja` with `add_generation_prompt: true` and a single BOS.
    static func messages(sourceCode: String, targetCode: String, text: String) throws -> [[String: any Sendable]] {
        [try message(sourceCode: sourceCode, targetCode: targetCode, text: text)]
    }

    /// The rendered prompt, exactly as the pinned template renders it for this request. Used by the Properties-tab
    /// preview (so a preview and a run cannot disagree) and by the tests.
    static func render(sourceCode: String, targetCode: String, text: String) throws -> String {
        let codes = try validatedCodes(sourceCode: sourceCode, targetCode: targetCode)
        guard let sourceName = TranslateGemmaLanguages.templateName(for: codes.source),
              let targetName = TranslateGemmaLanguages.templateName(for: codes.target) else {
            throw TranslateGemmaError.unknownLanguage(codes.source)
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)   // the template pipes `text` through `| trim`
        guard !trimmed.isEmpty else { throw TranslateGemmaError.emptyText }
        return """
        <bos><start_of_turn>user
        You are a professional \(sourceName) (\(codes.source)) to \(targetName) (\(codes.target)) translator. Your goal \
        is to accurately convey the meaning and nuances of the original \(sourceName) text while adhering to \
        \(targetName) grammar, vocabulary, and cultural sensitivities.
        Produce only the \(targetName) translation, without any additional explanations or commentary. Please \
        translate the following \(sourceName) text into \(targetName):


        \(trimmed)<end_of_turn>
        <start_of_turn>model

        """
    }

    /// Both codes must be in the pinned table (Trap 2) — an unknown code renders a broken prompt instead of raising.
    /// The returned codes are the table's own spelling (`en_gb` → `en-GB`).
    static func validatedCodes(sourceCode: String, targetCode: String) throws -> (source: String, target: String) {
        guard let source = TranslateGemmaLanguages.canonicalTemplateCode(sourceCode) else {
            throw TranslateGemmaError.unknownLanguage(sourceCode)
        }
        guard let target = TranslateGemmaLanguages.canonicalTemplateCode(targetCode) else {
            throw TranslateGemmaError.unknownLanguage(targetCode)
        }
        return (source, target)
    }
}
