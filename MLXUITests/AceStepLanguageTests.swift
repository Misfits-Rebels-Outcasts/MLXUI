import Foundation
import Testing
@testable import MLXUI

/// MC5-T-lang (owner ruling, 2026-11-21): the Run screen's lyric-language picker is English, Chinese, German, French and Other, and the code it picks reaches the prompt verbatim.
/// "de" and "fr" are in upstream's `VALID_LANGUAGES` (`acestep/constants.py`).
@MainActor
struct AceStepLanguageTests {

    /// v1 ships exactly five entries (owner ruling, 2026-11-21; do not add or remove any in MC5). The first four send upstream's own `vocal_language` codes (`VALID_LANGUAGES`). "Other" sends
    /// `unknown`, which is also in upstream's list ("used when language cannot be determined automatically"): the prompt's `# Languages` line then says `unknown` and the model is left to infer
    /// the language from the lyrics themselves, so "Other" covers every other language upstream supports, with less guidance than a named one.
    @Test func thePickerOffersExactlyTheseFiveWithTheseCodes() {
        #expect(AceStepRunModel.languages.count == 5)
        let options = AceStepRunModel.languages.map { "\($0.name)=\($0.code)" }
        #expect(options == ["English=en", "Chinese=zh", "German=de", "French=fr", "Other=unknown"], "\(options)")
    }

    @Test(arguments: ["en", "zh", "de", "fr", "unknown"])
    func theChosenCodeIsTheRequestsLanguageAndTheLyricPromptsLanguageLine(code: String) async throws {
        final class Box: @unchecked Sendable { var request: AceStepRequest? }
        let box = Box()
        let model = AceStepRunModel(generator: { request, _ in box.request = request; return AceStepAudio(samples: [0, 0]) })
        model.vocalsEnabled = true        // the Vocals (experimental) section is open: lyrics and the picker count
        model.lyrics = "la la"
        model.language = code
        model.generate()
        await model.finished()
        let request = try #require(box.request)
        #expect(request.language == code)
        #expect(request.isInstrumental == false)
        let prompt = AceStepPrompt.lyricString(lyrics: request.lyrics, language: request.language, instrumental: request.isInstrumental)
        #expect(prompt.hasPrefix("# Languages\n\(code)\n\n# Lyric\nla la"), "\(prompt)")
    }

    @Test func anInstrumentalSongStillSendsUnknownWhateverTheLanguage() {
        #expect(AceStepPrompt.lyricString(lyrics: "", language: "fr", instrumental: true).hasPrefix("# Languages\nunknown\n"))
    }
}
