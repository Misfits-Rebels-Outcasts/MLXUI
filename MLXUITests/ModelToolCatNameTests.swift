import Testing
import Foundation
@testable import MLXUI

/// MT1-3: display names a saved `.cat` cannot write. The parser reads a lowercase-leading name or one with an
/// extension-like dotted token as settings, so the row has no model (DA-8). Model Tools therefore builds its one-row
/// `Row` directly (ModelToolRunner.oneRowDocument). This pins today's parser behaviour; it matters to MT5 serialisation.
struct ModelToolCatNameTests {

    /// `nil` for a line that does not parse or whose row has no model.
    private func model(_ line: String) -> String? {
        (try? CatParser.parse("mlxflow 0.8\n\n\(line)\n"))?.rows.first?.model
    }

    @Test func todaysParserReadsTheseNamesAsSettingsNotAModel() {
        #expect(model("1. OCR   dots.ocr") == nil)
        #expect(model("1. OCR   olmOCR-2 7B") == nil)
        #expect(model("1. Generate Video   Wan2.1-T2V-1.3B") == nil)
    }

    /// The behaviour we would want: the written name is the row's model. Red today, SPEC-Q249 (catflow-mlx).
    @Test func aSavedCatShouldBeAbleToNameThem() {
        // SPEC-Q249
        withKnownIssue("SPEC-Q249: dots.ocr / olmOCR-2 7B parse as settings") {
            #expect(model("1. OCR   dots.ocr") == "dots.ocr")
            #expect(model("1. OCR   olmOCR-2 7B") == "olmOCR-2 7B")
        }
    }
}
