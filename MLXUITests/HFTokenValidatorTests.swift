import Testing
@testable import MLXUI

/// Covers the pure HF token validation/normalization gating the Settings Save button (T2).
struct HFTokenValidatorTests {

    @Test func normalizedTrimsSurroundingWhitespace() {
        #expect(HFTokenValidator.normalized("  hf_abc123 \n") == "hf_abc123")
    }

    @Test func plausibleForWellFormedToken() {
        #expect(HFTokenValidator.isPlausible("hf_abc123") == true)
    }

    @Test func plausibleAfterTrimmingPastedWhitespace() {
        #expect(HFTokenValidator.isPlausible("  hf_abc123  ") == true)
    }

    @Test func notPlausibleWhenEmpty() {
        #expect(HFTokenValidator.isPlausible("") == false)
    }

    @Test func notPlausibleWithoutHFPrefix() {
        #expect(HFTokenValidator.isPlausible("abc123") == false)
    }

    @Test func notPlausibleWithEmbeddedWhitespace() {
        #expect(HFTokenValidator.isPlausible("hf_abc 123") == false)
    }
}
