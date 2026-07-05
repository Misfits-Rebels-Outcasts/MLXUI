import Testing
@testable import MLXUI

/// Covers the pure display-string mappers in `DataNormalizer`.
struct DataNormalizerTests {

    // MARK: normalizeLicense

    @Test func licenseMapsKnownIdentifier() {
        #expect(DataNormalizer.normalizeLicense("apache-2.0") == "Apache 2.0")
        #expect(DataNormalizer.normalizeLicense("mit") == "MIT")
    }

    @Test func licenseFallsBackToUnknownForNil() {
        #expect(DataNormalizer.normalizeLicense(nil) == "Unknown")
    }

    @Test func licenseCapitalizesUnrecognizedValue() {
        #expect(DataNormalizer.normalizeLicense("bespoke") == "Bespoke")
    }

    // MARK: normalizeArchitecture

    @Test func architectureMapsKnownClass() {
        #expect(DataNormalizer.normalizeArchitecture("Qwen3ForCausalLM") == "Qwen3")
    }

    @Test func architectureDefaultsToTransformerForNil() {
        #expect(DataNormalizer.normalizeArchitecture(nil) == "Transformer")
    }

    @Test func architectureStripsCausalLMSuffixForUnknownClass() {
        #expect(DataNormalizer.normalizeArchitecture("NovelForCausalLM") == "Novel")
    }

    // MARK: normalizeTaskTags

    @Test func taskTagsMapRawHuggingFaceTags() {
        let tags = DataNormalizer.normalizeTaskTags(["text-generation"], domainId: "natural-language")
        #expect(tags == ["chat"])
    }

    @Test func taskTagsFallBackToDomainWhenEmpty() {
        let tags = DataNormalizer.normalizeTaskTags([], domainId: "vision")
        #expect(tags == ["vision"])
    }
}
