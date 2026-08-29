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

    // SA-AM2: segmentation tag mappings
    @Test func taskTagsMapsImageSegmentation() {
        let tags = DataNormalizer.normalizeTaskTags(["image-segmentation", "segment-anything"], domainId: "segmentation")
        #expect(tags == ["segmentation"])
    }

    @Test func taskTagsDomainFallbackSegmentation() {
        let tags = DataNormalizer.normalizeTaskTags([], domainId: "segmentation")
        #expect(tags == ["segmentation"])
    }

    // WAN-AM2: video tag mappings
    @Test func taskTagsMapsTextToVideo() {
        let tags = DataNormalizer.normalizeTaskTags(["text-to-video", "video-generation"], domainId: "videogen")
        #expect(tags == ["video"])
    }

    @Test func taskTagsDomainFallbackVideogen() {
        let tags = DataNormalizer.normalizeTaskTags([], domainId: "videogen")
        #expect(tags == ["video"])
    }

    @Test func architectureMapsWanDiT() {
        #expect(DataNormalizer.normalizeArchitecture("WanDiT") == "Wan DiT")
    }

    // SV-AM2: upscale tag mappings
    @Test func taskTagsMapsImageSuperResolution() {
        let tags = DataNormalizer.normalizeTaskTags(["image-super-resolution", "image-upscaling", "video-restoration"], domainId: "upscale")
        #expect(Set(tags) == Set(["upscaling", "restoration"]))
    }

    @Test func taskTagsDomainFallbackUpscale() {
        let tags = DataNormalizer.normalizeTaskTags([], domainId: "upscale")
        #expect(tags == ["upscaling"])
    }
}
