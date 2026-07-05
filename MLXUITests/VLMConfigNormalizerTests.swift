import Testing
import Foundation
@testable import MLXUI

/// `VLMConfigNormalizer` fills the Qwen2.5-VL special-token ids that olmOCR-2's verbose
/// `config.json` omits (only `vision_token_id` survives its serialization), so the stock
/// `MLXVLM` factory can decode it instead of failing with `keyNotFound "image_token_id"`.
struct VLMConfigNormalizerTests {

    /// The real olmOCR-2 shape: `model_type == qwen2_5_vl`, only `vision_token_id` present.
    private static let olmOCRConfig: [String: Any] = [
        "model_type": "qwen2_5_vl",
        "vision_token_id": 151654,
        "hidden_size": 3584,
    ]

    @Test func fillsMissingQwen25VLTokenIds() throws {
        let patched = try #require(VLMConfigNormalizer.patchedConfig(Self.olmOCRConfig))

        #expect(patched["image_token_id"] as? Int == 151655)
        #expect(patched["video_token_id"] as? Int == 151656)
        #expect(patched["vision_start_token_id"] as? Int == 151652)
        #expect(patched["vision_end_token_id"] as? Int == 151653)
        // A value the checkpoint already provides is preserved, never overwritten.
        #expect(patched["vision_token_id"] as? Int == 151654)
        // Unrelated keys ride through untouched.
        #expect(patched["hidden_size"] as? Int == 3584)
    }

    @Test func treatsExplicitNullAsMissing() throws {
        var config = Self.olmOCRConfig
        config["image_token_id"] = NSNull()
        let patched = try #require(VLMConfigNormalizer.patchedConfig(config))
        #expect(patched["image_token_id"] as? Int == 151655)
    }

    @Test func leavesCompleteConfigUnchanged() {
        var config = Self.olmOCRConfig
        for (key, value) in VLMConfigNormalizer.qwen25VLTokenDefaults { config[key] = value }
        // Nothing to add → nil, so the caller skips the rewrite.
        #expect(VLMConfigNormalizer.patchedConfig(config) == nil)
    }

    @Test func ignoresNonQwen25VLConfigs() {
        // Other VLM architectures decode fine on their own; don't touch them.
        let config: [String: Any] = ["model_type": "idefics3", "hidden_size": 2048]
        #expect(VLMConfigNormalizer.patchedConfig(config) == nil)
    }

    @Test func patchesInstalledConfigFileInPlaceAndIsIdempotent() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("olmocr-cfg-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: Self.olmOCRConfig).write(to: url)

        VLMConfigNormalizer.normalizeConfigIfNeeded(at: dir)

        let reloaded = try JSONSerialization.jsonObject(
            with: Data(contentsOf: url)) as? [String: Any]
        #expect(reloaded?["image_token_id"] as? Int == 151655)
        #expect(reloaded?["vision_token_id"] as? Int == 151654)

        // Second pass is a no-op (already complete) — the file stays valid + complete.
        VLMConfigNormalizer.normalizeConfigIfNeeded(at: dir)
        let again = try JSONSerialization.jsonObject(
            with: Data(contentsOf: url)) as? [String: Any]
        #expect(again?["image_token_id"] as? Int == 151655)
    }
}
