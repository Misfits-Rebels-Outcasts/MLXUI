import Foundation
import MLXLMCommon
import MLXVLM

/// One-shot registration of the custom `deepseekocr_2` model + `DeepseekVLV2Processor` into MLXVLM's
/// public factory registries (async actors → must be awaited before the first `loadContainer`).
/// Idempotent; mirrors `DotsOCRRegistration`.
enum DeepSeekOCRRegistration {
    private static let latch = OnceLatch()

    static func ensureRegistered() async {
        await latch.runOnce {
            await VLMTypeRegistry.shared.registerModelType("deepseekocr_2") { data in
                let config = try JSONDecoder().decode(DeepSeekOCRConfig.self, from: data)
                return DeepSeekOCR(config)
            }
            await VLMProcessorTypeRegistry.shared.registerProcessorType("DeepseekVLV2Processor") { _, tokenizer in
                DeepSeekOCRProcessor(tokenizer: tokenizer)
            }
        }
    }
}
