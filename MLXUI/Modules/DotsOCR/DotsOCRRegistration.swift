import Foundation
import MLXLMCommon
import MLXVLM

/// One-shot registration of the custom `dots_ocr` model + `DotsVLProcessor` into MLXVLM's public
/// factory registries. The registries are actors (async), so this must be awaited before the first
/// `loadContainer` — `DotsOCREngine.generate` does that via `ensureRegistered()`. Idempotent.
enum DotsOCRRegistration {
    private static let latch = OnceLatch()

    static func ensureRegistered() async {
        await latch.runOnce {
            // Model: config.json's `model_type == "dots_ocr"` → our DotsOCR.
            await VLMTypeRegistry.shared.registerModelType("dots_ocr") { data in
                let config = try JSONDecoder().decode(DotsOCRConfig.self, from: data)
                return DotsOCR(config)
            }
            // Processor: preprocessor_config.json's `processor_class == "DotsVLProcessor"`.
            await VLMProcessorTypeRegistry.shared.registerProcessorType("DotsVLProcessor") { data, tokenizer in
                let config = try JSONDecoder().decode(Qwen25VLProcessorConfiguration.self, from: data)
                return DotsOCRProcessor(config: config, tokenizer: tokenizer)
            }
        }
    }
}

/// Runs a block exactly once across the process, ordering-independent (guards the async registry
/// mutations so concurrent `generate` calls don't double-register).
actor OnceLatch {
    private var hasRun = false

    func runOnce(_ body: () async -> Void) async {
        if hasRun { return }
        hasRun = true
        await body()
    }
}
