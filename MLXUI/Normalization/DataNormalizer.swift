import Foundation

enum DataNormalizer {
    static func normalizeLicense(_ raw: String?) -> String {
        guard let raw else { return "Unknown" }
        let map: [String: String] = [
            "apache-2.0": "Apache 2.0", "apache-2": "Apache 2.0",
            "mit": "MIT",
            "cc-by-nc-4.0": "CC BY-NC 4.0", "cc-by-4.0": "CC BY 4.0",
            "llama3": "Llama 3 Community", "llama3.1": "Llama 3.1 Community",
            "gemma": "Gemma License", "other": "Custom", "openrail": "OpenRAIL",
            "bigcode-openrail-m": "BigCode OpenRAIL-M",
        ]
        let lower = raw.lowercased()
        for (key, value) in map where lower.contains(key) {
            return value
        }
        return raw.capitalized
    }

    static func normalizeArchitecture(_ raw: String?, config: [String: Any]? = nil) -> String {
        guard let raw else { return "Transformer" }
        let classMap: [String: String] = [
            "Qwen3ForCausalLM": "Qwen3", "Qwen2ForCausalLM": "Qwen2",
            "LlamaForCausalLM": "Llama", "GemmaForCausalLM": "Gemma",
            "Gemma2ForCausalLM": "Gemma 2", "MistralForCausalLM": "Mistral",
            "PhiForCausalLM": "Phi", "Phi3ForCausalLM": "Phi-3",
            "DeepseekV2ForCausalLM": "DeepSeek", "GPT2LMHeadModel": "GPT-2",
            "WhisperForConditionalGeneration": "Whisper", "BertModel": "BERT",
            "T5ForConditionalGeneration": "T5", "StableLmForCausalLM": "StableLM",
            "OlmoForCausalLM": "OLMo", "FalconForCausalLM": "Falcon",
        ]
        var result = classMap[raw] ?? raw
            .replacingOccurrences(of: "ForCausalLM", with: "")
            .replacingOccurrences(of: "ForConditionalGeneration", with: "")
        if let config {
            let numKV = config["num_key_value_heads"] as? Int
            let numHeads = config["num_attention_heads"] as? Int
            if let kv = numKV, let heads = numHeads {
                result += kv < heads ? " (GQA)" : " (MHA)"
            }
        }
        return result
    }

    static func normalizeTaskTags(_ raw: [String]?, domainId: String) -> [String] {
        let tagMap: [String: String] = [
            "text-generation": "chat", "conversational": "chat",
            "automatic-speech-recognition": "speech-to-text",
            "text-to-speech": "text-to-speech", "image-to-text": "ocr",
            "visual-question-answering": "vision", "fill-mask": "nlp",
            "token-classification": "nlp",
        ]
        var tags = (raw ?? []).compactMap { tagMap[$0] }
        tags = Array(Set(tags))
        if tags.isEmpty {
            let domainMap: [String: [String]] = [
                "natural-language": ["chat"], "conversational-ai": ["chat"],
                "code-software-engineering": ["code"],
                "vision": ["vision"], "audio": ["speech-to-text"],
                "music": ["music"], "video": ["video"],
            ]
            tags = domainMap[domainId] ?? []
        }
        return tags
    }
}
