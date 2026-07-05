import Foundation

enum ModelSource: String, Codable {
    case mlx, coreai, coreml, research
}

enum ModelType: String, Codable {
    case llm, asr, tts, embedding, vision, ocr, video
    var sfSymbol: String {
        switch self {
        case .llm: "bubble.left.and.bubble.right"
        case .asr: "waveform"
        case .tts: "mouth"
        case .embedding: "square.grid.3x3"
        case .vision: "eye"
        case .ocr: "doc.text.viewfinder"
        case .video: "film"
        }
    }
}

struct ModelVariant: Codable, Identifiable, Hashable {
    var id: String { hfModelId }
    let quantization: String
    let format: String
    let ramGB: Double
    let downloadSizeGB: Double
    let qualityPercent: Int
    let hfModelId: String
    let recommended: Bool?
}

struct ModelBenchmarks: Codable, Hashable {
    let mmlu: Double?
    let humanEval: Double?
    let gsm8k: Double?
    let hellaswag: Double?
    let arc: Double?
    let truthfulQA: Double?
}

struct ModelEntry: Codable, Identifiable, Hashable {
    let id: String
    let family: String
    let displayName: String
    let paramSize: String
    let paramCountB: Double?
    let modelType: ModelType

    let source: ModelSource
    let format: String
    let platforms: [String]
    let minMacOSVersion: String?

    let hfRepo: String
    let hfModelId: String

    let ramGB: Double
    let downloadSizeGB: Double
    let contextWindow: Int?

    let license: String?
    let licenseUrl: String?
    let description: String?
    /// One-line blurb for cards/lists. Falls back to `description`'s lead if absent.
    let summary: String?
    /// Provenance of `description`/`summary` (HF repo id, "curated", or "generated").
    let descriptionSource: String?
    let architecture: String?
    let languages: [String]?
    let lastUpdated: String?

    let taskTags: [String]?
    let benchmarks: ModelBenchmarks?

    let speedTokensPerSec: Double?
    let speedHardware: String?
    let speedEstimated: Bool?

    let communityDownloads: Int?
    let communityLikes: Int?

    let variants: [ModelVariant]?

    // ── Runtime-scaled speed ──
    func scaledSpeed(bandwidthGBps: Double) -> Double? {
        guard let base = speedTokensPerSec else { return nil }
        let ratio = bandwidthGBps / 400.0
        return round(base * ratio)
    }

    // ── Best variant ──
    func bestVariant(for availableRAMGB: Double) -> ModelVariant {
        guard let variants, !variants.isEmpty else {
            return ModelVariant(quantization: format, format: format,
                ramGB: ramGB, downloadSizeGB: downloadSizeGB,
                qualityPercent: 100, hfModelId: hfModelId, recommended: true)
        }
        return variants.first(where: { $0.recommended == true && $0.ramGB <= availableRAMGB })
            ?? variants.first(where: { $0.ramGB <= availableRAMGB })
            ?? variants.first!
    }

    // ── Quality ──
    var qualityScore: Double? {
        guard let b = benchmarks else { return nil }
        let scores: [Double] = [b.mmlu, b.humanEval, b.gsm8k, b.hellaswag, b.arc, b.truthfulQA].compactMap { $0 }
        guard !scores.isEmpty else { return nil }
        let avg = scores.reduce(0, +) / Double(scores.count)
        return max(1.0, min(5.0, avg / 20.0))
    }

    // ── Display helpers ──
    func exceedsRAM(_ systemRAMGB: Double) -> Bool {
        ramGB > systemRAMGB
    }

    /// Short blurb for cards/lists: prefer `summary`, else the first line of
    /// `description`, else a generated fallback.
    var displaySummary: String {
        if let s = summary?.trimmingCharacters(in: .whitespacesAndNewlines), !s.isEmpty {
            return s
        }
        if let first = description?.split(separator: "\n").first {
            let line = String(first).trimmingCharacters(in: .whitespacesAndNewlines)
            if !line.isEmpty { return line }
        }
        return "\(displayName) — \(modelType.rawValue.uppercased()) model (mlx-community)."
    }

    var displaySpeed: String {
        guard let speed = speedTokensPerSec else { return "—" }
        let prefix = (speedEstimated == true) ? "~" : ""
        return "\(prefix)\(Int(speed)) tok/s"
    }

    func formattedDownloads() -> String {
        guard let d = communityDownloads else { return "—" }
        if d >= 1_000_000 { return String(format: "%.1fM", Double(d) / 1_000_000) }
        if d >= 1_000 { return String(format: "%.1fK", Double(d) / 1_000) }
        return "\(d)"
    }
}
