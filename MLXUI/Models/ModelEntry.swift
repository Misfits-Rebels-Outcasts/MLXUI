import Foundation

nonisolated enum ModelSource: String, Codable {
    case mlx, coreai, coreml, research
}

nonisolated enum ModelType: String, Codable {
    case llm, asr, tts, embedding, vision, ocr, video, image, music, segmentation, upscale, rerank
    var sfSymbol: String {
        switch self {
        case .llm: "bubble.left.and.bubble.right"
        case .asr: "waveform"
        case .tts: "mouth"
        case .embedding: "square.grid.3x3"
        case .vision: "eye"
        case .ocr: "doc.text.viewfinder"
        case .video: "film"
        case .image: "photo.artframe"
        case .music: "music.note"
        case .segmentation: "lasso"
        case .upscale: "sparkle.magnifyingglass"
        case .rerank: "arrow.up.arrow.down.square"
        }
    }
}

nonisolated struct ModelVariant: Codable, Identifiable {
    var id: String { hfModelId }
    let quantization: String
    let format: String
    let ramGB: Double
    let downloadSizeGB: Double
    let qualityPercent: Int
    let hfModelId: String
    let recommended: Bool?
}

// Written explicitly, rather than relying on synthesis, so the conformance witnesses are
// unambiguously nonisolated under whole-module compilation (WMO surfaced a stray
// "main actor-isolated conformance" warning on the compiler-synthesized version).
nonisolated extension ModelVariant: Hashable {
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.quantization == rhs.quantization && lhs.format == rhs.format && lhs.ramGB == rhs.ramGB
            && lhs.downloadSizeGB == rhs.downloadSizeGB && lhs.qualityPercent == rhs.qualityPercent
            && lhs.hfModelId == rhs.hfModelId && lhs.recommended == rhs.recommended
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(quantization)
        hasher.combine(format)
        hasher.combine(ramGB)
        hasher.combine(downloadSizeGB)
        hasher.combine(qualityPercent)
        hasher.combine(hfModelId)
        hasher.combine(recommended)
    }
}

nonisolated struct ModelBenchmarks: Codable {
    let mmlu: Double?
    let humanEval: Double?
    let gsm8k: Double?
    let hellaswag: Double?
    let arc: Double?
    let truthfulQA: Double?
}

// Written explicitly, rather than relying on synthesis, so the conformance witnesses are
// unambiguously nonisolated under whole-module compilation (WMO surfaced a stray
// "main actor-isolated conformance" warning on the compiler-synthesized version).
nonisolated extension ModelBenchmarks: Hashable {
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.mmlu == rhs.mmlu && lhs.humanEval == rhs.humanEval && lhs.gsm8k == rhs.gsm8k
            && lhs.hellaswag == rhs.hellaswag && lhs.arc == rhs.arc && lhs.truthfulQA == rhs.truthfulQA
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(mmlu)
        hasher.combine(humanEval)
        hasher.combine(gsm8k)
        hasher.combine(hellaswag)
        hasher.combine(arc)
        hasher.combine(truthfulQA)
    }
}

nonisolated struct ModelEntry: Codable, Identifiable {
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

// Written explicitly, rather than relying on synthesis, so the conformance witnesses are
// unambiguously nonisolated under whole-module compilation (WMO surfaced a stray
// "main actor-isolated conformance" warning on the compiler-synthesized version).
nonisolated extension ModelEntry: Hashable {
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.id == rhs.id && lhs.family == rhs.family && lhs.displayName == rhs.displayName
            && lhs.paramSize == rhs.paramSize && lhs.paramCountB == rhs.paramCountB
            && lhs.modelType == rhs.modelType && lhs.source == rhs.source && lhs.format == rhs.format
            && lhs.platforms == rhs.platforms && lhs.minMacOSVersion == rhs.minMacOSVersion
            && lhs.hfRepo == rhs.hfRepo && lhs.hfModelId == rhs.hfModelId && lhs.ramGB == rhs.ramGB
            && lhs.downloadSizeGB == rhs.downloadSizeGB && lhs.contextWindow == rhs.contextWindow
            && lhs.license == rhs.license && lhs.licenseUrl == rhs.licenseUrl
            && lhs.description == rhs.description && lhs.summary == rhs.summary
            && lhs.descriptionSource == rhs.descriptionSource && lhs.architecture == rhs.architecture
            && lhs.languages == rhs.languages && lhs.lastUpdated == rhs.lastUpdated
            && lhs.taskTags == rhs.taskTags && lhs.benchmarks == rhs.benchmarks
            && lhs.speedTokensPerSec == rhs.speedTokensPerSec && lhs.speedHardware == rhs.speedHardware
            && lhs.speedEstimated == rhs.speedEstimated && lhs.communityDownloads == rhs.communityDownloads
            && lhs.communityLikes == rhs.communityLikes && lhs.variants == rhs.variants
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(family)
        hasher.combine(displayName)
        hasher.combine(paramSize)
        hasher.combine(paramCountB)
        hasher.combine(modelType)
        hasher.combine(source)
        hasher.combine(format)
        hasher.combine(platforms)
        hasher.combine(minMacOSVersion)
        hasher.combine(hfRepo)
        hasher.combine(hfModelId)
        hasher.combine(ramGB)
        hasher.combine(downloadSizeGB)
        hasher.combine(contextWindow)
        hasher.combine(license)
        hasher.combine(licenseUrl)
        hasher.combine(description)
        hasher.combine(summary)
        hasher.combine(descriptionSource)
        hasher.combine(architecture)
        hasher.combine(languages)
        hasher.combine(lastUpdated)
        hasher.combine(taskTags)
        hasher.combine(benchmarks)
        hasher.combine(speedTokensPerSec)
        hasher.combine(speedHardware)
        hasher.combine(speedEstimated)
        hasher.combine(communityDownloads)
        hasher.combine(communityLikes)
        hasher.combine(variants)
    }
}
