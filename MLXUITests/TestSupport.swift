import Foundation
@testable import MLXUI

/// Builds a `ModelEntry` with sane defaults so individual tests only specify the
/// fields they care about. `ModelEntry` has many `let` properties and only the
/// synthesized memberwise initializer, so this factory keeps the tests readable.
func makeEntry(
    id: String = "test-id",
    family: String = "TestFamily",
    displayName: String = "Test Model",
    modelType: ModelType = .llm,
    source: ModelSource = .mlx,
    ramGB: Double = 8.0,
    downloadSizeGB: Double = 4.0,
    lastUpdated: String? = nil,
    benchmarks: ModelBenchmarks? = nil,
    speedTokensPerSec: Double? = nil,
    speedEstimated: Bool? = nil,
    communityDownloads: Int? = nil,
    variants: [ModelVariant]? = nil
) -> ModelEntry {
    ModelEntry(
        id: id,
        family: family,
        displayName: displayName,
        paramSize: "4B",
        paramCountB: 4.0,
        modelType: modelType,
        source: source,
        format: "mlx",
        platforms: ["macos"],
        minMacOSVersion: nil,
        hfRepo: "mlx-community/Test",
        hfModelId: "mlx-community/Test-4bit",
        ramGB: ramGB,
        downloadSizeGB: downloadSizeGB,
        contextWindow: 4096,
        license: nil,
        licenseUrl: nil,
        description: nil,
        summary: nil,
        descriptionSource: nil,
        architecture: nil,
        languages: nil,
        lastUpdated: lastUpdated,
        taskTags: nil,
        benchmarks: benchmarks,
        speedTokensPerSec: speedTokensPerSec,
        speedHardware: nil,
        speedEstimated: speedEstimated,
        communityDownloads: communityDownloads,
        communityLikes: nil,
        variants: variants
    )
}
