import Foundation

/// Bundles the CLM SDK + UI and registers them. Backed by `ml-explore/mlx-swift-lm`'s
/// `MLXEmbedders` (the frozen Qwen3-8B encoder, run as an embedder) plus this module's own
/// `CLMHeads`/`CLMSchema` port of the CLM heads and question schema — CLM is Laya's bigger
/// sibling: an 8B encoder instead of 0.4B, 2048 tokens per text instead of 512, no
/// option-count ceiling. Runs the **installed** `RealityCat/CLM-v0.1-8B-MLX-8bit` checkpoint.
enum CLMModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "clm",
        displayName: "CLM (Decision)",
        modalities: ["text"],
        modelTypes: [.decision],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX"],
        notes: """
        A frozen Qwen3-8B encoder + two small projection heads (state/action), answering one \
        typed question (choice/score/yes-no) with a label, a probability per option, and a \
        confidence — never generative text. Ported from the Python `clm_mlx` package shipped \
        inside `RealityCat/CLM-v0.1-8B-MLX-8bit` (Apache-2.0). Claims source==.mlx decision \
        entries with family=="CLM" (CLM 8B) — discriminated from Laya's own decision entries \
        by family (CL-1 gate B), both sharing RunnerKind.decision.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: CLMSDK(), ui: CLMUI(), descriptor: descriptor)
    }
}
