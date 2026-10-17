import Foundation

/// Bundles Clef Flash's SDK + UI and registers them. Backed by `ml-explore/mlx-swift-lm`'s
/// `MLXLLM` (`Qwen35Model`, the 4-bit Qwen3.5-9B text backbone) plus this module's own
/// `ClefPrompt`/`ClefHead`/`ClefAnswer` port of `clef_mlx.py`'s joint schema head.
///
/// Clef is the third `.decision` family after Laya (0.4B encoder) and CLM (8B encoder): one forward
/// pass over an 8,192-token prompt (gate F2), then a small head that answers every question in the request at
/// once — choice, score and yes/no — with probabilities, never generative text. Runs the **installed**
/// `mlx-community/clef-flash-4bit` checkpoint (Apache-2.0, 11 files, ~6.2 GB).
enum ClefModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "clef",
        displayName: "Clef Flash (Decision)",
        modalities: ["text"],
        modelTypes: [.decision],
        backingPackage: "ml-explore/mlx-swift-lm",
        packageLicense: "MIT",
        maintainers: ["Apple MLX", "Cloudflare"],
        notes: """
        A 4-bit Qwen3.5-9B backbone plus a 244 MB joint schema head, answering typed questions \
        (choice/score/yes-no) with a label, a probability per option and a confidence — never \
        generative text. Ported from the Python `clef_mlx.py` shipped in \
        `mlx-community/clef-flash-4bit` (Apache-2.0). One forward pass answers every question in a \
        request together. Claims source==.mlx decision entries with family=="Clef", discriminated \
        from Laya's and CLM's own decision entries by family, all three sharing RunnerKind.decision.
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: ClefSDK(), ui: ClefUI(), descriptor: descriptor)
    }
}
