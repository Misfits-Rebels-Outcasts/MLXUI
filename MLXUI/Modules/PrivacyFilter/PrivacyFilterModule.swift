import Foundation

/// Bundles the Privacy Filter SDK and UI and registers them. OpenAI's `openai/privacy-filter` (Apache-2.0, bf16 as
/// published) ported to Swift/MLX: a bidirectional token-classification encoder with a 128-expert MoE feed-forward, a
/// 33-class BIOES head and `opf`'s constrained Viterbi span decoder, loading the published safetensors the installer
/// downloads — nothing is downloaded at run time, and no Python runs.
///
/// Claims only `.redact` privacy-filter-family entries (Trap 10: by family, never by architecture). The Run screen is
/// Redact (paste/open, side-by-side highlight, checkbox re-render, copy/save/export); the AI Workflows row is `Redact`
/// (SPEC-Q251).
enum PrivacyFilterModule: ModelModule {
    static let descriptor = ModelModuleDescriptor(
        id: "privacy-filter",
        displayName: "OpenAI Privacy Filter (PII redaction, MLX)",
        modalities: ["text"],
        modelTypes: [.redact],
        backingPackage: "openai/privacy-filter (Swift/MLX port)",
        packageLicense: "Apache-2.0",
        maintainers: ["OpenAI"],
        notes: """
        PII detection and redaction for text: eight categories (person, email, phone, URL, address, date, account \
        number, secret) are found by a 1.5 B-parameter bidirectional token classifier and each detected span is \
        replaced by its placeholder (<PRIVATE_PERSON>, …). A from-scratch Swift/MLX port of the official bf16 \
        checkpoint: 8 banded-attention blocks (band 128), 14 query / 2 KV heads, 128 experts with top-4 routing, a \
        33-class BIOES head and opf's constrained Viterbi decoder at the file's default operating point. Inputs run \
        up to 128,000 tokens with linear memory via 1,024-token-halo windows; Stop cancels between windows. Primarily \
        English, can miss personal data, and is a redaction aid, not an anonymisation or compliance guarantee. Claims \
        only privacy-filter-family .redact models; no other module claims that kind. The AI Workflows row "Redact" \
        runs it (text → redacted text, with an only= category subset).
        """
    )

    static func register(into registry: ModelRegistry) {
        registry.add(sdk: PrivacyFilterSDK(), ui: PrivacyFilterUI(), descriptor: descriptor)
    }
}
