// MLX forward-pass tests — commented out (slow; run manually with ⌘U)
/*
import Testing
import Foundation
import MLX
@testable import MLXUI

/// SD-ENG1 — shape/construction tests for the SDXL text encoders. Instantiate each encoder
/// with **unloaded** (random-init) weights, feed `[1, 77]` int32 token ids, and assert the
/// output shapes — CLIP-L `[1, 77, 768]`, OpenCLIP ViT-G/14 `[1, 77, 1280]`. No real weights:
/// this is a forward-pass plumbing check (the key-loading contract is verified by the engine's
/// `loadWeights` at run time; numerical parity is the human run-smoke).
///
/// Serialized like `DeepSeekOCRMLXTests`: MLX's default stream isn't safe under Swift Testing's
/// parallelism (SIGTRAP).
@Suite(.serialized)
struct SDXLTurboEncoderTests {

    @Test func clipLEncoderOutputsPenultimateHiddenStates() {
        let encoder = SDCLIPEncoder()
        let ids = MLXArray.arange(77).expandedDimensions(axis: 0)   // [1, 77] int32
        let out = encoder(ids)
        eval(out)
        #expect(out.shape == [1, 77, 768])
    }

    @Test func openCLIPEncoderOutputsPenultimateHiddenStates() {
        let encoder = SDOpenCLIPEncoder()
        let ids = MLXArray.arange(77).expandedDimensions(axis: 0)   // [1, 77] int32
        let out = encoder(ids)
        eval(out)
        #expect(out.shape == [1, 77, 1280])
    }

    @Test func openCLIPPooledAppliesTextProjection() {
        // SD-ENG4 — the UNet's `text_time` conditioning needs the pooled text_encoder_2 output
        // (EOS last-hidden state through `text_projection`) → [1, 1280].
        let encoder = SDOpenCLIPEncoder()
        let ids = MLXArray.arange(77).expandedDimensions(axis: 0)
        let pooled = encoder.pooled(ids)
        eval(pooled)
        #expect(pooled.shape == [1, 1280])
    }

    @Test func clipTokenizerMatchesFluxShapeContract() {
        // SDCLIPTokenizer is the FLUX CLIP tokenizer renamed; exercise the BPE + 77-window
        // contract directly on a tiny synthetic vocab so the rename didn't break anything.
        // "hello" → chars [h,e,l,l,o</w>], merged by rank: (h,e)→(he,l)→(hel,l)→(hell,o</w>).
        let vocabulary = ["<|startoftext|>": 0, "<|endoftext|>": 1, "hello</w>": 2]
        let ranks: [SDCLIPTokenizer.Bigram: Int] = [
            SDCLIPTokenizer.Bigram("h", "e"): 0,
            SDCLIPTokenizer.Bigram("he", "l"): 1,
            SDCLIPTokenizer.Bigram("hel", "l"): 2,
            SDCLIPTokenizer.Bigram("hell", "o</w>"): 3,
        ]
        let ids = SDCLIPTokenizer.tokenize("hello hello", bpeRanks: ranks, vocabulary: vocabulary)
        #expect(ids.first == 0)            // bos
        #expect(ids.last == 1)             // eos
        #expect(ids == [0, 2, 2, 1])       // bos + two "hello</w>" + eos
    }
}

*/
