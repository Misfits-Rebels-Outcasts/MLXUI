import Testing
import Foundation
import MLX
@testable import MLXUI

/// Pins the pooled output shape of `FluxCLIPEncoder` after the EOS pooling was rewritten from a
/// bare-Int / `MLXArray` subscript (a 0-d `Gather`, which aborts under Metal API Validation —
/// LY-5, journal 2026-336) to a host-Int position plus width-1 Range slices (a `Slice`).
///
/// Random-init weights, no checkpoint: a plumbing/shape check only. Numerical parity is the
/// human run-smoke, and **Metal API Validation is never active in the test host**, so this
/// cannot reproduce the original abort — it guards the shape contract the rewrite must keep.
///
/// Serialized like the other MLX suites: MLX's default stream isn't safe under Swift Testing's
/// parallelism (SIGTRAP).
@Suite(.serialized)
struct FluxCLIPEncoderTests {

    /// A full 77-token window, EOS (the largest id, as in the real vocab) at the last position.
    @Test func pooledOutputIsVectorOfModelWidthForFullWindow() {
        let encoder = FluxCLIPEncoder()
        let ids = MLXArray.arange(FluxCLIPEncoder.maxLength).asType(.int32).expandedDimensions(axis: 0)
        let pooled = encoder(ids)
        eval(pooled)
        #expect(pooled.shape == [768])
    }

    /// A short, unpadded prompt (what `FluxCLIPTokenizer.tokenize` actually returns): the EOS
    /// position is mid-window, so the host-Int slice path is exercised off the last index.
    @Test func pooledOutputIsVectorOfModelWidthForShortPrompt() {
        let encoder = FluxCLIPEncoder()
        let ids = MLXArray([49406, 320, 1125, 539, 320, 49407].map(Int32.init)).expandedDimensions(axis: 0)
        #expect(ids.shape == [1, 6])
        let pooled = encoder(ids)
        eval(pooled)
        #expect(pooled.shape == [768])
    }
}
