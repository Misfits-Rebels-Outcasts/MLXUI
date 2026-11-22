import MLX

extension MLXArray {
    /// `self[i]` for a literal integer `i`, without the Gather. mlx-swift turns a lone integer subscript into `take` with a 0-dim index,
    /// and `Gather::eval_gpu` hands Metal a nil `setBytes` for that (Metal API Validation asserts: "bytes argument cannot be nil").
    /// A one-row slice and a squeeze is a view: same values, no Gather (MC5-FIX-finish-gather). `i` must be in range (not negative).
    func row(_ i: Int) -> MLXArray {
        self[i ..< i + 1].squeezed(axis: 0)
    }
}
