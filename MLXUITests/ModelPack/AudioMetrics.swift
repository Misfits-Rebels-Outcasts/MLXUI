import Foundation

/// MP5-3a: the measuring tools the DeepFilterNet parity check (MP5-3b) uses, and a deterministic noise source for building the
/// noisy clip. Test target only. A parity check compares SNR improvement and RMS against a clean reference with a tolerance,
/// never bit for bit (owner ruling 2026-10-25).
enum AudioMetrics {
    /// Root mean square of `samples`; 0 for none.
    static func rms(_ samples: [Float]) -> Double {
        guard !samples.isEmpty else { return 0 }
        var sum = 0.0
        for s in samples { sum += Double(s) * Double(s) }
        return (sum / Double(samples.count)).squareRoot()
    }

    /// Signal-to-noise ratio in dB of `estimate` against the clean `reference`: 20 log10(rms(reference) / rms(estimate - reference)).
    /// Compares the common length. `.infinity` when the estimate is identical, `-.infinity` for a silent reference.
    static func snrDB(estimate: [Float], reference: [Float]) -> Double {
        let n = min(estimate.count, reference.count)
        guard n > 0 else { return 0 }
        var noise = [Float](repeating: 0, count: n)
        for i in 0..<n { noise[i] = estimate[i] - reference[i] }
        let signalRMS = rms(Array(reference[0..<n]))
        let noiseRMS = rms(noise)
        if signalRMS == 0 { return -.infinity }
        if noiseRMS == 0 { return .infinity }
        return 20 * log10(signalRMS / noiseRMS)
    }

    /// SNR improvement in dB: how much closer `cleaned` is to `reference` than `noisy` was.
    static func improvementDB(noisy: [Float], cleaned: [Float], reference: [Float]) -> Double {
        snrDB(estimate: cleaned, reference: reference) - snrDB(estimate: noisy, reference: reference)
    }

    /// `reference + noise`, with `noise` scaled so the mix has exactly `snrDB` against `reference`.
    static func mix(reference: [Float], noise: [Float], snrDB: Double) -> [Float] {
        let n = min(reference.count, noise.count)
        let noiseRMS = rms(Array(noise[0..<n]))
        guard noiseRMS > 0 else { return Array(reference[0..<n]) }
        let target = rms(Array(reference[0..<n])) / pow(10, snrDB / 20)
        let gain = Float(target / noiseRMS)
        return (0..<n).map { reference[$0] + noise[$0] * gain }
    }

    /// Steady broadband noise plus a mains hum (`humHz` and its second harmonic), fully determined by `seed` (SplitMix64).
    /// Unit RMS-ish before scaling by ``mix(reference:noise:snrDB:)``.
    static func noise(count: Int, sampleRate: Int, seed: UInt64, humHz: Double = 50) -> [Float] {
        var generator = SplitMix64(seed: seed)
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let t = Double(i) / Double(sampleRate)
            // Sum of 4 uniforms: a cheap bell shape, zero mean, deterministic.
            let u = (generator.unit() + generator.unit() + generator.unit() + generator.unit() - 2) * 0.866
            let hum = 0.6 * sin(2 * .pi * humHz * t) + 0.3 * sin(2 * .pi * 2 * humHz * t)
            out[i] = Float(0.7 * u + 0.5 * hum)
        }
        return out
    }
}

/// A small seeded generator, so the test noise is the same on every machine and run.
struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    /// A uniform value in [0, 1).
    mutating func unit() -> Double { Double(next() >> 11) / Double(1 << 53) }
}
