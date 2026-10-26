import Foundation
import MLX
import MLXAudioSTS

/// The real ``LFMAudioBackend``: `LFM2AudioModel` (+ its tokenizer processor) and `LFM2AudioDetokenizer`, both loaded from the
/// installed folder. Not `Sendable`: built and used inside one `LFMAudioEngine.withBackend` region on the audio thread, so no
/// array it makes ever crosses a thread (KICKOFF rule 17). Every array the library hands back is read through `MLXErrorGuard`
/// (17a); every array this file builds is evaluated before it is used elsewhere (17b).
nonisolated final class LFMAudioRealBackend: LFMAudioBackend {
    private let model: LFM2AudioModel
    private let processor: LFM2AudioProcessor
    private let directory: URL
    private var detokenizer: LFM2AudioDetokenizer?

    /// Token ids that are control tokens, not words: end of message, audio start, end of text.
    private static let controlTokens: Set<Int> = [lfmImEndToken, lfmAudioStartToken, lfmTextEndToken]

    private init(model: LFM2AudioModel, processor: LFM2AudioProcessor, directory: URL) {
        self.model = model
        self.processor = processor
        self.directory = directory
    }

    /// The files a load needs, checked first so a missing one is a readable sentence, not the library's path error. The root
    /// `tokenizer-*.safetensors` is deliberately not on the list (readme T7).
    static let requiredFiles = ["config.json", "model.safetensors", "tokenizer.json",
                                "audio_detokenizer/config.json", "audio_detokenizer/model.safetensors"]

    static func missingFiles(in directory: URL) -> [String] {
        requiredFiles.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    static func load(_ directory: URL) async throws -> LFMAudioRealBackend {
        guard missingFiles(in: directory).isEmpty else {
            throw LFMAudioEngine.EngineError.noWeights(directory.lastPathComponent)
        }
        // The engine feeds the model 16 kHz and reads 24 kHz back; refuse a checkpoint whose config says otherwise.
        let rates = try rates(inConfig: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard rates.input == LFMAudioEngine.inputSampleRate, rates.output == LFMAudioEngine.outputSampleRate else {
            throw LFMAudioEngine.EngineError.unexpectedRates(input: rates.input, output: rates.output)
        }
        try Task.checkCancellation()
        let model = try await LFM2AudioModel.fromModelDirectory(directory)
        guard let processor = model.processor else { throw LFMAudioError.tokenizerNotLoaded }
        return LFMAudioRealBackend(model: model, processor: processor, directory: directory)
    }

    /// The model's rates from its `config.json`: the preprocessor's `sample_rate` (input) and the top-level one (output, 24000 by
    /// the library's default when the file omits it).
    static func rates(inConfig data: Data) throws -> (input: Int, output: Int) {
        let config = try JSONDecoder().decode(LFM2AudioConfig.self, from: data)
        return (config.preprocessor.sampleRate, config.sampleRate)
    }

    /// Test seam: mel frames the library's preprocessor makes for `sampleCount` samples when told `claimedRate`. The library
    /// ignores that argument (it always uses the config's rate), which is why ``LFMAudioEngine`` resamples to 16 kHz itself.
    static func melFrameCount(inConfig data: Data, sampleCount: Int, claimedRate: Int) throws -> Int {
        let config = try JSONDecoder().decode(LFM2AudioConfig.self, from: data)
        let processor = LFM2AudioProcessor(config)
        let features = processor.preprocessAudio(MLXArray([Float](repeating: 0.01, count: sampleCount)), sampleRate: claimedRate)
        try MLXErrorGuard.evaluate(features)
        return features.dim(0)
    }

    func generate(_ prompt: LFMPrompt, mode: LFMGenerationMode, sampling: LFMSampling) async throws -> LFMGenerated {
        defer { Memory.clearCache() }
        let chat = ChatState(processor: processor)
        for turn in prompt.turns {
            chat.newTurn(role: turn.role)
            for part in turn.parts {
                switch part {
                case .text(let text):
                    chat.addText(text)
                case .audio(let samples):
                    let array = MLXArray(samples)
                    try MLXErrorGuard.evaluate(array)
                    chat.addAudio(array, sampleRate: LFMAudioEngine.inputSampleRate)
                }
            }
            chat.endTurn()
        }
        chat.newTurn(role: "assistant")
        if prompt.assistantStartsAudio { chat.addAudioStartToken() }

        let config = LFMGenerationConfig(maxNewTokens: sampling.maxNewTokens, temperature: sampling.temperature, topK: sampling.topK,
                                         topP: 1.0, audioTemperature: sampling.audioTemperature, audioTopK: sampling.audioTopK)
        let textTokens = chat.getTextTokens(), features = chat.getAudioFeatures(), modalities = chat.getModalities()
        // The library runs its whole loop inside this call (the stream's build closure is synchronous); iterating just drains it.
        let stream: AsyncThrowingStream<(MLXArray, LFMModality), Error>
        switch mode {
        case .interleaved:
            stream = model.generateInterleaved(textTokens: textTokens, audioFeatures: features, modalities: modalities, config: config)
        case .sequential:
            stream = model.generateSequential(textTokens: textTokens, audioFeatures: features, modalities: modalities, config: config)
        }

        var ids: [Int] = []
        var frames: [[Int32]] = []
        for try await (token, modality) in stream {
            try Task.checkCancellation()
            switch modality {
            case .text:
                ids.append(Int(try MLXErrorGuard.item(token.asType(.int32), as: Int32.self)))
            case .audioOut:
                let frame = try MLXErrorGuard.read(token.asType(.int32), as: Int32.self)
                // The end-of-audio frame carries 2048 in every codebook; it is not speech.
                if frame.first.map({ Int($0) == lfmAudioEOSToken }) != true { frames.append(frame) }
            case .audioIn:
                break
            }
        }
        let words = ids.filter { !Self.controlTokens.contains($0) }
        return LFMGenerated(text: words.isEmpty ? "" : processor.decodeText(words), audioFrames: frames)
    }

    func decode(frames: [[Int32]]) async throws -> [Float] {
        defer { Memory.clearCache() }
        guard let width = frames.first?.count, width > 0 else { return [] }
        let detokenizer = try loadedDetokenizer()
        let flat = frames.flatMap { $0 }
        // (frames, codebooks) -> (1, codebooks, frames), the layout the library's README feeds the detokenizer.
        let codes = MLXArray(flat, [frames.count, width]).transposed(1, 0).expandedDimensions(axis: 0)
        try MLXErrorGuard.evaluate(codes)
        let waveform = detokenizer(codes)
        return try MLXErrorGuard.read(waveform.asType(.float32), as: Float.self)
    }

    private func loadedDetokenizer() throws -> LFM2AudioDetokenizer {
        if let detokenizer { return detokenizer }
        let loaded = try LFM2AudioDetokenizer.fromPretrained(modelPath: directory)   // local files only, despite the name
        detokenizer = loaded
        return loaded
    }
}
