import Foundation
import MLX
import Tokenizers

public class PaddleOCRVLGenerator {
    let model: PaddleOCRVLModel
    let tokenizer: any Tokenizer
    let config: PaddleOCRVLConfig

    private let eosTokenId: Int
    private let visionTokenId: Int
    private let visionStartTokenId: Int
    private let visionEndTokenId: Int
    private let stopTokenIds: Set<Int>

    public init(model: PaddleOCRVLModel, tokenizer: any Tokenizer, config: PaddleOCRVLConfig) {
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.visionTokenId = config.visionTokenId
        self.visionStartTokenId = config.visionStartTokenId
        self.visionEndTokenId = config.visionEndTokenId
        self.eosTokenId = tokenizer.eosTokenId ?? config.eosTokenId
        // Stop on both the tokenizer's and the config's eos id (they can differ; the model
        // emits the config one, e.g. </s>=2, which the loop must catch to avoid runaway output).
        self.stopTokenIds = [eosTokenId, config.eosTokenId]
    }

    public func buildPrompt(task: PaddleOCRTask = .ocr) -> String {
        task.prompt
    }

    public func generate(
        processedImages: ProcessedImages,
        task: PaddleOCRTask = .ocr,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) -> GenerationResult {
        let numImageTokens = processedImages.numImageTokens

        // Build the prompt in the model's chat format (chat_template.jinja):
        //   <|begin_of_sentence|>User: <|IMAGE_START|>{image}<|IMAGE_END|>{task}\nAssistant:\n
        // The User:/Assistant: turn structure is what teaches the model to end its answer with
        // </s>; without it the model reads the image but never stops.
        let prefixText = "<|begin_of_sentence|>User: <|IMAGE_START|>"
        let suffixText = "<|IMAGE_END|>" + buildPrompt(task: task) + "\nAssistant:\n"
        let prefixIds = tokenizer.encode(text: prefixText, addSpecialTokens: false)
        let suffixIds = tokenizer.encode(text: suffixText, addSpecialTokens: false)

        var inputIds: [Int] = prefixIds
        let imageStart = inputIds.count
        inputIds.append(contentsOf: Array(repeating: visionTokenId, count: numImageTokens))
        inputIds.append(contentsOf: suffixIds)

        var inputIdArray = MLXArray(inputIds.map { Int32($0) }).reshaped(1, -1)

        // 3D mRoPE position ids: image tokens get 2D (row, col) positions; text is sequential.
        let patch = config.visionConfig.patchSize
        let gridMergedH = (processedImages.height / patch) / 2
        let gridMergedW = (processedImages.width / patch) / 2
        let (positionIds, genStart) = computePositionIds(
            seqLen: inputIds.count, imageStart: imageStart,
            gridMergedH: gridMergedH, gridMergedW: gridMergedW)

        let cache = model.newCache()

        var logits = model.forward(
            inputIds: inputIdArray,
            pixelValues: processedImages.pixelValues,
            cache: cache,
            positionIds: positionIds
        )

        var generatedTokens: [Int] = []
        var generatedText = ""

        for step in 0..<maxNewTokens {
            let lastLogits = logits[0, -1]

            let nextTokenId: Int
            if temperature <= 0 {
                nextTokenId = argMax(lastLogits).item(Int.self)
            } else {
                nextTokenId = sampleWithTemperature(
                    logits: lastLogits,
                    temperature: temperature,
                    topP: topP
                )
            }

            if stopTokenIds.contains(nextTokenId) {
                break
            }

            generatedTokens.append(nextTokenId)

            let decoded = tokenizer.decode(tokens: [nextTokenId])
            generatedText += decoded

            inputIdArray = MLXArray([Int32(nextTokenId)]).reshaped(1, 1)
            let p = Int32(genStart + step)
            let stepPos = MLXArray([p, p, p]).reshaped(3, 1)
            logits = model.forwardGeneration(inputIds: inputIdArray, cache: cache, positionIds: stepPos)
        }

        return GenerationResult(
            tokens: generatedTokens,
            text: generatedText,
            tokenCount: generatedTokens.count
        )
    }

    public func generate(
        pixelValues: MLXArray,
        task: PaddleOCRTask = .ocr,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) -> GenerationResult {
        let textPrompt = buildPrompt(task: task)
        let textIds = tokenizer.encode(text: textPrompt, addSpecialTokens: false)

        let imageHeight = pixelValues.dim(1)
        let imageWidth = pixelValues.dim(2)
        let patchSize = config.visionConfig.patchSize
        // 2×2 projector merge → quarter the token count.
        let numImageTokens = (imageHeight / patchSize / 2) * (imageWidth / patchSize / 2)

        var inputIds: [Int] = []

        if let bosId = tokenizer.bosTokenId {
            inputIds.append(bosId)
        }

        inputIds.append(visionStartTokenId)
        inputIds.append(contentsOf: Array(repeating: visionTokenId, count: numImageTokens))
        inputIds.append(visionEndTokenId)
        inputIds.append(contentsOf: textIds)

        var inputIdArray = MLXArray(inputIds.map { Int32($0) }).reshaped(1, -1)

        let cache = model.newCache()

        var logits = model.forward(
            inputIds: inputIdArray,
            pixelValues: pixelValues,
            cache: cache
        )

        var generatedTokens: [Int] = []
        var generatedText = ""

        for _ in 0..<maxNewTokens {
            let lastLogits = logits[0, -1]

            let nextTokenId: Int
            if temperature <= 0 {
                nextTokenId = argMax(lastLogits).item(Int.self)
            } else {
                nextTokenId = sampleWithTemperature(
                    logits: lastLogits,
                    temperature: temperature,
                    topP: topP
                )
            }

            if stopTokenIds.contains(nextTokenId) {
                break
            }

            generatedTokens.append(nextTokenId)

            let decoded = tokenizer.decode(tokens: [nextTokenId])
            generatedText += decoded

            inputIdArray = MLXArray([Int32(nextTokenId)]).reshaped(1, 1)
            logits = model.forwardGeneration(inputIds: inputIdArray, cache: cache)
        }

        return GenerationResult(
            tokens: generatedTokens,
            text: generatedText,
            tokenCount: generatedTokens.count
        )
    }

    /// Build Qwen2-VL-style 3D mRoPE position ids `[3, seqLen]` (temporal, height, width). Text
    /// tokens are sequential on all three axes; the contiguous image block at `imageStart` gets a
    /// shared temporal position and 2D (row, col) height/width. Returns the ids and the next
    /// position to start generated text from.
    private func computePositionIds(
        seqLen: Int, imageStart: Int, gridMergedH: Int, gridMergedW: Int
    ) -> (positionIds: MLXArray, nextPosition: Int) {
        var t = [Int32](), h = [Int32](), w = [Int32]()
        t.reserveCapacity(seqLen); h.reserveCapacity(seqLen); w.reserveCapacity(seqLen)
        let numImage = gridMergedH * gridMergedW
        var cur = 0
        var idx = 0
        while idx < seqLen {
            if idx == imageStart && numImage > 0 {
                let st = cur
                for r in 0 ..< gridMergedH {
                    for c in 0 ..< gridMergedW {
                        t.append(Int32(st)); h.append(Int32(st + r)); w.append(Int32(st + c))
                    }
                }
                cur = st + max(gridMergedH, gridMergedW)
                idx += numImage
            } else {
                t.append(Int32(cur)); h.append(Int32(cur)); w.append(Int32(cur))
                cur += 1
                idx += 1
            }
        }
        let arr = MLXArray(t + h + w).reshaped(3, seqLen)
        return (arr, cur)
    }

    private func sampleWithTemperature(logits: MLXArray, temperature: Float, topP: Float) -> Int {
        let scaledLogits = logits / temperature

        if topP < 1.0 {
            let probs = softmax(scaledLogits, axis: -1)
            let sortedIndices = argSort(probs, axis: -1)
            let sortedProbs = take(probs, sortedIndices, axis: -1).squeezed(axis: 0)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            let topProbs = MLX.where(
                cumulativeProbs .> (1 - topP), sortedProbs, zeros(like: sortedProbs))

            let sortedToken = categorical(log(topProbs + 1e-10))
            let token = sortedIndices.squeezed(axis: 0)[sortedToken]
            return token.item(Int.self)
        } else {
            let sample = categorical(scaledLogits)
            return sample.item(Int.self)
        }
    }

    public func generateBatch(
        batchImages: BatchProcessedImages,
        task: PaddleOCRTask = .ocr,
        maxNewTokens: Int = 1024,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) -> BatchGenerationResult {
        let results = batchImages.items.map { processedImages in
            generate(
                processedImages: processedImages,
                task: task,
                maxNewTokens: maxNewTokens,
                temperature: temperature,
                topP: topP
            )
        }
        return BatchGenerationResult(results: results)
    }
}

public struct GenerationResult: Sendable {
    public let tokens: [Int]
    public let text: String
    public let tokenCount: Int
}

public struct BatchGenerationResult: Sendable {
    public let results: [GenerationResult]

    public var texts: [String] {
        results.map { $0.text }
    }

    public var batchSize: Int {
        results.count
    }
}
