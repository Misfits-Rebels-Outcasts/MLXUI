import Foundation

/// MT1-1 — the v1 Model Tools table (`RSI/plan-chat-model-tools-2026-11.md` §3.4, as corrected by the MT1-0 rulings, journal
/// `2026-693`). **Data only, plus pure lookups.** A router never sees a model id: nothing in a name, description or parameter
/// description may name a model (backlog implementer rule 3).
///
/// Each row maps a tool to a served `TaskCatalog` task. Where a parameter lands on the one-row flow is explicit
/// (`ParamPlacement`), because MT1-0 showed the keys are not the plan's names: `language` is the row key `lang=`, a Describe
/// Image `question` is the row's first bare quoted token (not a `key=value`), and the text a tool speaks, translates or draws is
/// the **input asset**, never a row parameter.
nonisolated struct ModelToolSpec: Sendable, Equatable {

    /// What an argument may be. Ranges come from `RealExecutor.stageConfig` and the curated manifests.
    enum ParamType: Sendable, Equatable {
        case string
        /// A whole number, optionally bounded (inclusive).
        case integer(min: Int?, max: Int?)
    }

    /// Where an argument goes on the one-row flow.
    enum ParamPlacement: Sendable, Equatable {
        /// A `key=value` setting on the row (`rowKey` is the executor's key, which may differ from the tool's name).
        case setting(rowKey: String)
        /// The row's first bare quoted token (Describe Image's question).
        case rowPrompt
        /// Not on the row: consumed by the tool itself (`read_asset`'s paging).
        case tool
    }

    struct Param: Sendable, Equatable {
        let name: String
        let type: ParamType
        let required: Bool
        let placement: ParamPlacement
        /// One line, no model names.
        let description: String
    }

    /// The name the router sees.
    let name: String
    /// The `TaskCatalog` task the row runs; `nil` for the two tools that run no model.
    let task: String?
    /// The kind of the input asset (`nil` = the tool takes no asset).
    let input: Kind?
    /// The kind of the result.
    let output: Kind
    let params: [Param]
    /// Narrows the task's derived model pool to these runner kinds (translate: the dedicated model only).
    let kinds: Set<RunnerKind>?
    /// <= 140 characters.
    let description: String

    static let maxDescriptionLength = 140

    // MARK: - Lookups (pure)

    func param(named name: String) -> Param? { params.first { $0.name == name } }
    var requiredParams: [Param] { params.filter(\.required) }
    var paramNames: [String] { params.map(\.name) }

    /// Argument names the spec does not declare, in the order given.
    func unknownArguments(in names: [String]) -> [String] {
        let known = Set(paramNames)
        return names.filter { !known.contains($0) }
    }

    /// Required parameters absent from `names`.
    func missingRequired(in names: [String]) -> [Param] {
        let given = Set(names)
        return requiredParams.filter { !given.contains($0.name) }
    }

    static func spec(named name: String) -> ModelToolSpec? { all.first { $0.name == name } }
    static func specs(forTask task: String) -> [ModelToolSpec] { all.filter { $0.task == task } }
    /// The tools that run a model row.
    static var modelTools: [ModelToolSpec] { all.filter { $0.task != nil } }

    // MARK: - The table

    private static let seedType = ParamType.integer(min: 0, max: 4_294_967_295)   // manifest range; `RealExecutor` parses UInt64

    static let all: [ModelToolSpec] = [
        ModelToolSpec(
            name: "transcribe_audio", task: "Transcribe", input: .audio, output: .text,
            params: [Param(name: "language", type: .string, required: false, placement: .setting(rowKey: "lang"),
                           description: "Spoken language, e.g. ms or Malay. Omit for English.")],
            kinds: nil,
            description: "Speech to text. Set language to the spoken language unless it is English."),
        ModelToolSpec(
            name: "ocr_image", task: "OCR", input: .image, output: .text, params: [], kinds: nil,
            description: "Read the text in an image."),
        ModelToolSpec(
            name: "describe_image", task: "Describe Image", input: .image, output: .text,
            params: [Param(name: "question", type: .string, required: false, placement: .rowPrompt,
                           description: "What to ask about the image. Omit for a general description.")],
            kinds: nil,
            description: "Describe an image or answer a question about it."),
        ModelToolSpec(
            name: "speak", task: "Speak", input: .text, output: .audio,
            params: [Param(name: "voice", type: .string, required: false, placement: .setting(rowKey: "voice"),
                           description: "Optional voice name. Omit it for the default voice; never send \"default\".")],
            kinds: nil,
            description: "Turn text into spoken audio."),
        ModelToolSpec(
            name: "translate", task: "Translate", input: .text, output: .text,
            params: [Param(name: "to", type: .string, required: true, placement: .setting(rowKey: "to"),
                           description: "Target language name or code, e.g. Malay or ms.")],
            kinds: [.translate],
            description: "Translate text into another language. The source language is detected."),
        ModelToolSpec(
            name: "generate_image", task: "Generate Image", input: .text, output: .image,
            params: [
                Param(name: "width", type: .integer(min: 1, max: nil), required: false, placement: .setting(rowKey: "width"),
                      description: "Width in pixels."),
                Param(name: "height", type: .integer(min: 1, max: nil), required: false, placement: .setting(rowKey: "height"),
                      description: "Height in pixels."),
                Param(name: "steps", type: .integer(min: 1, max: 50), required: false, placement: .setting(rowKey: "steps"),
                      description: "Number of denoising steps, 1 to 50."),
                Param(name: "seed", type: seedType, required: false, placement: .setting(rowKey: "seed"),
                      description: "Random seed, for a repeatable picture."),
            ],
            kinds: nil,
            description: "Draw a picture from a text prompt."),
        ModelToolSpec(
            name: "upscale_image", task: "Upscale", input: .image, output: .image, params: [], kinds: nil,
            description: "Make an image larger and sharper."),
        ModelToolSpec(
            name: "segment_image", task: "Segment", input: .image, output: .image, params: [], kinds: nil,
            description: "Cut out the main subject of an image as a mask."),
        ModelToolSpec(
            name: "estimate_depth", task: "Estimate Depth", input: .image, output: .image, params: [], kinds: nil,
            description: "Make a grayscale depth map of an image: near is bright, far is dark."),
        ModelToolSpec(
            name: "generate_music", task: "Generate Sound", input: .text, output: .audio, params: [], kinds: nil,
            description: "Make a short (about 30-second) instrumental clip from a text prompt."),
        ModelToolSpec(
            name: "list_capabilities", task: nil, input: nil, output: .text, params: [], kinds: nil,
            description: "List what you can do with the user's models, and whether each is ready."),
        ModelToolSpec(
            name: "read_asset", task: nil, input: .text, output: .text,
            params: [
                Param(name: "asset", type: .string, required: true, placement: .tool, description: "Asset handle, e.g. a3."),
                Param(name: "start", type: .integer(min: 0, max: nil), required: false, placement: .tool,
                      description: "First character to read. Default 0."),
                Param(name: "length", type: .integer(min: 1, max: nil), required: false, placement: .tool,
                      description: "How many characters to read."),
            ],
            kinds: nil,
            description: "Read part of a long text asset, a page at a time."),
    ]
}
