import Foundation

/// MT4-1 — which tools a chat turn offers its router (`RSI/plan-chat-model-tools-2026-11.md` §3.6), as a pure function.
///
/// Two jobs:
/// 1. **Model tools** (the v1 table, `ModelToolSpec`): rules 1-4 of §3.6 under the router's tool budget (Apple Intelligence 6,
///    MLX 10), ordered, with a reason for everything left out. Not wired into the chat until MT4-2 lands the assets; table-tested
///    here.
/// 2. **The tool-picking policy** (owner ruling, Boo, 2026-11-10: "don't offer read_clipboard, write_clipboard or fetch_url
///    unless the turn calls for them"). `turnCalls(_:prompt:)` is a conservative keyword/shape heuristic over what the user typed.
///    The same rule is extended, as the implementer's call pending owner confirmation, to `calculator`, `run_javascript` and
///    `datetime`, and (MT4-2b, owner 2026-11-11) to `embed_text`, `summarize` and `semantic_search`. Only those nine names are ever gated; file tools and every other enabled tool keep today's behaviour.
///    `gateAmbiguousTools` switches the policy on: on for Apple Intelligence, off for MLX (MT4-6's eval decides MLX).
///
/// Deterministic: the same input gives the same output, in the same order.
nonisolated enum ToolExposurePolicy {

    /// What the router can take.
    struct Router: Sendable, Equatable {
        let toolBudget: Int
        let hasVision: Bool
        static let appleIntelligence = Router(toolBudget: AppleFoundationToolBudget.maxTools, hasVision: false)
        static func mlx(hasVision: Bool) -> Router { Router(toolBudget: mlxToolBudget, hasVision: hasVision) }
    }

    /// The most tools an MLX router is given (plan §3.6).
    static let mlxToolBudget = 10

    struct Input: Sendable {
        var router: Router
        /// Ruling D: on in the Chat page, off in the bench sheet. Off: no model tool is offered.
        var modelToolsEnabled: Bool
        /// Kinds of every asset in the conversation / attached to this turn.
        var conversationAssetKinds: Set<Kind> = []
        var turnAssetKinds: Set<Kind> = []
        /// Some text asset is longer than the inline budget (rule 1: `read_asset`).
        var hasOversizedTextAsset = false
        /// What the user typed this turn.
        var prompt: String = ""
        /// Model-tool name to its status; a missing name is `.none`.
        var statuses: [String: ModelToolStatus] = [:]
        /// Tools the user declined to install this turn (ruling F).
        var declinedThisTurn: Set<String> = []
        /// The user's enabled non-model tools, in today's order.
        var enabledTools: [String] = []
        var gateAmbiguousTools = false
    }

    enum DropReason: Sendable, Equatable {
        case overBudget
        case declinedThisTurn
        /// Candidates exist but none fits this Mac; `list_capabilities` can say why.
        case cantRun(String)
        case notAvailable
        /// The router sees the image itself.
        case routerHasVision
        /// An ambiguous tool the turn does not call for.
        case notCalledFor
    }

    struct Dropped: Sendable, Equatable {
        let name: String
        let reason: DropReason
    }

    struct Output: Sendable, Equatable {
        let tools: [String]
        let dropped: [Dropped]
    }

    // MARK: - Tables

    /// Rule 2: input kind to tools, in the plan's order.
    static let toolsForInputKind: [Kind: [String]] = [
        .image: ["ocr_image", "describe_image", "upscale_image", "segment_image", "estimate_depth"],
        .audio: ["transcribe_audio"],
    ]
    /// Rule 3: generator tools, in the plan's order.
    static let generatorTools = ["speak", "translate", "generate_image", "generate_music"]

    /// The non-model tools the policy may withhold.
    static let ambiguousTools: Set<String> = ["fetch_url", "read_clipboard", "write_clipboard", "calculator", "run_javascript", "datetime",
                                              "embed_text", "summarize", "semantic_search"]

    // MARK: - The policy

    static func select(_ input: Input) -> Output {
        var tools: [String] = []
        var dropped: [Dropped] = []
        var seen = Set<String>()

        func offer(_ name: String, status: ModelToolStatus?) {
            guard seen.insert(name).inserted else { return }
            if input.declinedThisTurn.contains(name) { dropped.append(Dropped(name: name, reason: .declinedThisTurn)); return }
            if let status {
                switch status {
                case .ready, .needsDownload: break
                case .cantRun(let reason): dropped.append(Dropped(name: name, reason: .cantRun(reason))); return
                case .none: dropped.append(Dropped(name: name, reason: .notAvailable)); return
                }
            }
            tools.append(name)
        }

        /// A model tool by name, status looked up; the two model-free tools have no status.
        func modelTool(_ name: String) {
            guard let spec = ModelToolSpec.spec(named: name) else { return }
            offer(name, status: spec.task == nil ? nil : (input.statuses[name] ?? ModelToolStatus.none))
        }

        /// Ready before needsDownload, otherwise the plan's order.
        func ordered(_ names: [String]) -> [String] {
            func rank(_ n: String) -> Int {
                switch input.statuses[n] { case .ready?: return 0; case .needsDownload?: return 1; default: return 2 }
            }
            return names.enumerated().sorted { (rank($0.element), $0.offset) < (rank($1.element), $1.offset) }.map(\.element)
        }

        if input.modelToolsEnabled {
            // 1. Always.
            modelTool("list_capabilities")
            if input.hasOversizedTextAsset { modelTool("read_asset") }
            // 2. Input kind of what was attached this turn. Audio: its tool. Image (MT4-2b): `ocr_image` by default; the other
            //    image tools only after the generators below, and only when the prompt calls for them.
            var cuedImageTools: [String] = []
            for kind in [Kind.image, .audio] where input.turnAssetKinds.contains(kind) {
                for name in ordered(toolsForInputKind[kind] ?? []) {
                    if kind == .image, name != "ocr_image" {
                        if name == "describe_image", input.router.hasVision { dropped.append(Dropped(name: name, reason: .routerHasVision)); seen.insert(name); continue }
                        if imageToolCalled(name, prompt: input.prompt) { cuedImageTools.append(name) }
                        continue
                    }
                    modelTool(name)
                }
            }
            // 3. Generators, only when the prompt calls for them (ahead of the cued image tools).
            for name in ordered(generatorTools) where generatorCalled(name, prompt: input.prompt) { modelTool(name) }
            for name in cuedImageTools { modelTool(name) }
            // 4. Kinds already in the conversation.
            //    A kind attached this turn is already handled by rule 2 (the conversation's assets include this turn's), so only
            //    earlier-turn kinds land here; otherwise the five image tools would return on the turn the image arrives.
            for kind in [Kind.image, .audio] where input.conversationAssetKinds.contains(kind) && !input.turnAssetKinds.contains(kind) {
                for name in ordered(toolsForInputKind[kind] ?? []) {
                    if name == "describe_image", input.router.hasVision { continue }
                    modelTool(name)
                }
            }
        }
        // 5. The user's enabled non-model tools fill what is left, in today's order.
        for name in input.enabledTools {
            if input.gateAmbiguousTools, ambiguousTools.contains(name), !turnCalls(name, prompt: input.prompt) {
                if seen.insert(name).inserted { dropped.append(Dropped(name: name, reason: .notCalledFor)) }
                continue
            }
            offer(name, status: nil)
        }

        // Budget (the legacy set, with model tools off and the gate off, is never cut: it is "today's" set).
        if input.modelToolsEnabled || input.gateAmbiguousTools, tools.count > input.router.toolBudget {
            for name in tools[input.router.toolBudget...] { dropped.append(Dropped(name: name, reason: .overBudget)) }
            tools = Array(tools.prefix(input.router.toolBudget))
        }
        return Output(tools: tools, dropped: dropped)
    }

    // MARK: - turnCalls (the tool-picking policy)

    /// Does what the user typed call for this tool? Only the nine ambiguous names are ever refused; every other name is true.
    static func turnCalls(_ tool: String, prompt: String) -> Bool {
        let text = prompt.lowercased()
        switch tool {
        case "fetch_url": return matches(text, urlCue)
        case "write_clipboard": return matches(text, writeClipboardCue)
        case "read_clipboard": return !matches(text, writeClipboardCue) && matches(text, readClipboardCue)
        case "calculator": return matches(text, calculatorCue)
        case "run_javascript": return matches(text, javascriptCue)
        case "datetime": return matches(text, dateTimeCue)
        case "embed_text": return matches(text, embedCue)
        case "summarize": return matches(text, summarizeCue)
        case "semantic_search": return matches(text, searchCue)
        default: return true
        }
    }

    /// MT4-2b: does the prompt call for this (non-OCR) image tool?
    static func imageToolCalled(_ tool: String, prompt: String) -> Bool {
        let text = prompt.lowercased()
        switch tool {
        case "describe_image": return matches(text, #"\b(describe|description|caption|what('s| is| are) (in|on|shown|happening)|what do you see|look at|tell me about (this|the) (image|picture|photo))\b"#)
        case "upscale_image": return matches(text, #"\b(upscal\w*|enlarg\w*|super.?resolution|higher resolution|sharpen|increase (the )?resolution)\b"#)
        case "segment_image": return matches(text, #"\b(segment\w*|cut ?out|mask|isolate|outline|remove (the )?background)\b"#)
        case "estimate_depth": return matches(text, #"\b(depth|how far|distance|depth map|3d)\b"#)
        default: return false
        }
    }

    static func generatorCalled(_ tool: String, prompt: String) -> Bool {
        let text = prompt.lowercased()
        switch tool {
        case "speak": return matches(text, #"\b(speak|say (it|this|that) aloud|read (it|this|that|aloud)|out loud|aloud|text.to.speech|narrat\w*|voice)\b"#)
        case "translate": return matches(text, #"\b(translat\w*)\b"#)
        case "generate_image": return matches(text, #"\b(draw|paint|sketch|illustrat\w*|render|(generate|create|make)\b.{0,20}\b(image|picture|photo|illustration)|(image|picture|photo) of)\b"#)
        case "generate_music": return matches(text, #"\b(music|song|melody|soundtrack|sound effect|jingle|beat)\b"#)
        default: return false
        }
    }

    private static let urlCue = #"(https?://|www\.|\b[a-z0-9-]+\.(com|org|net|io|dev|app|ai|edu|gov|co|uk|de|fr|jp|info|me)\b|\b(website|web page|webpage|url|link|web site)\b)"#
    private static let writeClipboardCue = #"\b(copy|put|place|save|write|set|send|add)\b.{0,80}\bclipboard\b|\bcopy (this|that|it)\b"#
    private static let readClipboardCue = #"(\bclipboard\b|\b(what|that|text) i (just )?copied\b|\bpaste what i\b)"#
    private static let calculatorCue = #"(\d\s*[+\-*/×÷^x]\s*\d|\d\s*%|\b(calculate|compute|calculator|percent\w*|sum of|square root|multipl\w*|divided?\b|plus|minus|times)\b.*\d|\d.*\b(plus|minus|times|divided by|multiplied by|percent)\b)"#
    private static let embedCue = #"\b(embed\w*|embedding\w*|vectori[sz]\w*|cosine|similarity)\b"#
    private static let summarizeCue = #"\b(summari[sz]\w*|summary|summaries|tl;?dr|recap|condense)\b"#
    private static let searchCue = #"(\b(search\w*|index\w*|find (in|across|among)|look (up )?in)\b.{0,60}\b(my |the |our )?(documents?|docs|notes?|files?|library|knowledge base|folder)\b|\bmy (documents?|notes|files)\b|\bsemantic\b|\bindex\w*\b)"#
    private static let javascriptCue = #"\b(javascript|js|script|run (this |the |some )?code|node\.?js)\b"#
    private static let dateTimeCue = #"\b(time|date|today|tonight|now|day|days|week|month|year|when|yesterday|tomorrow|o'clock|clock|weekday)\b"#

    private static func matches(_ text: String, _ pattern: String) -> Bool {
        text.range(of: pattern, options: .regularExpression) != nil
    }
}
