import Foundation
import FoundationModels
import enum MLXLMCommon.JSONValue   // the tool-argument type the app's tools already take (only this symbol: MLXLMCommon has its own `Tool`)

/// AFM-2 — the seam between `RealExecutor`'s Apple Foundation Models dispatch (availability-
/// agnostic) and the real `FoundationModels` framework (macOS 26+). **This is the one file in
/// the app that knows about the macOS 26 floor** — every `@available(macOS 26, *)` for Apple
/// Foundation Models lives here, per the owner's ask, rather than `if #available` scattered
/// through views or `RealExecutor`. Verified against the installed macOS 26.5 SDK
/// (`xcrun swiftc -typecheck`, journal `2026-258`) — every signature below type-checks against
/// the real framework, not a guess from documentation that may have moved on.
///
/// Two protocols, neither macOS-26-tagged, are the whole surface `RealExecutor` touches:
/// - `AppleFoundationExecuting` — prompt in, text out; a constrained variant for deciders.
/// - `AppleFoundationAvailabilityChecking` — the current readiness, one property.
/// `AppleFoundationAvailability` is the factory: it does the `#available` check and hands
/// back an existential, or `nil` below macOS 26 — a slot simply absent from the pool, never
/// an error and never a greyed row (AFM-1). Two test-only override points let
/// `CatFlowAppleFoundationTests` exercise every branch without a real device or an old OS.

/// One method to generate free text, one to generate a tag constrained to a declared set.
/// `Sendable`, no availability tag — `RealExecutor` holds this as `any AppleFoundationExecuting`
/// with no idea it's macOS-26-gated underneath.
nonisolated protocol AppleFoundationExecuting: Sendable {
    /// Plain-text generation for the frame-backed tasks (Summarize, Rewrite, Translate, …).
    /// `instructions` is the session's system prompt (empty when the frame's own rendered
    /// text already carries everything, which is every case today — RealExecutor's frame
    /// rendering matches the MLX path's).
    func generate(instructions: String, prompt: String) async throws -> String

    /// Constrained generation for the deciders (Classify, Gate, Score, Judge, Decide): the
    /// model returns exactly one of `tags`, via guided generation — never Spec §12.2's
    /// strict-parse-plus-one-retry path (`RealExecutor.fireTag`), and never **F010** (AFM-2:
    /// "does not fall to the parse-and-retry path").
    func generateTag(instructions: String, prompt: String, tags: [String]) async throws -> String
}

/// Just the readiness question, so `AppleFoundationAvailability` can be tested without a real
/// device: a fixture conformer returns any `Readiness` value; only the real one touches
/// `SystemLanguageModel`.
nonisolated protocol AppleFoundationAvailabilityChecking: Sendable {
    var readiness: Readiness { get }
}

/// S1-4 — **streaming** chat for the Local Server. Same rule as the two protocols above: nothing
/// here carries `@available`, so `Server/Backends/AFMBackend.swift` never touches the macOS 26
/// floor. Verified against the installed SDK (journal `2026-369`): `LanguageModelSession`,
/// `streamResponse(to:options:)` and `GenerationOptions(temperature:maximumResponseTokens:)` all
/// type-check for macOS 26.0.
///
/// **Multi-turn history is NOT passed as a `Transcript`.** `Transcript(entries:)` /
/// `LanguageModelSession(model:transcript:)` type-check, but on a real Mac (macOS 27, Apple
/// Intelligence on) a session built from *constructed* entries ignores the assistant turns in it —
/// "My name is Sam." / "Nice to meet you, Sam." / "What is my name?" is answered "I'm a foundation
/// model developed by Apple" — while a copy of a *live* session's own transcript is honoured. The
/// model only trusts responses it generated. So the backend folds earlier turns into the prompt text
/// instead (journal `2026-369`, finding 1).
nonisolated enum AppleFoundationChatEvent: Sendable, Equatable {
    /// The **whole reply so far** — Apple Foundation Models streams snapshots, not deltas (readme
    /// T7). The consumer diffs consecutive snapshots.
    case snapshot(String)
    /// Token counts, when the OS can report them (`SystemLanguageModel.tokenCount`, macOS 26.4+).
    case usage(promptTokens: Int, completionTokens: Int)
}

/// The failures the server maps to HTTP statuses, lifted out of `FoundationModels`' own error
/// types (which differ by OS: `LanguageModelSession.GenerationError` on 26.x, `LanguageModelError`
/// on 27) so the backend never sees either.
nonisolated enum AppleFoundationChatError: Error, Sendable, Equatable {
    case contextWindowExceeded(limit: Int?, tokenCount: Int?)
    case guardrailViolation
    case unsupportedLanguage
    case failed
}

nonisolated protocol AppleFoundationChatStreaming: Sendable {
    /// `instructions` = the session's system prompt; `prompt` = the user turn (earlier turns already
    /// folded in by the caller). Unset sampling params are the model's own.
    func streamChat(instructions: String, prompt: String,
                    temperature: Double?, maximumResponseTokens: Int?) -> AsyncThrowingStream<AppleFoundationChatEvent, Error>

    /// R23 (S2-6): the tokens `instructions` + `prompt` will occupy, so an over-limit request can be
    /// a clean `400` **before** the 200 head instead of an in-stream error. `nil` — the default —
    /// means "this streamer can't count", which is the honest answer below macOS 26.4 where
    /// `SystemLanguageModel.tokenCount(for:)` doesn't exist. **Never an estimate** (rule: the
    /// ~4 chars/token figure is for *usage*, not for refusing a request).
    func promptTokenCount(instructions: String, prompt: String) async -> Int?

    /// The model's context window (`SystemLanguageModel.contextSize`), or `nil` when unknown.
    var contextSize: Int? { get }
}

extension AppleFoundationChatStreaming {
    /// Streamers that can't count (fixtures, and any SDK without `tokenCount(for:)`).
    func promptTokenCount(instructions: String, prompt: String) async -> Int? { nil }
    var contextSize: Int? { nil }
}

// MARK: - MT3-2: tool-calling chat seam (untagged)

/// MT3-2 — one tool, as the router describes it to Apple Foundation Models. Plain data (no
/// `FoundationModels` types, no availability tag): the router builds these from `AgentTool`s and the
/// real seam turns them into runtime-built `GenerationSchema`s (MT3-0: `GenerationSchema(type:
/// GeneratedContent.self, properties:)`, `Arguments == GeneratedContent`).
nonisolated struct AppleFoundationToolSpec: Sendable, Equatable {
    enum Kind: Sendable, Equatable { case string, integer, number, boolean }
    struct Property: Sendable, Equatable {
        let name: String
        let description: String
        let kind: Kind
        let required: Bool
    }
    let name: String
    let description: String
    let properties: [Property]
}

/// Runs one tool call for the session. Called from inside `Tool.call`; **must not throw** (a model-caused
/// error is a corrective string, MT3-0: a throwing tool surfaces from `respond` as `ToolCallError` and ends the turn).
typealias AppleFoundationToolHandler = @Sendable (_ toolName: String, _ arguments: [String: JSONValue]) async -> String

/// A session that may be reused across turns (`LanguageModelSession` keeps its transcript). One
/// `respond` at a time: a second while one is running is a programmer error in the SDK, so the router serialises.
nonisolated protocol AppleFoundationToolSession: AnyObject, Sendable {
    /// Snapshots (whole reply so far) and a trailing `.usage`. Failures are `AppleFoundationToolError`.
    func respond(prompt: String) -> AsyncThrowingStream<AppleFoundationChatEvent, Error>
}

/// Typed failures for the chat router (superset of `AppleFoundationChatError`, which the Local Server's
/// exhaustive switch still uses unchanged).
nonisolated enum AppleFoundationToolError: Error, Sendable, Equatable {
    case contextWindowExceeded(limit: Int?, tokenCount: Int?)
    case guardrailViolation
    case unsupportedLanguage
    case refusal
    case rateLimited
    case timeout
    case assetsUnavailable
    case failed
}

nonisolated protocol AppleFoundationToolChatting: Sendable {
    /// `SystemLanguageModel.contextSize`, or `nil` when unknown.
    var contextSize: Int? { get }
    func makeSession(instructions: String?, tools: [AppleFoundationToolSpec],
                     handler: @escaping AppleFoundationToolHandler) -> any AppleFoundationToolSession
    /// Tokens `instructions` + `tools` occupy (`SystemLanguageModel.tokenCount(for:)`, macOS 26.4+); `nil` when it can't count.
    func toolTokenCount(instructions: String?, tools: [AppleFoundationToolSpec]) async -> Int?
}

/// The real executor. `@available(macOS 26, *)` because `LanguageModelSession` and
/// `DynamicGenerationSchema` are — confined to this one type so nothing else in the app
/// carries the annotation.
@available(macOS 26, *)
nonisolated struct AppleFoundationModelExecutor: AppleFoundationExecuting {
    func generate(instructions: String, prompt: String) async throws -> String {
        let session = LanguageModelSession(instructions: instructions)
        let response = try await session.respond(to: prompt)
        return response.content
    }

    func generateTag(instructions: String, prompt: String, tags: [String]) async throws -> String {
        let session = LanguageModelSession(instructions: instructions)
        // A runtime-determined enum-of-strings schema (the decider's declared tags aren't
        // known until the row is read) — `DynamicGenerationSchema(name:description:anyOf:)`,
        // verified against the real SDK, not `@Generable`'s compile-time macro (which needs a
        // static type the tag set can't be here).
        let root = DynamicGenerationSchema(name: "Tag", description: "one of the row's declared tags",
                                           anyOf: tags)
        let schema = try GenerationSchema(root: root, dependencies: [])
        let response = try await session.respond(to: prompt, schema: schema)
        return try response.content.value(String.self)
    }
}

@available(macOS 26, *)
extension AppleFoundationModelExecutor: AppleFoundationChatStreaming {
    func streamChat(instructions: String, prompt: String,
                    temperature: Double?, maximumResponseTokens: Int?) -> AsyncThrowingStream<AppleFoundationChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let session = instructions.isEmpty ? LanguageModelSession() : LanguageModelSession(instructions: instructions)
                    let options = GenerationOptions(temperature: temperature, maximumResponseTokens: maximumResponseTokens)
                    for try await snapshot in session.streamResponse(to: prompt, options: options) {
                        let text: String = snapshot.content           // the whole reply so far
                        continuation.yield(.snapshot(text))
                    }
                    if !Task.isCancelled, let usage = await Self.usage(of: session) { continuation.yield(usage) }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: AppleFoundationErrorMapper.map(error))
                }
            }
            // The consumer going away (client disconnect) cancels the generation.
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// R23 (S2-6): the real count, from the SDK's own tokenizer — `tokenCount(for:)`, macOS 26.4+.
    /// Instructions and prompt are counted separately because that is how the session is built
    /// (`LanguageModelSession(instructions:)` + `streamResponse(to:)`); their sum is what has to fit.
    /// Typechecked against the installed SDK (macOS 27.0) before writing this: rule 11.
    func promptTokenCount(instructions: String, prompt: String) async -> Int? {
        guard #available(macOS 26.4, *) else { return nil }
        let model = SystemLanguageModel.default
        var total = 0
        if !instructions.isEmpty {
            guard let count = try? await model.tokenCount(for: Instructions(instructions)) else { return nil }
            total += count
        }
        guard let promptCount = try? await model.tokenCount(for: prompt) else { return nil }
        return total + promptCount
    }

    /// `SystemLanguageModel.contextSize` (macOS 26.0+, back-deployed before 26.4). Measured at
    /// 8192 tokens on this Mac — journal `2026-369`, and the number the Connect sheet's fitness
    /// line already uses.
    var contextSize: Int? { SystemLanguageModel.default.contextSize }

    /// Prompt = everything before the reply; completion = the reply entry. `tokenCount(for:)` is
    /// macOS 26.4+, so older systems report no usage and the backend estimates.
    private static func usage(of session: LanguageModelSession) async -> AppleFoundationChatEvent? {
        guard #available(macOS 26.4, *) else { return nil }
        let entries = Array(session.transcript)
        guard let reply = entries.last, case .response = reply else { return nil }
        let model = SystemLanguageModel.default
        guard let prompt = try? await model.tokenCount(for: Array(entries.dropLast())),
              let completion = try? await model.tokenCount(for: [reply]) else { return nil }
        return .usage(promptTokens: prompt, completionTokens: completion)
    }
}

/// `FoundationModels`' errors → `AppleFoundationChatError`. Two generations of error type exist:
/// macOS 26.x throws `LanguageModelSession.GenerationError`; **macOS 27 throws `LanguageModelError`**
/// (observed for a context overflow on this Mac — journal `2026-369`), so both are mapped.
@available(macOS 26, *)
nonisolated enum AppleFoundationErrorMapper {
    static func map(_ error: Error) -> AppleFoundationChatError {
        if let error = error as? LanguageModelSession.GenerationError {
            switch error {
            case .exceededContextWindowSize:
                // 26.x's error carries no size; the model's own `contextSize` is the limit.
                return .contextWindowExceeded(limit: SystemLanguageModel.default.contextSize, tokenCount: nil)
            case .guardrailViolation: return .guardrailViolation
            case .unsupportedLanguageOrLocale: return .unsupportedLanguage
            default: return .failed
            }
        }
        if #available(macOS 27, *), let error = error as? LanguageModelError {
            switch error {
            case .contextSizeExceeded(let exceeded):
                return .contextWindowExceeded(limit: exceeded.contextSize, tokenCount: exceeded.tokenCount)
            case .guardrailViolation: return .guardrailViolation
            case .unsupportedLanguageOrLocale: return .unsupportedLanguage
            default: return .failed
            }
        }
        return .failed
    }
}

/// MT3-2 — the typed error map for the chat router. Same two generations as `map(_:)`: 26.x
/// `GenerationError`, 27 `LanguageModelError` / `SystemLanguageModel.Error`.
@available(macOS 26, *)
nonisolated extension AppleFoundationErrorMapper {
    static func mapDetailed(_ error: Error) -> AppleFoundationToolError {
        if let error = error as? AppleFoundationToolError { return error }
        // MT3-FIX-3: a throwing tool wraps its error; look through it.
        if let wrapped = error as? LanguageModelSession.ToolCallError { return mapDetailed(wrapped.underlyingError) }
        if let error = error as? LanguageModelSession.GenerationError {
            switch error {
            case .exceededContextWindowSize:
                return .contextWindowExceeded(limit: SystemLanguageModel.default.contextSize, tokenCount: nil)
            case .guardrailViolation: return .guardrailViolation
            case .unsupportedLanguageOrLocale: return .unsupportedLanguage
            case .refusal: return .refusal
            case .rateLimited: return .rateLimited
            case .assetsUnavailable: return .assetsUnavailable
            default: return .failed
            }
        }
        if #available(macOS 27, *) {
            if let error = error as? LanguageModelError {
                switch error {
                case .contextSizeExceeded(let exceeded):
                    return .contextWindowExceeded(limit: exceeded.contextSize, tokenCount: exceeded.tokenCount)
                case .guardrailViolation: return .guardrailViolation
                case .unsupportedLanguageOrLocale: return .unsupportedLanguage
                case .refusal: return .refusal
                case .rateLimited: return .rateLimited
                case .timeout: return .timeout
                default: return .failed
                }
            }
            if let error = error as? SystemLanguageModel.Error {
                switch error {
                case .assetsUnavailable: return .assetsUnavailable
                @unknown default: return .failed
                }
            }
        }
        // MT3-FIX-3: with tools armed, a huge prompt came back as a private `GenerativeFunctionsFoundation.GenerativeError`
        // ("Provided 570,100 tokens, but the maximum allowed is 8,192."), a type this file cannot name. Read its sentence.
        if let limit = overflowLimit(inDescription: String(describing: error)) {
            return .contextWindowExceeded(limit: limit, tokenCount: nil)
        }
        return .failed
    }

    /// The window size in an "…tokens, but the maximum allowed is N…" sentence, else `nil`.
    static func overflowLimit(inDescription text: String) -> Int? {
        guard let range = text.range(of: "tokens, but the maximum allowed is ") else { return nil }
        let digits = text[range.upperBound...].prefix { $0.isNumber || $0 == "," }.replacingOccurrences(of: ",", with: "")
        return Int(digits)
    }
}

/// MT3-2 — an `AgentTool`-shaped FoundationModels `Tool` whose schema is built at runtime (MT3-0 item 1).
@available(macOS 26, *)
nonisolated struct AppleFoundationRuntimeTool: Tool {
    typealias Arguments = GeneratedContent
    typealias Output = String

    let name: String
    let description: String
    let parameters: GenerationSchema
    let handler: AppleFoundationToolHandler

    init(spec: AppleFoundationToolSpec, handler: @escaping AppleFoundationToolHandler) {
        var props: [GenerationSchema.Property] = []
        for p in spec.properties {
            switch p.kind {
            case .string:
                props.append(p.required
                    ? GenerationSchema.Property(name: p.name, description: p.description, type: String.self)
                    : GenerationSchema.Property(name: p.name, description: p.description, type: String?.self))
            case .integer:
                props.append(p.required
                    ? GenerationSchema.Property(name: p.name, description: p.description, type: Int.self)
                    : GenerationSchema.Property(name: p.name, description: p.description, type: Int?.self))
            case .number:
                props.append(p.required
                    ? GenerationSchema.Property(name: p.name, description: p.description, type: Double.self)
                    : GenerationSchema.Property(name: p.name, description: p.description, type: Double?.self))
            case .boolean:
                props.append(p.required
                    ? GenerationSchema.Property(name: p.name, description: p.description, type: Bool.self)
                    : GenerationSchema.Property(name: p.name, description: p.description, type: Bool?.self))
            }
        }
        self.name = spec.name
        self.description = spec.description
        self.parameters = GenerationSchema(type: GeneratedContent.self, description: spec.description, properties: props)
        self.handler = handler
    }

    func call(arguments: GeneratedContent) async throws -> String {
        let decoded = try? JSONDecoder().decode([String: JSONValue].self, from: Data(arguments.jsonString.utf8))
        return await handler(name, decoded ?? [:])
    }
}

@available(macOS 26, *)
nonisolated final class AppleFoundationToolSessionImpl: AppleFoundationToolSession, @unchecked Sendable {
    private let session: LanguageModelSession

    init(instructions: String?, tools: [AppleFoundationToolSpec], handler: @escaping AppleFoundationToolHandler) {
        let runtimeTools: [any Tool] = tools.map { AppleFoundationRuntimeTool(spec: $0, handler: handler) }
        if let instructions, !instructions.isEmpty {
            session = LanguageModelSession(tools: runtimeTools, instructions: instructions)
        } else {
            session = LanguageModelSession(tools: runtimeTools)
        }
    }

    func respond(prompt: String) -> AsyncThrowingStream<AppleFoundationChatEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // MT3-FIX-3: a prompt far past the window can come back from the framework as `guardrailViolation`
                    // ("May contain unsafe content", seen at ~450K tokens) instead of `contextSizeExceeded`. Count first.
                    if let over = await Self.overflow(of: prompt) { throw over }
                    for try await snapshot in session.streamResponse(to: prompt) {
                        let text: String = snapshot.content
                        continuation.yield(.snapshot(text))
                    }
                    if !Task.isCancelled, let usage = await usage() { continuation.yield(usage) }
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: AppleFoundationErrorMapper.mapDetailed(error))
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// `.contextWindowExceeded` when the prompt alone cannot fit the window: the model's own count where it has one
    /// (macOS 26.4+), otherwise a byte estimate no real text beats (16 bytes per token).
    static func overflow(of prompt: String) async -> AppleFoundationToolError? {
        let limit = SystemLanguageModel.default.contextSize
        if #available(macOS 26.4, *), let count = try? await SystemLanguageModel.default.tokenCount(for: prompt) {
            return count > limit ? .contextWindowExceeded(limit: limit, tokenCount: count) : nil
        }
        return prompt.utf8.count > limit * 16 ? .contextWindowExceeded(limit: limit, tokenCount: nil) : nil
    }

    /// Context in use after the turn (everything before the reply) and the reply itself (macOS 26.4+).
    private func usage() async -> AppleFoundationChatEvent? {
        guard #available(macOS 26.4, *) else { return nil }
        let entries = Array(session.transcript)
        guard let reply = entries.last, case .response = reply else { return nil }
        let model = SystemLanguageModel.default
        guard let prompt = try? await model.tokenCount(for: Array(entries.dropLast())),
              let completion = try? await model.tokenCount(for: [reply]) else { return nil }
        return .usage(promptTokens: prompt, completionTokens: completion)
    }
}

@available(macOS 26, *)
nonisolated struct AppleFoundationToolChatter: AppleFoundationToolChatting {
    var contextSize: Int? { SystemLanguageModel.default.contextSize }

    func makeSession(instructions: String?, tools: [AppleFoundationToolSpec],
                     handler: @escaping AppleFoundationToolHandler) -> any AppleFoundationToolSession {
        AppleFoundationToolSessionImpl(instructions: instructions, tools: tools, handler: handler)
    }

    func toolTokenCount(instructions: String?, tools: [AppleFoundationToolSpec]) async -> Int? {
        guard #available(macOS 26.4, *) else { return nil }
        let model = SystemLanguageModel.default
        let runtimeTools: [any Tool] = tools.map { AppleFoundationRuntimeTool(spec: $0, handler: { _, _ in "" }) }
        var total = 0
        if let instructions, !instructions.isEmpty {
            guard let n = try? await model.tokenCount(for: Instructions(instructions)) else { return nil }
            total += n
        }
        guard let n = try? await model.tokenCount(for: runtimeTools) else { return nil }
        return total + n
    }
}

/// The real availability checker. `@available(macOS 26, *)` for `SystemLanguageModel`.
/// Maps `SystemLanguageModel.Availability` to `Readiness` per AFM-1's table, verbatim.
@available(macOS 26, *)
nonisolated struct SystemLanguageModelAvailabilityChecker: AppleFoundationAvailabilityChecking {
    var readiness: Readiness {
        switch SystemLanguageModel.default.availability {
        case .available:
            return .ready
        case .unavailable(.appleIntelligenceNotEnabled):
            return .needsSetup(reason: "Turn on Apple Intelligence in System Settings",
                               action: .enableAppleIntelligence)
        case .unavailable(.deviceNotEligible):
            return .unavailable(reason: "This Mac can't run Apple Intelligence")
        case .unavailable(.modelNotReady):
            return .needsSetup(reason: "macOS is still downloading the model", action: nil)
        @unknown default:
            return .unavailable(reason: "Apple Intelligence isn't available on this Mac")
        }
    }
}

/// The factory `TaskModels`/`RealExecutor` actually call. Neither function carries
/// `@available` — each does its own `#available` check internally, so callers never need one
/// either (the whole point of confining the annotation to this file).
nonisolated enum AppleFoundationAvailability {
    /// AFM-FOLLOWUP-1 — **"mock by default" applied to the OS.** `false` unless the real app
    /// opts in exactly once (`MLXUIApp.init()`). Without this flip, `currentReadiness()`/
    /// `makeExecutor()` return a deterministic "absent" regardless of what this build
    /// machine's actual OS version or Apple Intelligence state is — so a test run's default
    /// is never host-dependent, the same guarantee "mock by default" gives every real-model
    /// path. A test that genuinely wants the real framework sets this explicitly (none do
    /// today, matching the real-model-path convention of staying out of CI).
    static var useRealSystem = false
    /// Test-only: when set, both factories return it instead of touching the real framework —
    /// lets a test drive every `Readiness` state and every generation path without a real
    /// device, regardless of `useRealSystem`. Production code never sets this.
    static var checkerOverride: (any AppleFoundationAvailabilityChecking)?
    static var executorOverride: (any AppleFoundationExecuting)?
    /// Test-only, as above — for the Local Server's streaming chat seam (S1-4).
    static var chatStreamerOverride: (any AppleFoundationChatStreaming)?
    /// Test-only, as above — for the chat router's tool-calling seam (MT3-2).
    static var toolChatterOverride: (any AppleFoundationToolChatting)?

    /// The current machine's Apple Intelligence state, mapped to `Readiness`, or `nil` when
    /// `useRealSystem` is off or the OS is below macOS 26 — a slot simply **absent** from the
    /// pool (AFM-1), not an error.
    static func currentReadiness(forceReal: Bool = false) -> Readiness? {
        if let checkerOverride { return checkerOverride.readiness }
        guard useRealSystem || forceReal else { return nil }
        guard #available(macOS 26, *) else { return nil }
        return SystemLanguageModelAvailabilityChecker().readiness
    }

    /// The real executor, or `nil` when `useRealSystem` is off, the OS is below macOS 26, or
    /// Apple Intelligence genuinely isn't reachable (`RealExecutor` treats a `nil` here as a
    /// stage failure naming the row, never a crash).
    static func makeExecutor() -> (any AppleFoundationExecuting)? {
        if let executorOverride { return executorOverride }
        guard useRealSystem else { return nil }
        guard #available(macOS 26, *) else { return nil }
        return AppleFoundationModelExecutor()
    }

    /// The real streaming-chat seam, or `nil` on the same conditions as `makeExecutor()` (S1-4).
    static func makeChatStreamer() -> (any AppleFoundationChatStreaming)? {
        if let chatStreamerOverride { return chatStreamerOverride }
        guard useRealSystem else { return nil }
        guard #available(macOS 26, *) else { return nil }
        return AppleFoundationModelExecutor()
    }

    /// The real tool-calling chat seam (MT3-2), or `nil` on the same conditions as `makeExecutor()`.
    /// `forceReal` skips the `useRealSystem` opt-in (the owner-run real tests); it still needs the OS and
    /// still returns `nil` when the OS is below macOS 26. Readiness is the caller's question (`currentReadiness`).
    static func makeToolChatter(forceReal: Bool = false) -> (any AppleFoundationToolChatting)? {
        if let toolChatterOverride { return toolChatterOverride }
        guard useRealSystem || forceReal else { return nil }
        guard #available(macOS 26, *) else { return nil }
        return AppleFoundationToolChatter()
    }
}
