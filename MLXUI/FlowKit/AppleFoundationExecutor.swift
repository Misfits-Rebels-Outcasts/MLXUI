import Foundation
import FoundationModels

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
    /// Test-only: when set, `currentReadiness()` returns `nil` immediately, simulating
    /// "macOS < 26" on a machine that's actually running 26+ (`CatFlowAppleFoundationTests`).
    /// Production code never sets this; always `false` outside a test run.
    static var simulateOSUnavailable = false
    /// Test-only: when set, both factories return it instead of touching `#available`/the
    /// real framework — lets a test drive every `Readiness` state and every generation path
    /// without a real device. Production code never sets this.
    static var checkerOverride: (any AppleFoundationAvailabilityChecking)?
    static var executorOverride: (any AppleFoundationExecuting)?

    /// The current machine's Apple Intelligence state, mapped to `Readiness`, or `nil` below
    /// macOS 26 — a slot simply **absent** from the pool (AFM-1), not an error.
    static func currentReadiness() -> Readiness? {
        if simulateOSUnavailable { return nil }
        if let checkerOverride { return checkerOverride.readiness }
        guard #available(macOS 26, *) else { return nil }
        return SystemLanguageModelAvailabilityChecker().readiness
    }

    /// The real executor, or `nil` below macOS 26 (or when Apple Intelligence genuinely
    /// isn't reachable — `RealExecutor` treats a `nil` here as a stage failure naming the
    /// row, never a crash).
    static func makeExecutor() -> (any AppleFoundationExecuting)? {
        if simulateOSUnavailable { return nil }
        if let executorOverride { return executorOverride }
        guard #available(macOS 26, *) else { return nil }
        return AppleFoundationModelExecutor()
    }
}
