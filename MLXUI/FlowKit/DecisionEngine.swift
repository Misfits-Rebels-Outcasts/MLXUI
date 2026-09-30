import Foundation

/// CL-6 (`RSI/DelegateCLMBacklog.md` §CL-0 gate C, ruled (1) — one shared decider path): the
/// seam `RealExecutor.runDecisionDecider` (LY-7's `runLayaDecider`, generalised) calls after
/// picking a conformer by `modelEntry.family`. Extracted now that there are two conformers to
/// shape it against (`LayaDecisionEngine`, `CLMDecisionEngine`) — the gate's own words: "the
/// refactor happens once there are two conformers to shape it (not one guessed-at)."
///
/// Question-*building* stays inside each conformer, not here — Laya and CLM's wire shapes
/// genuinely differ (CL-5-FIX-1: CLM's `choice` criteria must be a JSON *object*, `{tag: ""}`
/// per declared tag; Laya's stays a bare array). What's shared is everything `DecisionAsk`
/// carries (the plain, engine-agnostic question: type, tag-clause-stripped instructions, and
/// the declared tags in order) and the call itself — `runDecisionDecider` builds one
/// `DecisionAsk` and every step around it (refusals, tag firing, the pass-through payload,
/// the run-log detail, the truncation flag) unchanged, regardless of which engine answers.
nonisolated struct DecisionAsk: Sendable {
    let type: LayaQuestionType
    /// The criterion with any `tags: a, b, c` clause already stripped
    /// (`RealExecutor.layaInstructions`) — computed once, shared by both engines.
    let instructions: String
    /// `RealExecutor.declaredTags(row)`, in declared order.
    let tags: [String]
}

nonisolated protocol DecisionEngine: Sendable {
    /// `directory` is the installed model's root (`ModelStore.shared.directory(forModelID:)`).
    /// `state` is the gathered input text (`DeciderFrame.flatTexts`). Throws `LayaPromptError`
    /// on a validation failure — `runDecisionDecider` wraps it as `FlowError.stageFailure`
    /// with the row's own path, the same sentence LY-7 already produced for Laya.
    func decide(directory: URL, state: String, ask: DecisionAsk) async throws -> LayaAnswer
}

/// LY-7's body, moved as-is (gate C, option 1): builds Laya's own `LayaQuestion` — criteria as
/// a bare label array (`dict.fromkeys`-shaped, matching `LayaRunView`'s own Choice criteria)
/// — then calls the injected `askLaya` closure. `LayaQuestionDefinition.resolve()` is where
/// Laya's own "too many options for the token budget" validation lives
/// (`LayaPromptError.tooManyOptions`) — CLM has no equivalent ceiling, so that check must stay
/// here, not move to the shared `DecisionAsk` step.
nonisolated struct LayaDecisionEngine: DecisionEngine {
    let askLaya: @Sendable (URL, String, [LayaQuestion]) async throws -> [LayaAnswer]

    func decide(directory: URL, state: String, ask: DecisionAsk) async throws -> LayaAnswer {
        let criteria = LayaJSON.array(ask.tags.map { .string($0) })
        let question = try LayaQuestionDefinition(
            type: ask.type, instructions: .string(ask.instructions), criteria: criteria
        ).resolve()
        guard let first = try await askLaya(directory, state, [question]).first else {
            throw LayaPromptError.invalidCriteria("Laya produced no answer")
        }
        return first
    }
}

/// CL-6 — builds CLM's own question the way CL-5-FIX-1 learned the hard way (`CLMRunView`'s
/// own copy of this same choice made the identical mistake first): `Classify`/`Gate` send
/// `choice` criteria as a JSON **object**, `{tag: ""}` per declared tag, in order — an empty
/// description per `CLMSchema.candidates`' `isEmptyValue` check means "the candidate text is
/// the tag itself." `Score` sends `score` criteria as a JSON **array** of the declared tags,
/// in order — the ordered levels `CLMSchema.answer`'s `.score` case reads back for
/// `scoreLevels`. No option-count ceiling: each option is embedded on its own (CLM's action
/// head, not a slice of a fixed token budget), so this builds no equivalent to Laya's
/// `LayaQuestionDefinition.resolve()` validation.
// SPEC-Q234 (`catflow-mlx/SPEC_QUESTIONS.md`, filed 2026-09-29: "a decider may be served by a
// non-generative decision engine") — CL-7 (`RSI/DelegateCLMBacklog.md`), marked here the same
// way LY-8 marked Laya's own path: CLM is this spec question's second conformer, not a new one.
nonisolated struct CLMDecisionEngine: DecisionEngine {
    /// CL-6 — `CLMDecisionEngine`'s only call into the actual engine, injected like `askLaya`
    /// so a test can fake `CLMEngine` without loading a real checkpoint. The default calls the
    /// real `CLMEngineCache`/`CLMEngine.answer` for a single question.
    let askCLM: @Sendable (URL, CLMJSON, LayaQuestionType, CLMJSON, CLMJSON) async throws -> LayaAnswer

    func decide(directory: URL, state: String, ask: DecisionAsk) async throws -> LayaAnswer {
        let criteria: CLMJSON
        switch ask.type {
        case .choice:
            criteria = .object(ask.tags.map { CLMJSONField(key: $0, value: .string("")) })
        case .score:
            criteria = .array(ask.tags.map { .string($0) })
        case .noul:
            criteria = .object([])   // unreachable — Classify/Gate/Score never build .noul here
        }
        return try await askCLM(directory, .string(state), ask.type, .string(ask.instructions), criteria)
    }
}
