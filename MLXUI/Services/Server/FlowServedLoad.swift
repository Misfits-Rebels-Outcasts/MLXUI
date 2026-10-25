import Foundation
import Observation

/// SC-5b (gate D, owner 2026-10-07: "I would prefer Flows to ask first too."): a flow row whose model load would stop
/// keeping a served model ready asks first. The row's executor throws this; the interpreter **parks** the run on it the
/// way it parks a human row (HR-4: `ParkRun` → `runParked` → `FlowRunSession.parked`), the session shows the question
/// and re-runs with the answer. It is never auto-answered.
nonisolated struct ServedLoadQuestion: Error, Sendable, Equatable {
    /// The `runParked` policy that tells this park from a human row's.
    static let policy = "served-load"
    /// "Loading X will stop keeping Y ready for connected apps. It stays served and reloads on the next request (about N s)."
    let sentence: String
    let servedIDs: [String]
}

/// The answer to a parked served-load question, handed to the re-run (it outlives the park: the session records it, the
/// row's gate takes it). Keyed by flow and the interpreter's activation path; one-shot, and stale after two minutes.
nonisolated final class ServedLoadDecisions: @unchecked Sendable {
    enum Decision: Sendable, Equatable { case loadAnyway, cancel }

    static let shared = ServedLoadDecisions()
    static let lifetime: TimeInterval = 120

    private let lock = NSLock()
    private var entries: [String: (decision: Decision, at: Date)] = [:]

    private func key(_ flowID: String, _ path: String) -> String { "\(flowID)|\(path)" }

    func set(_ decision: Decision, flowID: String, path: String) {
        lock.lock(); defer { lock.unlock() }
        entries[key(flowID, path)] = (decision, Date())
    }

    /// The recorded answer for this row, removed as it is read; nil when none (or it went stale).
    func take(flowID: String, path: String) -> Decision? {
        lock.lock(); defer { lock.unlock() }
        guard let entry = entries.removeValue(forKey: key(flowID, path)) else { return nil }
        return Date().timeIntervalSince(entry.at) <= Self.lifetime ? entry.decision : nil
    }

    func clear(flowID: String) {
        lock.lock(); defer { lock.unlock() }
        entries = entries.filter { !$0.key.hasPrefix("\(flowID)|") }
    }
}

/// What `RealExecutor` calls before a model row loads (`servedLoadGate`). Passes through (returns) when nothing kept ready
/// is in the way, when serving isn't enabled, or after a recorded "Load anyway" (having unpinned and pushed those models
/// out); throws the row's failure on a recorded "Cancel"; and otherwise throws the `ServedLoadQuestion` that parks the run.
nonisolated struct FlowServedLoadGate: Sendable {
    let flowID: String
    /// What loading `model` adds to the ledger (0 when it is already resident).
    let bytesNeeded: @Sendable (ModelEntry) async -> Int64
    let check: @Sendable (Int64) async -> [String]
    let pushOut: @Sendable ([String]) async -> Void
    let reloadSeconds: @Sendable ([String]) async -> Int?
    var decisions: ServedLoadDecisions = .shared
    var isEnabled: @Sendable () -> Bool = { LocalServerGate.isEnabled }
    var refusal: @Sendable () -> RunAdmission.Refusal? = { RunAdmission.canLoadModel(surface: .flowRow) }

    func callAsFunction(path: String, model: ModelEntry) async throws {
        // SC-7 (gate F): in Dedicated server mode a model row fails with the sentence — also an armed firing's, which is
        // logged as a failed run and skipped (it doesn't queue). Asked before anything else, whatever the bytes.
        if let refusal = refusal() { throw FlowError.stageFailure(row: path, message: refusal.sentence) }
        guard isEnabled() else { return }                                     // gate closed: no check at all
        let bytes = await bytesNeeded(model)
        guard bytes > 0 else { return }
        let ids = await check(bytes)
        guard !ids.isEmpty else { return }
        let sentence = ServedLoadConfirmer.Question(modelName: model.displayName, servedIDs: ids,
                                                    reloadSeconds: await reloadSeconds(ids)).message
        switch decisions.take(flowID: flowID, path: path) {
        case .loadAnyway?:
            await pushOut(ids)
        case .cancel?:
            throw FlowError.stageFailure(row: path, message: sentence)
        case nil:
            throw ServedLoadQuestion(sentence: sentence, servedIDs: ids)
        }
    }

    /// The app's gate: the real check against the controller's pool, the real push-out as `.flow`.
    @MainActor
    static func live(flowID: String, cache: EngineCache = .shared,
                     booksWeights: @escaping @Sendable (ModelEntry) async -> Bool) -> FlowServedLoadGate {
        FlowServedLoadGate(
            flowID: flowID,
            bytesNeeded: { model in await Self.bytesNeeded(model, cache: cache, booksWeights: await booksWeights(model)) },
            check: { bytes in await ServeMemory.wouldDisplaceServed(loadBytes: bytes, requester: .flow) },
            pushOut: { ids in await LocalServerController.shared.pushOutServed(ids, for: .flow) },
            reloadSeconds: { ids in
                let measured = await MainActor.run { ids.compactMap { LocalServerController.shared.measuredLoadSeconds(for: $0) } }
                return measured.isEmpty ? nil : Int(measured.reduce(0, +).rounded())
            })
    }

    /// What a row's load adds to the ledger, by who owns the weights: the LLM pool, the Laya / CLM caches, the VLM pool
    /// (own-weights stages), or the engine cache.
    static func bytesNeeded(_ model: ModelEntry, cache: EngineCache, booksWeights: Bool) async -> Int64 {
        let directory = ModelStore.shared.directory(forHFModelID: model.hfModelId)
        let fallback = Int64(model.ramGB * 1_073_741_824)
        switch model.runnerKind {
        case .llm:
            if await ModelContainerPool.shared.residentBytes(directory: directory) != nil { return 0 }
            return ModelWeightCharge.bytes(directory: directory, catalogFallback: fallback)
        case .decision:
            return model.family == "CLM" ? await CLMEngineCache.shared.bytesNeeded(directory: directory)
                                         : await LayaEngineCache.shared.bytesNeeded(directory: directory)
        case .diarization:
            // MP4-3a: the engine cache books it like the ASR engines (per-run load, `ModelLoadProfile` row unmeasured).
            return await cache.bytesNeeded(for: model, config: .default, booksWeights: booksWeights)
        default:
            return await cache.bytesNeeded(for: model, config: .default, booksWeights: booksWeights)
        }
    }
}

/// SC-5b: parked served-load questions, for the Local Server screen ("A flow is waiting: … Load anyway / Cancel"). A
/// session registers one when its run parks and removes it when answered or stopped; the answer closure re-runs the flow.
@MainActor
@Observable
final class ServedLoadWaiting {
    struct Item: Identifiable {
        let id: UUID
        let flow: String
        let sentence: String
        let answer: (Bool) -> Void
    }

    static let shared = ServedLoadWaiting()
    private(set) var items: [Item] = []

    @discardableResult
    func add(flow: String, sentence: String, answer: @escaping (Bool) -> Void) -> UUID {
        let id = UUID()
        items.append(Item(id: id, flow: flow, sentence: sentence, answer: answer))
        return id
    }

    func remove(_ id: UUID?) {
        guard let id else { return }
        items.removeAll { $0.id == id }
    }
}
