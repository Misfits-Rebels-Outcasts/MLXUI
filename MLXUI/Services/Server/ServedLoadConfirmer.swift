import Foundation
import Observation
import SwiftUI

/// SC-5a (gate D, owner 2026-10-07: "I would prefer Flows to ask first too."): the question an interactive load asks
/// before it pushes out a model other apps are being served. One per surface (the chat runner, each Run window);
/// the surface shows `question` as a sheet (`.servedLoadConfirmation(_:)`).
///
/// `confirm` returns `true` to go ahead — at once when the Local Server isn't enabled, when the load needs nothing,
/// or when nothing kept ready is in the way; after "Load anyway" otherwise, having unpinned and pushed out exactly the
/// models the question named. `false` is Cancel: the caller loads nothing and starts nothing.
@MainActor
@Observable
final class ServedLoadConfirmer {
    /// What the sheet shows.
    nonisolated struct Question: Identifiable, Equatable, Sendable {
        let id = UUID()
        let modelName: String
        let servedIDs: [String]
        /// About how long the pushed-out models take to come back, when the request log has measured them.
        let reloadSeconds: Int?

        var servedNames: [String] { servedIDs.map { $0.split(separator: "/").last.map(String.init) ?? $0 } }

        /// "Loading X will stop keeping Y ready for connected apps. It stays served and reloads on the next request
        /// (about N s)." — plural for several.
        var message: String {
            let several = servedIDs.count > 1
            let about = reloadSeconds.map { " (about \($0) s)" } ?? ""
            let tail = several
                ? "They stay served and reload on the next request\(about)."
                : "It stays served and reloads on the next request\(about)."
            return "Loading \(modelName) will stop keeping \(Self.list(servedNames)) ready for connected apps. \(tail)"
        }

        static func list(_ names: [String]) -> String {
            guard names.count > 1, let last = names.last else { return names.first ?? "" }
            return names.dropLast().joined(separator: ", ") + " and " + last
        }
    }

    private(set) var question: Question?
    @ObservationIgnored private var answer: CheckedContinuation<Bool, Never>?

    /// Which served models `loadBytes` would stop keeping ready (`ServeMemory.wouldDisplaceServed`).
    @ObservationIgnored private let check: (Int64, MemoryBudget.HolderKind?) async -> [String]
    /// "Load anyway": unpin and push exactly these out.
    @ObservationIgnored private let pushOut: ([String], MemoryBudget.HolderKind) async -> Void
    @ObservationIgnored private let reloadSeconds: ([String]) -> Int?

    /// The defaults are the app's controller; tests inject their own.
    init(check: @escaping (Int64, MemoryBudget.HolderKind?) async -> [String] = { bytes, requester in
             await ServeMemory.wouldDisplaceServed(loadBytes: bytes, requester: requester)
         },
         pushOut: @escaping ([String], MemoryBudget.HolderKind) async -> Void = { ids, requester in
             await LocalServerController.shared.pushOutServed(ids, for: requester)
         },
         reloadSeconds: @escaping ([String]) -> Int? = { ids in
             let measured = ids.compactMap { LocalServerController.shared.measuredLoadSeconds(for: $0) }
             return measured.isEmpty ? nil : Int(measured.reduce(0, +).rounded())
         }) {
        self.check = check
        self.pushOut = pushOut
        self.reloadSeconds = reloadSeconds
    }

    func confirm(modelName: String, loadBytes: Int64, requester: MemoryBudget.HolderKind) async -> Bool {
        guard LocalServerGate.isEnabled, loadBytes > 0 else { return true }          // gate closed: no check at all
        let ids = await check(loadBytes, requester)
        guard !ids.isEmpty else { return true }
        // one question at a time on a surface
        while question != nil {
            try? await Task.sleep(for: .milliseconds(100))
            if Task.isCancelled { return false }
        }
        let asked = Question(modelName: modelName, servedIDs: ids, reloadSeconds: reloadSeconds(ids))
        let loadAnyway = await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
                if Task.isCancelled { continuation.resume(returning: false); return }
                question = asked
                answer = continuation
            }
        } onCancel: {
            Task { @MainActor in self.respond(false, to: asked) }
        }
        guard loadAnyway else { return false }
        await pushOut(ids, requester)
        return true
    }

    /// The sheet's buttons (and its dismissal, which is Cancel).
    func respond(_ loadAnyway: Bool, to asked: Question? = nil) {
        if let asked, question != asked { return }          // a late cancel for a question already answered
        guard let continuation = answer else { return }
        answer = nil
        question = nil
        continuation.resume(returning: loadAnyway)
    }
}

/// The confirm sheet: one sentence, two buttons.
struct ServedLoadConfirmSheet: View {
    let message: String
    let respond: (Bool) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Stop keeping a served model ready?")
                .font(.headline)
            Text(message)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { respond(false) }
                    .keyboardShortcut(.cancelAction)
                Button("Load anyway") { respond(true) }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 420)
    }
}

extension View {
    /// Presents `confirmer`'s question; dismissing the sheet any other way is Cancel.
    func servedLoadConfirmation(_ confirmer: ServedLoadConfirmer) -> some View {
        sheet(item: Binding(get: { confirmer.question },
                            set: { if $0 == nil { confirmer.respond(false) } })) { question in
            ServedLoadConfirmSheet(message: question.message) { confirmer.respond($0) }
        }
    }
}

private struct ServedLoadConfirmerKey: EnvironmentKey {
    static let defaultValue: ServedLoadConfirmer? = nil
}

extension EnvironmentValues {
    /// SC-5a: the confirmer of the Run window this view is in (nil outside one).
    var servedLoadConfirmer: ServedLoadConfirmer? {
        get { self[ServedLoadConfirmerKey.self] }
        set { self[ServedLoadConfirmerKey.self] = newValue }
    }
}
