//
//  ModelInstallCoordinator.swift
//  MLXUI — Model Tools, MT4-3 (plan §3.8, install and continue).
//
//  `ModelToolHost.installRequest` lands here. A tool call whose model is not on this Mac **waits** on an install card
//  (the same pattern as `PendingApproval`): Download & continue, Choose another..., or Cancel. The call resumes with `true`
//  once the `.installed` marker is there (the runner then resolves again, with the same arguments) or with `false` on a
//  decline, a Stop or leaving the conversation. A download starts only from the card's own button — never from a router.
//
//  The install engine is behind `ModelInstalling`, so the whole state machine is tested with a fake.
//

import Foundation
import Observation

/// What the card needs from the install engine (`AppModelInstaller` in the app; a fake in tests). Main actor.
@MainActor
protocol ModelInstalling: AnyObject {
    func entry(forModelID id: String) -> ModelEntry?
    /// `.installed` when the marker is on disk; otherwise the engine's current state for this model.
    func state(of entry: ModelEntry) -> InstallState
    func start(_ entry: ModelEntry)
    /// Stop the download and clear its scratch files.
    func cancel(_ entry: ModelEntry)
    func availableDiskGB() -> Double
    /// Returns once the state of `entry` may have changed.
    func changed(_ entry: ModelEntry) async
    /// A model finished installing: refresh whatever the tool resolution reads (the claim table).
    func installFinished(_ entry: ModelEntry)
}

/// One install card on screen.
struct ModelInstallCard: Identifiable, Equatable {
    enum Phase: Equatable {
        /// "Transcribing needs X. 0.6 GB download." [Download & continue] [Choose another...] [Cancel]
        case offering
        /// The other fitting candidates, with their sizes.
        case choosing
        case downloading
        /// "Installed. Continuing..." until the tool call picks up again.
        case installed
        /// The download failed (network, disk, ...): the engine's sentence and a Retry.
        case failed(String)
        /// The model is gated (401/403): the card links to the Hugging Face token setting.
        case gated(String)
        /// There is no room for it: no download button; alternatives that do fit are offered.
        case wontFit(String)
    }

    let id = UUID()
    var conversationID: UUID?
    let toolName: String
    var target: ModelInstallOption
    var alternatives: [ModelInstallOption]
    var phase: Phase
    /// This card started the download (so Cancel / Stop may stop it); false when it joined one already running.
    var startedDownload = false
}

@MainActor
@Observable
final class ModelInstallCoordinator {
    /// The cards on screen, oldest first.
    private(set) var cards: [ModelInstallCard] = []

    @ObservationIgnored var installer: (any ModelInstalling)?
    @ObservationIgnored private var waiters: [UUID: [CheckedContinuation<Bool, Never>]] = [:]
    @ObservationIgnored private var watchers: [UUID: Task<Void, Never>] = [:]

    init() {}

    // MARK: - The tool call's side

    /// Suspends until the user has decided and, on Download, the model is installed. `true` = installed, go on.
    func request(_ need: ModelInstallNeed, conversationID: UUID?) async -> Bool {
        guard let installer, !Task.isCancelled else { return false }
        guard let entry = installer.entry(forModelID: need.modelID) else { return false }
        // An earlier call (or Browse) finished it: nothing to ask.
        if installer.state(of: entry) == .installed { return true }

        let cardID: UUID
        if let existing = cards.first(where: { $0.target.id == need.modelID && !$0.isFinished }) {
            cardID = existing.id          // two calls want the same model: one card, one install
        } else {
            var card = ModelInstallCard(
                conversationID: conversationID, toolName: need.toolName,
                target: ModelInstallOption(id: need.modelID, displayName: need.modelDisplayName, downloadGB: need.downloadGB),
                alternatives: need.alternatives.filter { diskFits($0, installer) }, phase: .offering)
            if let reason = diskRefusal(gb: need.downloadGB, installer) { card.phase = .wontFit(reason) }
            cardID = card.id
            cards.append(card)
        }
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Bool, Never>) in
                waiters[cardID, default: []].append(continuation)
            }
        } onCancel: {
            Task { @MainActor in self.decline(cardID) }
        }
    }

    // MARK: - The card's buttons

    /// "Download & continue" (and "Retry", and "Try again" after a token was added).
    func startDownload(_ cardID: UUID) {
        guard let installer, let index = index(of: cardID) else { return }
        switch cards[index].phase {
        case .offering, .failed, .gated: break
        default: return
        }
        guard let entry = installer.entry(forModelID: cards[index].target.id) else {
            cards[index].phase = .failed("This model is no longer in the catalog.")
            return
        }
        if let reason = diskRefusal(gb: entry.downloadSizeGB, installer) {
            cards[index].phase = .wontFit(reason)
            return
        }
        cards[index].startedDownload = !installer.state(of: entry).isActive
        installer.start(entry)
        cards[index].phase = .downloading
        watch(cardID, entry)
    }

    /// "Choose another...": show the candidates.
    func showChoices(_ cardID: UUID) {
        guard let index = index(of: cardID), !cards[index].alternatives.isEmpty else { return }
        switch cards[index].phase {
        case .offering, .failed, .gated, .wontFit: cards[index].phase = .choosing
        default: break
        }
    }

    /// Back from the candidates to the offer.
    func hideChoices(_ cardID: UUID) {
        guard let index = index(of: cardID), cards[index].phase == .choosing else { return }
        cards[index].phase = .offering
    }

    /// A candidate was picked: it becomes the card's model, and the card offers it (the previous one stays a candidate).
    func choose(_ cardID: UUID, option optionID: String) {
        guard let installer, let index = index(of: cardID),
              let picked = cards[index].alternatives.first(where: { $0.id == optionID }) else { return }
        switch cards[index].phase {
        case .offering, .choosing, .failed, .gated, .wontFit: break
        default: return
        }
        let previous = cards[index].target
        cards[index].alternatives.removeAll { $0.id == optionID }
        if diskFits(previous, installer) { cards[index].alternatives.append(previous) }
        cards[index].target = picked
        cards[index].phase = diskRefusal(gb: picked.downloadGB, installer).map { .wontFit($0) } ?? .offering
    }

    /// "Cancel": the tool call gets the decline sentence; a download this card started is stopped.
    func decline(_ cardID: UUID) {
        guard let index = index(of: cardID) else { return }
        let card = cards[index]
        watchers[cardID]?.cancel()
        watchers[cardID] = nil
        if card.startedDownload, card.phase == .downloading,
           let entry = installer?.entry(forModelID: card.target.id) {
            installer?.cancel(entry)
        }
        cards.remove(at: index)
        resume(cardID, with: false)
    }

    // MARK: - The turn's side

    /// Stop, leaving the conversation, the end of a turn: every card resolves as declined, its download is stopped, nothing stays.
    func cancelAll() {
        for card in cards { decline(card.id) }
    }

    /// The tool call picked up again after an install (`.running`): "Installed. Continuing..." has done its job.
    func toolResumed(_ toolName: String) {
        cards.removeAll { $0.toolName == toolName && $0.phase == .installed }
    }

    // MARK: - Internals

    private func index(of cardID: UUID) -> Int? { cards.firstIndex { $0.id == cardID } }

    private func resume(_ cardID: UUID, with value: Bool) {
        let pending = waiters.removeValue(forKey: cardID) ?? []
        for continuation in pending { continuation.resume(returning: value) }
    }

    private func setPhase(_ cardID: UUID, _ phase: ModelInstallCard.Phase) {
        guard let index = index(of: cardID) else { return }
        cards[index].phase = phase
    }

    private func complete(_ cardID: UUID, _ entry: ModelEntry) {
        guard index(of: cardID) != nil else { return }
        installer?.installFinished(entry)
        setPhase(cardID, .installed)
        watchers[cardID] = nil
        resume(cardID, with: true)
    }

    private func watch(_ cardID: UUID, _ entry: ModelEntry) {
        watchers[cardID]?.cancel()
        watchers[cardID] = Task { [weak self] in
            while !Task.isCancelled {
                guard let self, let installer = self.installer, self.index(of: cardID) != nil else { return }
                switch installer.state(of: entry) {
                case .installed:
                    self.complete(cardID, entry)
                    return
                case .error(let message, _):
                    self.setPhase(cardID, .failed(message))
                    self.watchers[cardID] = nil
                    return
                case .needsAuth(let message):
                    self.setPhase(cardID, .gated(message))
                    self.watchers[cardID] = nil
                    return
                case .idle:
                    // Cancelled elsewhere (Browse's own Cancel): not a failure of ours, but the call cannot go on.
                    self.setPhase(cardID, .failed("The download was cancelled."))
                    self.watchers[cardID] = nil
                    return
                case .resolving, .downloading, .verifying:
                    await installer.changed(entry)
                }
            }
        }
    }

    private func diskRefusal(gb: Double, _ installer: any ModelInstalling) -> String? {
        DiskSpace.preflightError(downloadSizeGB: gb, availableDiskGB: installer.availableDiskGB())
    }

    private func diskFits(_ option: ModelInstallOption, _ installer: any ModelInstalling) -> Bool {
        diskRefusal(gb: option.downloadGB, installer) == nil
    }
}

extension ModelInstallCard {
    /// Installed cards are done (a new request for the same model gets `true` straight away).
    var isFinished: Bool { phase == .installed }
}

/// The card's words (pure, so a test can pin them). Plain facts: what the tool needs, how big, that it stays on this Mac.
extension ModelInstallCard {
    /// "Transcribing", "Describing the image", ... for the headline "‹verb› needs ‹model›".
    static func action(forTool name: String) -> String {
        switch name {
        case "transcribe_audio": return "Transcribing"
        case "ocr_image": return "Reading the image"
        case "describe_image": return "Describing the image"
        case "speak": return "Speaking"
        case "translate": return "Translating"
        case "generate_image": return "Drawing"
        case "generate_music": return "Making sound"
        case "upscale_image": return "Upscaling"
        case "segment_image": return "Cutting out"
        case "estimate_depth": return "Estimating depth"
        default: return "This"
        }
    }

    var headline: String { "\(Self.action(forTool: toolName)) needs \(target.displayName)" }

    /// "0.6 GB download · runs on this Mac only".
    var detail: String { "\(Self.sizeText(target.downloadGB)) download · runs on this Mac only" }

    static func sizeText(_ gb: Double) -> String {
        gb < 10 ? String(format: "%.1f GB", max(gb, 0.1)) : String(format: "%.0f GB", gb)
    }

    /// "62% · 0.4 of 0.6 GB" from the engine's byte counts.
    static func progressText(downloaded: Int64, total: Int64) -> String {
        let percent = Int((DownloadProgress.fraction(downloaded: downloaded, total: total) * 100).rounded())
        return "\(percent)% · \(sizeText(Double(downloaded) / 1e9)) of \(sizeText(Double(total) / 1e9))"
    }
}
