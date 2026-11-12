//
//  AppModelInstaller.swift
//  MLXUI — Model Tools, MT4-3: the install card's engine, on the app's own install path
//  (`AppState.installModel` -> `InstallManager.install`). Nothing here downloads by itself.
//

import Foundation
import Observation

@MainActor
final class AppModelInstaller: ModelInstalling {
    private unowned let appState: AppState

    init(appState: AppState) { self.appState = appState }

    func entry(forModelID id: String) -> ModelEntry? {
        appState.browserData?.domains.flatMap { $0.allModels }.first { $0.id == id }
    }

    func state(of entry: ModelEntry) -> InstallState {
        if appState.installManager.isInstalled(entry) { return .installed }
        return appState.installManager.modelStates[entry.id] ?? .idle
    }

    func start(_ entry: ModelEntry) { appState.installModel(entry) }

    func cancel(_ entry: ModelEntry) { appState.cancelInstall(entry) }

    func availableDiskGB() -> Double {
        let free = try? ModelStore.shared.modelsDirectory.resourceValues(forKeys: [.volumeAvailableCapacityKey])
            .volumeAvailableCapacity
        return Double(free ?? 0) / 1_000_000_000.0
    }

    func changed(_ entry: ModelEntry) async {
        let manager = appState.installManager
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            withObservationTracking {
                _ = manager.modelStates[entry.id]
            } onChange: {
                continuation.resume()
            }
        }
    }

    func installFinished(_ entry: ModelEntry) {
        // The tool resolves again against the live tables: refresh the claim table the pool derivation reads.
        appState.recomputeClaimableModelIDs()
    }
}
