//
//  HomeStatusFooter.swift
//  MLXUI — Home chat, HC-2 (ruling E).
//
//  One quiet line under the grid in the Home chat page's empty states: chip · RAM · free disk, plus the
//  Local Server sentence and link while that card exists. Page surface only; callers never show it on the bench.
//

import SwiftUI

struct HomeStatusFooter: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        let line = HomeStatusLine(chipName: appState.systemInfo.chipName, formattedRAM: appState.systemInfo.formattedRAM,
                                  formattedDisk: appState.systemInfo.formattedDisk,
                                  macOSVersion: appState.systemInfo.macOSVersion)
        let card = appState.localServer.homeCard
        HStack(spacing: 4) {
            Text(HomeStatusLine.footer(system: line.sentence, serverSentence: card?.sentence))
            if card != nil {
                Text("·")
                Button(HomeServerCard.linkTitle) {
                    appState.route = []
                    appState.selectedSection = HomeServerCard.linkTarget
                }
                .buttonStyle(.link)
            }
        }
        .font(.footnote)
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
        .frame(maxWidth: .infinity)
        .padding(.top, 8)
    }
}
