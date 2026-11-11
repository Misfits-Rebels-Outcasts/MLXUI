//
//  ModelInstallCardView.swift
//  MLXUI — Model Tools, MT4-3: the install card a model tool waits on (plan §3.8).
//
//  Offer -> downloading -> "Installed. Continuing..." — or a failure with Retry, a gated model's link to the token setting, a
//  won't-fit with nothing to press but a fitting alternative, and Choose another.... The only thing that starts a download
//  is the Download & continue (or Retry) button here.
//

import SwiftUI

struct ModelInstallCardView: View {
    let card: ModelInstallCard
    let coordinator: ModelInstallCoordinator
    /// The engine's live state for the card's model, for the progress line.
    let installState: InstallState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .top, spacing: 8) {
                Image(systemName: "arrow.down.circle").foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 2) {
                    Text(card.headline).font(.callout.weight(.semibold))
                    Text(card.detail).font(.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }
            content
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 8))
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder private var content: some View {
        switch card.phase {
        case .offering:
            HStack(spacing: 8) {
                Button("Download & continue") { coordinator.startDownload(card.id) }
                    .buttonStyle(.borderedProminent)
                if !card.alternatives.isEmpty {
                    Button("Choose another…") { coordinator.showChoices(card.id) }
                }
                Button("Cancel") { coordinator.decline(card.id) }
            }
            .controlSize(.small)
        case .choosing:
            VStack(alignment: .leading, spacing: 6) {
                ForEach(card.alternatives) { option in
                    Button {
                        coordinator.choose(card.id, option: option.id)
                    } label: {
                        Text("\(option.displayName) · \(ModelInstallCard.sizeText(option.downloadGB))")
                    }
                }
                HStack(spacing: 8) {
                    Button("Back") { coordinator.hideChoices(card.id) }
                    Button("Cancel") { coordinator.decline(card.id) }
                }
            }
            .controlSize(.small)
        case .downloading:
            VStack(alignment: .leading, spacing: 6) {
                switch installState {
                case .downloading(let progress, let downloaded, let total):
                    ProgressView(value: progress)
                    Text(ModelInstallCard.progressText(downloaded: downloaded, total: total))
                        .font(.caption).foregroundStyle(.secondary)
                case .verifying:
                    ProgressView()
                    Text("Verifying…").font(.caption).foregroundStyle(.secondary)
                default:
                    ProgressView()
                    Text("Starting the download…").font(.caption).foregroundStyle(.secondary)
                }
                Button("Cancel") { coordinator.decline(card.id) }.controlSize(.small)
            }
        case .installed:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Installed. Continuing…").font(.caption).foregroundStyle(.secondary)
            }
        case .failed(let message):
            VStack(alignment: .leading, spacing: 6) {
                Text(message).font(.caption).foregroundStyle(.red)
                HStack(spacing: 8) {
                    Button("Retry") { coordinator.startDownload(card.id) }.buttonStyle(.borderedProminent)
                    if !card.alternatives.isEmpty { Button("Choose another…") { coordinator.showChoices(card.id) } }
                    Button("Cancel") { coordinator.decline(card.id) }
                }
                .controlSize(.small)
            }
        case .gated(let message):
            VStack(alignment: .leading, spacing: 6) {
                Label(message, systemImage: "lock.fill").font(.caption).foregroundStyle(.orange)
                HStack(spacing: 8) {
                    SettingsOpener(pane: .models) { Text("Open Settings") }.buttonStyle(.borderedProminent)
                    Button("Try again") { coordinator.startDownload(card.id) }
                    Button("Cancel") { coordinator.decline(card.id) }
                }
                .controlSize(.small)
            }
        case .wontFit(let reason):
            VStack(alignment: .leading, spacing: 6) {
                Text(reason).font(.caption).foregroundStyle(.orange)
                HStack(spacing: 8) {
                    if !card.alternatives.isEmpty { Button("Choose another…") { coordinator.showChoices(card.id) } }
                    Button("Cancel") { coordinator.decline(card.id) }
                }
                .controlSize(.small)
            }
        }
    }
}
