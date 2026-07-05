import SwiftUI

struct HomeView: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("AI Browser")
                .font(.largeTitle)
                .fontWeight(.bold)
                .padding(.horizontal, 24)
                .padding(.top, 24)

            Text("Discover and run AI models locally on your Mac")
                .foregroundStyle(.secondary)
                .padding(.horizontal, 24)
                .padding(.bottom, 24)

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 32) {
                    systemStatusSection
                    quickStartSection
                    trendingSection
                }
                .padding(24)
            }
        }
    }

    private var systemStatusSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Your Mac")
                .font(.headline)

            HStack(spacing: 24) {
                statusBadge(icon: "cpu.fill", label: appState.systemInfo.chipName)
                statusBadge(icon: "memorychip.fill", label: appState.systemInfo.formattedRAM)
                statusBadge(icon: "internaldrive.fill", label: "\(appState.systemInfo.formattedDisk) free")
                statusBadge(icon: "macbook", label: "macOS \(appState.systemInfo.macOSVersion)")
            }
        }
        .padding(20)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
    }

    private func statusBadge(icon: String, label: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
            Text(label)
                .font(.subheadline)
        }
    }

    private var quickStartSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Quick Start")
                .font(.headline)

            HStack(spacing: 12) {
                ForEach(appState.visibleSections.prefix(3)) { section in
                    Button {
                        appState.selectedSection = .browse(section.id)
                        appState.resetFilters()
                    } label: {
                        VStack(spacing: 8) {
                            Image(systemName: section.sfSymbol)
                                .font(.title)
                            Text(section.name)
                                .font(.subheadline)
                                .fontWeight(.medium)
                            Text("\(section.modelCount(in: appState.browserData!)) models")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(16)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var trendingSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Trending")
                .font(.headline)

            if let data = appState.browserData {
                let trending = data.domains
                    .flatMap { $0.allModels }
                    .sorted { ($0.communityDownloads ?? 0) > ($1.communityDownloads ?? 0) }
                    .prefix(8)

                LazyVGrid(columns: [GridItem(.adaptive(minimum: 200), spacing: 12)], spacing: 12) {
                    ForEach(Array(trending)) { model in
                        NavigationLink(destination: ModelDetailView(model: model)) {
                            ModelCard(model: model, showCompare: false)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
}
