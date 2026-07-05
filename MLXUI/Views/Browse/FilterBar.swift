import SwiftUI

struct FilterBar: View {
    @Environment(AppState.self) private var appState

    var body: some View {
        VStack(spacing: 8) {
            HStack(spacing: 16) {
                // Only show the source picker when the catalog actually contains more
                // than one source — otherwise every non-present segment is a dead tab.
                if appState.availableSources.count > 1 {
                    Picker("Source", selection: Binding(
                        get: { appState.filterSource },
                        set: { appState.filterSource = $0 }
                    )) {
                        Text("All Sources").tag(nil as ModelSource?)
                        ForEach(appState.availableSources, id: \.self) { src in
                            Text(src.rawValue.capitalized).tag(src as ModelSource?)
                        }
                    }
                    .pickerStyle(.segmented)
                    .labelsHidden()

                    Spacer()
                } else {
                    Spacer()
                }

                HStack(spacing: 4) {
                    Image(systemName: "memorychip")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("RAM: ≤")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(Int(appState.filterRAMLimitGB)) GB")
                        .font(.caption)
                        .monospacedDigit()
                    Slider(value: Binding(
                        get: { appState.filterRAMLimitGB },
                        set: { appState.filterRAMLimitGB = $0 }
                    ), in: 2...appState.ramSliderMax, step: 2)
                    .frame(width: 120)
                }

                if appState.filterSource != nil || appState.filterRAMLimitGB < appState.systemInfo.totalRAMGB || !appState.filterCapabilities.isEmpty {
                    Button("Reset") {
                        appState.resetFilters()
                    }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(.blue)
                }
            }

            if !appState.availableCapabilities.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(appState.availableCapabilities, id: \.self) { cap in
                            Button {
                                if appState.filterCapabilities.contains(cap) {
                                    appState.filterCapabilities.remove(cap)
                                } else {
                                    appState.filterCapabilities.insert(cap)
                                }
                            } label: {
                                Text(cap)
                                    .font(.caption)
                                    .padding(.horizontal, 10)
                                    .padding(.vertical, 4)
                                    .background(
                                        appState.filterCapabilities.contains(cap)
                                        ? Color.accentColor
                                        : Color.secondary.opacity(0.15)
                                    )
                                    .foregroundStyle(
                                        appState.filterCapabilities.contains(cap)
                                        ? .white
                                        : .primary
                                    )
                                    .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 16)
                }
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.bar)
    }
}
