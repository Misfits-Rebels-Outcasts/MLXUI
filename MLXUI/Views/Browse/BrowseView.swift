import SwiftUI

struct BrowseView: View {
    @Environment(AppState.self) private var appState
    @State private var showComparison = false

    // No maximum cap: columns stretch to absorb leftover width so the gutters stay an
    // even 10pt instead of widening into gaps at in-between window sizes.
    private let gridColumns = [GridItem(.adaptive(minimum: 200), spacing: 10)]

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            FilterBar()
            Divider()
            modelList
        }
        .onAppear { appState.loadFilters() }
        .onDisappear { appState.saveFilters() }
    }

    private var headerBar: some View {
        HStack {
            if case .browse(let sectionId) = appState.selectedSection,
               let section = appState.browserData?.sidebarSections.first(where: { $0.id == sectionId }) {
                Label(section.name, systemImage: section.sfSymbol)
                    .font(.title2)
                    .fontWeight(.semibold)
            }
            Spacer()
            if !appState.comparisonSet.isEmpty {
                Button {
                    showComparison = true
                } label: {
                    Label("Compare (\(appState.comparisonSet.count))", systemImage: "tablecells")
                        .font(.caption)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
            Text(appState.modelCountLabel)
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker("Sort", selection: Binding(get: { appState.sortOrder }, set: { appState.sortOrder = $0 })) {
                ForEach(SortOrder.allCases) { order in
                    Text(order.label).tag(order)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 130)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .sheet(isPresented: $showComparison) {
            ComparisonTableView(models: appState.comparisonSet)
        }
    }

    private var modelList: some View {
        Group {
            if appState.filteredModels.isEmpty {
                emptyState
            } else {
                ScrollView {
                    // A single LazyVGrid holding one Section per domain group keeps every
                    // card in one lazy container. Multiple sibling LazyVGrids in one
                    // ScrollView mis-estimate offscreen height and blank out cells while
                    // scrolling — using one grid avoids that. Section headers in a
                    // LazyVGrid automatically span all columns.
                    LazyVGrid(columns: gridColumns, spacing: 10, pinnedViews: [.sectionHeaders]) {
                        ForEach(appState.groupedModels, id: \.domain.id) { group in
                            Section {
                                ForEach(group.models) { model in
                                    NavigationLink(destination: ModelDetailView(model: model)) {
                                        ModelCard(model: model, showCompare: true)
                                    }
                                    .buttonStyle(.plain)
                                }
                            } header: {
                                sectionHeader(group)
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 12)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 32))
                .foregroundStyle(.secondary)
            Text("No models match your filters")
                .font(.headline)
            Text("Try adjusting the source, RAM limit, or capability toggles above.")
                .font(.caption)
                .foregroundStyle(.secondary)
            Button("Reset All Filters") {
                appState.resetFilters()
            }
            .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func sectionHeader(_ group: (domain: DomainNode, models: [ModelEntry])) -> some View {
        HStack {
            Image(systemName: group.domain.sfSymbol)
                .foregroundStyle(.secondary)
            Text(group.domain.name)
                .font(.headline)
            Text("(\(group.models.count))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.bar)
    }
}
