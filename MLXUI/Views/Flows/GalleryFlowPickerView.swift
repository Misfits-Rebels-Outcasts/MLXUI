import SwiftUI

/// KW-3-1: the "Copy flow here" picker — every bundled gallery flow, searchable, one tap to
/// copy into the workspace that opened it. `.catpipeline` flows are not offered: `.catpipeline`
/// inside a workspace is still an open R17 decision, deliberately out of scope for this phase
/// (`FlowEditRoute.copyIntoWorkspace` refuses one too, as the store-side backstop).
struct GalleryFlowPickerView: View {
    @Environment(\.dismiss) private var dismiss
    let onPick: (GalleryFlowMetadata) -> Void

    @State private var searchText = ""

    private var entries: [GalleryFlowMetadata] {
        GalleryLoader.loadMetadata()
            .filter { !$0.filename.hasSuffix(".catpipeline") }
            .filter { searchText.isEmpty || $0.title.localizedCaseInsensitiveContains(searchText) }
            .sorted { $0.number < $1.number }
    }

    var body: some View {
        NavigationStack {
            List(entries) { entry in
                Button {
                    onPick(entry)
                    dismiss()
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(entry.title)
                            .font(.headline)
                        Text(entry.category)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
            }
            .searchable(text: $searchText, prompt: "Search flows")
            .navigationTitle("Copy a Flow Here")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
        }
        .frame(minWidth: 380, minHeight: 440)
    }
}
