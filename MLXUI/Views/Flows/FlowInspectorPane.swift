import SwiftUI
import AVFoundation
import AppKit

/// The inspector pane (CFM-R3-3): a right-hand pane bound to the selected row, split into
/// two tabs.
///
/// - **Output**: the selected row's cached `Asset`, or a `Save *` row's written file,
///   presented per modality the way the run views do — the presentation half of
///   `TTSRunView`/`ASRRunView`/`OCRRunView`/`ImageQARunView`/`EmbeddingRunView`
///   (design doc §6), extracted rather than duplicated, and the standalone Run sheets are
///   untouched. Also surfaces the `CatalogBridge` substitution note (CFM-R2-2 rule 3) for a
///   row whose model is `.sameFamily`/`.substitute`.
/// - **Properties**: the row's editable details — model, instruction, inputs, settings,
///   decisions. The flow editor embeds a live `FlowRowInspectorView`; the read-only flow
///   list embeds a frozen one, so a gallery flow's properties are browsable (scrollable)
///   but never mutable.
struct FlowInspectorPane<Properties: View>: View {
    enum Tab: Hashable {
        case properties, output
    }

    /// The selected row's cached output, or nil when nothing is selected / no run yet.
    let output: Asset?
    /// The row's title (task name) for the output pane header.
    let rowTitle: String
    /// A `CatalogBridge` substitution note (e.g. "running Kokoro 82M as …"), or nil.
    let substitutionNote: String?
    /// The file a selected `Save *` row wrote (resolved by the caller), so the saved result
    /// is playable/viewable rather than just a status sentence.
    var savedFile: URL?
    /// The saved file's kind, driving the presentation.
    var savedKind: Kind?
    /// The row-properties pane — editable in the editor, frozen in the read-only flow list.
    private let properties: () -> Properties

    @State private var tab: Tab
    @State private var player: AVAudioPlayer?

    init(output: Asset?, rowTitle: String, substitutionNote: String?,
         savedFile: URL? = nil, savedKind: Kind? = nil,
         initialTab: Tab = .output,
         @ViewBuilder properties: @escaping () -> Properties) {
        self.output = output
        self.rowTitle = rowTitle
        self.substitutionNote = substitutionNote
        self.savedFile = savedFile
        self.savedKind = savedKind
        self.properties = properties
        _tab = State(initialValue: initialTab)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            tabBar
            Divider()
                .padding(.bottom, 10)
            switch tab {
            case .properties:
                properties()
            case .output:
                outputPane
            }
        }
        .padding(14)
        .frame(minWidth: 260, maxWidth: 320, maxHeight: .infinity, alignment: .topLeading)
        .background(.quaternary.opacity(0.18))
    }

    // MARK: - Tabs

    private var tabBar: some View {
        HStack(spacing: 4) {
            tabButton(.properties, "Properties", systemImage: "slider.horizontal.3")
            tabButton(.output, "Output", systemImage: "sidebar.right")
            Spacer()
        }
        .padding(.bottom, 8)
    }

    private func tabButton(_ target: Tab, _ title: String, systemImage: String) -> some View {
        let isActive = tab == target
        return Button {
            tab = target
        } label: {
            Label(title, systemImage: systemImage)
                .font(.subheadline.weight(.semibold))
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(isActive ? Color.accentColor.opacity(0.15) : Color.clear,
                            in: Capsule())
                .foregroundStyle(isActive ? Color.accentColor : Color.secondary)
        }
        .buttonStyle(.plain)
    }

    // MARK: - Output tab

    private var outputPane: some View {
        VStack(alignment: .leading, spacing: 12) {
            outputHeader
            Divider()
            if let savedFile, let savedKind {
                savedContent(for: savedFile, kind: savedKind)
            } else if let output {
                if let item = output.items.first {
                    content(for: item)
                } else {
                    placeholder("No output")
                }
            } else {
                placeholder("Run the flow, then select a row to inspect its output.")
            }
            if let substitutionNote {
                Label(substitutionNote, systemImage: "arrow.triangle.swap")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
            Spacer(minLength: 0)
        }
    }

    private var outputHeader: some View {
        HStack(spacing: 6) {
            Image(systemName: "sidebar.right")
                .foregroundStyle(.secondary)
            Text(rowTitle.isEmpty ? "Inspector" : rowTitle)
                .font(.headline)
            Spacer()
        }
    }

    // MARK: - Per-modality presentation

    @ViewBuilder
    private func content(for item: Item) -> some View {
        switch item.kind {
        case .audio:
            audioContent(item)
        case .text:
            textContent(item)
        case .status:
            statusContent(item)
        case .image:
            imageContent(item)
        default:
            placeholder("This \(item.kind.rawValue) output has no viewer in this version.")
        }
    }

    /// Present a `Save *` row's written file with the same viewers as a live output, plus a
    /// "Show in Finder" affordance.
    @ViewBuilder
    private func savedContent(for url: URL, kind: Kind) -> some View {
        let item = Item(kind: kind, value: nil, path: url, sourceText: nil)
        switch kind {
        case .audio:
            VStack(alignment: .leading, spacing: 6) {
                audioContent(item)
                openInFinderButton(url)
            }
        case .text:
            VStack(alignment: .leading, spacing: 6) {
                textContent(item)
                openInFinderButton(url)
            }
        case .image:
            VStack(alignment: .leading, spacing: 6) {
                imageContent(item)
                openInFinderButton(url)
            }
        default:
            VStack(alignment: .leading, spacing: 6) {
                fileContent(url)
                openInFinderButton(url)
            }
        }
    }

    private func fileContent(_ url: URL) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Saved file", systemImage: "doc")
                .font(.subheadline.bold())
            Text(url.lastPathComponent)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func openInFinderButton(_ url: URL) -> some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } label: {
            Label("Show in Finder", systemImage: "folder")
        }
        .controlSize(.small)
    }

    private func audioContent(_ item: Item) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: "waveform")
                    .foregroundStyle(.secondary)
                Text("Audio").font(.subheadline.bold())
                Spacer()
                Button { play(item) } label: {
                    Label("Play", systemImage: "play.fill")
                }
                .disabled(item.path == nil)
            }
            if let path = item.path {
                Text(path.lastPathComponent)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func textContent(_ item: Item) -> some View {
        let text = item.value
            ?? (item.path.flatMap { try? String(contentsOf: $0, encoding: .utf8) })
            ?? ""
        return ScrollView {
            Text(text)
                .font(.body)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func statusContent(_ item: Item) -> some View {
        HStack(spacing: 8) {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
            Text(item.value ?? "")
                .font(.callout)
                .textSelection(.enabled)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func imageContent(_ item: Item) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Image").font(.subheadline.bold())
            if let path = item.path,
               let data = try? Data(contentsOf: path),
               let cg = ImageLoader.decodedCGImage(from: data) {
                Image(nsImage: NSImage(cgImage: cg, size: .zero))
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: .infinity, maxHeight: 220)
            } else {
                placeholder("Couldn't load the image.")
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary.opacity(0.4), in: RoundedRectangle(cornerRadius: 8))
    }

    private func placeholder(_ text: String) -> some View {
        Text(text)
            .font(.callout)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .padding(8)
    }

    // MARK: - Audio playback (the same AVAudioPlayer-on-WAVData the run views use)

    private func play(_ item: Item) {
        guard let path = item.path else { return }
        do {
            let buffer = try AudioFileReader.read(path)
            let data = AudioWriter.wavData(buffer)
            let newPlayer = try AVAudioPlayer(data: data)
            self.player = newPlayer        // retain so it isn't deallocated mid-play
            newPlayer.play()
        } catch {
            // No error surface in the pane — silence is fine for an inspect pane.
        }
    }
}
