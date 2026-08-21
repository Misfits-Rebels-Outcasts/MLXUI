import SwiftUI
import AVFoundation
import AppKit

/// The inspector pane (CFM-R3-3): a right-hand pane bound to the selected row's cached
/// `Asset`, presenting each output modality the way the run views do — the presentation
/// half of `TTSRunView`/`ASRRunView`/`OCRRunView`/`ImageQARunView`/`EmbeddingRunView`
/// (design doc §6), extracted rather than duplicated, and the standalone Run sheets are
/// untouched. Also surfaces the `CatalogBridge` substitution note (CFM-R2-2 rule 3) for a
/// row whose model is `.sameFamily`/`.substitute`.
struct FlowInspectorPane: View {
    /// The selected row's cached output, or nil when nothing is selected / no run yet.
    let output: Asset?
    /// The row's title (task name) for the pane header.
    let rowTitle: String
    /// A `CatalogBridge` substitution note (e.g. "running Kokoro 82M as …"), or nil.
    let substitutionNote: String?

    @State private var player: AVAudioPlayer?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            if let output {
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
        .padding(14)
        .frame(minWidth: 260, maxWidth: 320, maxHeight: .infinity, alignment: .topLeading)
        .background(.quaternary.opacity(0.18))
    }

    // MARK: - Header

    private var header: some View {
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
        ScrollView {
            Text(item.value ?? "")
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
