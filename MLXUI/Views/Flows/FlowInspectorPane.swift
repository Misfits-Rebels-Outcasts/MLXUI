import SwiftUI
import AVFoundation
import AppKit
import UniformTypeIdentifiers

/// The inspector pane (CFM-R3-3): a right-hand pane bound to the selected row, split into
/// three tabs.
///
/// - **Flow** (FH-3): the flow's own identity — name, header, version, extension, row count —
///   moved here out of the editor's toolbar. Editable only in the editor (`FlowTabInfo
///   .editable`); the My Workflows list and both gallery shelves render it read-only.
/// - **Step** (renamed from Properties, FH-3 — a pane that showed "Properties" for both a row
///   and the flow itself would be unreadable): the row's editable details — model, instruction,
///   inputs, settings, decisions. The flow editor embeds a live `FlowRowInspectorView`; the
///   read-only flow list embeds a frozen one, so a gallery flow's properties are browsable
///   (scrollable) but never mutable.
/// - **Output**: the selected row's cached `Asset`, or a `Save *` row's written file,
///   presented per modality the way the run views do — the presentation half of
///   `TTSRunView`/`ASRRunView`/`OCRRunView`/`ImageQARunView`/`EmbeddingRunView`
///   (design doc §6), extracted rather than duplicated, and the standalone Run sheets are
///   untouched. Also surfaces the `CatalogBridge` substitution note (CFM-R2-2 rule 3) for a
///   row whose model is `.sameFamily`/`.substitute`.
struct FlowInspectorPane<Properties: View>: View {
    enum Tab: Hashable {
        case flow, step, output
    }

    /// FH-3: the flow's own identity, shown in the Flow tab.
    let flowInfo: FlowTabInfo
    /// The selected row's cached output, or nil when nothing is selected / no run yet.
    let output: Asset?
    /// The row's title (task name) for the output pane header.
    let rowTitle: String
    /// A `CatalogBridge` substitution note (e.g. "running Kokoro 82M as …"), or nil.
    let substitutionNote: String?
    /// DA-10-FIX-1: the selected row's failure or skip sentence — shown prominently at the top
    /// of the Output tab, since a skipped/failed row has no output of its own. `isSkip` picks
    /// the amber "Skipped" framing over the red "Failed" one.
    var statusNote: (text: String, isSkip: Bool)?
    /// The file a selected `Save *` row wrote (resolved by the caller), so the saved result
    /// is playable/viewable rather than just a status sentence.
    var savedFile: URL?
    /// The saved file's kind, driving the presentation.
    var savedKind: Kind?
    /// OV-1: the saved file's presentation (by its own extension, task name as fallback) —
    /// drives whether the Output tab offers a Quick Look button (OV-2).
    var savedPresentation: SavedFilePresentation?
    /// The row-properties pane — editable in the editor, frozen in the read-only flow list.
    private let properties: () -> Properties

    @State private var tab: Tab
    @State private var audio = InspectorAudioController()
    @State private var isShowingQuickLook = false

    init(flowInfo: FlowTabInfo, output: Asset?, rowTitle: String, substitutionNote: String?,
         statusNote: (text: String, isSkip: Bool)? = nil,
         savedFile: URL? = nil, savedKind: Kind? = nil,
         savedPresentation: SavedFilePresentation? = nil,
         initialTab: Tab = .output,
         @ViewBuilder properties: @escaping () -> Properties) {
        self.flowInfo = flowInfo
        self.output = output
        self.rowTitle = rowTitle
        self.substitutionNote = substitutionNote
        self.statusNote = statusNote
        self.savedFile = savedFile
        self.savedKind = savedKind
        self.savedPresentation = savedPresentation
        self.properties = properties
        _tab = State(initialValue: initialTab)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            tabBar
            Divider()
                .padding(.bottom, 10)
            switch tab {
            case .flow:
                flowPane
            case .step:
                properties()
            case .output:
                outputPane
            }
        }
        .padding(14)
        .frame(minWidth: 260, maxWidth: 320, maxHeight: .infinity, alignment: .topLeading)
        .background(.quaternary.opacity(0.18))
        // Stop playback when the inspected row changes (or a re-run swaps the output) and
        // when the pane goes away, so the Play/Stop button never lies about a stale player.
        .onChange(of: displayedAudioPath) { audio.stop() }
        .onDisappear { audio.stop() }
        .sheet(isPresented: $isShowingQuickLook) {
            if let savedFile { QuickLookSheet(url: savedFile) }
        }
    }

    /// The audio file the Output tab is currently showing a Play button for, if any.
    private var displayedAudioPath: URL? {
        if let savedFile, savedKind == .audio { return savedFile }
        if let item = output?.items.first, item.kind == .audio { return item.path }
        return nil
    }

    // MARK: - Tabs

    private var tabBar: some View {
        HStack(spacing: 4) {
            tabButton(.flow, "Flow", systemImage: "flowchart")
            tabButton(.step, "Step", systemImage: "slider.horizontal.3")
            tabButton(.output, "Output", systemImage: "sidebar.right")
            Spacer()
        }
        .padding(.bottom, 8)
    }

    // MARK: - Flow tab (FH-3)

    private var flowPane: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                TextField("Flow name", text: flowInfo.name)
                    .textFieldStyle(.roundedBorder)
                    .disabled(!flowInfo.editable)
                HStack(spacing: 6) {
                    Text("\(flowInfo.headerKeyword) \(flowInfo.version)")
                        .font(.caption.monospaced())
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary.opacity(0.6), in: RoundedRectangle(cornerRadius: 5))
                    Text("." + flowInfo.fileExtension)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                Text("\(flowInfo.rowCount) row\(flowInfo.rowCount == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let savedURL = flowInfo.savedURL {
                    Text(savedURL.lastPathComponent)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
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
            if let statusNote {
                Label(statusNote.text,
                      systemImage: statusNote.isSkip ? "exclamationmark.triangle" : "xmark.octagon.fill")
                    .font(.callout)
                    .foregroundStyle(statusNote.isSkip ? .orange : .red)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background((statusNote.isSkip ? Color.orange : Color.red).opacity(0.09),
                               in: RoundedRectangle(cornerRadius: 6))
            }
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

    /// Present a `Save *` row's written file with the same viewers as a live output, plus
    /// (OV-5) the shared action row every branch gets alike.
    @ViewBuilder
    private func savedContent(for url: URL, kind: Kind) -> some View {
        let item = Item(kind: kind, value: nil, path: url, sourceText: nil)
        let presentation = savedPresentation ?? .other
        switch kind {
        case .audio:
            VStack(alignment: .leading, spacing: 6) {
                audioContent(item)
                savedFileActions(url: url, presentation: presentation)
            }
        case .text:
            VStack(alignment: .leading, spacing: 6) {
                textContent(item)
                savedFileActions(url: url, presentation: presentation)
            }
        case .image:
            VStack(alignment: .leading, spacing: 6) {
                imageContent(item)
                savedFileActions(url: url, presentation: presentation)
            }
        default:
            VStack(alignment: .leading, spacing: 6) {
                fileContent(url, isFolder: presentation == .folder)
                savedFileActions(url: url, presentation: presentation)
            }
        }
    }

    /// OV-5 (revised after owner feedback on the shipped four-button row reading as crowded):
    /// one primary button plus a "···" overflow, instead of a fixed row or a two-line wrap.
    /// Quick Look is the backlog's own "primary affordance" (OV-2 — it needs no LaunchServices
    /// hand-off), so it stays a visible button; "Open in ‹app›", "Export a Copy…" and Show in
    /// Finder collapse into one menu. `.folder` has no Quick Look, so Show in Finder — its only
    /// action — is the visible button there, not tucked behind a menu for one item.
    @ViewBuilder
    private func savedFileActions(url: URL, presentation: SavedFilePresentation) -> some View {
        HStack(spacing: 8) {
            if presentation.allowsQuickLook {
                quickLookButton
                moreActionsMenu(url: url, presentation: presentation)
            } else {
                openInFinderButton(url)
            }
        }
    }

    /// Everything besides Quick Look, behind one "···" menu — Open in ‹app› (only when macOS
    /// has a handler), Export a Copy…, Show in Finder, in that order.
    @ViewBuilder
    private func moreActionsMenu(url: URL, presentation: SavedFilePresentation) -> some View {
        Menu {
            if presentation.allowsOpenInApp,
               let appURL = NSWorkspace.shared.urlForApplication(toOpen: url),
               let label = openLabel(appDisplayName: FileManager.default.displayName(atPath: appURL.path)) {
                openInAppButton(url: url, label: label)
            }
            if presentation.allowsExport {
                exportCopyButton(url)
            }
            openInFinderButton(url)
        } label: {
            Image(systemName: "ellipsis.circle")
        }
        .menuStyle(.borderlessButton)
        .help("More actions for this saved file")
    }

    private var quickLookButton: some View {
        Button {
            isShowingQuickLook = true
        } label: {
            Label("Quick Look", systemImage: "eye")
        }
        .controlSize(.small)
    }

    /// OV-3's button — always rendered when called; the caller (`actionButtons`) is what
    /// decides whether it belongs in the row at all, so hiding it for "no handler" or
    /// `.folder` never means a disabled button, only its absence.
    private func openInAppButton(url: URL, label: String) -> some View {
        Button {
            NSWorkspace.shared.open(url)
        } label: {
            Label(label, systemImage: "arrow.up.forward.app")
        }
        .controlSize(.small)
    }

    /// OV-4's button — copies the saved file to wherever the user picks, never moves the
    /// original. `.folder` is excluded by the caller (`actionButtons`): no zip, no recursive
    /// copy, Show in Finder is the folder's answer.
    private func exportCopyButton(_ url: URL) -> some View {
        Button {
            exportCopy(of: url)
        } label: {
            Label("Export a Copy…", systemImage: "square.and.arrow.up")
        }
        .controlSize(.small)
    }

    private func exportCopy(of url: URL) {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = url.lastPathComponent
        if let type = UTType(filenameExtension: url.pathExtension) {
            panel.allowedContentTypes = [type]
        }
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        try? FileManager.default.copyItem(at: url, to: destination)
    }

    /// OV-5: `.folder`'s card reads "Saved folder" (§9, mirrors the existing "Saved file")
    /// — everything else that reaches this generic card (today, only a saved video) keeps the
    /// existing title. Icon unchanged either way — §9 names only the title string.
    private func fileContent(_ url: URL, isFolder: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Label(isFolder ? "Saved folder" : "Saved file", systemImage: "doc")
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
                Button { audio.toggle(item.path) } label: {
                    let playing = audio.isPlaying(item.path)
                    Label(playing ? "Stop" : "Play",
                          systemImage: playing ? "stop.fill" : "play.fill")
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

}

/// Plays one audio file at a time for the Output tab, with a Play/Stop toggle that flips
/// back to "Play" the moment playback ends. One instance per `FlowInspectorPane`; the pane
/// stops it when the inspected row changes or the pane disappears. Same
/// `AVAudioPlayer`-on-WAV-data path the run views use; mirrors `MusicGenRunModel`'s
/// `AVAudioPlayerDelegate` reset.
@MainActor
@Observable
final class InspectorAudioController: NSObject, AVAudioPlayerDelegate {
    /// The file currently playing, or nil. Drives the button's Play/Stop label.
    private(set) var playingPath: URL?
    private var player: AVAudioPlayer?

    /// True when `path` is the file playing right now.
    func isPlaying(_ path: URL?) -> Bool {
        path != nil && playingPath == path
    }

    /// Stop if `path` is already playing, otherwise start it (replacing whatever was playing).
    func toggle(_ path: URL?) {
        guard let path else { return }
        if playingPath == path {
            stop()
            return
        }
        stop()
        do {
            let buffer = try AudioFileReader.read(path)
            let newPlayer = try AVAudioPlayer(data: AudioWriter.wavData(buffer))
            newPlayer.delegate = self       // reset the toggle when playback completes
            self.player = newPlayer          // retain so it isn't deallocated mid-play
            newPlayer.play()
            self.playingPath = path
        } catch {
            // No error surface in the pane — silence is fine for an inspect pane.
        }
    }

    func stop() {
        player?.stop()
        player = nil
        playingPath = nil
    }

    // MARK: - AVAudioPlayerDelegate

    /// Playback reached the end — flip the button back to "Play".
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        stop()
    }
}

/// FH-3: the flow's own identity, decoupled from `FlowEditorModel` — a small value type so the
/// read-only flow list can feed the same Flow tab the editor does (fact 3,
/// `RSI/DelegateOutputViewerBacklog.md`). `name` is a real `Binding`, not a plain `String`: in
/// the editor it's `$model.name` itself, so typing in the Flow tab writes straight into the
/// same property `FlowEditorModel.save()`'s stale-sibling guard reads — never a disconnected
/// copy that would silently stop the guard from firing (§7 trap 8). The read-only list passes
/// `.constant(_:)`; `editable` disables the field either way.
struct FlowTabInfo {
    var name: Binding<String>
    var headerKeyword: String
    var version: String
    var fileExtension: String
    var savedURL: URL?
    var rowCount: Int
    /// FH-4 renders these as checkbox rows with reasons; FH-3 only carries them through so
    /// that later work doesn't need to touch either call site again.
    var flagsOrder: [CapabilityFlag]
    var editable: Bool
}
