//
//  ModelToolChipView.swift
//  MLXUI — Model Tools, MT4-4 (plan §3.9).
//
//  A model-tool call in the transcript: the "‹Model› · ‹Task› · 2.1 s" chip (click for the input and output handles, the
//  row text and the timing), and under it the output: an audio player, the image, or the video. A text output is read
//  by the model and shown through its answer, so it gets no inline view. Edition-neutral.
//

import SwiftUI
import AppKit
import AVKit

struct ModelToolChipView: View {
    let chip: ModelToolChip
    /// The output asset and where it lives, when the call made one that is still on disk.
    let output: (record: ChatAssetRecord, url: URL)?

    @State private var showsDetails = false

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button {
                showsDetails.toggle()
            } label: {
                HStack(spacing: 5) {
                    if chip.state == .running {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: icon).font(.caption2)
                    }
                    Text(chip.title).font(.caption)
                    if let handle = chip.outputHandle {
                        Text(handle).font(.caption2.monospaced()).foregroundStyle(.secondary)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(tint.opacity(0.14), in: Capsule())
                .overlay(Capsule().strokeBorder(tint.opacity(0.35), lineWidth: 1))
            }
            .buttonStyle(.plain)
            .help("Show the input, output and timing")
            .popover(isPresented: $showsDetails, arrowEdge: .bottom) { detailsPopover }

            if case .failed(let message) = chip.state {
                Text(message)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let output, chip.state == .done {
                ModelToolOutputView(record: output.record, url: output.url)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var tint: Color {
        switch chip.state {
        case .running: return .blue
        case .done: return .green
        case .failed: return .red
        case .declined: return .orange
        }
    }

    private var icon: String {
        switch chip.state {
        case .failed: return "exclamationmark.triangle.fill"
        case .declined: return "hand.raised.fill"
        default: break
        }
        switch chip.toolName {
        case "transcribe_audio": return "waveform"
        case "ocr_image": return "text.viewfinder"
        case "describe_image": return "eye"
        case "speak": return "speaker.wave.2"
        case "translate": return "character.bubble"
        case "generate_image": return "photo"
        case "upscale_image": return "arrow.up.left.and.arrow.down.right"
        case "segment_image": return "scissors"
        case "estimate_depth": return "square.3.layers.3d"
        case "generate_music": return "music.note"
        default: return "wand.and.stars"
        }
    }

    // MARK: - Details

    private var detailsPopover: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(chip.title).font(.headline)
            if chip.details.isEmpty {
                Text("Nothing more to show yet.").font(.caption).foregroundStyle(.secondary)
            } else {
                Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 10, verticalSpacing: 4) {
                    ForEach(Array(chip.details.enumerated()), id: \.offset) { _, line in
                        GridRow {
                            Text(line.label).font(.caption).foregroundStyle(.secondary)
                            Text(line.value).font(.caption.monospaced()).textSelection(.enabled)
                        }
                    }
                }
            }
            if let output {
                HStack(spacing: 10) {
                    Button("Open") { NSWorkspace.shared.open(output.url) }
                    Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([output.url]) }
                }
                .buttonStyle(.link)
                .font(.caption)
            }
        }
        .padding(12)
        .frame(minWidth: 240, maxWidth: 360, alignment: .leading)
    }
}

/// The inline rendering of one output asset.
struct ModelToolOutputView: View {
    let record: ChatAssetRecord
    let url: URL

    var body: some View {
        switch record.kind {
        case "audio":
            InlineAudioPlayer(url: url)
        case "image":
            InlineImageView(url: url)
        case "video":
            InlineVideoView(url: url)
        default:
            EmptyView()
        }
    }
}

/// A downsampled thumbnail, decoded off the main actor; a click opens the full file.
private struct InlineImageView: View {
    let url: URL
    @State private var image: NSImage?
    @State private var missing = false

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 360, maxHeight: 280)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .onTapGesture { NSWorkspace.shared.open(url) }
                    .help("Open")
            } else if missing {
                Label("This image is no longer on disk.", systemImage: "photo")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .task(id: url) {
            if let loaded = await Self.load(url) { image = loaded.image } else { missing = true }
        }
    }

    private struct Loaded: @unchecked Sendable { let image: NSImage }

    @concurrent private static func load(_ url: URL) async -> Loaded? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 1024,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
        return Loaded(image: NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height)))
    }
}

/// `VideoPreview` (an `AVPlayerView`), never SwiftUI's `VideoPlayer`, which aborted the process on macOS 27.
private struct InlineVideoView: View {
    let url: URL
    @State private var player: AVPlayer?

    var body: some View {
        Group {
            if FileManager.default.fileExists(atPath: url.path), let player {
                VideoPreview(player: player)
                    .frame(width: 360, height: 220)
                    .clipShape(RoundedRectangle(cornerRadius: 8))
            } else if !FileManager.default.fileExists(atPath: url.path) {
                Label("This video is no longer on disk.", systemImage: "film")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .onAppear { if player == nil { player = AVPlayer(url: url) } }
        .onDisappear { player?.pause() }
    }
}
