import Foundation
import AVFoundation
import CoreGraphics

// CFM-R12-7 group e — the video/audio tools, **AVFoundation subset** (owner ruling 2026-08-25: "AVFoundation subset (Recommended)" —
// no ffmpeg, no entitlement). Trim / Extract Frame / Extract Audio / Mux for common formats;
// `Read Video` is file-backed; `Join Video` was the last unported instant tool (R13-3).

/// Shared AVFoundation plumbing for the video tools.
nonisolated enum VideoTools {
    /// Parse an ffmpeg-style time ("00:00:05", "1:30", "0", "1.5") to seconds.
    static func parseTime(_ text: String) -> Double? {
        let parts = text.split(separator: ":")
        switch parts.count {
        case 1: return Double(parts[0])
        case 2:
            guard let m = Double(parts[0]), let s = Double(parts[1]) else { return nil }
            return m * 60 + s
        case 3:
            guard let h = Double(parts[0]), let m = Double(parts[1]), let s = Double(parts[2]) else { return nil }
            return h * 3600 + m * 60 + s
        default: return nil
        }
    }

    static func cmTime(_ seconds: Double, asset: AVAsset) -> CMTime {
        let scale = asset.duration.timescale > 0 ? asset.duration.timescale : 600
        return CMTime(seconds: seconds, preferredTimescale: scale)
    }

    /// Await an export session to completion.
    static func export(_ session: AVAssetExportSession) async throws {
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            session.exportAsynchronously {
                switch session.status {
                case .completed: cont.resume()
                case .cancelled: cont.resume(throwing: FlowError.stageFailure(row: "video", message: "cancelled"))
                default:
                    cont.resume(throwing: session.error ?? FlowError.stageFailure(row: "video", message: "export failed"))
                }
            }
        }
    }
}

/// `Read Video` (file → video): resolve the path and return a file-backed `.video` item.
nonisolated struct ReadVideoTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.video) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        guard let raw = FlowSettings(settings).pathValue() else {
            throw FlowError.missingInlineValue(row: "Read Video", kind: .file)
        }
        let url = try workspace.resolve(raw, flowID: flowID)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw FlowError.fileReadFailed(row: "Read Video", path: raw)
        }
        progress(1.0)
        return Asset(items: [Item(kind: .video, value: nil, path: url, sourceText: nil)])
    }
}

/// `Extract Frame` (video → image): grab one frame at `at=` (default the first frame).
nonisolated struct ExtractFrameTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.video) }
    var produces: Shape { .single(.image) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let path = try input.items.first?.path ?? { throw FlowError.missingInlineValue(row: "Extract Frame", kind: .video) }()
        let at = FlowSettings(settings).value(for: "at") ?? FlowSettings(settings).firstBare() ?? "0"
        guard let seconds = VideoTools.parseTime(at) else {
            throw FlowError.invalidSettings(row: "Extract Frame", setting: "at", detail: "isn't a time")
        }
        let asset = AVURLAsset(url: path)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        let time = VideoTools.cmTime(seconds, asset: asset)
        let cg: CGImage
        do {
            cg = try generator.copyCGImage(at: time, actualTime: nil)
        } catch {
            throw FlowError.stageFailure(row: "Extract Frame", message: "couldn't grab a frame (\(error.localizedDescription))")
        }
        progress(0.8)
        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let out = blobDir.appendingPathComponent("frame-\(UUID().uuidString).png")
        guard let data = PNGEncoder.pngData(from: cg) else {
            throw FlowError.writeFailed(row: "Extract Frame", path: out.lastPathComponent)
        }
        try data.write(to: out)
        progress(1.0)
        return Asset(items: [Item(kind: .image, value: nil, path: out, sourceText: nil)])
    }
}

/// `Extract Audio` (video → audio): export the audio track as AAC `.m4a`.
nonisolated struct ExtractAudioTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.video) }
    var produces: Shape { .single(.audio) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let path = try input.items.first?.path ?? { throw FlowError.missingInlineValue(row: "Extract Audio", kind: .video) }()
        let asset = AVURLAsset(url: path)
        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let out = blobDir.appendingPathComponent("audio-\(UUID().uuidString).m4a")
        guard let session = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetAppleM4A) else {
            throw FlowError.stageFailure(row: "Extract Audio", message: "no compatible export preset")
        }
        session.outputURL = out
        session.outputFileType = .m4a
        try await VideoTools.export(session)
        progress(1.0)
        return Asset(items: [Item(kind: .audio, value: nil, path: out, sourceText: nil)])
    }
}

/// `Trim` (audio/video → same): re-encode the `in=`/`out=` time range.
nonisolated struct TrimTool: AssetStage {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    var accepts: Shape { .single(.file) }
    var produces: Shape { .single(.file) }

    func run(_ input: Asset, progress: @Sendable @escaping (Double) -> Void) async throws -> Asset {
        let item = try input.items.first ?? { throw FlowError.missingInlineValue(row: "Trim", kind: .file) }()
        let path = try item.path ?? { throw FlowError.missingInlineValue(row: "Trim", kind: .file) }()
        let s = FlowSettings(settings)
        let start = s.value(for: "in"), end = s.value(for: "out")
        guard start != nil || end != nil else {
            throw FlowError.stageFailure(row: "Trim", message: "needs at least one of `in=`/`out=`")
        }
        let asset = AVURLAsset(url: path)
        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let isAudio = item.kind == .audio
        let out = blobDir.appendingPathComponent("trim-\(UUID().uuidString).\(isAudio ? "m4a" : "mp4")")
        let preset = isAudio ? AVAssetExportPresetAppleM4A : AVAssetExportPresetHighestQuality
        guard let session = AVAssetExportSession(asset: asset, presetName: preset) else {
            throw FlowError.stageFailure(row: "Trim", message: "no compatible export preset")
        }
        var range = CMTimeRange(start: .zero, end: asset.duration)
        let duration = asset.duration.seconds
        if let start, let t = VideoTools.parseTime(start) {
            range = CMTimeRange(start: VideoTools.cmTime(min(t, duration), asset: asset), duration: range.duration)
        }
        if let end, let t = VideoTools.parseTime(end) {
            let length = max(t - range.start.seconds, 0)
            range = CMTimeRange(start: range.start, duration: CMTime(seconds: length, preferredTimescale: range.duration.timescale))
        }
        session.timeRange = range
        session.outputURL = out
        session.outputFileType = isAudio ? .m4a : .mp4
        try await VideoTools.export(session)
        progress(1.0)
        return Asset(items: [Item(kind: item.kind, value: nil, path: out, sourceText: nil)])
    }
}

/// `Mux` (video+audio → video): combine a video and an audio track into one clip.
nonisolated struct MuxTool {
    let workspace: FlowWorkspace
    let flowID: String
    let settings: String

    func run(inputs: [Asset]) async throws -> Asset {
        let allItems = inputs.flatMap { $0.items }
        guard let videoItem = allItems.first(where: { $0.kind == .video }),
              let audioItem = allItems.first(where: { $0.kind == .audio }),
              let videoPath = videoItem.path, let audioPath = audioItem.path else {
            throw FlowError.badInputCardinality(row: "Mux", expected: "(video, audio)", got: allItems.count)
        }
        let videoAsset = AVURLAsset(url: videoPath)
        let audioAsset = AVURLAsset(url: audioPath)

        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid),
              let sourceVideo = try await videoAsset.loadTracks(withMediaType: .video).first else {
            throw FlowError.stageFailure(row: "Mux", message: "no video track")
        }
        try videoTrack.insertTimeRange(CMTimeRange(start: .zero, duration: videoAsset.duration),
                                       of: sourceVideo, at: .zero)
        if let sourceAudio = try await audioAsset.loadTracks(withMediaType: .audio).first,
           let audioTrack = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) {
            try audioTrack.insertTimeRange(CMTimeRange(start: .zero, duration: audioAsset.duration),
                                           of: sourceAudio, at: .zero)
        }

        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let out = blobDir.appendingPathComponent("mux-\(UUID().uuidString).mp4")
        guard let session = AVAssetExportSession(asset: composition, presetName: AVAssetExportPresetHighestQuality) else {
            throw FlowError.stageFailure(row: "Mux", message: "no compatible export preset")
        }
        session.outputURL = out
        session.outputFileType = .mp4
        try await VideoTools.export(session)
        return Asset(items: [Item(kind: .video, value: nil, path: out, sourceText: nil)])
    }
}

/// `Join Video` (video list → video): concatenate clips in list order — the last unported
/// instant tool (R13-3). Same-size clips insert into one `AVMutableComposition` (video +
/// audio) and export re-encoded as `.mp4`. A clip whose **display size** disagrees with the
/// first is refused with a plain sentence naming the mismatch — never silently letterboxed,
/// cropped, or dropped. The Python's `ffmpeg -c copy` concat has the same constraint: the
/// concat demuxer requires identical parameters, and ffmpeg refuses loudly when they differ
/// (the App Store build has no ffmpeg, so this is the AVFoundation equivalent of that
/// loud refusal, ahead of any export work).
nonisolated struct JoinVideoTool {
    let workspace: FlowWorkspace
    let flowID: String

    func run(inputs: [Asset]) async throws -> Asset {
        let paths = inputs.flatMap { $0.items }.compactMap { $0.path }
        guard paths.count >= 2 else {
            throw FlowError.stageFailure(row: "Join Video",
                                         message: "needs at least two videos to join")
        }

        // Load every clip's video track and display size up front, so the honest refusal
        // happens before any composition work (rule 5: refuse, never approximate).
        var clips: [(url: URL, track: AVAssetTrack, size: CGSize)] = []
        for (index, path) in paths.enumerated() {
            let asset = AVURLAsset(url: path)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw FlowError.stageFailure(row: "Join Video",
                                             message: "clip \(index + 1) has no video track")
            }
            let natural = try await track.load(.naturalSize)
            let transform = try await track.load(.preferredTransform)
            let rotated = natural.applying(transform)
            let size = CGSize(width: abs(rotated.width), height: abs(rotated.height))
            clips.append((path, track, size))
        }
        let first = clips[0].size
        for clip in clips.dropFirst() where abs(clip.size.width - first.width) > 1
            || abs(clip.size.height - first.height) > 1 {
            throw FlowError.stageFailure(row: "Join Video",
                message: "the clips don't match — clip 1 is \(Int(first.width))×\(Int(first.height)), "
                    + "another is \(Int(clip.size.width))×\(Int(clip.size.height)). Join Video needs same-size clips.")
        }

        let composition = AVMutableComposition()
        guard let videoTrack = composition.addMutableTrack(withMediaType: .video,
                                                           preferredTrackID: kCMPersistentTrackID_Invalid) else {
            throw FlowError.stageFailure(row: "Join Video", message: "couldn't build the composition")
        }
        var cursor = CMTime.zero
        var audioTrack: AVMutableCompositionTrack?  // added only if any clip carries audio
        for clip in clips {
            let asset = AVURLAsset(url: clip.url)
            let duration = try await asset.load(.duration)
            let range = CMTimeRange(start: .zero, duration: duration)
            if let sourceVideo = try await asset.loadTracks(withMediaType: .video).first {
                try videoTrack.insertTimeRange(range, of: sourceVideo, at: cursor)
            }
            if let sourceAudio = try await asset.loadTracks(withMediaType: .audio).first {
                if audioTrack == nil {
                    audioTrack = composition.addMutableTrack(withMediaType: .audio,
                                                             preferredTrackID: kCMPersistentTrackID_Invalid)
                }
                if let audioTrack {
                    try audioTrack.insertTimeRange(range, of: sourceAudio, at: cursor)
                }
            }
            cursor = CMTimeAdd(cursor, duration)
        }

        let blobDir = workspace.directory(for: flowID).appendingPathComponent(".blobs")
        try FileManager.default.createDirectory(at: blobDir, withIntermediateDirectories: true)
        let out = blobDir.appendingPathComponent("joined-\(UUID().uuidString).mp4")
        guard let session = AVAssetExportSession(asset: composition,
                                                 presetName: AVAssetExportPresetHighestQuality) else {
            throw FlowError.stageFailure(row: "Join Video", message: "no compatible export preset")
        }
        // An explicit video composition — AVFoundation can't always infer render size /
        // frame duration for a multi-insert composition, and a plain export then fails with
        // "The operation is not supported for this media." The render size is the first
        // clip's (the size check guarantees they all agree).
        let fps = max(Int(clips[0].track.nominalFrameRate), 1)
        let videoComposition = AVMutableVideoComposition()
        videoComposition.renderSize = clips[0].size
        videoComposition.frameDuration = CMTime(value: 1, timescale: CMTimeScale(fps))
        let instruction = AVMutableVideoCompositionInstruction()
        instruction.timeRange = CMTimeRange(start: .zero, duration: cursor)
        let layerInstruction = AVMutableVideoCompositionLayerInstruction(assetTrack: videoTrack)
        instruction.layerInstructions = [layerInstruction]
        videoComposition.instructions = [instruction]
        session.videoComposition = videoComposition
        session.outputURL = out
        session.outputFileType = .mp4
        try await VideoTools.export(session)
        return Asset(items: [Item(kind: .video, value: nil, path: out, sourceText: nil)])
    }
}
