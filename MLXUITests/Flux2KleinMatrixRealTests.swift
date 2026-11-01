import AppKit
import CoreGraphics
import Foundation
import MLX
import Testing
@testable import MLXUI

// MP7-4, smoke row 140, OPT-IN: the 12-image matrix. 4 prompts (2 short under 20 tokens, 1 long, 1 CJK) x 3 configs, SAME seed, 1024 x 1024, 4 steps:
//   A  DiT 8 + text encoder 8-bit (the production default)
//   B  DiT 8 + text encoder bf16  (test-only switch: `Flux2KleinInstalledStages(modelDirectory:encoderUnquantizedForTesting:)`)
//   C  DiT 4 + text encoder 8-bit
// Writes one PNG per image (named prompt__config), a labelled contact sheet, `matrix-results.md` and `matrix-results.json` (peak footprint and MLX peak per config, time per
// stage) into TEST_RUNNER_MLXUI_TEST_KLEIN_MATRIX_OUT. Unsandboxed Direct entitlements (reads ~/Library/Application Support/AI Browser or TEST_RUNNER_MLXUI_TEST_KLEIN_DIR).
// ONE process, the app CLOSED (`pgrep -fl MLXUI` empty). Stop line: 14 GB footprint in any run. The test does NOT decide encoder 8 vs bf16 or whether int4 stays: it records;
// the owner decides from the pictures.

enum Klein4Matrix {
    nonisolated static let out: URL? = Klein3hReal.env["MLXUI_TEST_KLEIN_MATRIX_OUT"].map { URL(fileURLWithPath: $0, isDirectory: true) }
    nonisolated static let enabled = Klein3hReal.installedAllComponents && out != nil
    nonisolated static let skip: Comment = "needs the installed Klein model, TEST_RUNNER_MLXUI_TEST_KLEIN_MATRIX_OUT=<dir> and an unsandboxed host; 12 full runs, about 10 minutes, app closed"
    nonisolated static let seed: UInt64 = 7
    nonisolated static let size = 1024
    nonisolated static let steps = 4

    struct PromptCase: Sendable { let id: String; let label: String; let text: String }
    struct Config: Sendable { let id: String; let label: String; let quantize: Int; let encoderBF16: Bool }

    nonisolated static let prompts: [PromptCase] = [
        .init(id: "p1-short-fox", label: "short 1", text: "a red fox in the snow"),
        .init(id: "p2-short-lighthouse", label: "short 2", text: "a lighthouse at night"),
        .init(id: "p3-long-market", label: "long", text: "A bustling open-air night market in a narrow old-town street, photographed from slightly above: rows of wooden stalls hung with paper lanterns in warm orange and red, steam rising from a vendor's pot of noodles in the foreground, a woman in a green raincoat holding a yellow umbrella while she hands coins to a smiling street-food seller, wet cobblestones reflecting the neon signs, a small black cat watching from a crate of oranges, shallow depth of field, cinematic colour grading, fine film grain, highly detailed, natural light mixed with lantern glow."),
        .init(id: "p4-cjk-fox", label: "CJK", text: "一只红狐狸站在雪地森林里，清晨的阳光穿过树林，写实摄影风格"),
    ]
    nonisolated static let configs: [Config] = [
        .init(id: "dit8-enc8", label: "DiT 8 + encoder 8 (default)", quantize: 8, encoderBF16: false),
        .init(id: "dit8-encbf16", label: "DiT 8 + encoder bf16", quantize: 8, encoderBF16: true),
        .init(id: "dit4-enc8", label: "DiT 4 + encoder 8", quantize: 4, encoderBF16: false),
    ]

    struct RunRecord: Codable {
        var prompt: String, config: String, file: String
        var promptTokens: Int, templateTokens: Int, realTokens: Int
        var encoderSeconds: Double, ditSeconds: Double, vaeSeconds: Double, totalSeconds: Double
        var encoderMLXPeakGB: Double, ditMLXPeakGB: Double, vaeMLXPeakGB: Double
        var footprintPeakGB: Double, footprintBeforeGB: Double
    }

    /// The tokens the prompt itself takes (template removed) and the real length the DiT sees.
    static func tokenCounts(_ text: String, in dir: URL) async throws -> (prompt: Int, template: Int, real: Int) {
        let tokenizer = try await HFTokenizerLoader().load(from: dir.appendingPathComponent("tokenizer"))
        let real = try Flux2KleinPrompt.encode(text, tokenizer: tokenizer).realLength
        let template = try Flux2KleinPrompt.encode("", tokenizer: tokenizer).realLength
        return (real - template, template, real)
    }

    // MARK: Contact sheet

    @MainActor static func contactSheet(images: [String: CGImage], records: [RunRecord]) -> CGImage? {
        let cell = 384, gap = 8, labelW = 230, headH = 44, footH = 30
        let cols = configs.count, rows = prompts.count
        let width = labelW + cols * (cell + gap) + gap
        let height = headH + rows * (cell + gap) + gap + footH
        guard let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                                         colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
              let gctx = NSGraphicsContext(bitmapImageRep: rep) else { return nil }
        let saved = NSGraphicsContext.current
        NSGraphicsContext.current = gctx
        defer { NSGraphicsContext.current = saved }
        NSColor.white.setFill()
        NSRect(x: 0, y: 0, width: width, height: height).fill()
        let head: [NSAttributedString.Key: Any] = [.font: NSFont.boldSystemFont(ofSize: 15), .foregroundColor: NSColor.black]
        let small: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: 12), .foregroundColor: NSColor.darkGray]
        for (c, config) in configs.enumerated() {
            let x = labelW + gap + c * (cell + gap)
            (config.label as NSString).draw(in: NSRect(x: x, y: height - headH + 6, width: cell, height: 32), withAttributes: head)
        }
        for (r, prompt) in prompts.enumerated() {
            let top = height - headH - r * (cell + gap)       // AppKit origin is bottom-left
            let text = "\(prompt.label)\n\n" + (prompt.text.count > 120 ? String(prompt.text.prefix(117)) + "..." : prompt.text)
            (text as NSString).draw(in: NSRect(x: 6, y: top - cell, width: labelW - 12, height: cell), withAttributes: small)
            for (c, config) in configs.enumerated() {
                let x = labelW + gap + c * (cell + gap)
                if let image = images["\(prompt.id)__\(config.id)"] {
                    NSImage(cgImage: image, size: NSSize(width: cell, height: cell)).draw(in: NSRect(x: x, y: top - cell, width: cell, height: cell))
                }
            }
        }
        let foot = "seed \(seed), \(size) x \(size), \(steps) steps, VAE bf16; one row per prompt, one column per configuration"
        (foot as NSString).draw(in: NSRect(x: 6, y: 6, width: width - 12, height: footH - 8), withAttributes: small)
        return rep.cgImage
    }
}

@Suite(.enabled(if: Klein4Matrix.enabled, Klein4Matrix.skip), .serialized)
struct Flux2KleinMatrixRealTests {
    @Test @MainActor func twelveImagesSameSeedThreeConfigsFourPrompts() async throws {
        let dir = try #require(Klein3hReal.modelDirectory)
        let out = try #require(Klein4Matrix.out)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let limitBefore = Memory.cacheLimit
        var images: [String: CGImage] = [:]
        var records: [Klein4Matrix.RunRecord] = []
        var tokenInfo: [String: (prompt: Int, template: Int, real: Int)] = [:]
        for p in Klein4Matrix.prompts { tokenInfo[p.id] = try await Klein4Matrix.tokenCounts(p.text, in: dir) }
        for p in Klein4Matrix.prompts.prefix(2) { #expect(try #require(tokenInfo[p.id]).prompt < 20, "short prompt \(p.id) must be under 20 tokens") }

        for config in Klein4Matrix.configs {
            let stages = Flux2KleinInstalledStages(modelDirectory: dir, encoderUnquantizedForTesting: config.encoderBF16)
            let engine = Flux2KleinEngine(stages: stages)
            for prompt in Klein4Matrix.prompts {
                Memory.clearCache()
                let before = KleinFootprint.gb(KleinFootprint.now())
                let sampler = KleinFootprint(); sampler.start()
                let request = Flux2KleinRequest(prompt: prompt.text, width: Klein4Matrix.size, height: Klein4Matrix.size, steps: Klein4Matrix.steps, seed: Klein4Matrix.seed,
                                                quantize: config.quantize, row: "Matrix")
                let started = Date()
                let result = try await MLXErrorGuard.guardedRun { try await engine.generate(request) }
                let total = Date().timeIntervalSince(started)
                let peak = KleinFootprint.gb(sampler.stop())
                func stage(_ c: Flux2KleinComponent) -> Flux2KleinStageReport? { result.stages.first { $0.component == c } }
                let key = "\(prompt.id)__\(config.id)"
                let file = "\(key).png"
                try Klein3hReal.writePNG(result.image, to: out.appendingPathComponent(file))
                images[key] = result.image
                let tokens = try #require(tokenInfo[prompt.id])
                let record = Klein4Matrix.RunRecord(
                    prompt: prompt.id, config: config.id, file: file, promptTokens: tokens.prompt, templateTokens: tokens.template, realTokens: tokens.real,
                    encoderSeconds: stage(.textEncoder)?.seconds ?? 0, ditSeconds: stage(.transformer)?.seconds ?? 0, vaeSeconds: stage(.vae)?.seconds ?? 0, totalSeconds: total,
                    encoderMLXPeakGB: KleinFootprint.gb(stage(.textEncoder)?.mlxPeakBytes ?? 0), ditMLXPeakGB: KleinFootprint.gb(stage(.transformer)?.mlxPeakBytes ?? 0),
                    vaeMLXPeakGB: KleinFootprint.gb(stage(.vae)?.mlxPeakBytes ?? 0), footprintPeakGB: peak, footprintBeforeGB: before)
                records.append(record)
                print(String(format: "KLEIN4-MATRIX %@ %@: %.1f s (enc %.1f, dit %.1f, vae %.1f), footprint peak %.2f GB, mlx peaks %.2f / %.2f / %.2f GB",
                             prompt.id, config.id, total, record.encoderSeconds, record.ditSeconds, record.vaeSeconds, peak, record.encoderMLXPeakGB, record.ditMLXPeakGB, record.vaeMLXPeakGB))
                #expect(Memory.cacheLimit == limitBefore, "the process-global cache limit must be put back")
                try #require(peak < 14.0, "STOP LINE: footprint peaked at \(peak) GB in \(key)")
            }
        }

        let sheet = try #require(Klein4Matrix.contactSheet(images: images, records: records))
        try Klein3hReal.writePNG(sheet, to: out.appendingPathComponent("contact-sheet.png"))
        try JSONEncoder().encode(records).write(to: out.appendingPathComponent("matrix-results.json"))
        try table(records, tokenInfo: tokenInfo).write(to: out.appendingPathComponent("matrix-results.md"), atomically: true, encoding: .utf8)
        #expect(records.count == 12 && images.count == 12)
        print("KLEIN4-MATRIX DONE: \(out.path)")
    }

    private func table(_ records: [Klein4Matrix.RunRecord], tokenInfo: [String: (prompt: Int, template: Int, real: Int)]) -> String {
        var md = "# MP7-4 matrix: 4 prompts x 3 configs, seed \(Klein4Matrix.seed), \(Klein4Matrix.size) x \(Klein4Matrix.size), \(Klein4Matrix.steps) steps, VAE bf16\n\n"
        md += "## Per configuration (footprint = whole-process phys_footprint peak over that configuration's four runs; MLX peak = largest stage peak)\n\n"
        md += "| config | footprint peak GB | MLX peak GB (enc / DiT / VAE, max over prompts) | mean seconds (enc / DiT incl. load / VAE / total) |\n|---|---|---|---|\n"
        for config in Klein4Matrix.configs {
            let rs = records.filter { $0.config == config.id }
            guard !rs.isEmpty else { continue }
            func mean(_ f: (Klein4Matrix.RunRecord) -> Double) -> Double { rs.map(f).reduce(0, +) / Double(rs.count) }
            func max(_ f: (Klein4Matrix.RunRecord) -> Double) -> Double { rs.map(f).max() ?? 0 }
            md += String(format: "| %@ | %.2f | %.2f / %.2f / %.2f | %.1f / %.1f / %.1f / %.1f |\n", config.label, max { $0.footprintPeakGB }, max { $0.encoderMLXPeakGB },
                         max { $0.ditMLXPeakGB }, max { $0.vaeMLXPeakGB }, mean { $0.encoderSeconds }, mean { $0.ditSeconds }, mean { $0.vaeSeconds }, mean { $0.totalSeconds })
        }
        md += "\n## Per image\n\n| prompt | tokens (prompt / template / real) | config | enc s | DiT s | VAE s | total s | footprint GB | MLX GB enc / DiT / VAE | file |\n|---|---|---|---|---|---|---|---|---|---|\n"
        for r in records {
            md += String(format: "| %@ | %d / %d / %d | %@ | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f / %.2f / %.2f | %@ |\n", r.prompt, r.promptTokens, r.templateTokens, r.realTokens, r.config,
                         r.encoderSeconds, r.ditSeconds, r.vaeSeconds, r.totalSeconds, r.footprintPeakGB, r.encoderMLXPeakGB, r.ditMLXPeakGB, r.vaeMLXPeakGB, r.file)
        }
        md += "\nPrompts:\n"
        for p in Klein4Matrix.prompts { md += "- \(p.id) (\(p.label)): \(p.text)\n" }
        return md
    }
}
