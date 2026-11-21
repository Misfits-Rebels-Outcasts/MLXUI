import CryptoKit
import Testing
import Foundation
@testable import MLXUI

/// MC5-3a: the ACE-Step 1.5 catalog entry. Facts are from the pinned HF repo
/// (`ACE-Step/Ace-Step1.5` @ 19671f4, tree and api captured 2026-11-21 under
/// `RSI/notes/modelcandidates/hf/ACE-Step_Ace-Step1.5/`) and the MC5-0b rulings (R3, R7, R8). The engine, the
/// download exclusion of `acestep-5Hz-lm-1.7B/` and the workflow wiring land in MC5-3b … MC5-W; this file pins what
/// the catalog must say.
@MainActor
struct AceStepCatalogTests {

    private static let hfID = "ACE-Step/Ace-Step1.5"

    private func catalog() throws -> BrowserData {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .deletingLastPathComponent()   // repo root
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data)
    }

    private func entry() throws -> ModelEntry {
        let all = try catalog().domains.flatMap { $0.allModels }
        return try #require(all.first { $0.hfModelId == Self.hfID })
    }

    @Test func entryFactsMatchTheRepo() throws {
        let e = try entry()
        #expect(e.id == "ACE-Step--Ace-Step1.5")
        #expect(e.displayName == "ACE-Step 1.5")
        #expect(e.family == "ace-step")
        #expect(e.modelType == .music)
        #expect(e.source == .mlx)
        #expect(e.hfRepo == "ACE-Step")
        #expect(e.architecture == "AceStepConditionGenerationModel")
        #expect(e.license == "mit")
        #expect(e.lastUpdated == "2026-02-03")
        #expect(e.taskTags == ["text-to-music", "music-generation"])
        #expect(e.communityDownloads == 93_349)
        #expect(e.communityLikes == 937)
        // R3: bf16, nothing quantized; one variant.
        #expect(e.format == "mlx-bf16")
        #expect(e.variants?.count == 1)
        #expect(e.variants?.first?.quantization == "bf16")
        #expect(e.variants?.first?.format == "mlx-bf16")
        #expect(e.variants?.first?.hfModelId == Self.hfID)
        #expect(e.variants?.first?.recommended == true)
        #expect(e.variants?.first?.qualityPercent == 100, "published precision, not a benchmark")
        #expect(e.variants?.first?.ramGB == e.ramGB)
        #expect(e.variants?.first?.downloadSizeGB == e.downloadSizeGB)
    }

    /// The download figure is the bytes `ModelFileSelector` selects from the pinned sibling list (MC5-3b corrected the
    /// MC5-3a figure 6.34, the unfiltered sibling sum). The selection itself is pinned in `AceStepInstallTests`.
    @Test func downloadSizeIsTheSelectedBytes() throws {
        let selected = AceStepInstallTests.selectedBytes()
        #expect(selected == 6_332_732_359)
        let e = try entry()
        #expect(e.downloadSizeGB == 6.33)
        #expect((Double(selected) / 1e9 * 100).rounded() / 100 == e.downloadSizeGB)
        #expect(try catalogNotes().contains("6,332,732,359"))
    }

    /// AM-T (MC5-T run 1, 2026-11-21): `ramGB` is the measured whole-process peak at 240 s, 10,378 MiB = 10.88 GB, rounded up to 0.1 GB. It replaced the AM1 provisional 8.9
    /// (weights 5.897 GB x 1.5). Entry and variant carry the same figure.
    @Test func ramGBIsTheMeasuredPeakRoundedUp() throws {
        let e = try entry()
        let peakGB = 10_378.0 * 1_048_576 / 1e9
        #expect((peakGB * 10).rounded(.up) / 10 == 10.9)
        #expect(e.ramGB == 10.9)
        #expect((e.variants ?? []).allSatisfy { $0.ramGB == 10.9 }, "the variant agrees with the entry")
        #expect(e.ramGB >= peakGB && e.ramGB >= e.downloadSizeGB, "never under the measured peak or the download (CatalogGuardTests floor)")
        #expect(e.description?.contains("provisional estimate") == false)
        #expect(e.description?.contains("RAM figure (10.9 GB) is measured") == true)
        let notes = try catalogNotes()
        #expect(!notes.contains("PROVISIONAL"))
        #expect(notes.contains("ramGB 10.9 is MEASURED (MC5-T run 1") && notes.contains("10,378 MiB"))
    }

    @Test func fieldsThatDoNotApplyAreNil() throws {
        let e = try entry()
        // The LLM weights heuristic and the tokens/s baseline do not describe a music diffusion model.
        #expect(e.speedTokensPerSec == nil)
        #expect(e.speedHardware == nil)
        #expect(e.speedEstimated == nil)
        #expect(e.displaySpeed == "—")
        #expect(e.benchmarks == nil)
        #expect(e.qualityScore == nil)
        #expect(e.paramCountB == nil)
        #expect(e.contextWindow == nil)
        #expect(e.languages == nil)
    }

    /// v1 is a style prompt and optional lyrics to music, with no planner LM. The text must not claim anything that
    /// is not built, and names the bundled Apache-2.0 text encoder (R8). It does not raise the VAE licence question.
    @Test func aboutTextMakesNoClaimBeyondV1() throws {
        let e = try entry()
        let text = ((e.description ?? "") + "\n" + (e.summary ?? "")).lowercased()
        #expect(!text.isEmpty)
        for banned in ["language model", " lm ", "lm-", "planner", "cover", "edit", "repaint", "extract", "lego",
                       "50+", "vocal-to", "chain-of-thought", "10 minute", "10-minute"] {
            #expect(!text.contains(banned), "About text must not mention '\(banned)'")
        }
        #expect(text.contains("qwen3-embedding-0.6b"))
        #expect(text.contains("apache-2.0"))
        #expect(!text.contains("vae licen"))
        #expect(!text.contains("stable audio"))
        // MC5-L (owner, 2026-11-21) forces two changes here: "stability" and "commercial" were banned before the VAE
        // was ruled to ship under the Stability AI Community License. They are now REQUIRED (see the notice test below),
        // and only the exact notice sentences may use them.
        let withoutNotice = text.replacingOccurrences(of: Self.enterpriseNote.lowercased(), with: "")
        #expect(!withoutNotice.contains("commercial"), "commercial appears only inside the enterprise note")
        #expect(text.contains("mit"))
    }

    static let stabilityLine = "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved."
    static let enterpriseNote = "Commercial use by organisations with annual revenue over US $1M may require a Stability AI Enterprise licence for this component."

    private func bundledResource(_ name: String) throws -> Data {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        return try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/AceStep/\(name)"))
    }

    /// MC5-L: the catalog licence stays mit; the notice, About text and byte-pinned licence text carry the Stability terms.
    @Test func stabilityNoticeAndLicenseTextAreShipped() throws {
        let e = try entry()
        #expect(e.license == "mit")
        let notice = try #require(String(data: bundledResource("NOTICE.txt"), encoding: .utf8))
        #expect(notice.contains(Self.stabilityLine))
        #expect(notice.contains("applies to the VAE component of ACE-Step 1.5"))
        #expect(notice.contains("Apache License, Version 2.0 (Apache-2.0). Copyright Alibaba Cloud / Qwen Team."))
        #expect(notice.contains("Qwen3-Embedding-0.6B"))
        let license = try bundledResource("StabilityAICommunityLicense.txt")
        #expect(license.count == 11852)
        #expect(SHA256.hash(data: license).map { String(format: "%02x", $0) }.joined()
                == "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d", "byte-for-byte pin")
        #expect(notice.contains("f21265c1e2710b3bd2386596943f0007f55f802e") && notice.contains("d6f6b1a4dce5c852"), "provenance recorded")
        for text in [e.description ?? "", AceStepModule.descriptor.notes] {
            #expect(text.contains("Powered by Stability AI"))
            #expect(text.contains(Self.enterpriseNote))
        }
        #expect(AceStepModule.descriptor.notes.contains("Powered by Stability AI"))
    }

    /// The bundle (both targets) carries the same files when the test host has them.
    @Test func noticeResourcesAreInTheAppBundle() {
        for name in ["NOTICE", "StabilityAICommunityLicense"] {
            #expect(Bundle.main.url(forResource: name, withExtension: "txt") != nil
                    || Bundle(for: BundleToken.self).url(forResource: name, withExtension: "txt") != nil, "\(name).txt must be bundled")
        }
    }
    private final class BundleToken {}

    /// MC5-T-notice item 3: the model-page disclosure shows the BUNDLED files verbatim, collapsed by default, on ACE-Step only.
    @Test func licenceNoticesDisclosureShowsTheBundledFiles() throws {
        let bundle = Bundle(for: BundleToken.self)
        let host = Bundle.main
        let b = host.url(forResource: "NOTICE", withExtension: "txt") != nil ? host : bundle
        let sections = AceStepLicenceNotices.sections(bundle: b)
        #expect(sections.map(\.title) == ["Notice", "Stability AI Community License"])
        let noticeBytes = try bundledResource("NOTICE.txt")
        let licenseBytes = try bundledResource("StabilityAICommunityLicense.txt")
        #expect(sections[0].text == String(data: noticeBytes, encoding: .utf8))
        #expect(sections[1].text == String(data: licenseBytes, encoding: .utf8))
        #expect(SHA256.hash(data: Data(sections[1].text.utf8)).map { String(format: "%02x", $0) }.joined()
                == "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d")
        #expect(sections[0].text.contains(Self.stabilityLine))
        #expect(sections[0].text.contains("Apache License, Version 2.0 (Apache-2.0)"), "the Qwen Apache-2.0 notice is visibly present")
        #expect(AceStepLicenceNotices.expandedByDefault == false)
        #expect(AceStepLicenceNotices.applies(toModelID: Self.hfID))
        #expect(!AceStepLicenceNotices.applies(toModelID: "jasonvassallo/mlx-musicgen-small"))
        #expect(AceStepLicenceNotices.load("no-such-file", bundle: b) == nil)
    }

    /// MC5-T-vocalsui (owner, 2026-11-21): vocals ship as EXPERIMENTAL, so nothing user-visible about the model may claim or imply voice. The user-visible text is the entry's description, summary, display
    /// name, task tags and variant labels, and the module's About notes (`AceStepModule.descriptor`); the entry's `notes` is a maintainer record that is not decoded into `ModelEntry` and not shown. The
    /// word list is whole words: vocal(s), lyric(s), sing(s) / singing / singer(s), song(s), voice(s), choir. (The `ModelEntry` decode check below is the same text the Browse card renders.)
    @Test func userVisibleTextMakesNoVocalClaim() throws {
        let e = try entry()
        let visible = ([e.description, e.summary, e.displayName, AceStepModule.descriptor.displayName, AceStepModule.descriptor.notes]
                       + (e.taskTags ?? []).map(Optional.some) + (e.variants ?? []).map { $0.quantization }.map(Optional.some)).compactMap { $0 }.joined(separator: "\n").lowercased()
        #expect(!visible.isEmpty)
        // Whole words only ("using" contains "sing").
        for banned in ["vocal", "vocals", "lyric", "lyrics", "sing", "sings", "singing", "singer", "singers", "song", "songs", "voice", "voices", "choir"] {
            let hit = visible.range(of: "\\b\(banned)\\b", options: .regularExpression) != nil
            #expect(!hit, "user-visible ACE-Step text must not contain '\(banned)'")
        }
        #expect(visible.contains("instrumental"), "the claim is instrumental music")
    }

    @Test func liveInTheMusicLeafAppendedLast() throws {
        let data = try catalog()
        let leaf = try #require(data.domains.first { $0.id == "music" })
        #expect(leaf.allModels.map(\.hfModelId) == ["jasonvassallo/mlx-musicgen-small", Self.hfID], "append, never reorder")
        let section = try #require(data.sidebarSections.first { $0.id == "audio" })
        #expect(section.domainIds.contains("music"))
        #expect(data.sidebarSections.filter { $0.domainIds.contains("music") }.map(\.id) == ["audio"])
        #expect(data.domainIndex["music"] != nil)
    }

    /// The MusicGen entry is untouched by MC5-3a.
    @Test func musicGenEntryIsUnchanged() throws {
        let all = try catalog().domains.flatMap { $0.allModels }
        let m = try #require(all.first { $0.hfModelId == "jasonvassallo/mlx-musicgen-small" })
        #expect(m.id == "jasonvassallo--mlx-musicgen-small")
        #expect(m.displayName == "MusicGen-small")
        #expect(m.family == "MusicGen")
        #expect(m.modelType == .music)
        #expect(m.license == "cc-by-nc-4.0")
        #expect(m.format == "mlx-f32")
        #expect(m.ramGB == 3.7)
        #expect(m.downloadSizeGB == 2.5)
        #expect(m.paramCountB == 0.615)
        #expect(m.variants?.count == 1)
        #expect(m.variants?.first?.qualityPercent == 100)
        #expect(m.summary?.hasPrefix("Meta's smallest MusicGen text-to-music model") == true)
    }

    /// `notes` is not decoded into `ModelEntry`; read it from the raw JSON.
    private func catalogNotes() throws -> String {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let data = try Data(contentsOf: repoRoot.appendingPathComponent("MLXUI/Resources/browser.json"))
        let root = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let domains = try #require(root["domains"] as? [[String: Any]])
        let leaf = try #require(domains.first { $0["id"] as? String == "music" })
        let models = try #require(leaf["models"] as? [[String: Any]])
        let m = try #require(models.first { $0["hfModelId"] as? String == Self.hfID })
        return try #require(m["notes"] as? String)
    }
}
