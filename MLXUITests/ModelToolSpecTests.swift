import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

/// MT1-1 — drift tests for `ModelToolSpec` (`RSI/DelegateModelToolsBacklog.md`).
struct ModelToolSpecTests {

    private func catalog() throws -> [ModelEntry] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("MLXUI/Resources/browser.json"))
        return try JSONDecoder().decode(BrowserData.self, from: data).domains.flatMap { $0.allModels }
    }

    @Test func namesAreUnique() {
        let names = ModelToolSpec.all.map(\.name)
        #expect(Set(names).count == names.count)
        #expect(Set(names) == ["transcribe_audio", "ocr_image", "describe_image", "speak", "translate", "generate_image",
                               "upscale_image", "segment_image", "estimate_depth", "generate_music",
                               "list_capabilities", "read_asset"])
    }

    @Test func everyTaskIsServedWithARunnerKindAndMatchingKinds() throws {
        for spec in ModelToolSpec.all {
            // list_capabilities and read_asset run no model (task nil): skipped here on purpose, pinned by the assertion below.
            guard let task = spec.task else { continue }
            let desc = try #require(TaskCatalog.get(task), "\(spec.name): \(task) is not a task")
            #expect(TaskModels.isServedByExecutor(task), "\(spec.name)")
            #expect(TaskModels.runnerKind(forTask: task) != nil, "\(spec.name)")
            #expect(desc.accepts == spec.input.map { Shape.single($0) }, "\(spec.name) input kind drifted from the descriptor")
            #expect(desc.gives == Shape.single(spec.output), "\(spec.name) output kind drifted from the descriptor")
        }
        #expect(ModelToolSpec.all.filter { $0.task == nil }.map(\.name).sorted() == ["list_capabilities", "read_asset"])
    }

    @Test func descriptionsAreShort() {
        for spec in ModelToolSpec.all {
            #expect(spec.description.count <= ModelToolSpec.maxDescriptionLength, "\(spec.name): \(spec.description.count)")
            #expect(!spec.description.isEmpty)
            for p in spec.params {
                #expect(p.description.count <= ModelToolSpec.maxDescriptionLength, "\(spec.name).\(p.name)")
            }
        }
    }

    @Test func noModelNamesAnywhereTheRouterReads() throws {
        let entries = try catalog()
        let fragments = ["whisper", "parakeet", "kokoro", "flux", "glm", "paddle", "sam", "depth anything", "seedvr", "musicgen",
                         "voxtral", "qwen", "gemma", "llama", "wan", "lfm", "sdxl", "z-image", "olmocr", "dots", "deepseek",
                         "mlx-community", "huggingface"]
        var texts: [(String, String)] = []
        for s in ModelToolSpec.all {
            texts.append((s.name, s.name))
            texts.append((s.name, s.description))
            for p in s.params { texts.append(("\(s.name).\(p.name)", p.name + " " + p.description)) }
        }
        for (label, text) in texts {
            let lower = text.lowercased()
            for fragment in fragments {
                let pattern = "(?<![a-z0-9])" + NSRegularExpression.escapedPattern(for: fragment) + "(?![a-z0-9])"
                #expect(lower.range(of: pattern, options: .regularExpression) == nil, "\(label) mentions '\(fragment)'")
            }
            for entry in entries {
                #expect(!lower.contains(entry.id.lowercased()) && !lower.contains(entry.hfModelId.lowercased()),
                        "\(label) names \(entry.id)")
                #expect(!lower.contains(TaskModels.displayName(for: entry).lowercased()), "\(label) names \(entry.displayName)")
            }
        }
    }

    @Test func transcribeDescriptionCarriesTheLanguageRule() throws {
        let spec = try #require(ModelToolSpec.spec(named: "transcribe_audio"))
        #expect(spec.description.contains("Set language to the spoken language unless it is English."))
        let language = try #require(spec.param(named: "language"))
        #expect(language.placement == .setting(rowKey: "lang"))
        #expect(!language.required)
    }

    @Test func translateRequiresToAndUsesTheDedicatedPool() throws {
        let spec = try #require(ModelToolSpec.spec(named: "translate"))
        #expect(spec.param(named: "to")?.required == true)
        #expect(spec.missingRequired(in: []).map(\.name) == ["to"])
        #expect(spec.kinds == [.translate])
        #expect(ModelToolSpec.all.filter { $0.kinds != nil }.map(\.name) == ["translate"])
    }

    @Test func paramPlacementsMatchTheRowKeys() throws {
        #expect(ModelToolSpec.spec(named: "describe_image")?.param(named: "question")?.placement == .rowPrompt)
        #expect(ModelToolSpec.spec(named: "speak")?.param(named: "voice")?.placement == .setting(rowKey: "voice"))
        let image = try #require(ModelToolSpec.spec(named: "generate_image"))
        #expect(image.paramNames == ["width", "height", "steps", "seed"])
        for p in image.params { #expect(p.placement == .setting(rowKey: p.name)); #expect(!p.required) }
        // The text to speak / draw / translate is the input asset, never a row parameter.
        for name in ["speak", "translate", "generate_image", "generate_music"] {
            #expect(ModelToolSpec.spec(named: name)?.param(named: "text") == nil)
            #expect(ModelToolSpec.spec(named: name)?.param(named: "prompt") == nil)
            #expect(ModelToolSpec.spec(named: name)?.input == .text)
        }
    }

    @Test func rowKeysAreKeysTheExecutorReads() throws {
        // Each setting key must be one `RealExecutor.stageConfig` actually reads for that task, so a drift fails here.
        let exec = RealExecutor(workspace: FlowWorkspace(root: FileManager.default.temporaryDirectory), flowID: "mt1-1",
                                blobDirectory: FileManager.default.temporaryDirectory,
                                makeModelStage: { _, _ in throw StageError.modelNotInstalled(id: "x") },
                                installedModelIDs: [], catalog: [])
        let probes: [(String, String, (StageConfig) -> Bool)] = [
            ("transcribe_audio", "lang=ms", { $0.language == "ms" }),
            ("speak", "voice=bf_emma", { $0.voice == "bf_emma" }),
            ("generate_image", "seed=7", { $0.seed == 7 }),
            ("generate_image", "width=512", { $0.width == 512 }),
            ("generate_image", "height=256", { $0.height == 256 }),
            ("generate_image", "steps=4", { $0.steps == 4 }),
            ("describe_image", "\"Is it red?\"", { $0.prompt == "Is it red?" }),
        ]
        for (tool, settings, check) in probes {
            let spec = try #require(ModelToolSpec.spec(named: tool))
            let task = try #require(spec.task)
            let desc = try #require(TaskCatalog.get(task))
            let config = try exec.stageConfig(for: desc, row: Row(task: desc.name, model: "m", settings: settings))
            #expect(check(config), "\(tool) \(settings)")
        }
        // Spec keys all appear in the probes (a new placement must come with a probe).
        let probed = Set(probes.map { "\($0.0)|\($0.1)" })
        #expect(probed.count == probes.count)
    }

    @Test func unknownArgumentsAreReported() throws {
        let spec = try #require(ModelToolSpec.spec(named: "generate_image"))
        #expect(spec.unknownArguments(in: ["seed", "seconds", "prompt"]) == ["seconds", "prompt"])
        #expect(ModelToolSpec.spec(named: "ocr_image")?.unknownArguments(in: ["language"]) == ["language"])
    }

    @Test func generateImageLimits() throws {
        let image = try #require(ModelToolSpec.spec(named: "generate_image"))
        #expect(image.param(named: "steps")?.type == .integer(min: 1, max: 50))
        for name in ["width", "height"] {
            #expect(image.param(named: name)?.type == .integer(min: 1, max: nil), "\(name) has no upper bound")
        }
    }

    @Test func droppedThingsStayDropped() {
        #expect(ModelToolSpec.spec(named: "generate_video") == nil)
        #expect(!ModelToolSpec.all.contains { $0.task == "Generate Video" })
        for spec in ModelToolSpec.all {
            #expect(spec.param(named: "seconds") == nil, "\(spec.name)")
            if spec.name == "translate" { #expect(spec.param(named: "from") == nil) }
            #expect(!spec.description.lowercased().contains("video"), "\(spec.name)")
        }
    }

    // HT-FIX-music-cue (R2): the clip is about 30 s, and a request for another length is answered, never dropped.
    @Test func generateMusicSaysItMakesAboutThirtySeconds() {
        #expect(ModelToolSpec.spec(named: "generate_music")?.description == "Make a short (about 30-second) instrumental clip from a text prompt.")
        #expect(ModelToolSpec.spec(named: "generate_music")?.params.isEmpty == true)
    }

    @Test func aRequestedLengthGetsTheThirtySecondLine() throws {
        let spec = try #require(ModelToolSpec.spec(named: "generate_music"))
        for text in ["guitar in rain for 15 seconds", "a 2 min lofi loop", "rain 45s", "10 secs of piano"] {
            #expect(ModelToolRunner.lengthNote(for: spec, arguments: ["text": .string(text)])
                    == "ACE-Step makes a 30-second clip; the requested length isn't supported yet.", "\(text)")
        }
        for text in ["guitar in rain", "a 4/4 drum loop", "top 10 songs"] {
            #expect(ModelToolRunner.lengthNote(for: spec, arguments: ["text": .string(text)]) == nil, "\(text)")
        }
        let speak = try #require(ModelToolSpec.spec(named: "speak"))
        #expect(ModelToolRunner.lengthNote(for: speak, arguments: ["text": .string("for 15 seconds")]) == nil)
    }
}
