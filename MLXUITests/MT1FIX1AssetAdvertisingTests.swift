//
//  MT1FIX1AssetAdvertisingTests.swift
//  MT1-FIX-1: a model tool advertises `asset` only when constructed for a host-bound chat.
//

import Testing
import Foundation
import MLXLMCommon
@testable import MLXUI

@Suite("MT1FIX1 asset advertising")
struct MT1FIX1AssetAdvertisingTests {
    private func properties(_ tool: ModelTool) -> [String: Any] {
        let fn = tool.toolSpec["function"] as? [String: any Sendable]
        let params = fn?["parameters"] as? [String: any Sendable]
        return params?["properties"] as? [String: Any] ?? [:]
    }

    /// Everything the router reads for one tool, as one string.
    private func text(_ tool: ModelTool) -> String {
        tool.toolDescription + String(describing: tool.toolSpec["function"] ?? "")
    }

    @Test func noHostModeNeverMentionsAsset() {
        for tool in ModelTool.all(advertisingAssets: false) where tool.name != "read_asset" {
            #expect(properties(tool)["asset"] == nil, "\(tool.name) advertises asset")
            #expect(!tool.parameters.map(\.name).contains("asset"), "\(tool.name) parameters")
            #expect(!text(tool).lowercased().contains("asset"), "\(tool.name) schema text mentions asset")
        }
    }

    @Test func theDefaultIsNoHostMode() {
        #expect(ModelTool.all.map { $0.advertisesAssets } == Array(repeating: false, count: ModelTool.all.count))
        #expect(ModelTool(spec: ModelToolSpec.all[0]).advertisesAssets == false)
    }

    @Test func noHostModeKeepsPathUrlAndText() throws {
        let tools = ModelTool.all(advertisingAssets: false)
        let transcribe = try #require(tools.first { $0.name == "transcribe_audio" })
        #expect(Set(transcribe.parameters.map(\.name)) == ["path", "url", "language"])
        let speak = try #require(tools.first { $0.name == "speak" })
        #expect(Set(speak.parameters.map(\.name)) == ["text", "voice"])
    }

    @Test func hostModeAdvertisesAssetOnEveryInputTool() throws {
        let tools = ModelTool.all(advertisingAssets: true)
        for tool in tools {
            let inputNames = ModelToolRunner.inputArgumentNames(for: tool.spec)
            if inputNames.contains("asset") {
                #expect(properties(tool)["asset"] != nil, "\(tool.name) should advertise asset")
            }
        }
        let transcribe = try #require(tools.first { $0.name == "transcribe_audio" })
        #expect(Set(transcribe.parameters.map(\.name)) == ["asset", "path", "url", "language"])
        let speak = try #require(tools.first { $0.name == "speak" })
        #expect(Set(speak.parameters.map(\.name)) == ["text", "asset", "voice"])
        let asset = try #require(properties(transcribe)["asset"] as? [String: Any])
        #expect((asset["description"] as? String)?.contains("asset") == true)
    }

    @Test func readAssetIsUnchangedInBothModes() throws {
        for mode in [false, true] {
            let tool = try #require(ModelTool.all(advertisingAssets: mode).first { $0.name == "read_asset" })
            #expect(properties(tool)["asset"] != nil)
        }
    }

    @Test func bothModesListTheSameTools() {
        #expect(ModelTool.all(advertisingAssets: false).map(\.name) == ModelTool.all(advertisingAssets: true).map(\.name))
    }

    @Test func executionStillRefusesAHandleWithoutAHost() async throws {
        // Unchanged execute path: a no-asset schema does not change what execute accepts.
        let tool = try #require(ModelTool.all(advertisingAssets: false).first { $0.name == "transcribe_audio" })
        let out = try await tool.execute(arguments: ["asset": .string("a1")])
        #expect(!out.isEmpty)
    }

    @Test func theChatRegisteredBackCompatToolsFollowTheSameMode() {
        // ModelRunner registers these two; default = no host = no `asset`.
        for tool: any AgentTool in [TranscribeAudioTool(), OCRImageTool()] {
            let names = tool.parameters.map(\.name)
            #expect(!names.contains("asset") && names.contains("url") && names.contains("path"), "\(tool.name)")
            #expect(!String(describing: tool.toolSpec["function"] ?? "").lowercased().contains("asset"), "\(tool.name)")
        }
        for tool: any AgentTool in [TranscribeAudioTool(advertisesAssets: true), OCRImageTool(advertisesAssets: true)] {
            let names = tool.parameters.map(\.name)
            #expect(names.contains("asset") && names.contains("url") && names.contains("path"), "\(tool.name)")
        }
        #expect(TranscribeAudioTool().name == "transcribe_audio" && OCRImageTool().name == "ocr_image")
    }
}
