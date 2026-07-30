//
//  ModelStore.swift
//  MLXUI — the single source of truth for on-disk locations.
//
//  Seventeen call sites used to rebuild `Application Support/AI Browser/…` by
//  hand, each with its own `first!` force-unwrap and its own spelling of the
//  layout. That was survivable while there was one app; it isn't now, because
//  the expression resolves to *different places* in the two editions:
//
//    App Store (sandboxed) …/Library/Containers/net.connectcode.mlxui/Data/
//                             Library/Application Support/AI Browser/
//    Direct (un-sandboxed)  ~/Library/Application Support/AI Browser/
//
//  One type means the Direct edition's "adopt the App Store container if it
//  exists, instead of re-downloading 30 GB of weights" behaviour is a single
//  override rather than seventeen. See Design/dual-distribution.md §2 and §6.
//
//  Layout (unchanged — see CLAUDE.md § "Install & storage"):
//
//    installed.json            registry
//    models/{model-id}/        config.json, *.safetensors, tokenizer.json, .installed
//    downloads/{model-id}/     temp, deleted on completion or cancel
//    WhisperKit/               WhisperKit's own CoreML + tokenizer download home
//    agent-index/index.json    semantic_search embedding index
//

import Foundation

nonisolated struct ModelStore: Sendable {

    /// The app-wide instance. Injectable via `init(baseDirectory:)` for tests and
    /// for the direct edition's container-adoption override.
    static let shared = ModelStore()

    static let folderName = "AI Browser"

    /// `…/Application Support/AI Browser`.
    let baseDirectory: URL

    init(baseDirectory: URL? = nil) {
        if let baseDirectory {
            self.baseDirectory = baseDirectory
            return
        }
        // `.first` rather than `first!`: an empty result is vanishingly unlikely
        // but a crash on launch is not a proportionate response to it.
        let appSupport = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        self.baseDirectory = appSupport.appendingPathComponent(
            Self.folderName,
            isDirectory: true
        )
    }

    // MARK: - Well-known locations

    var modelsDirectory: URL {
        baseDirectory.appendingPathComponent("models", isDirectory: true)
    }

    var downloadsDirectory: URL {
        baseDirectory.appendingPathComponent("downloads", isDirectory: true)
    }

    var installedRegistryURL: URL {
        baseDirectory.appendingPathComponent("installed.json")
    }

    /// WhisperKit insists on its own writable home for the CoreML model *and* the
    /// tokenizer config — see the note in `WhisperEngine`.
    var whisperKitDirectory: URL {
        baseDirectory.appendingPathComponent("WhisperKit", isDirectory: true)
    }

    /// Persisted embedding index for the `semantic_search` agent tool.
    var agentIndexDirectory: URL {
        baseDirectory.appendingPathComponent("agent-index", isDirectory: true)
    }

    // MARK: - Per-model

    /// Where `InstallManager` places a model's files. `id` is the catalog id with
    /// `/` already replaced by `--` (e.g. `mlx-community--Qwen3-4B-4bit`).
    func directory(forModelID id: String) -> URL {
        modelsDirectory.appendingPathComponent(id, isDirectory: true)
    }

    /// Scratch directory a download writes into before the atomic move.
    func downloadDirectory(forModelID id: String) -> URL {
        downloadsDirectory.appendingPathComponent(id, isDirectory: true)
    }

    /// The `.installed` marker — the atomic "install succeeded" signal the launch
    /// reconciliation checks the registry against.
    func installedMarker(forModelID id: String) -> URL {
        directory(forModelID: id).appendingPathComponent(".installed")
    }

    /// A loose file directly under the base directory (demo output and similar).
    func file(named name: String) -> URL {
        baseDirectory.appendingPathComponent(name)
    }
}
